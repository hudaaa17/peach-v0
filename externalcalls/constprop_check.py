"""Conservative same-function and same-file constant propagation for URLs."""

from .ast_access import _walk, find_node_at, load_tree
from .ast_access import find_call_node, get_url_argument
from .ast_grep_check import CATALOGUE
from .edge_pairing import pair_edges_with_calls
from .url_eval import decimal_integer, evaluate, host_from_text, redact_userinfo
from .ast_access import unwrap


MAX_CHAIN_DEPTH = 3
_FUNCTIONS = {"function_definition", "lambda", "function_declaration", "function_expression",
              "arrow_function", "method_definition", "generator_function_declaration", "generator_function"}
_BLOCKS = {"block", "statement_block", "module", "program"}


def _scope(node):
    while node.parent is not None and node.type not in _FUNCTIONS:
        node = node.parent
    return node


def _contains(outer, inner):
    return outer.start_byte <= inner.start_byte and inner.end_byte <= outer.end_byte


def _targets(node):
    if node is None:
        return []
    if node.type in ("identifier", "type_identifier", "shorthand_property_identifier_pattern"):
        return [node]
    field = {"pair_pattern": "value", "assignment_pattern": "left", "object_assignment_pattern": "left",
             "default_parameter": "name", "typed_default_parameter": "name", "required_parameter": "pattern"}.get(node.type)
    if field:
        return _targets(node.child_by_field_name(field))
    if node.type == "typed_parameter":
        return _targets(node.named_children[0])
    if node.type in ("pattern_list", "list_pattern", "tuple_pattern", "object_pattern", "array_pattern",
                     "rest_pattern", "list_splat_pattern", "dictionary_splat_pattern", "as_pattern_target"):
        return [target for child in node.named_children for target in _targets(child)]
    return []


def _bindings(root, source):
    """Collect binding sites, retaining nested-function sites conservatively."""
    result = {}
    def text(node):
        return source[node.start_byte:node.end_byte].decode("utf-8", "ignore")
    def add(target, node, kind, value=None, scope=None):
        for leaf in _targets(target):
            result.setdefault(text(leaf), []).append({"node": node, "kind": kind,
                                                      "value": value, "scope": scope or _scope(node)})
    for node in _walk(root):
        kind = node.type
        if kind in ("assignment", "variable_declarator", "assignment_expression",
                    "augmented_assignment", "augmented_assignment_expression", "named_expression"):
            target = node.child_by_field_name("name" if kind in ("variable_declarator", "named_expression") else "left")
            value = node.child_by_field_name("value" if kind in ("variable_declarator", "named_expression") else "right")
            simple = target is not None and target.type == "identifier" and kind in (
                "assignment", "variable_declarator", "assignment_expression")
            if value is not None and value.type in ("assignment", "assignment_expression"):
                simple = False
            if node.parent is not None and node.parent.type in ("assignment", "assignment_expression"):
                simple = False
            if kind == "assignment_expression" and node.parent.type != "expression_statement":
                simple = False
            imported = (kind == "variable_declarator" and value is not None
                        and value.type == "call_expression"
                        and text(value.child_by_field_name("function")) == "require")
            add(target, node, "imported_constant" if imported else "simple" if simple else "complex_binding", value)
        elif kind in ("for_statement", "for_in_statement"):
            add(node.child_by_field_name("left"), node, "complex_binding")
        elif kind == "as_pattern":
            add(node.child_by_field_name("alias"), node, "complex_binding")
        elif kind == "catch_clause":
            add(node.child_by_field_name("parameter"), node, "complex_binding")
        elif kind in ("global_statement", "nonlocal_statement", "delete_statement"):
            for child in node.named_children:
                add(child, node, "complex_binding")
        if kind in _FUNCTIONS:
            params = node.child_by_field_name("parameters")
            if params is not None:
                for param in params.named_children:
                    add(param, node, "parameter", scope=node)
            else:
                add(node.child_by_field_name("parameter"), node, "parameter", scope=node)
        if kind in _FUNCTIONS | {"class_definition", "class_declaration"}:
            add(node.child_by_field_name("name"), node, "complex_binding", scope=_scope(node.parent))
        if kind in ("import_statement", "import_from_statement"):
            if node.child_by_field_name("source") is None:
                for i, child in enumerate(node.children):
                    if node.field_name_for_child(i) == "name":
                        target = child.child_by_field_name("alias") if child.type == "aliased_import" else None
                        if target is None:
                            name = child.child_by_field_name("name") if child.type == "aliased_import" else child
                            target = name.named_children[0] if name and name.named_children else name
                        add(target, node, "imported_constant")
            else:
                for child in _walk(node):
                    if child.type == "import_specifier":
                        add(child.child_by_field_name("alias") or child.child_by_field_name("name"), node, "imported_constant")
                    elif child.type == "namespace_import":
                        for target in child.named_children:
                            add(target, node, "imported_constant")
                    elif child.type == "import_clause":
                        for target in child.named_children:
                            if target.type == "identifier":
                                add(target, node, "imported_constant")
    return result


def _statement(node):
    while node.parent is not None and node.parent.type not in _BLOCKS:
        node = node.parent
    return node


def _block(node):
    while node.parent is not None and node.type not in _BLOCKS:
        node = node.parent
    return node


def run_constprop_check(edges, records, repo_root, literal_report, ctx=None):
    report = {"aborted": False, "edges_total": 0, "edges_considered": 0, "outcomes": {},
              "samples": {}, "reasons": [], "resolved_edges": {}, "still_pending": {},
              "env_refs": {}, "imported_constants": {}, "userinfo_redacted": 0}
    try:
        calls, problem = pair_edges_with_calls(edges, records)
        report["edges_total"] = len(edges)
        if problem:
            report["aborted"] = True
            report["reasons"].append(problem)
            return report
        languages = {rec.path: rec.language for rec in records}
        pending = literal_report.get("pending_constprop", {})
        cache = {}
        fields = ("literal", "literal_method", "constprop_via", "host", "needs_review", "review_reason")
        # All edges are accounted for, including those outside the pending side table.
        for index, edge in enumerate(edges):
            report["edges_considered"] += 1
            before = None
            resolved = None
            redacted = False
            failure = None
            chain = []
            env_refs = []
            imported = []
            try:
                before = {field: getattr(edge, field) for field in fields}
                changes = {}
                if (index not in pending or edge.status != "external_candidate"
                        or (edge.host is not None and edge.review_reason != "port_boundary_host")):
                    outcome = "skipped_not_pending"
                else:
                    tree, source, error = load_tree(repo_root, edge.file, cache)
                    if error:
                        outcome = error
                    else:
                        info = pending[index]
                        # Phase 3B stores a position, so recover its unique expression type at that position.
                        argument = None
                        call = calls[index]
                        call_node = find_call_node(tree, call.callee_line, call.callee_col)
                        if not isinstance(call_node, str):
                            entry = next((e for e in CATALOGUE if e.id == edge.external_pattern), None)
                            if entry is not None:
                                candidate, error = get_url_argument(call_node, entry, languages[edge.file])
                                if candidate is not None:
                                    argument = find_node_at(tree, info["arg_line"], info["arg_col"], candidate.type)
                                    # Nested left-associative '+' nodes can share type and start.
                                    # The catalogue-selected outer argument disambiguates that position.
                                    if tuple(candidate.start_point) == (info["arg_line"] - 1, info["arg_col"]):
                                        if argument is None or argument == candidate:
                                            argument = candidate
                                    else:
                                        argument = None
                        if argument is None:
                            outcome = "argument_not_found"
                        else:
                            bindings = _bindings(tree.root_node, source)
                            active = []
                            resolved_names = []
                            language = languages[edge.file]
                            def fail(kind, detail=None):
                                nonlocal failure
                                if failure is None:
                                    failure = {"kind": kind, "detail": detail}
                                return None
                            def resolve(name, use, depth):
                                nonlocal failure
                                if name in active:
                                    return fail("cycle")
                                if depth > MAX_CHAIN_DEPTH:
                                    return fail("chain_too_deep")
                                scope = _scope(use)
                                sites = bindings.get(name, [])
                                local = [s for s in sites if _contains(scope, s["node"])
                                         and not (s["node"] == scope and s["kind"] != "parameter")]
                                if any(s["kind"] == "parameter" and s["scope"] == scope for s in local):
                                    return fail("parameter")
                                if any(s["node"].type in ("global_statement", "nonlocal_statement", "delete_statement") for s in local):
                                    return fail("complex_binding")
                                choices = local if local else sites
                                if any(s["kind"] == "imported_constant" for s in choices):
                                    imported.append(name)
                                    return fail("imported_constant")
                                if not choices:
                                    return fail("not_found")
                                if len(choices) > 1:
                                    return fail("multiple_assignments")
                                site = choices[0]
                                statement = _statement(site["node"])
                                if not local and statement.parent != tree.root_node:
                                    # Captured outer-function variables are deliberately outside scope.
                                    if site["scope"] != tree.root_node and _contains(site["scope"], scope):
                                        return fail("not_found")
                                    return fail("not_module_level")
                                if site["kind"] != "simple":
                                    return fail("complex_binding")
                                value = site["value"]
                                if value is None:
                                    return fail("no_value")
                                active.append(name)
                                chain.append(name)
                                typed_value = None
                                try:
                                    number = decimal_integer(value, source, language)
                                    if number is not None:
                                        typed_value = int(number)
                                        result = {"text": number, "complete": True, "blocked_by": None}
                                    elif unwrap(value).type == "identifier":
                                        nested = source[unwrap(value).start_byte:unwrap(value).end_byte].decode("utf-8", "ignore")
                                        typed_value = resolve(nested, value, depth + 1)
                                        result = {"text": str(typed_value) if typed_value is not None else "",
                                                  "complete": typed_value is not None, "blocked_by": None}
                                    else:
                                        result = evaluate(value, source, language,
                                                          lambda nested: resolve(nested, value, depth + 1))
                                finally:
                                    active.pop()
                                if not result["complete"]:
                                    blocked = result["blocked_by"]
                                    if blocked and blocked["kind"] == "env_var":
                                        env_refs.append({"var": blocked["name"]})
                                    if result["text"] and (failure is None or failure["kind"] in (
                                            "not_found", "value_not_static", "no_host_after_constprop")):
                                        value_host, value_reason = host_from_text(result["text"], False)
                                        if value_host is None:
                                            failure = {"kind": "no_host_after_constprop", "detail": value_reason}
                                            return None
                                    return fail("value_not_static", blocked["kind"] if blocked else None)
                                if site["scope"] == scope:
                                    if site["node"].end_byte > use.start_byte:
                                        return fail("assigned_after_use")
                                    if not _contains(_block(statement.parent), _block(use)):
                                        return fail("conditional_assignment")
                                resolved_names.append(name)
                                return typed_value if typed_value is not None else result["text"]
                            result = evaluate(argument, source, language,
                                              lambda name: resolve(name, argument, 1))
                            host, reason = host_from_text(result["text"], result["complete"])
                            if host and (edge.review_reason != "port_boundary_host" or resolved_names):
                                text = redact_userinfo(result["text"])
                                redacted = text != result["text"]
                                changes = {"literal": text, "literal_method": "const_prop", "constprop_via": "ast",
                                           "host": host, "needs_review": not result["complete"],
                                           "review_reason": None if result["complete"] else reason or "partial_url"}
                                resolved = {"literal": text, "host": host, "partial": not result["complete"], "chain": chain}
                                outcome = "full_url" if result["complete"] else "partial_url"
                                if reason == "port_boundary_host":
                                    outcome = reason
                            elif result["text"] and resolved_names and (failure is None or failure["kind"] in (
                                    "not_found", "value_not_static", "no_host_after_constprop")):
                                outcome = "no_host_after_constprop"
                                failure = {"kind": outcome, "detail": reason}
                            elif failure:
                                outcome = failure["kind"]
                            else:
                                outcome = "no_host_after_constprop"
                                failure = {"kind": outcome, "detail": reason}
                for field, value in changes.items():
                    setattr(edge, field, value)
            except Exception as exc:
                if before is not None:
                    for field, value in before.items():
                        setattr(edge, field, value)
                outcome = "edge_failed"
                failure = {"kind": outcome, "detail": f"{type(exc).__name__}: {exc}"}
                resolved = None
                redacted = False
                env_refs = []
                imported = []
            if resolved is not None:
                report["resolved_edges"][index] = resolved
            elif index in pending:
                report["still_pending"][index] = outcome
            if env_refs:
                report["env_refs"][index] = env_refs[0]
            if imported:
                report["imported_constants"][index] = imported[0]
            if failure:
                report["reasons"].append({"cause": failure["kind"], "detail": failure["detail"],
                                          "sample": f"{edge.file}:{edge.line}"})
            report["userinfo_redacted"] += redacted
            report["outcomes"][outcome] = report["outcomes"].get(outcome, 0) + 1
            samples = report["samples"].setdefault(outcome, [])
            sample = f"{edge.file}:{edge.line}"
            if sample not in samples and len(samples) < 5:
                samples.append(sample)
    except Exception as exc:
        report["aborted"] = True
        report["reasons"].append({"cause": "stage_failed", "detail": f"{type(exc).__name__}: {exc}"})
    try:
        if ctx is not None:
            ctx.bump("constprop_check.edges_considered", report["edges_considered"])
            for outcome, count in report["outcomes"].items():
                ctx.bump(f"constprop_check.outcome.{outcome}", count)
    except Exception:
        pass
    return report
