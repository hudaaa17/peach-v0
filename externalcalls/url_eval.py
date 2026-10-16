"""Conservative static URL prefixes, without decoding source escapes."""

import re

from .ast_access import unwrap


def decimal_integer(node, source_bytes, language):
    node = unwrap(node)
    if node.type != ("integer" if language == "python" else "number"):
        return None
    text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", "ignore")
    return text if re.fullmatch(r"[0-9]+", text) else None


def evaluate(node, source_bytes, language, resolve_name=None):
    numeric = set()
    def spelling(n):
        return source_bytes[n.start_byte:n.end_byte].decode("utf-8", "ignore")

    def stop(n, kind, name=None):
        return {"text": "", "complete": False, "blocked_by": {
            "kind": kind, "name": name, "line": n.start_point[0] + 1,
            "col": n.start_point[1]}}

    def static(text):
        return {"text": text, "complete": True, "blocked_by": None}

    def sequence(nodes):
        text = ""
        for part in nodes:
            result = visit(part)
            text += result["text"]
            if not result["complete"]:
                return dict(result, text=text)
        return static(text)

    def member(n):
        if n is None:
            return None
        if n.type == "identifier":
            return spelling(n)
        if n.type in ("attribute", "member_expression"):
            left = member(n.child_by_field_name("object"))
            right = n.child_by_field_name("attribute" if n.type == "attribute" else "property")
            return left + "." + spelling(right) if left and right else None
        return None

    def env(n):
        key = None
        matched = False
        if n.type in ("subscript", "subscript_expression"):
            base = n.child_by_field_name("value" if n.type == "subscript" else "object")
            matched = member(base) in ("os.environ", "process.env")
            key = n.child_by_field_name("subscript" if n.type == "subscript" else "index")
        elif n.type in ("call", "call_expression"):
            matched = member(n.child_by_field_name("function")) in ("os.environ.get", "os.getenv")
            args = n.child_by_field_name("arguments")
            key = args.named_children[0] if args and args.named_children else None
        elif n.type == "member_expression" and member(n.child_by_field_name("object")) == "process.env":
            return True, spelling(n.child_by_field_name("property"))
        name = None
        if matched and key is not None and key.type == "string":
            result = visit(key)
            start = next((c for c in key.children if c.type == "string_start"), None)
            if result["complete"] and (start is None or "f" not in spelling(start).lower()):
                name = result["text"]
        return matched, name

    def visit(n, number_context=False):
        n = unwrap(n)
        position = (n.start_byte, n.end_byte)
        number = decimal_integer(n, source_bytes, language)
        if number_context and number is not None:
            numeric.add(position)
            return static(number)
        matched, name = env(n)
        if matched:
            return stop(n, "env_var", name)
        kind = n.type
        if kind in ("identifier", "shorthand_property_identifier"):
            name = spelling(n)
            if resolve_name is None:
                return stop(n, "identifier", name)
            value = resolve_name(name)
            if type(value) is int:
                if number_context:
                    numeric.add(position)
                    return static(str(value))
                return stop(n, "other_expression")
            return static(value) if isinstance(value, str) else stop(n, "identifier_unresolved", name)
        if kind == "concatenated_string":
            return sequence(n.named_children)
        if kind in ("binary_operator", "binary_expression"):
            operator = n.child_by_field_name("operator")
            if operator is not None and spelling(operator) == "+":
                if language != "python":
                    left, right = n.child_by_field_name("left"), n.child_by_field_name("right")
                    first = visit(left, True)
                    if not first["complete"]:
                        return first
                    second = visit(right, True)
                    if ((unwrap(left).start_byte, unwrap(left).end_byte) in numeric
                            and (unwrap(right).start_byte, unwrap(right).end_byte) in numeric):
                        return stop(n, "other_expression")
                    return dict(second, text=first["text"] + second["text"])
                return sequence([n.child_by_field_name("left"), n.child_by_field_name("right")])
        if kind == "call" and language == "python":
            function = n.child_by_field_name("function")
            arguments = n.child_by_field_name("arguments")
            args = [c for c in arguments.named_children if c.type != "comment"] if arguments else []
            if (function is not None and function.type == "identifier" and spelling(function) == "str"
                    and len(args) == 1 and args[0].type in ("identifier", "integer")):
                return visit(args[0], True)
        if kind in ("string", "template_string"):
            if language == "python":
                start = next(c for c in n.children if c.type == "string_start")
                if "b" in spelling(start).lower():
                    return stop(n, "bytes_literal")
            return sequence([c for c in n.named_children if c.type not in ("string_start", "string_end")])
        if kind in ("string_content", "string_fragment"):
            raw = spelling(n)
            cut = raw.find("\\")
            end = n.end_byte if cut < 0 else n.start_byte + len(raw[:cut].encode("utf-8"))
            parts = []
            cursor = n.start_byte
            for child in n.named_children:
                if child.type == "escape_interpolation" and child.end_byte <= end:
                    parts.append(source_bytes[cursor:child.start_byte].decode("utf-8", "ignore"))
                    parts.append(spelling(child)[0])
                    cursor = child.end_byte
            parts.append(source_bytes[cursor:end].decode("utf-8", "ignore"))
            prefix = "".join(parts)
            if cut >= 0:
                return dict(stop(n, "escape_in_string"), text=prefix)
            return static(prefix)
        if kind == "escape_sequence":
            return stop(n, "escape_in_string")
        if kind in ("interpolation", "template_substitution"):
            expression = n.child_by_field_name("expression") if kind == "interpolation" else (
                n.named_children[0] if len(n.named_children) == 1 else None)
            if (expression is not None and len(n.named_children) == 1
                    and (expression.type == "identifier"
                         or decimal_integer(expression, source_bytes, language) is not None)):
                return visit(expression, True)
            return stop(n, "other_expression")
        return stop(n, "member_access" if kind in ("attribute", "member_expression") else
                    "call_expression" if kind in ("call", "call_expression") else "other_expression")

    return visit(node)


def redact_userinfo(text):
    """Remove only userinfo visible in the static authority."""
    scheme = next((s for s in ("http://", "https://", "ws://", "wss://") if text.startswith(s)), None)
    if scheme is None:
        return text
    remainder = text[len(scheme):]
    ends = [remainder.index(c) for c in "/?#" if c in remainder]
    end = min(ends) if ends else len(remainder)
    at = remainder.rfind("@", 0, end)
    return scheme + remainder[at + 1:] if at >= 0 else text


def host_from_text(text, complete):
    scheme = next((s for s in ("http://", "https://", "ws://", "wss://") if text.startswith(s)), None)
    if scheme is None:
        return None, "relative_url"
    remainder = text[len(scheme):]
    ends = [remainder.index(c) for c in "/?#" if c in remainder]
    if not ends and not complete:
        authority = remainder.rsplit("@", 1)[-1]
        if "[" in authority or "]" in authority:
            return None, "unsupported_authority"
        if re.fullmatch(r"[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?:", authority):
            return authority[:-1].lower(), "port_boundary_host"
        return None, "host_incomplete"
    authority = remainder[:min(ends)] if ends else remainder
    if authority.count("@") > 1:
        return None, "unsupported_authority"
    authority = authority.rsplit("@", 1)[-1]
    if ":" in authority:
        authority, port = authority.rsplit(":", 1)
        if not port or any(c not in "0123456789" for c in port):
            return None, "unsupported_authority"
    labels = authority.split(".")
    if not authority or any(not label or label.startswith("-") or label.endswith("-")
                            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for c in label)
                            for label in labels):
        return None, "unsupported_authority"
    return authority.lower(), None
