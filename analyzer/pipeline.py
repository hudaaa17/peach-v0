import shutil
import time
from pathlib import Path

from .clone import clone_repo, CloneError
from .parser import parse_repo
from .resolver import resolve_calls
from .external_calls import flag_external_calls
from .literal_check import check_literals
from .constprop_check import check_constant_propagation
from .config_check import check_config_references
from .joern_check import check_joern_escalation
from .llm_fallback import check_llm_fallback
from .obs import RunContext, get_logger, log, run_context, stage

LOG = get_logger("pipeline")


def _import_target_file(raw: str, all_paths: list):
    """Best-effort match of an import string to a file in the repo."""
    tail = raw.replace(".", "/").strip("/").split("/")[-1]
    if not tail:
        return None
    for p in all_paths:
        stem = Path(p).stem
        if stem == tail:
            return p
    return None


def analyze_repo(url: str, progress=None, run_id: str = None) -> dict:
    """Run the full extraction pipeline.

    Every stage is wrapped in `obs.stage(...)`, which logs entry/exit,
    records wall-clock duration, and attributes any exception to the stage
    that raised it. Stage counters and diagnostics accumulate on the shared
    RunContext and are attached to the returned graph under
    `meta.run` — so whatever the logs say, the caller can see too.
    """
    ctx = RunContext(url, run_id=run_id)

    def note(msg):
        if progress:
            progress(msg)

    with run_context(ctx):
        log(LOG, "info", "analysis starting", repo=url)

        note("Cloning repository…")
        with stage(ctx, "clone", LOG, repo=url):
            repo_root = clone_repo(url, ctx=ctx)

        t0 = time.time()
        try:
            note("Parsing source files (tree-sitter style pass)…")
            with stage(ctx, "parse", LOG) as st:
                records = parse_repo(repo_root, ctx=ctx)
                code_records = [
                    r for r in records
                    if not r.is_config and r.language not in ("unknown", "other")
                ]
                config_records = [r for r in records if r.is_config]
                records_by_path = {r.path: r for r in records}
                st["files"] = len(records)
                st["code_files"] = len(code_records)
                st["config_files"] = len(config_records)

            note("Resolving internal symbols (SCIP-style)…")
            with stage(ctx, "resolve", LOG) as st:
                edges = resolve_calls(code_records, ctx=ctx)
                st["edges"] = len(edges)

            note("Flagging external client calls (ast-grep pattern match)…")
            with stage(ctx, "external", LOG) as st:
                flag_external_calls(edges, ctx=ctx)
                st["candidates"] = sum(
                    1 for e in edges if e.status == "external_candidate")

            note("Checking for literal arguments…")
            with stage(ctx, "literal", LOG) as st:
                check_literals(edges, ctx=ctx)
                st["resolved"] = ctx.count("literal.resolved")

            note("Tracing constants back to their literal value…")
            with stage(ctx, "constprop", LOG) as st:
                check_constant_propagation(edges, records_by_path, ctx=ctx)
                st["resolved"] = ctx.count("constprop.resolved")

            note("Cross-referencing config & manifests…")
            with stage(ctx, "config", LOG) as st:
                st["matches"] = len(check_config_references(
                    edges, config_records, ctx=ctx, pass_label="initial"))

            note("Escalating unresolved calls to Joern (scoped cross-function trace)…")
            with stage(ctx, "joern", LOG) as st:
                escalated = check_joern_escalation(
                    edges, records_by_path, records, ctx=ctx)
                st["escalated"] = len(escalated)
                st["resolved"] = sum(1 for e in escalated if e.joern_resolved)
                # newly-resolved hosts may match a manifest too
                check_config_references(
                    edges, config_records, ctx=ctx, pass_label="post_joern")

            note("Running SLM fallback (Qwen2.5-Coder-1.5B) on anything still unresolved…")
            with stage(ctx, "slm", LOG) as st:
                touched = check_llm_fallback(edges, records_by_path, ctx=ctx)
                st["sent"] = len(touched)
                st["resolved"] = ctx.count("slm.resolved")
                # and so may SLM-inferred hosts
                check_config_references(
                    edges, config_records, ctx=ctx, pass_label="post_slm")

            note("Assembling graph…")
            with stage(ctx, "graph", LOG) as st:
                graph = _build_graph(code_records, config_records, edges)
                st["nodes"] = len(graph["nodes"])
                st["edges"] = len(graph["edges"])

            graph["meta"] = {
                "repo": url,
                "files_parsed": len(code_records),
                "config_files": len(config_records),
                "elapsed_seconds": round(time.time() - t0, 2),
                "joern_escalated": sum(1 for e in edges if e.escalated_to_joern),
                "joern_resolved": sum(1 for e in edges if e.joern_resolved),
                "llm_fallback_used": sum(1 for e in edges if e.sent_to_llm),
                "llm_resolved": sum(1 for e in edges if e.literal_method == "llm_inferred"),
                "needs_review": sum(1 for e in edges if e.needs_review),
                # Full observability payload: stage timings, counters, and
                # human-readable diagnostics. Safe to ignore in the UI, but
                # it means a surprising run can be explained from the API
                # response alone, without reading the server log.
                "run": ctx.summary(),
            }
            _log_run_report(ctx, graph["meta"])
            return graph
        finally:
            shutil.rmtree(repo_root, ignore_errors=True)
            log(LOG, "debug", "scratch directory removed", path=str(repo_root))


def _log_run_report(ctx: RunContext, meta: dict):
    """One consolidated end-of-run line plus any diagnostics, so a single
    grep for 'run report' gives the shape of the whole analysis."""
    log(LOG, "info", "run report",
        repo=meta["repo"],
        files=meta["files_parsed"],
        config_files=meta["config_files"],
        external_candidates=ctx.count("external.candidates"),
        resolved_literal=ctx.count("literal.resolved"),
        resolved_constprop=ctx.count("constprop.resolved"),
        joern_escalated=meta["joern_escalated"],
        joern_resolved=meta["joern_resolved"],
        slm_sent=meta["llm_fallback_used"],
        slm_resolved=meta["llm_resolved"],
        needs_review=meta["needs_review"],
        seconds=meta["elapsed_seconds"])

    slowest = sorted(ctx.stages.items(), key=lambda kv: kv[1]["seconds"], reverse=True)[:3]
    log(LOG, "info", "slowest stages",
        **{name: f"{data['seconds']}s" for name, data in slowest})

    for d in ctx.notes:
        log(LOG, d["level"] if d["level"] in ("info", "warning", "error") else "info",
            f"diagnostic[{d['stage']}]: {d['message']}", **d.get("fields", {}))


def _build_graph(code_records, config_records, edges):
    nodes = {}
    graph_edges = []
    all_paths = [r.path for r in code_records]

    def add_node(node_id, **attrs):
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, **attrs}
        return node_id

    # file + definition nodes, "contains" edges
    for rec in code_records:
        file_id = f"file:{rec.path}"
        add_node(file_id, label=Path(rec.path).name, type="file", group="file",
                 path=rec.path, language=rec.language)

        for d in rec.definitions:
            def_id = f"def:{d.qualified_name}"
            add_node(def_id, label=d.name, type=d.kind, group=d.kind,
                     file=rec.path, line=d.start_line)
            graph_edges.append({"from": file_id, "to": def_id, "kind": "contains"})

        for imp in rec.imports:
            target = _import_target_file(imp.raw, all_paths)
            if target and target != rec.path:
                graph_edges.append({
                    "from": file_id, "to": f"file:{target}", "kind": "imports", "label": imp.raw,
                })

    def_ids_present = {n for n, a in nodes.items() if a.get("type") in ("function", "method", "class")}

    def ensure_caller_node(caller_qn, file_path):
        node_id = f"def:{caller_qn}"
        if node_id in nodes:
            return node_id
        if caller_qn.endswith("::<module>"):
            add_node(node_id, label="(module level)", type="module_scope", group="module_scope",
                      file=file_path, line=0)
            file_id = f"file:{file_path}"
            if file_id in nodes:
                graph_edges.append({"from": file_id, "to": node_id, "kind": "contains"})
            return node_id
        return None

    ext_nodes = {}
    cfg_node_ids = set()

    for edge in edges:
        caller_id = ensure_caller_node(edge.caller, edge.file)
        if not caller_id:
            continue

        if edge.status == "internal" and edge.resolved_targets:
            for target_qn in edge.resolved_targets[:1]:  # keep graph readable
                target_id = f"def:{target_qn}"
                if target_id in nodes and target_id != caller_id:
                    graph_edges.append({
                        "from": caller_id, "to": target_id, "kind": "calls", "line": edge.line,
                    })

        elif edge.status == "external_candidate":
            label = edge.host or edge.callee_expr
            ext_id = f"ext:{label}"
            if ext_id not in ext_nodes:
                add_node(ext_id, label=label, type="external_service", group="external_service",
                          pattern=edge.external_pattern)
                ext_nodes[ext_id] = True
            graph_edges.append({
                "from": caller_id, "to": ext_id, "kind": "calls_external",
                "pattern": edge.external_pattern,
                "method": edge.literal_method,   # "literal" | "const_prop" | "joern" | "llm_inferred" | None
                "line": edge.line,
                "needs_review": edge.needs_review,
                "review_reason": edge.review_reason,
            })
            for cfg_path in edge.config_matches:
                cfg_id = f"cfg:{cfg_path}"
                if cfg_id not in cfg_node_ids:
                    add_node(cfg_id, label=Path(cfg_path).name, type="config", group="config", path=cfg_path)
                    cfg_node_ids.add(cfg_id)
                graph_edges.append({"from": ext_id, "to": cfg_id, "kind": "configured_in"})

    # config files that had no matches still show up if repo is small enough,
    # but to avoid clutter we only add config nodes that were actually matched.

    # dedupe edges
    seen = set()
    deduped = []
    for e in graph_edges:
        key = (e["from"], e["to"], e["kind"], e.get("label", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)

    return {"nodes": list(nodes.values()), "edges": deduped}
