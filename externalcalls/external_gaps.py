"""Convert external-call reports into gaps for failures and unchecked work."""

from analyzer_new.gaps import Gap


def build_external_call_gaps(run_report, join_report, literal_report, constprop_report, config_report=None) -> list[Gap]:
    """Never mutate reports; preserve gaps already built if conversion fails."""
    gaps = []
    try:
        for stage_report, mapping, is_join in (
            (run_report, {"ast_grep_unavailable": "ast_grep_unavailable",
                          "ast_grep_language_failed": "ast_grep_language_failed",
                          "stage_failed": "ast_grep_stage_failed"}, False),
            (join_report, {"join_aborted": "join_aborted", "stage_failed": "join_stage_failed",
                           "hit_ambiguous": "join_hit_ambiguous", "hit_without_edge": "join_hit_without_edge",
                           "hit_ignored": "join_hit_ignored", "regex_fallback_used": "regex_fallback_used"}, True),
        ):
            if stage_report is None:
                continue
            for reason in stage_report.get("reasons", []):
                kind = mapping.get(reason["kind"])
                count = reason.get("count", 1)
                if kind is None or count <= 0:
                    continue
                language = reason.get("language", "")
                impact = ("Calls in this language may be missing from the external-call results; missing means unknown, not 'no external calls'."
                          if language else "External calls may be missing from the repo results; missing means unknown, not 'no external calls'.")
                if is_join:
                    impact = ("NO edge was marked external because the join aborted." if stage_report.get("aborted")
                              else "External-call classification may be incomplete or based on callee spelling alone.")
                gaps.append(Gap(stage="ast_grep", scope="language" if language else "repo", kind=kind,
                                count=count, samples=list(reason.get("samples", [])),
                                detail=f"cause={reason.get('cause', '')} language={language}: {reason.get('detail', '')}",
                                impact=impact))
            if not is_join:
                for record in stage_report.get("client_instances", []):
                    if record["count"] > 0:
                        gaps.append(Gap(stage="ast_grep", scope="file", kind="client_instance_untracked",
                                        path=record["file"], count=record["count"], samples=list(record.get("samples", [])),
                                        detail=f"factories={record.get('factories', [])}; positions={record.get('samples', [])}",
                                        impact=record["impact"]))
            elif stage_report.get("edges_without_position", 0) > 0:
                gaps.append(Gap(stage="ast_grep", scope="repo", kind="edges_not_checkable",
                                count=stage_report["edges_without_position"],
                                detail="unresolved calls whose callee position or spelling is unusable (opaque expression or position -1) were not checked for external calls",
                                impact=("NO edge was marked external because the join aborted."
                                        if stage_report.get("aborted") else
                                        "Some external calls may be missed (for example chained calls such as <expr>.then).")))
        for stage, stage_report, mapping, impact in (
            ("literal", literal_report,
             {"file_unreadable": "literal_file_unreadable", "grammar_unavailable": "literal_grammar_unavailable",
              "node_not_found": "literal_call_not_found", "ambiguous_node": "literal_call_ambiguous",
              "edge_failed": "literal_edge_failed", "no_position": "literal_no_position"},
             "The URL of these external calls was not examined; host unknown."),
            ("constprop", constprop_report,
             {"file_unreadable": "constprop_file_unreadable", "grammar_unavailable": "constprop_grammar_unavailable",
              "argument_not_found": "constprop_argument_not_found", "edge_failed": "constprop_edge_failed"},
             "Variable URLs of these calls were not examined; host unknown."),
        ):
            if stage_report is None:
                continue
            if stage_report.get("aborted"):
                gaps.append(Gap(stage=stage, scope="repo", kind=f"{stage}_aborted", count=1,
                                detail=f"{stage} stage aborted: {stage_report.get('reasons', [])}", impact=impact))
            for outcome, kind in mapping.items():
                count = stage_report.get("outcomes", {}).get(outcome, 0)
                if count > 0:
                    gaps.append(Gap(stage=stage, scope="repo", kind=kind, count=count,
                                    samples=list(stage_report.get("samples", {}).get(outcome, []))[:5],
                                    detail=f"outcome={outcome}", impact=impact))
        if config_report is not None:
            for path in config_report.get("unparsed_files", []):
                error = config_report.get("unparsed_errors", {}).get(path, "error text unavailable")
                gaps.append(Gap(stage="config", scope="file", kind="config_file_unparsed", path=path, count=1,
                                detail=f"YAML could not be parsed (for example a Helm template): {error}",
                                impact="Config values defined in this file are unknown; env-based URLs of external calls may stay unresolved."))
    except Exception as exc:
        gaps.append(Gap(stage="ast_grep", scope="repo", kind="gap_conversion_failed",
                        detail=f"{type(exc).__name__}: {exc}",
                        impact="External-call gaps may be incomplete.", count=1))
    return gaps
