"""Pure-Python tests of pairing and the ast-grep/CallEdge join."""

from copy import deepcopy
from dataclasses import asdict, replace
import re
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from analyzer_new.parser import CallSite, FileRecord
from analyzer_new.resolver import CallEdge
from externalcalls.ast_grep_check import RawHit
from externalcalls.ast_grep_join import join_ast_grep_hits
from externalcalls import ast_grep_join as join_module
from externalcalls import ast_grep_check as runner
from analyzer_new import parser as parser_module
from analyzer_new.resolver import resolve_calls


RUN_REPORT = {"binary": "ast-grep", "version": "0.45.3", "reasons": [], "languages": {}}
REPORT_KEYS = {"aborted", "edges_total", "edges_eligible", "edges_without_position",
               "hits_total", "hits_marked", "hits_unmatched", "hits_ambiguous",
               "per_pattern", "fallback_languages", "fallback_marked", "reasons", "marked"}


def call(callee="fetch", col=0, line=1, file="a.js", caller="<module>:a.js", callee_line=None):
    return CallSite(caller, callee, line, "x", file,
                    line if callee_line is None else callee_line, col)


def edge(site, status="unresolved"):
    return CallEdge(site.caller, site.callee_expr, site.file, site.line, site.arg_text, status=status)


def record(*calls, file="a.js", language="javascript"):
    return FileRecord(file, language, calls=list(calls))


def hit(callee="fetch", start=(1, 0), end=(1, 8), pattern="js.fetch", file="a.js", alias=False):
    return RawHit(file, *start, *end, callee + "(x)", pattern,
                  "python" if file.endswith(".py") else "javascript", callee, alias)


class PairingTests(unittest.TestCase):
    def test_empty_inputs_and_exact_report_keys(self):
        report = join_ast_grep_hits([], [], [], RUN_REPORT)
        self.assertEqual(set(report), REPORT_KEYS)
        self.assertFalse(report["aborted"])
        self.assertEqual(report["reasons"], [])

    def test_count_mismatch_changes_nothing(self):
        site = call()
        edges = [edge(site)]
        before = deepcopy(edges)
        report = join_ast_grep_hits(edges, [record()], [hit()], RUN_REPORT)
        self.assertEqual(edges, before)
        self.assertTrue(report["aborted"])
        self.assertEqual(report["reasons"][0]["kind"], "join_aborted")
        self.assertEqual(report["reasons"][0]["cause"], "edge_count_mismatch")
        self.assertEqual(report["reasons"][0]["detail"], "edges=1, CallSites=0")
        self.assertEqual(report["marked"], {})
        print("PAIR COUNT", report)

    def test_each_pair_field_mismatch_changes_nothing(self):
        for field, value in [("file", "wrong.js"), ("line", 99),
                             ("callee_expr", "wrong"), ("caller", "wrong_caller")]:
            with self.subTest(field):
                sites = [call(), call(col=10)]
                edges = [edge(s) for s in sites]
                setattr(edges[1], field, value)
                before = deepcopy(edges)
                report = join_ast_grep_hits(edges, [record(*sites)], [hit()], RUN_REPORT)
                self.assertEqual(edges, before)
                self.assertTrue(report["aborted"])
                reason, = report["reasons"]
                self.assertEqual((reason["kind"], reason["cause"]), ("join_aborted", "pair_mismatch"))
                self.assertEqual(reason["detail"],
                                 f"index=1, {field}: edge={value!r}, CallSite={getattr(sites[1], field)!r}")
                self.assertEqual(report["marked"], {})
                print("PAIR FIELD", field, reason, "edges_unchanged", edges == before)

    def test_pairing_walks_records_then_calls(self):
        a, b = call(), call(file="b.js", caller="<module>:b.js")
        edges = [edge(a), edge(b)]
        report = join_ast_grep_hits(edges, [record(a), record(b, file="b.js")], [], RUN_REPORT)
        self.assertFalse(report["aborted"])
        self.assertEqual(report["edges_total"], 2)

    def test_malformed_pair_is_a_stage_failure(self):
        site = call()
        edges = [edge(site)]
        before = deepcopy(edges)
        report = join_ast_grep_hits(edges, [object()], [], RUN_REPORT)
        self.assertTrue(report["aborted"])
        self.assertEqual(report["reasons"][0]["kind"], "stage_failed")
        self.assertEqual(edges, before)


class JoinTests(unittest.TestCase):
    def join(self, sites, hits, statuses=None):
        edges = [edge(s, statuses[i] if statuses else "unresolved") for i, s in enumerate(sites)]
        records = [record(*sites)]
        report = join_ast_grep_hits(edges, records, hits, RUN_REPORT)
        self.assertFalse(report["aborted"], report)
        self.assertEqual(set(report), REPORT_KEYS)
        return edges, report

    def test_unresolved_mark_changes_exactly_three_fields(self):
        site = call()
        original = edge(site)
        original.host = "already-recorded"
        original.resolved_targets = ["preserve"]
        before = asdict(original)
        report = join_ast_grep_hits([original], [record(site)], [hit()], RUN_REPORT)
        expected = dict(before, status="external_candidate", external_pattern="js.fetch", external_via="ast_grep")
        self.assertEqual(asdict(original), expected)
        self.assertEqual(report["marked"], {0: {"pattern_id": "js.fetch", "via": "ast_grep", "via_alias": False}})
        self.assertEqual(report["per_pattern"], {"js.fetch": 1})
        self.assertEqual((report["edges_eligible"], report["hits_marked"]), (1, 1))
        self.assertEqual(report["reasons"], [])

    def test_other_statuses_ignored_without_diagnostics(self):
        for status in ("internal", "internal_unmapped", "external_candidate"):
            with self.subTest(status):
                site = call()
                original = edge(site, status)
                before = deepcopy(original)
                report = join_ast_grep_hits([original], [record(site)], [hit()], RUN_REPORT)
                self.assertEqual(original, before)
                self.assertEqual(report["reasons"], [])
                self.assertEqual(report["marked"], {})
                self.assertEqual(report["per_pattern"], {})
                self.assertEqual(tuple(report[k] for k in ("edges_eligible", "edges_without_position",
                                                          "hits_marked", "hits_unmatched", "hits_ambiguous")),
                                 (0, 0, 0, 0, 0))

    def test_nested_fetch_claims_distinct_innermost_hits(self):
        outer, inner = call(col=0), call(col=6)
        hits = [hit(start=(1, 0), end=(1, 15)), hit(start=(1, 6), end=(1, 14))]
        edges, report = self.join([outer, inner], hits)
        self.assertEqual([e.status for e in edges], ["external_candidate"] * 2)
        self.assertEqual(report["hits_marked"], 2)
        self.assertEqual(report["hits_ambiguous"], 0)
        self.assertEqual(report["hits_unmatched"], 0)
        self.assertEqual(set(report["marked"]), {0, 1})
        # Reversed call order also proves that the inner edge claims the inner
        # hit, rather than claiming the outer hit and blocking the outer edge.
        reversed_edges, reversed_report = self.join([inner, outer], hits)
        self.assertEqual([e.status for e in reversed_edges], ["external_candidate"] * 2)
        self.assertEqual(reversed_report["hits_marked"], 2)
        self.assertEqual(reversed_report["hits_ambiguous"], 0)
        print("NESTED", {"positions": [(s.callee_line, s.callee_col) for s in [outer, inner]],
                         "hit_ranges": [(h.start_line, h.start_col, h.end_line, h.end_col) for h in hits],
                         "edges": [asdict(e) for e in edges], "report": report,
                         "reversed_order_marked": reversed_report["marked"]})

    def test_build_url_inside_axios_hit_is_not_marked(self):
        sites = [call("axios.get", col=6), call("buildUrl", col=10)]
        edges, report = self.join(sites, [hit("axios.get", end=(1, 22), pattern="js.axios.method")])
        self.assertEqual([e.status for e in edges], ["external_candidate", "unresolved"])
        self.assertEqual(report["marked"], {0: {"pattern_id": "js.axios.method", "via": "ast_grep", "via_alias": False}})
        self.assertEqual(report["hits_unmatched"], 0)
        self.assertEqual(report["reasons"], [])
        print("BUILD URL", {"positions": [(s.callee_expr, s.callee_line, s.callee_col) for s in sites],
                            "range": (1, 0, 1, 22), "edges": [asdict(e) for e in edges], "report": report})

    def test_awaited_generic_starts_at_callee(self):
        edges, report = self.join([call("axios.get", col=12, line=3)],
                                  [hit("axios.get", start=(3, 6), end=(3, 40), pattern="js.axios.method")])
        self.assertEqual(edges[0].status, "external_candidate")
        self.assertEqual(report["hits_marked"], 1)

    def test_alias_provenance_and_missing_alias_default(self):
        edges, report = self.join([call("r.get", col=2, file="a.py")],
                                  [hit("r.get", end=(1, 12), pattern="py.requests", file="a.py", alias=True)])
        self.assertEqual(edges[0].external_pattern, "py.requests")
        self.assertIs(report["marked"][0]["via_alias"], True)
        old_hit = vars(hit()).copy()
        del old_hit["via_alias"]
        _, report = self.join([call()], [SimpleNamespace(**old_hit)])
        self.assertIs(report["marked"][0]["via_alias"], False)

    def test_different_callee_text_is_unmatched(self):
        edges, report = self.join([call("buildUrl", col=2)], [hit()])
        self.assertEqual(edges[0].status, "unresolved")
        self.assertEqual(report["hits_unmatched"], 1)
        self.assertEqual(report["reasons"][0]["cause"], "no_matching_edge")

    def test_hit_file_without_calls_including_absent_record(self):
        for records in ([], [record()], [FileRecord("a.js", "javascript", parse_mode="fallback")]):
            with self.subTest(records=records):
                report = join_ast_grep_hits([], records, [hit()], RUN_REPORT)
                self.assertFalse(report["aborted"])
                self.assertEqual(report["hits_unmatched"], 1)
                self.assertEqual(report["reasons"], [{
                    "kind": "hit_without_edge", "cause": "file_without_call_sites", "language": "",
                    "detail": "hit has no CallSite with matching callee text and position",
                    "count": 1, "samples": ["a.js:1:0 fetch js.fetch"]}])

    def test_equal_range_tie_is_ambiguous(self):
        edges, report = self.join([call()], [hit(), hit(pattern="js.axios")])
        self.assertEqual(edges[0].status, "unresolved")
        self.assertEqual((report["hits_marked"], report["hits_ambiguous"], report["hits_unmatched"]), (0, 1, 0))
        self.assertEqual(report["reasons"][0]["cause"], "equal_range_tie")

    def test_same_hit_claimed_twice_is_ambiguous(self):
        edges, report = self.join([call(), call()], [hit()])
        self.assertEqual([e.status for e in edges], ["external_candidate", "unresolved"])
        self.assertEqual((report["hits_marked"], report["hits_ambiguous"]), (1, 1))
        self.assertEqual(report["reasons"][0]["cause"], "claimed_twice")

    def test_innermost_tie_breaks_by_smallest_end(self):
        edges, report = self.join([call(col=0), call(col=10)],
                                  [hit(end=(1, 20)), hit(end=(1, 8))])
        self.assertEqual([e.status for e in edges], ["external_candidate"] * 2)
        self.assertEqual(report["hits_ambiguous"], 0)

    def test_unknown_positions_and_opaque_callees_are_ineligible(self):
        edges, report = self.join([call(col=-1), call("<expr>.get", col=0)],
                                  [hit(), hit("<expr>.get")])
        self.assertEqual([e.status for e in edges], ["unresolved"] * 2)
        self.assertEqual(report["edges_without_position"], 2)
        self.assertEqual(report["edges_eligible"], 0)
        self.assertEqual(report["hits_marked"], 0)
        # The opaque call is ineligible but still accounts for a matching hit.
        self.assertEqual(report["hits_unmatched"], 1)

    def test_utf8_byte_columns_not_character_columns(self):
        prefix = 'label = "é"; '
        byte_col, char_col = len(prefix.encode()), len(prefix)
        self.assertGreater(byte_col, char_col)
        matching_hit = hit(start=(1, byte_col), end=(1, byte_col + 8))
        edges, report = self.join([call(col=byte_col)], [matching_hit])
        self.assertEqual(edges[0].status, "external_candidate")
        self.assertEqual(report["hits_marked"], 1)
        edges, report = self.join([call(col=char_col)], [matching_hit])
        self.assertEqual(edges[0].status, "unresolved")
        self.assertEqual(report["hits_unmatched"], 1)

    def test_multiline_ranges_and_exclusive_end(self):
        for position, marked in [((2, 0), 1), ((3, 4), 0), ((1, 3), 1), ((1, 2), 0)]:
            with self.subTest(position):
                _, report = self.join([call(col=position[1], callee_line=position[0])],
                                      [hit(start=(1, 3), end=(3, 4))])
                self.assertEqual(report["hits_marked"], marked)

    def test_factory_and_unknown_patterns_are_guarded_and_counted(self):
        edges, report = self.join([call("requests.Session", col=9)],
                                  [hit("requests.Session", pattern="py.requests.Session"), hit(pattern="unknown")])
        self.assertEqual(edges[0].status, "unresolved")
        self.assertEqual((report["hits_total"], report["hits_marked"], report["hits_unmatched"]), (2, 0, 0))
        self.assertEqual([(r["kind"], r["cause"], r["count"]) for r in report["reasons"]],
                         [("hit_ignored", "non_external_call", 1), ("hit_ignored", "unknown_pattern_id", 1)])

    def test_join_is_idempotent(self):
        site = call()
        edges, records, hits = [edge(site)], [record(site)], [hit()]
        first = join_ast_grep_hits(edges, records, hits, RUN_REPORT)
        self.assertEqual(first["hits_marked"], 1)
        before = deepcopy(edges)
        second = join_ast_grep_hits(edges, records, hits, RUN_REPORT)
        self.assertEqual(edges, before)
        self.assertEqual((second["edges_eligible"], second["hits_marked"], second["hits_unmatched"]), (0, 0, 0))
        self.assertEqual(second["marked"], {})
        self.assertEqual(second["reasons"], [])

    def test_report_side_table_is_complete_and_samples_capped(self):
        sites = [call(line=i + 1) for i in range(12)]
        hits = [hit(start=(i + 1, 0), end=(i + 1, 8), alias=bool(i % 2)) for i in range(12)]
        _, report = self.join(sites, hits)
        self.assertEqual(set(report["marked"]), set(range(12)))
        self.assertEqual(report["per_pattern"], {"js.fetch": 12})
        self.assertEqual([report["marked"][i]["via_alias"] for i in range(12)], [bool(i % 2) for i in range(12)])
        empty_report = join_ast_grep_hits([], [], hits, RUN_REPORT)
        reason, = empty_report["reasons"]
        self.assertEqual(reason["count"], 12)
        self.assertEqual(reason["samples"], [f"a.js:{i + 1}:0 fetch js.fetch" for i in range(5)])

    def test_context_counters(self):
        ctx = mock.Mock()
        site = call()
        report = join_ast_grep_hits([edge(site)], [record(site)], [hit()], RUN_REPORT, ctx)
        self.assertEqual(report["hits_marked"], 1)
        ctx.bump.assert_any_call("ast_grep_join.hits_marked", 1)
        ctx.bump.assert_any_call("ast_grep_join.edges_eligible", 1)

    def test_unexpected_error_discards_planned_marks(self):
        sites = [call(), call(line=2)]
        edges = [edge(s) for s in sites]
        before = deepcopy(edges)
        bad = hit(start=(2, 0), end=(2, 8))
        bad.end_line = None
        report = join_ast_grep_hits(edges, [record(*sites)], [hit(), bad], RUN_REPORT)
        self.assertEqual(edges, before)
        self.assertTrue(report["aborted"])
        self.assertEqual(report["marked"], {})
        self.assertEqual(report["hits_marked"], 0)
        self.assertEqual(report["reasons"][0]["kind"], "stage_failed")

    def test_context_failure_does_not_abort_or_discard_marks(self):
        site = call()
        edges = [edge(site)]
        before = deepcopy(edges)
        ctx = mock.Mock()
        ctx.bump.side_effect = RuntimeError("telemetry failed")
        report = join_ast_grep_hits(edges, [record(site)], [hit()], RUN_REPORT, ctx)
        expected = deepcopy(before)
        expected[0].status = "external_candidate"
        expected[0].external_pattern = "js.fetch"
        expected[0].external_via = "ast_grep"
        self.assertEqual(edges, expected)
        self.assertFalse(report["aborted"])
        self.assertEqual(report["reasons"], [])
        self.assertEqual(report["hits_marked"], 1)
        self.assertEqual(report["marked"], {0: {"pattern_id": "js.fetch", "via": "ast_grep", "via_alias": False}})

    def test_partial_commit_is_rolled_back(self):
        sites = [call(), call(line=2)]
        edges = [edge(s) for s in sites]
        before = deepcopy(edges)
        def fail_second(instance, name, value):
            if instance is edges[1] and name == "external_pattern":
                raise RuntimeError("commit failed")
            object.__setattr__(instance, name, value)
        with mock.patch.object(CallEdge, "__setattr__", fail_second):
            report = join_ast_grep_hits(edges, [record(*sites)],
                                        [hit(), hit(start=(2, 0), end=(2, 8))], RUN_REPORT)
        self.assertEqual(edges, before)
        self.assertTrue(report["aborted"])
        self.assertEqual(report["marked"], {})
        self.assertEqual(report["hits_marked"], 0)
        self.assertEqual(report["reasons"][0]["kind"], "stage_failed")


class FallbackTests(unittest.TestCase):
    def run_join(self, sites, run_report, hits=(), statuses=None):
        edges = [edge(s, statuses[i] if statuses else "unresolved") for i, s in enumerate(sites)]
        report = join_ast_grep_hits(edges, [record(*sites)], list(hits), run_report)
        self.assertFalse(report["aborted"], report)
        self.assertEqual(set(report), REPORT_KEYS)
        return edges, report

    def test_missing_binary_marks_python_and_javascript(self):
        sites = [call("requests.get", file="a.py", col=9), call()]
        edges, report = self.run_join(sites, dict(RUN_REPORT, binary=None))
        self.assertEqual([(e.status, e.external_pattern, e.external_via) for e in edges],
                         [("external_candidate", "py.requests", "regex_fallback"),
                          ("external_candidate", "js.fetch", "regex_fallback")])
        self.assertEqual(report["fallback_languages"], ["javascript", "python"])
        self.assertEqual(report["fallback_marked"], {"javascript": 1, "python": 1})
        self.assertEqual(report["hits_marked"], 2)
        self.assertEqual(report["hits_total"], 0)
        self.assertTrue(all(m["via_alias"] is False for m in report["marked"].values()))
        self.assertEqual([(r["kind"], r["cause"], r["language"], r["count"], r["detail"])
                          for r in report["reasons"]], [
            ("regex_fallback_used", "binary_missing", language, 1,
             "matched by callee spelling only; aliases and call shape are not checked")
            for language in ["javascript", "python"]])

    def test_any_language_failure_cause_covers_all_files_not_samples(self):
        for cause in ("timeout", "missing_file", "unknown_language", "alias_discovery_failed", "alias_rules_too_large"):
            with self.subTest(cause):
                sites = [call("requests.get", file="a.py"), call("requests.post", file="b.py"), call()]
                failure = {"kind": "ast_grep_language_failed", "cause": cause, "language": "python",
                           "detail": "failed", "count": 1, "samples": ["a.py"]}
                edges, report = self.run_join(sites, dict(RUN_REPORT, reasons=[failure]))
                self.assertEqual([e.status for e in edges], ["external_candidate", "external_candidate", "unresolved"])
                self.assertEqual(report["fallback_languages"], ["python"])
                self.assertEqual(report["fallback_marked"], {"python": 2})
                self.assertEqual(report["reasons"][0]["count"], 2)
                self.assertEqual(report["reasons"][0]["cause"], "language_failed")

    def test_unavailable_or_stage_failure_degrades_every_language(self):
        for kind in ("ast_grep_unavailable", "stage_failed"):
            for language in ("", "python"):
                with self.subTest(kind=kind, language=language):
                    reasons = [{"kind": kind, "language": language, "cause": "anything"}]
                    _, report = self.run_join([call(), call("requests.get", file="a.py")],
                                              dict(RUN_REPORT, reasons=reasons))
                    self.assertEqual(report["fallback_languages"], ["javascript", "python"])
                    self.assertEqual(report["fallback_marked"], {"javascript": 1, "python": 1})

    def test_empty_language_failure_degrades_all_languages(self):
        _, report = self.run_join([call(), call("requests.get", file="a.py")], dict(RUN_REPORT, reasons=[
            {"kind": "ast_grep_language_failed", "language": "", "cause": "anything"}]))
        self.assertEqual(report["fallback_marked"], {"javascript": 1, "python": 1})

    def test_clean_and_unrelated_reasons_do_not_enable_fallback(self):
        for reasons in ([], [{"kind": "hit_without_edge", "language": "javascript"}],
                        [{"kind": "ast_grep_language_failed", "language": "python"}]):
            with self.subTest(reasons=reasons):
                edges, report = self.run_join([call()], dict(RUN_REPORT, reasons=reasons))
                self.assertEqual(edges[0].status, "unresolved")
                self.assertEqual(report["fallback_languages"], [])
                self.assertEqual(report["fallback_marked"], {})

    def test_ast_grep_marks_not_overwritten_or_double_counted(self):
        sites = [call(), call("axios.get", col=6, line=2)]
        _, report = self.run_join(sites, dict(RUN_REPORT, binary=None), [hit(alias=True)])
        self.assertEqual(report["marked"], {
            0: {"pattern_id": "js.fetch", "via": "ast_grep", "via_alias": True},
            1: {"pattern_id": "js.axios.method", "via": "regex_fallback", "via_alias": False}})
        self.assertEqual(report["hits_marked"], 2)
        self.assertEqual(report["fallback_marked"], {"javascript": 1})
        self.assertEqual(report["per_pattern"], {"js.fetch": 1, "js.axios.method": 1})

    def test_alias_factory_unknown_position_and_opaque_are_not_marked(self):
        sites = [call("r.get", file="a.py"), call("requests.Session", file="a.py"), call("unknown"),
                 call(col=-1), call("<expr>.fetch"), call("fetch", col=0)]
        edges, report = self.run_join(sites, dict(RUN_REPORT, binary=None), statuses=["unresolved"] * 5 + ["internal"])
        self.assertEqual([e.status for e in edges], ["unresolved"] * 5 + ["internal"])
        self.assertEqual(report["hits_marked"], 0)
        self.assertEqual(report["edges_without_position"], 2)
        self.assertEqual(report["fallback_marked"], {"javascript": 0, "python": 0})
        self.assertEqual([r["count"] for r in report["reasons"]], [0, 0])

    def test_multiple_catalogue_entries_leave_one_edge_ambiguous(self):
        original = join_module.CATALOGUE
        fetch = next(e for e in original if e.id == "js.fetch")
        with mock.patch.object(join_module, "CATALOGUE", original + (replace(fetch, id="js.duplicate"),)):
            edges, report = self.run_join([call()], dict(RUN_REPORT, binary=None))
        self.assertEqual(edges[0].status, "unresolved")
        self.assertEqual(report["hits_ambiguous"], 1)
        self.assertEqual(report["hits_marked"], 0)
        self.assertEqual(report["reasons"][0]["cause"], "multiple_catalogue_entries")

    def test_fallback_can_resolve_ast_ambiguity_without_double_counting(self):
        edges, report = self.run_join([call()], dict(RUN_REPORT, binary=None), [hit(), hit(pattern="js.axios")])
        self.assertEqual(edges[0].external_via, "regex_fallback")
        self.assertEqual(report["hits_ambiguous"], 0)
        self.assertEqual(report["hits_marked"], 1)

    def test_telemetry_failure_does_not_discard_fallback(self):
        site = call()
        edges = [edge(site)]
        ctx = mock.Mock()
        ctx.bump.side_effect = RuntimeError("telemetry failed")
        report = join_ast_grep_hits(edges, [record(site)], [], dict(RUN_REPORT, binary=None), ctx)
        self.assertFalse(report["aborted"])
        self.assertEqual(edges[0].external_via, "regex_fallback")
        self.assertEqual(report["fallback_marked"], {"javascript": 1})

    def test_external_regex_compile_and_spelling_table(self):
        entries = [e for e in join_module.CATALOGUE if e.kind == "external_call"]
        for entry in entries:
            with self.subTest(entry.id):
                self.assertIsInstance(re.compile(entry.callee_regex), re.Pattern)
                print("REGEX COMPILE OK", entry.id, entry.callee_regex)
        spellings = [
            ("requests.get", "py.requests"), ("requests.options", "py.requests"),
            ("requests.request", "py.requests.request"), ("httpx.get", "py.httpx"),
            ("httpx.request", "py.httpx.request"), ("httpx.stream", "py.httpx.stream"),
            ("urllib.request.urlopen", "py.urllib.urlopen"), ("fetch", "js.fetch"),
            ("window.fetch", "js.window.fetch"), ("globalThis.fetch", "js.globalThis.fetch"),
            ("axios.get", "js.axios.method"), ("axios.options", "js.axios.method"),
            ("axios.request", "js.axios.request"), ("axios", "js.axios"), ("got", "js.got"),
            ("got.get", "js.got.method"), ("got.head", "js.got.method"), ("ky", "js.ky"),
            ("ky.post", "js.ky.method"), ("ky.delete", "js.ky.method"), ("superagent.get", "js.superagent"),
            ("superagent.patch", "js.superagent"), ("http.request", "js.http.request"),
            ("https.request", "js.https.request"), ("http.get", "js.http.get"), ("https.get", "js.https.get"),
            ("requests . post", "py.requests"),
        ]
        for spelling, expected in spellings:
            with self.subTest(spelling):
                self.assertEqual([e.id for e in entries if re.search(e.callee_regex, spelling)], [expected])


class RealIntegrationTests(unittest.TestCase):
    def require_binary(self):
        binary, version, _ = runner.find_binary()
        if binary is None:
            self.skipTest("real integration requires ast-grep 0.45.3; binary is absent")
        self.assertEqual(version, "ast-grep 0.45.3")

    def test_constructor_call_is_parsed_but_not_a_runner_hit(self):
        self.require_binary()
        root = Path(__file__).parent / "fixtures" / "ast_grep"
        rel = "join_constructor.js"
        rec = parser_module.parse_file(root, root / rel)
        self.assertEqual(rec.parse_mode, "treesitter")
        self.assertEqual([(c.callee_expr, c.callee_line, c.callee_col) for c in rec.calls], [("fetch", 1, 4)])
        edges = resolve_calls([rec], repo_root=None)
        hits, run_report = runner.run_patterns(root, {"javascript": [rel]})
        for item in edges + hits:
            self.assertNotIn("\\", item.file)
        self.assertEqual(hits, [])
        self.assertEqual(run_report["reasons"], [])
        report = join_ast_grep_hits(edges, [rec], hits, run_report)
        self.assertEqual([e.status for e in edges], ["unresolved"])
        self.assertEqual(report["hits_marked"], 0)
        print("CONSTRUCTOR", {"parser": [(c.callee_expr, c.callee_line, c.callee_col) for c in rec.calls],
                              "runner_hits": hits, "join": report, "edge_status": edges[0].status})

    def test_existing_fixtures_join_every_kept_hit(self):
        self.require_binary()
        root = Path(__file__).parent / "fixtures" / "ast_grep"
        files = ["client.py", "client.js", "client.ts", "client.tsx", "generics.ts", "generics.tsx",
                 "chain.js", "nested.js", "twice.py", "unicode.py", "alias.py", "alias.js", "alias.ts", "alias.tsx",
                 "alias_cross/requests_a.py", "alias_cross/httpx_b.py", "alias_cross/no_import.py", "alias_cross/other.py"]
        records = [parser_module.parse_file(root, root / rel) for rel in files]
        self.assertTrue(all(rec.parse_mode == "treesitter" for rec in records),
                        [(r.path, r.parse_mode) for r in records])
        edges = resolve_calls(records, repo_root=None)
        groups = {}
        for rec in records:
            groups.setdefault(rec.language, []).append(rec.path)
        hits, run_report = runner.run_patterns(root, groups)
        self.assertEqual(run_report["reasons"], [])
        for item in edges + hits:
            self.assertNotIn("\\", item.file)
        print("FIXTURE EDGE PATHS", sorted(set(e.file for e in edges))[:5])
        print("FIXTURE HIT PATHS", sorted(set(h.file for h in hits))[:5])
        before = deepcopy(edges)
        report = join_ast_grep_hits(edges, records, hits, run_report)
        calls = [c for rec in records for c in rec.calls]
        for h in hits:
            same_text = [(i, c.callee_line, c.callee_col, before[i].status)
                         for i, c in enumerate(calls) if c.file == h.file and c.callee_expr == h.callee_text]
            inside = [v for v in same_text if (h.start_line, h.start_col) <= (v[1], v[2]) < (h.end_line, h.end_col)]
            if not inside:
                print("POSITION DISAGREEMENT", h, "parser_calls", same_text)
            elif not any(v[0] in report["marked"] for v in inside):
                print("UNMARKED HIT", h, "parser_calls", inside)
        print("FIXTURE JOIN SUMMARY", {k: v for k, v in report.items() if k != "marked"})
        self.assertFalse(report["aborted"])
        self.assertEqual(report["hits_marked"], len(hits))
        self.assertEqual(report["hits_unmatched"], 0)
        self.assertEqual(report["hits_ambiguous"], 0)
        for rel, count in (("twice.py", 2), ("nested.js", 2), ("unicode.py", 1)):
            indices = [i for i in report["marked"] if edges[i].file == rel]
            self.assertEqual(len(indices), count)
            self.assertEqual(len(set(indices)), count)
            print("SPECIAL FIXTURE", rel, "distinct marked edge indices", indices)
        for h in hits:
            if h.via_alias:
                index, = [i for i, c in enumerate(calls) if c.file == h.file and c.callee_expr == h.callee_text
                          and (h.start_line, h.start_col) <= (c.callee_line, c.callee_col) < (h.end_line, h.end_col)]
                self.assertIs(report["marked"][index]["via_alias"], True)
        print("ALIASES MARKED", sum(mark["via_alias"] for mark in report["marked"].values()))
        for rel in ("client.py", "client.js"):
            with self.subTest(local_get=rel):
                local = [e for e in edges if e.file == rel and e.callee_expr == "get"
                         and "https://not-http.example.com" in e.arg_text]
                print("LOCAL GET", rel, [asdict(e) for e in local])
                self.assertEqual(len(local), 1)
                self.assertEqual(local[0].status, "internal")
                self.assertEqual(local[0].resolved_via, "lexical_fallback")
                self.assertNotEqual(local[0].status, "external_candidate")
                self.assertIsNone(local[0].external_pattern)
                self.assertIsNone(local[0].external_via)


if __name__ == "__main__":
    unittest.main()
