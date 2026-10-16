"""Pure-Python checks for edge/CallSite alignment."""

from copy import deepcopy
import unittest

from analyzer_new.parser import CallSite, FileRecord
from analyzer_new.resolver import CallEdge
from externalcalls.edge_pairing import pair_edges_with_calls


class EdgePairingTests(unittest.TestCase):
    def setUp(self):
        self.first = CallSite("<module>:a.py", "requests.get", 3, '"a"', "a.py", 3, 9)
        self.second = CallSite("<module>:b.js", "fetch", 5, '"b"', "b.js", 5, 0)
        self.records = [FileRecord("a.py", "python", calls=[self.first]),
                        FileRecord("b.js", "javascript", calls=[self.second])]
        self.edges = [CallEdge(c.caller, c.callee_expr, c.file, c.line, c.arg_text)
                      for c in (self.first, self.second)]

    def test_correct_pairing_preserves_order_identity_and_inputs(self):
        before_edges, before_records = deepcopy(self.edges), deepcopy(self.records)
        calls, problem = pair_edges_with_calls(self.edges, self.records)
        self.assertIsNone(problem)
        self.assertEqual(calls, [self.first, self.second])
        self.assertIs(calls[0], self.first)
        self.assertIs(calls[1], self.second)
        self.assertEqual(self.edges, before_edges)
        self.assertEqual(self.records, before_records)

    def test_count_mismatch_in_both_directions(self):
        for edges in (self.edges[:1], self.edges + [deepcopy(self.edges[1])]):
            with self.subTest(edge_count=len(edges)):
                before_edges, before_records = deepcopy(edges), deepcopy(self.records)
                calls, problem = pair_edges_with_calls(edges, self.records)
                self.assertIsNone(calls)
                self.assertEqual(problem, {"cause": "edge_count_mismatch",
                                           "detail": f"edges={len(edges)}, CallSites=2"})
                self.assertEqual(edges, before_edges)
                self.assertEqual(self.records, before_records)

    def test_each_pair_field_mismatch_reports_first_index_and_values(self):
        for field, value in (("file", "wrong.js"), ("line", 99),
                             ("callee_expr", "wrong"), ("caller", "wrong_caller")):
            with self.subTest(field):
                edges = deepcopy(self.edges)
                setattr(edges[1], field, value)
                before_edges, before_records = deepcopy(edges), deepcopy(self.records)
                calls, problem = pair_edges_with_calls(edges, self.records)
                self.assertIsNone(calls)
                self.assertEqual(problem, {"cause": "pair_mismatch",
                                           "detail": f"index=1, {field}: edge={value!r}, CallSite={getattr(self.second, field)!r}"})
                self.assertEqual(edges, before_edges)
                self.assertEqual(self.records, before_records)

    def test_two_identical_calls_on_one_line_keep_distinct_columns(self):
        first = CallSite("<module>:a.js", "fetch", 1, '"x"', "a.js", 1, 3)
        second = CallSite("<module>:a.js", "fetch", 1, '"x"', "a.js", 1, 19)
        records = [FileRecord("a.js", "javascript", calls=[first, second])]
        edges = [CallEdge(c.caller, c.callee_expr, c.file, c.line, c.arg_text) for c in (first, second)]
        before_edges, before_records = deepcopy(edges), deepcopy(records)
        calls, problem = pair_edges_with_calls(edges, records)
        self.assertIsNone(problem)
        self.assertIs(calls[0], first)
        self.assertIs(calls[1], second)
        self.assertEqual([c.callee_col for c in calls], [3, 19])
        self.assertEqual(edges, before_edges)
        self.assertEqual(records, before_records)

    def test_empty_input(self):
        self.assertEqual(pair_edges_with_calls([], []), ([], None))

    def test_malformed_inputs_return_a_problem_without_raising(self):
        for edges, records in (([], [object()]), ([object()], self.records[:1]), (None, [])):
            with self.subTest(edges=edges, records=records):
                calls, problem = pair_edges_with_calls(edges, records)
                self.assertIsNone(calls)
                self.assertEqual(set(problem), {"cause", "detail"})
                self.assertEqual(problem["cause"], "pair_mismatch")
                self.assertIsInstance(problem["detail"], str)
                self.assertTrue(problem["detail"])


if __name__ == "__main__":
    unittest.main()
