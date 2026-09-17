"""
Regression tests for the extraction pipeline.

The first test in `TestArgumentRecovery` is the one that matters most: it
pins the bug that made the Joern stage resolve 0 of 3 escalated calls on
a real repo. The argument of every f-string/concatenation call was being
replaced with the placeholder "<expr>" during parsing, which is
unresolvable by construction — no trace can match a URL, an identifier or
an assignment line against it.

Run with:  python -m unittest discover -s tests -t .
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("PEACH_LOG_LEVEL", "CRITICAL")

from analyzer.argtext import (  # noqa: E402
    first_argument, split_top_level, identifier_roots, string_literal_value, classify,
)
from analyzer.constprop_check import check_constant_propagation  # noqa: E402
from analyzer.external_calls import flag_external_calls  # noqa: E402
from analyzer.joern_check import check_joern_escalation  # noqa: E402
from analyzer.literal_check import check_literals  # noqa: E402
from analyzer.obs import RunContext  # noqa: E402
from analyzer.parser import parse_repo  # noqa: E402
from analyzer.resolver import resolve_calls  # noqa: E402


def build(files: dict):
    """Write a throwaway repo and run the deterministic stages over it."""
    tmp = tempfile.mkdtemp(prefix="peach_test_")
    root = Path(tmp)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    ctx = RunContext("test://fixture")
    records = parse_repo(root, ctx=ctx)
    code = [r for r in records
            if not r.is_config and r.language not in ("unknown", "other")]
    by_path = {r.path: r for r in records}
    edges = resolve_calls(code, ctx=ctx)
    flag_external_calls(edges, ctx=ctx)
    check_literals(edges, ctx=ctx)
    check_constant_propagation(edges, by_path, ctx=ctx)
    escalated = check_joern_escalation(edges, by_path, records, ctx=ctx)
    external = [e for e in edges if e.status == "external_candidate"]
    return ctx, external, escalated


def host_at(edges, line):
    for e in edges:
        if e.line == line:
            return e.host
    raise AssertionError(f"no external edge at line {line}: "
                         f"{[(e.line, e.callee_expr) for e in edges]}")


class TestArgumentRecovery(unittest.TestCase):
    """The root cause of 'Joern escalated 3, resolved 0'."""

    def test_fstring_argument_is_not_opaque(self):
        _, external, _ = build({"a.py": (
            "import requests\n"
            "KEY = 'abc'\n"
            "def go():\n"
            "    return requests.post(f'https://identitytoolkit.googleapis.com/v1/x?key={KEY}')\n"
        )})
        edge = external[0]
        self.assertNotIn("<expr>", edge.arg_text,
                         "f-string argument was replaced by a placeholder during parsing")
        self.assertEqual(edge.host, "identitytoolkit.googleapis.com")
        self.assertEqual(edge.literal_method, "literal")
        self.assertFalse(edge.escalated_to_joern,
                         "a statically readable host should never reach the Joern stage")

    def test_nested_call_attribute_is_not_opaque(self):
        _, external, _ = build({"a.py": (
            "import requests\n"
            "def go(thing):\n"
            "    return requests.get(thing.resolve().url)\n"
        )})
        self.assertEqual(external[0].arg_text, "thing.resolve().url")

    def test_concatenation_does_not_yield_a_fragment_host(self):
        """`BASE + '?key=' + k` must not resolve to the host '?key='."""
        _, external, _ = build({"a.py": (
            "import requests\n"
            "TOKEN_URL = 'https://securetoken.googleapis.com/v1/token'\n"
            "def go(k):\n"
            "    return requests.post(TOKEN_URL + '?key=' + k)\n"
        )})
        self.assertEqual(host_at(external, 4), "securetoken.googleapis.com")

    def test_interpolated_host_is_left_for_tracing(self):
        """f'{base}/v1/x' carries no static host, so the literal stage
        must decline rather than claim a bogus one."""
        _, external, escalated = build({"a.py": (
            "import requests\n"
            "def go(base):\n"
            "    return requests.get(f'{base}/v1/users')\n"
        )})
        self.assertNotEqual(external[0].literal_method, "literal")
        self.assertTrue(escalated)


class TestJoernTraces(unittest.TestCase):
    """Each of the four widened traces, end to end."""

    def test_env_var_via_variable(self):
        _, external, _ = build({"a.py": (
            "import os, requests\n"
            "def go():\n"
            "    endpoint = os.environ.get('HEALTH_SERVICE_URL')\n"
            "    return requests.get(endpoint)\n"
        )})
        self.assertEqual(host_at(external, 4), "HEALTH_SERVICE_URL")
        self.assertEqual(external[0].literal_method, "joern")

    def test_env_var_inline(self):
        _, external, _ = build({"a.py": (
            "import os, requests\n"
            "def go():\n"
            "    return requests.get(os.environ['SVC_URL'])\n"
        )})
        self.assertEqual(host_at(external, 3), "SVC_URL")

    def test_self_attribute_cross_function(self):
        _, external, _ = build({"a.py": (
            "import requests\n"
            "class Notifier:\n"
            "    def __init__(self):\n"
            "        self.base_url = 'https://notify.acme.io'\n"
            "    def send(self, body):\n"
            "        return requests.post(self.base_url, json=body)\n"
        )})
        edge = [e for e in external if e.line == 6][0]
        self.assertEqual(edge.host, "notify.acme.io")
        self.assertEqual(edge.literal_method, "joern",
                         "instance attributes belong to the cross-function trace")

    def test_imported_constant_cross_file(self):
        _, external, _ = build({
            "settings.py": "PAYMENTS_URL = 'https://payments.acme.io'\n",
            "svc.py": (
                "import requests\n"
                "from settings import PAYMENTS_URL\n"
                "def charge(n):\n"
                "    return requests.post(PAYMENTS_URL, json={'n': n})\n"
            ),
        })
        self.assertEqual(host_at(external, 4), "payments.acme.io")

    def test_cross_function_argument_propagation(self):
        _, external, _ = build({"a.py": (
            "import requests\n"
            "def fetch_from(service_url):\n"
            "    return requests.get(service_url)\n"
            "def caller():\n"
            "    return fetch_from('https://reports.acme.io/v2')\n"
        )})
        self.assertEqual(host_at(external, 3), "reports.acme.io")


class TestDiagnostics(unittest.TestCase):
    """The logging/monitoring layer has to explain a zero result."""

    def test_genuinely_dynamic_call_is_explained_not_silent(self):
        ctx, _, escalated = build({"a.py": (
            "import requests\n"
            "def go(thing):\n"
            "    return requests.get(thing.resolve().url)\n"
        )})
        self.assertEqual(len(escalated), 1)
        self.assertFalse(escalated[0].joern_resolved)
        # Traces were attempted rather than skipped wholesale.
        self.assertGreater(sum(ctx.counters_with_prefix("joern.trace_ran.").values()), 0)
        # And a human-readable note explains the zero.
        notes = [n for n in ctx.notes if n["stage"] == "joern"]
        self.assertTrue(notes, "a fully-unresolved Joern stage must leave a diagnostic")
        self.assertIn("bail reasons", notes[0]["message"].lower())

    def test_bail_reasons_are_recorded_per_trace(self):
        ctx, _, _ = build({"a.py": (
            "import requests\n"
            "def go(thing):\n"
            "    return requests.get(thing.resolve().url)\n"
        )})
        reasons = ctx.counters_with_prefix("joern.bail.")
        self.assertTrue(reasons)
        self.assertTrue(any(r.startswith("env_var.") for r in reasons))

    def test_uncovered_client_calls_are_reported(self):
        ctx, _, _ = build({"a.py": (
            "import smtplib\n"
            "def go():\n"
            "    return smtplib.SMTP('smtp.gmail.com', 587)\n"
        )})
        self.assertGreater(ctx.count("external.uncovered_client_calls"), 0,
                           "an SMTP client should be reported as a pattern-catalogue gap")


class TestCrossFunctionGuards(unittest.TestCase):
    """The trace that produced `host="email"` on a real repo."""

    def test_literal_is_matched_by_parameter_position(self):
        """`send_reset(email, url)` called with an address first must not
        bind the address to `url`."""
        _, external, _ = build({
            "auth/auth_functions.py": (
                "import requests\n"
                "def send_reset(email, url):\n"
                "    return requests.post(url, json={'email': email})\n"
            ),
            "auth/admin_page.py": (
                "from auth.auth_functions import send_reset\n"
                "def on_click():\n"
                "    send_reset('user@example.com', 'https://identitytoolkit.googleapis.com/v1/x')\n"
            ),
        })
        self.assertEqual(host_at(external, 3), "identitytoolkit.googleapis.com")

    def test_implausible_host_is_rejected(self):
        """A bare word at the right position is still not a host."""
        _, external, escalated = build({
            "auth/auth_functions.py": (
                "import requests\n"
                "def send_reset(email, url):\n"
                "    return requests.post(url, json={'email': email})\n"
            ),
            "auth/admin_page.py": (
                "from auth.auth_functions import send_reset\n"
                "def on_click(email):\n"
                "    send_reset(email, 'admin')\n"
            ),
        })
        self.assertIsNone(external[0].host,
                          "a non-host literal must not be claimed as a host")
        self.assertFalse(escalated[0].joern_resolved)

    def test_resolution_is_flagged_for_review(self):
        """One caller's literal doesn't prove the value for all callers."""
        _, external, _ = build({"a.py": (
            "import requests\n"
            "def fetch_from(service_url):\n"
            "    return requests.get(service_url)\n"
            "def caller():\n"
            "    return fetch_from('https://reports.acme.io/v2')\n"
        )})
        edge = [e for e in external if e.line == 3][0]
        self.assertTrue(edge.needs_review)
        self.assertIn("verify", (edge.review_reason or "").lower())

    def test_self_is_skipped_when_locating_parameters(self):
        _, external, _ = build({"a.py": (
            "import requests\n"
            "class C:\n"
            "    def call(self, endpoint_url):\n"
            "        return requests.get(endpoint_url)\n"
            "def go(c):\n"
            "    return c.call('https://svc.acme.io')\n"
        )})
        self.assertEqual(host_at(external, 4), "svc.acme.io")


class TestPathNormalisation(unittest.TestCase):
    def test_record_paths_use_forward_slashes(self):
        """Backslash paths break the "/"-splitting import resolution, which
        silently disabled the cross-file trace on Windows."""
        _, external, _ = build({
            "settings.py": "PAYMENTS_URL = 'https://payments.acme.io'\n",
            "auth/svc.py": (
                "import requests\n"
                "from settings import PAYMENTS_URL\n"
                "def charge():\n"
                "    return requests.post(PAYMENTS_URL)\n"
            ),
        })
        self.assertNotIn("\\", external[0].file)
        self.assertEqual(external[0].file, "auth/svc.py")
        self.assertEqual(host_at(external, 4), "payments.acme.io")


class TestLogFormatting(unittest.TestCase):
    def test_trace_list_is_not_truncated(self):
        import logging
        from analyzer.obs import _TextFormatter

        traces = [
            {"trace": "env_var", "ran": True, "resolved": False,
             "reason": "no_env_assignment_found", "detail": None},
            {"trace": "self_attr", "ran": False, "resolved": False,
             "reason": "arg_is_not_an_instance_attribute", "detail": None},
            {"trace": "imported_constant", "ran": True, "resolved": False,
             "reason": "no_import_matches_the_traced_identifier", "detail": None},
            {"trace": "cross_function_param", "ran": True, "resolved": False,
             "reason": "enclosing_function_has_no_known_callers", "detail": None},
        ]
        record = logging.LogRecord("peach.joern", logging.INFO, "f", 1,
                                   "escalation exhausted", None, None)
        record.fields = {"traces": traces}
        out = _TextFormatter().format(record)
        self.assertNotIn("…", out)
        for t in traces:
            self.assertIn(t["reason"], out)

    def test_empty_collections_are_omitted(self):
        import logging
        from analyzer.obs import _TextFormatter

        record = logging.LogRecord("peach.joern", logging.INFO, "f", 1,
                                   "done", None, None)
        record.fields = {"resolved": 0, "traces_resolved": {}}
        out = _TextFormatter().format(record)
        self.assertIn("resolved=0", out)
        self.assertNotIn("traces_resolved", out)


class TestArgText(unittest.TestCase):
    def test_split_respects_nesting_and_quotes(self):
        self.assertEqual(
            split_top_level("f(a, b), {'x': 1, 'y': 2}, 'a,b'"),
            ["f(a, b)", "{'x': 1, 'y': 2}", "'a,b'"],
        )

    def test_first_argument_not_broken_by_inner_comma(self):
        self.assertEqual(first_argument("url.format(a, b), timeout=3"), "url.format(a, b)")

    def test_string_literal_value_requires_whole_expression(self):
        self.assertEqual(string_literal_value("'https://x.io'"), "https://x.io")
        self.assertIsNone(string_literal_value("BASE + '?key=' + k"))

    def test_identifier_roots_prefers_endpointish_names(self):
        self.assertEqual(identifier_roots("BASE_URL + '/v1/' + path")[0], "BASE_URL")

    def test_classify(self):
        self.assertEqual(classify("<expr>"), "opaque_placeholder")
        self.assertEqual(classify("self.base"), "self_attribute")
        self.assertEqual(classify("'https://x.io'"), "string_literal")


if __name__ == "__main__":
    unittest.main(verbosity=2)
