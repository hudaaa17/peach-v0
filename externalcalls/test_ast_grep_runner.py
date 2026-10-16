"""
Tests for ast_grep_check (runner + callee catalogue).

Two kinds of test:

  * Catalogue / Rule / Discovery / FakeBinary tests need no ast-grep. The
    FakeBinary tests drive the real subprocess code path with a tiny fake
    `ast-grep` script that prints canned JSON. They test OUR parsing, positions,
    failure handling and command line, not ast-grep's behaviour.
  * RealBinary / ParserAgreement tests run the REAL ast-grep over
    fixtures/ast_grep/ and SKIP with a message when it (or the Tree-sitter
    grammar package) is not installed. They are what verifies the rule syntax,
    multi-rule processes, rule ids and JSON shape.

Written with unittest so they run with `python -m unittest` and are collected
by pytest unchanged.

Set PEACH_PACKAGE (or edit the default below) to the package that holds
ast_grep_check.py, obs.py, parser.py, gaps.py.
"""
import importlib
import importlib.util
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from externalcalls import ast_grep_check
from analyzer_new import parser as parser_module

mod = ast_grep_check

FIX = Path(__file__).parent / "fixtures" / "ast_grep"
REAL_BIN = mod.find_binary()[0]

NO_REAL = "ast-grep is not installed (put `ast-grep` on PATH or set PEACH_AST_GREP_BIN)"

try:
    import tree_sitter_language_pack
    HAVE_TS = True
except ImportError:
    HAVE_TS = False

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False


class Ctx:
    def __init__(self):
        self.counts = {}

    def bump(self, key, n=1):
        self.counts[key] = self.counts.get(key, 0) + n


class TmpCase(unittest.TestCase):
    """A temp dir, and an os.environ that is restored after each test."""

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmp = Path(td.name)
        patcher = mock.patch.dict(os.environ, {})
        patcher.start()
        self.addCleanup(patcher.stop)

    def env(self, **kw):
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)

    def write(self, name, data):
        path = self.tmp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
        return path


def family(parser_language):
    wanted = "python" if parser_language == "python" else "javascript"
    return [e for e in mod.CATALOGUE if e.language == wanted]


def rules_of(args):
    """The rule documents (dicts) passed in `--inline-rules` of a recorded argv."""
    text = args[args.index("--inline-rules") + 1]
    return [json.loads(doc) for doc in text.split("\n---\n")]


def check_reason_shape(test, reason):
    test.assertEqual(set(reason), {"kind", "cause", "language", "detail", "count", "samples"})
    test.assertIn(reason["kind"], ("ast_grep_unavailable", "ast_grep_language_failed", "stage_failed"))
    test.assertIsInstance(reason["count"], int)
    test.assertGreaterEqual(reason["count"], 1)
    test.assertIsInstance(reason["samples"], list)
    test.assertLessEqual(len(reason["samples"]), mod.MAX_SAMPLES)


# ----------------------------------------------------------------- catalogue ----

class CatalogueTests(unittest.TestCase):
    def test_triviality_comes_from_plain_catalogue_spellings(self):
        entries = family("python")
        self.assertTrue(mod._binding_is_trivial(("requests", "requests", None), entries))
        self.assertTrue(mod._binding_is_trivial(("urllib.request", "urllib.request", None), entries))
        for binding in (("r", "requests", None), ("get", "requests", "get"),
                        ("request", "urllib.request", None), ("requests", "httpx", None)):
            self.assertFalse(mod._binding_is_trivial(binding, entries))
        custom = mod.Entry("custom", "python", "^client\\.get$", module="pkg", members=("get",))
        self.assertTrue(mod._binding_is_trivial(("client", "pkg", None), [custom]))
        self.assertFalse(mod._binding_is_trivial(("pkg", "pkg", None), [custom]))

    def test_alias_metadata_agrees_with_every_plain_callee(self):
        for e in mod.CATALOGUE:
            with self.subTest(e.id):
                if e.id in ("js.fetch", "js.window.fetch", "js.globalThis.fetch"):
                    self.assertIsNone(e.module)
                    self.assertEqual(e.members, ())
                else:
                    self.assertTrue(e.module)
                    self.assertIsInstance(e.members, tuple)
                    for member in e.members or (None,):
                        spelling = e.module + "." + member if member else e.module
                        self.assertIsNotNone(re.fullmatch(e.callee_regex, spelling))

    def test_entries_are_well_formed(self):
        ids = [e.id for e in mod.CATALOGUE]
        self.assertEqual(len(ids), len(set(ids)), "duplicate catalogue ids")
        for e in mod.CATALOGUE:
            with self.subTest(e.id):
                self.assertIn(e.language, ("python", "javascript"))
                self.assertTrue(e.description)
                self.assertTrue(e.callee_regex.startswith("^") and e.callee_regex.endswith("$"))
                re.compile(e.callee_regex)
                # ast-grep uses the Rust regex engine: no lookaround, no backreferences
                for bad in ("(?=", "(?!", "(?<=", "(?<!", "\\1", "\\2"):
                    self.assertNotIn(bad, e.callee_regex)
                self.assertIn(e.kind, ("external_call", "client_factory"))
                if e.kind == "client_factory":
                    self.assertEqual((e.url_arg_index, e.url_arg_keyword, e.url_arg_property),
                                     (None, None, None))
                else:
                    self.assertTrue(e.url_arg_index is not None or e.url_arg_keyword or e.url_arg_property,
                                    "an entry must say where the URL argument is")
                    if e.language == "python":
                        self.assertEqual((e.url_arg_keyword, e.url_arg_property), ("url", None))
                    else:
                        self.assertIsNone(e.url_arg_keyword)

    def test_url_argument_conventions(self):
        by_id = {e.id: e for e in mod.CATALOGUE}
        for i in ("py.requests.request", "py.httpx.request", "py.httpx.stream"):
            self.assertEqual(by_id[i].url_arg_index, 1, i)
        for i in ("py.requests", "py.httpx", "py.urllib.urlopen", "js.fetch", "js.axios.method"):
            self.assertEqual(by_id[i].url_arg_index, 0, i)
        # URL inside an options object: property `url` of the first object-literal argument
        self.assertEqual((by_id["js.axios.request"].url_arg_index, by_id["js.axios.request"].url_arg_property),
                         (None, "url"))
        for i in ("js.axios", "js.got"):
            self.assertEqual((by_id[i].url_arg_index, by_id[i].url_arg_property), (0, "url"), i)

    def test_factory_catalogue_has_exact_modules_members_and_no_url_arguments(self):
        expected = {
            "py.requests.Session": ("requests", ("Session",)),
            "py.requests.session": ("requests", ("session",)),
            "py.httpx.Client": ("httpx", ("Client",)),
            "py.httpx.AsyncClient": ("httpx", ("AsyncClient",)),
            "js.axios.create": ("axios", ("create",)),
            "js.got.extend": ("got", ("extend",)),
            "js.ky.create": ("ky", ("create",)),
            "js.ky.extend": ("ky", ("extend",)),
            "js.superagent.agent": ("superagent", ("agent",)),
        }
        factories = [e for e in mod.CATALOGUE if e.kind == "client_factory"]
        self.assertEqual({e.id: (e.module, e.members) for e in factories}, expected)
        for e in factories:
            self.assertEqual((e.url_arg_index, e.url_arg_keyword, e.url_arg_property), (None, None, None))

    def test_requested_entries_exist(self):
        ids = {e.id for e in mod.CATALOGUE}
        self.assertTrue({"py.httpx.stream", "js.http.get", "js.https.get",
                         "js.window.fetch", "js.globalThis.fetch"} <= ids)

    def test_callee_regexes(self):
        by_id = {e.id: e for e in mod.CATALOGUE}
        cases = [
            ("py.requests", "requests.get", True), ("py.requests", "requests.head", True),
            ("py.requests", "requests.options", True), ("py.requests", "requests.Session", False),
            ("py.requests", "requests.request", False), ("py.requests", "myrequests.get", False),
            ("py.requests", "requests.get.x", False), ("py.requests", "requests .get", True),
            ("py.requests.request", "requests.request", True),
            ("py.httpx", "httpx.post", True), ("py.httpx", "httpx.Client", False),
            ("py.httpx", "httpx.options", False),
            ("py.httpx.stream", "httpx.stream", True),
            ("py.urllib.urlopen", "urllib.request.urlopen", True),
            ("py.urllib.urlopen", "urllib.request.Request", False),
            ("js.fetch", "fetch", True), ("js.fetch", "refetch", False), ("js.fetch", "window.fetch", False),
            ("js.window.fetch", "window.fetch", True), ("js.globalThis.fetch", "globalThis.fetch", True),
            ("js.axios", "axios", True), ("js.axios", "axios.get", False),
            ("js.axios.method", "axios.get", True), ("js.axios.method", "axios.create", False),
            ("js.axios.method", "axios\n  .get", True), ("js.axios.method", "axios.\n   get", True),
            ("js.axios.request", "axios.request", True),
            ("js.got", "got", True), ("js.got.method", "got.post", True),
            ("js.got.method", "got.extend", False),
            ("js.ky", "ky", True), ("js.ky.method", "ky.delete", True),
            ("js.superagent", "superagent.get", True),
            ("js.http.request", "http.request", True), ("js.http.request", "https.request", False),
            ("js.https.request", "https.request", True),
            ("js.http.get", "http.get", True), ("js.http.get", "https.get", False),
            ("js.https.get", "https.get", True),
        ]
        for entry_id, callee, expected in cases:
            with self.subTest(entry=entry_id, callee=callee):
                self.assertEqual(bool(re.search(by_id[entry_id].callee_regex, callee)), expected)

    def test_callee_helper(self):
        self.assertEqual(mod._callee("axios", "(get|post)"), r"^axios\s*\.\s*(get|post)$")
        self.assertEqual(mod._callee("fetch"), "^fetch$")
        self.assertEqual(mod._callee("urllib", "request", "urlopen"), r"^urllib\s*\.\s*request\s*\.\s*urlopen$")


# --------------------------------------------------------------------- rules ----

class RuleTests(unittest.TestCase):
    def test_filter_checks_head_module_callable_binding_and_member_in_order(self):
        by_id = {e.id: e for e in mod.CATALOGUE}
        cases = [
            ("r.get", "py.requests", [], "check 1"),
            ("r.get", "py.requests", [("r", "httpx", None)], "check 2"),
            ("get", "js.axios", [("get", "axios", "get")], "check 3"),
            ("r.nope", "py.requests", [("r", "requests", None)], "check 4"),
            ("g", "py.requests", [("g", "requests", "Session")], "check 4"),
            ("r.get", "py.requests", [("r", "requests", "get")], "check 4"),
            ("r.get", "py.requests", [("r", "requests", None)], None),
            ("g", "py.requests", [("g", "requests", "get")], None),
            ("ax", "js.axios", [("ax", "axios", None)], None),
            ("r.Session", "py.requests.Session", [("r", "requests", None)], None),
            ("Session", "py.requests.Session", [("Session", "requests", "Session")], None),
        ]
        for callee, entry_id, bindings, expected in cases:
            with self.subTest(callee=callee, entry=entry_id, expected=expected):
                hit = mod.RawHit("f", 1, 0, 1, 1, "", entry_id, "python", callee)
                reason = mod._alias_filter_reason(hit, by_id[entry_id], {"f": bindings})
                self.assertEqual(reason.split(":")[0] if reason else None, expected)

    def test_alias_rules_are_unions_with_reversible_ids_and_no_unknown_members(self):
        bindings = {"a.ts": [("$ax", "axios", None), ("get", "axios", "get"),
                              ("AxiosError", "axios", "AxiosError")],
                    "b.ts": [("second", "axios", None)]}
        for lang in ("javascript", "typescript", "tsx"):
            docs = mod._alias_rule_docs(family("javascript"), bindings, lang)
            self.assertEqual([d["id"] for d in docs], ["js.axios.method#alias", "js.axios.request#alias",
                                                       "js.axios#alias", "js.axios.create#alias"])
            self.assertNotIn("AxiosError", json.dumps(docs))
            self.assertEqual(len(docs), len({d["id"] for d in docs}))
            for doc in docs:
                canonical, alias = mod._decode_rule_id(doc["id"])
                self.assertTrue(alias)
                self.assertEqual(canonical + "#alias", doc["id"])
                self.assertEqual(mod._decode_rule_id(canonical), (canonical, False))
                original = mod._rule_doc(next(e for e in mod.CATALOGUE if e.id == canonical), lang)
                self.assertEqual(doc["rule"], original["rule"])
            regex = docs[0]["constraints"]["F"]["regex"]
            self.assertIsNotNone(re.fullmatch(regex, "$ax.get"))
            self.assertIsNone(re.fullmatch(regex, "ax.get"))
            self.assertIsNotNone(re.fullmatch(regex, "get"))
            self.assertIsNone(re.fullmatch(docs[2]["constraints"]["F"]["regex"], "get"))

    def test_all_original_rules_are_byte_for_byte_unchanged(self):
        fingerprints = {
            "python": "f2aec8cf9cc71512d2635ff254e2929520baad1b4e045de602008856327493c9",
            "javascript": "70917764f028711a5b5c7551fddd630d86dd0a0c7ab75500f9ef668fec5b3ddf",
            "typescript": "dedeb0e9c95d71138b7f24cc6c48c0692c20768811618f84e30823702a4d35c2",
            "tsx": "1120a3b90e7e75aca7dbc29667e209014cecddce33b1078b7d7b4769993ef491",
        }
        for lang, expected in fingerprints.items():
            entries = family("python" if lang == "python" else "javascript")
            original = [e for e in entries if e.kind == "external_call"]
            self.assertEqual(len(original), 6 if lang == "python" else 15)
            self.assertEqual(hashlib.sha256(mod._rules_text(original, lang).encode()).hexdigest(), expected)
            docs = [json.loads(d) for d in mod._rules_text(entries, lang).split("\n---\n")]
            self.assertEqual([d["id"] for d in docs], [e.id for e in entries])
            self.assertEqual(len(docs), 10 if lang == "python" else 20)
            self.assertEqual(docs[:len(original)], [mod._rule_doc(e, lang) for e in original])

    def test_one_rule_per_entry_with_the_shape_we_rely_on(self):
        for ast_lang, parser_lang in (("python", "python"), ("javascript", "javascript"),
                                      ("typescript", "typescript"), ("tsx", "typescript")):
            with self.subTest(ast_lang):
                entries = family(parser_lang)
                docs = [json.loads(d) for d in mod._rules_text(entries, ast_lang).split("\n---\n")]
                self.assertEqual([d["id"] for d in docs], [e.id for e in entries])
                kind = "call" if ast_lang == "python" else "call_expression"
                for d, e in zip(docs, entries):
                    self.assertEqual(d["language"], ast_lang)
                    function = {"field": "function", "pattern": "$F"}
                    if ast_lang in ("typescript", "tsx"):
                        # Exact equality still checks the entire rule, including
                        # the disjoint await branch; not merely its presence.
                        function = {"field": "function", "any": [
                            {"all": [{"not": {"kind": "await_expression"}}, {"pattern": "$F"}]},
                            {"kind": "await_expression", "pattern": "$AWAIT",
                             "has": {"any": [{"kind": "member_expression"}, {"kind": "identifier"}],
                                     "pattern": "$F"}},
                        ]}
                    self.assertEqual(d["rule"], {"kind": kind, "has": function})
                    self.assertEqual(d["constraints"], {"F": {"regex": e.callee_regex}})
                    self.assertNotIn("$$$", json.dumps(d))        # nothing about arguments

    @unittest.skipUnless(HAVE_YAML, "PyYAML is not installed")
    def test_rule_text_is_valid_multi_document_yaml(self):
        import yaml
        entries = family("typescript")
        docs = list(yaml.safe_load_all(mod._rules_text(entries, "tsx")))
        self.assertEqual([d["id"] for d in docs], [e.id for e in entries])
        self.assertEqual(docs[0]["constraints"]["F"]["regex"], entries[0].callee_regex)


# ----------------------------------------------------------- binary discovery ----

class DiscoveryTests(TmpCase):
    def only_path(self, **scripts):
        """PATH containing only the given fake executables (name -> body)."""
        bindir = self.tmp / "pathdir"
        for name, body in scripts.items():
            p = self.write(f"pathdir/{name}", body)
            p.chmod(p.stat().st_mode | stat.S_IXUSR)
        bindir.mkdir(exist_ok=True)
        self.env(PATH=bindir, PEACH_AST_GREP_BIN=None)

    def test_shadow_utils_sg_is_rejected(self):
        # what `sg --version` prints on Linux (shadow-utils "sg group command")
        self.only_path(sg="#!/bin/sh\necho 'Usage: sg group [[-c] command]'\nexit 1\n")
        path, version, tried = mod.find_binary()
        self.assertIsNone(path)
        self.assertEqual(version, "")
        self.assertTrue(any("sg" in t and "does not identify ast-grep" in t for t in tried), tried)

    def test_sg_that_is_ast_grep_is_accepted(self):
        self.only_path(sg="#!/bin/sh\necho 'ast-grep 0.99.0'\n")
        path, version, _ = mod.find_binary()
        self.assertTrue(path and path.endswith("sg"))
        self.assertEqual(version, "ast-grep 0.99.0")

    def test_ast_grep_is_preferred_over_sg(self):
        self.only_path(**{"ast-grep": "#!/bin/sh\necho 'ast-grep 1.0.0'\n",
                          "sg": "#!/bin/sh\necho 'ast-grep 2.0.0'\n"})
        path, version, _ = mod.find_binary()
        self.assertTrue(path.endswith("ast-grep"))
        self.assertEqual(version, "ast-grep 1.0.0")

    def test_nothing_on_path(self):
        self.only_path()
        path, _, tried = mod.find_binary()
        self.assertIsNone(path)
        self.assertEqual(len(tried), 2)

    def test_override_wins_and_does_not_fall_through(self):
        self.only_path(**{"ast-grep": "#!/bin/sh\necho 'ast-grep 1.0.0'\n"})
        self.env(PEACH_AST_GREP_BIN=self.tmp / "no-such-binary")
        path, _, tried = mod.find_binary()
        self.assertIsNone(path)                      # the good ast-grep on PATH is NOT used
        self.assertEqual(len(tried), 1)

    def test_override_must_identify_as_ast_grep(self):
        script = self.write("bin/other", "#!/bin/sh\necho 'other tool 1.0'\n")
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        self.env(PEACH_AST_GREP_BIN=script)
        self.assertIsNone(mod.find_binary()[0])


# ------------------------------------------------- the runner, with a fake binary ----

FAKE_AST_GREP = """#!{python}
import json, os, sys, time
args = sys.argv[1:]
if args[:1] == ["--version"]:
    print(os.environ.get("FAKE_VERSION", "ast-grep 9.9.9-fake"))
    sys.exit(0)
log = os.environ.get("FAKE_ARGV_LOG")
if log:
    with open(log, "a") as fh:
        fh.write(json.dumps(args) + "\\n")
discovery = ("--inline-rules" in args and
             json.loads(args[args.index("--inline-rules") + 1].split("\\n---\\n")[0])["id"].startswith("import."))
alias = "--rule" in args
mode = os.environ.get("FAKE_ALIAS_MODE" if alias else "FAKE_DISCOVERY_MODE" if discovery else "FAKE_MODE", "ok")
if mode == "sleep":
    time.sleep(30)
if mode == "fail":
    sys.stderr.write("boom: something went wrong\\n")
    sys.exit(2)
if mode == "exit1_stderr":
    sys.stderr.write("error: bad rule\\n")
    sys.exit(1)
if mode == "exit1_silent":
    sys.exit(1)
out = os.environ.get("FAKE_ALIAS_STDOUT" if alias else "FAKE_DISCOVERY_STDOUT" if discovery else "FAKE_STDOUT")
if out:
    sys.stdout.write(open(out).read())
if mode == "exit1_with_output":
    sys.exit(1)
"""


class FakeCase(TmpCase):
    """A fake ast-grep selected through PEACH_AST_GREP_BIN."""

    def setUp(self):
        super().setUp()
        self.fake = self.write("bin/fake-ast-grep", FAKE_AST_GREP.format(python=sys.executable))
        self.fake.chmod(self.fake.stat().st_mode | stat.S_IXUSR)
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        self.env(PEACH_AST_GREP_BIN=self.fake)
        self.ctx = None

    def emit(self, *objs, raw=None):
        """Make every fake run print these JSON objects, one per line (or `raw`)."""
        text = raw if raw is not None else "\n".join(json.dumps(o) for o in objs) + "\n"
        self.env(FAKE_STDOUT=self.write("fake_stdout.jsonl", text))

    def match(self, rel, needle, rule_id, callee=None, byte_offset=True, shift=0, file=None,
              captured=True, with_rule_id=True):
        """An ast-grep style JSON match of `needle` in repo/rel, as `scan` prints it."""
        raw = (self.repo / rel).read_bytes()
        s = raw.index(needle.encode("utf-8"))
        e = s + len(needle.encode("utf-8"))
        line_start = raw.rfind(b"\n", 0, s) + 1
        rng = {"start": {"line": raw.count(b"\n", 0, s),            # 0-based, as ast-grep prints
                         "column": len(raw[line_start:s].decode("utf-8"))},   # characters
               "end": {"line": raw.count(b"\n", 0, e),
                       "column": len(raw[raw.rfind(b"\n", 0, e) + 1:e].decode("utf-8"))}}
        if byte_offset:
            rng["byteOffset"] = {"start": s + shift, "end": e + shift}
        obj = {"text": needle, "range": rng, "file": file or rel, "language": "Python",
               "severity": "hint", "message": "x"}
        if with_rule_id:
            obj["ruleId"] = rule_id
        if captured:
            callee = callee if callee is not None else needle.split("(")[0]
            obj["metaVariables"] = {"single": {"F": {"text": callee}}, "multi": {}, "transformed": {}}
        return obj

    def scan(self, files, language="python"):
        return mod.run_patterns(self.repo, {language: list(files)}, ctx=self.ctx)

    def reasons(self, report, cause):
        return [r for r in report["reasons"] if r["cause"] == cause]


class FakeBinaryRunTests(FakeCase):
    def test_alias_deduplication_prefers_plain_and_counts_filtered_drops(self):
        self.write("repo/c.py", 'requests.get("u")\nr.get("u")\nh.get("u")\n')
        plain_obj = self.match("c.py", 'requests.get("u")', "py.requests")
        by_id = {e.id: e for e in family("python")}
        views = {}
        plain, problem = mod._hit_from_json(plain_obj, by_id, "python", self.repo, views)
        self.assertIsNone(problem)
        seen = {mod._match_key(plain_obj, plain)}
        duplicate = dict(plain_obj, ruleId="py.requests#alias")
        objects = [duplicate, self.match("c.py", 'r.get("u")', "py.requests#alias"),
                   self.match("c.py", 'h.get("u")', "py.requests#alias"),
                   self.match("c.py", 'h.get("u")', "py.httpx#alias")]
        report = {"client_instances": [], "reasons": []}
        hits = [plain]
        stats = mod._process_alias_matches(objects, "python", {"c.py": [
            ("requests", "requests", None), ("h", "httpx", None)]}, self.repo, report, hits, views, seen)
        self.assertEqual(report["reasons"], [])
        self.assertEqual([(h.callee_text, h.pattern_id, h.via_alias) for h in hits],
                         [("requests.get", "py.requests", False), ("h.get", "py.httpx", True)])
        self.assertEqual((stats["alias_hits_raw"], stats["alias_hits_kept"], stats["alias_hits_dropped"]),
                         (4, 1, 3))

    def test_oversized_alias_rules_preserve_plain_results(self):
        self.write("repo/c.py", 'import requests as r\nrequests.get("u")\n')
        self.emit(self.match("c.py", 'requests.get("u")', "py.requests"))
        discovery = self.match("c.py", "import requests as r", "import.python", captured=False)
        self.env(FAKE_DISCOVERY_STDOUT=self.write("discovery.jsonl", json.dumps(discovery)))
        with mock.patch.object(mod, "MAX_ALIAS_RULE_BYTES", 1):
            hits, report = self.scan(["c.py"])
        self.assertEqual([(h.callee_text, h.via_alias) for h in hits], [("requests.get", False)])
        self.assertEqual([r["cause"] for r in report["reasons"]], ["alias_rules_too_large"])
        self.assertEqual(report["languages"]["python"]["runs"], 1)
        self.assertEqual(report["languages"]["python"]["alias_pass_launches"], 0)

    def test_alias_pass_failure_preserves_plain_results(self):
        self.write("repo/c.py", 'import requests as r\nrequests.get("u")\n')
        self.emit(self.match("c.py", 'requests.get("u")', "py.requests"))
        discovery = self.match("c.py", "import requests as r", "import.python", captured=False)
        self.env(FAKE_DISCOVERY_STDOUT=self.write("discovery.jsonl", json.dumps(discovery)), FAKE_ALIAS_MODE="fail")
        hits, report = self.scan(["c.py"])
        self.assertEqual([h.callee_text for h in hits], ["requests.get"])
        self.assertEqual([r["cause"] for r in report["reasons"]], ["exit_nonzero"])
        self.assertIn("alias pass", report["reasons"][0]["detail"])

    def test_alias_transport_cleans_up_temporary_rules_on_success_failure_and_exception(self):
        import subprocess
        rules = mod._rules_text(family("python")[:2], "python")
        paths = []
        for outcome in (0, 2, "exception"):
            def run(cmd, cwd=None):
                self.assertEqual(cmd[:3], [str(self.fake), "scan", "--rule"])
                self.assertNotIn("--inline-rules", cmd)
                path = Path(cmd[3])
                paths.append(path)
                self.assertEqual(path.read_text(), rules)
                if outcome == "exception":
                    raise RuntimeError("test launch exception")
                return subprocess.CompletedProcess(cmd, outcome, "", "bad" if outcome else ""), None
            with mock.patch.object(mod, "_run", side_effect=run):
                if outcome == "exception":
                    with self.assertRaisesRegex(RuntimeError, "test launch exception"):
                        mod._run_one(str(self.fake), rules, ["a.py"], self.repo, rules_file=True)
                else:
                    objs, failure = mod._run_one(str(self.fake), rules, ["a.py"], self.repo, rules_file=True)
                    self.assertEqual(objs, [])
                    self.assertEqual(failure is None, outcome == 0)
            self.assertFalse(paths[-1].exists())

    def test_alias_pass_scans_binding_files_only_with_exact_chunking_and_limits(self):
        bindings = {"a.py": [("r", "requests", None)], "b.py": [],
                    "c.py": [("h", "httpx", None)], "d.py": [("g", "requests", "get")]}
        report = {"reasons": []}
        with mock.patch.object(mod, "FILES_PER_CALL", 2), mock.patch.object(
                mod, "_run_one", return_value=([], None)) as run:
            _, stats = mod._run_alias_pass(str(self.fake), "python", "python", bindings, self.repo, report)
            self.assertEqual([c.args[2] for c in run.call_args_list], [["a.py", "c.py"], ["d.py"]])
            self.assertTrue(all(c.kwargs == {"rules_file": True} for c in run.call_args_list))
        self.assertEqual((stats["alias_files"], stats["alias_pass_launches"]), (3, 2))
        with mock.patch.object(mod, "_run_one") as run:
            _, empty = mod._run_alias_pass(str(self.fake), "python", "python", {"b.py": []}, self.repo, report)
            run.assert_not_called()
        self.assertEqual(empty["alias_pass_launches"], 0)
        with mock.patch.object(mod, "MAX_ALIAS_RULE_BYTES", 1), mock.patch.object(mod, "_run_one") as run:
            _, limited = mod._run_alias_pass(str(self.fake), "python", "python", bindings, self.repo, report)
            run.assert_not_called()
        self.assertEqual(limited["alias_pass_launches"], 0)
        self.assertEqual([r["cause"] for r in report["reasons"]], ["alias_rules_too_large"])

    def test_client_instances_do_not_change_smoke_exit_status(self):
        import io
        from contextlib import redirect_stdout
        from externalcalls import ast_grep_smoke as smoke
        self.write("repo/c.py", "requests.Session()\n")
        self.emit(self.match("c.py", "requests.Session()", "py.requests.Session"))
        hits, report = self.scan(["c.py"])
        self.assertEqual(hits, [])
        self.assertEqual(report["reasons"], [])
        self.assertEqual(len(report["client_instances"]), 1)
        with mock.patch.object(smoke, "run_patterns", return_value=(hits, report)), redirect_stdout(io.StringIO()):
            self.assertEqual(smoke.main([str(self.repo)]), 0)
            report["client_instances"] = []
            self.assertEqual(smoke.main([str(self.repo)]), 0)

    def test_factories_are_reported_separately_with_sorted_capped_samples(self):
        self.write("repo/z.py", 's = requests.Session()\ns.get("u")\n')
        self.write("repo/a.py", 's = requests.Session()\nwith requests.Session() as c: pass\n'
                   'a = httpx.AsyncClient()\nb = requests.session()\nrequests.get("u")\n')
        objects = [self.match("z.py", "requests.Session()", "py.requests.Session"),
                   self.match("a.py", "requests.session()", "py.requests.session"),
                   self.match("a.py", "httpx.AsyncClient()", "py.httpx.AsyncClient")]
        first = self.match("a.py", "requests.Session()", "py.requests.Session")
        second = json.loads(json.dumps(first))
        raw = (self.repo / "a.py").read_bytes()
        start = raw.index(b"requests.Session()", first["range"]["byteOffset"]["end"])
        second["range"]["byteOffset"] = {"start": start, "end": start + len(b"requests.Session()")}
        objects.extend([second, first, self.match("a.py", 'requests.get("u")', "py.requests")])
        self.emit(*objects)
        hits, report = self.scan(["z.py", "a.py"])
        self.assertEqual(report["reasons"], [])
        self.assertEqual([h.callee_text for h in hits], ["requests.get"])
        self.assertEqual(report["client_instances"], [
            {"file": "a.py", "language": "python", "count": 4, "via_alias_count": 0,
             "factories": ["py.httpx.AsyncClient", "py.requests.Session", "py.requests.session"],
             "samples": ["1:4 requests.Session", "2:5 requests.Session", "3:4 httpx.AsyncClient"],
             "kind": "client_instance_untracked",
             "impact": "calls made through this client object are not detected"},
            {"file": "z.py", "language": "python", "count": 1, "via_alias_count": 0, "factories": ["py.requests.Session"],
             "samples": ["1:4 requests.Session"], "kind": "client_instance_untracked",
             "impact": "calls made through this client object are not detected"}])
        self.assertEqual(report["languages"]["python"]["client_factory_hits"], 5)
        self.assertEqual(report["languages"]["python"]["hits"], 1)
        self.assertEqual(report["languages"]["python"]["runs"], 1)

    def test_factory_with_bad_offsets_is_dropped_instead_of_reported(self):
        self.write("repo/c.py", "requests.Session()\n")
        self.emit(self.match("c.py", "requests.Session()", "py.requests.Session", shift=1))
        hits, report = self.scan(["c.py"])
        self.assertEqual(hits, [])
        self.assertEqual(report["client_instances"], [])
        self.assertEqual([r["cause"] for r in report["reasons"]], ["offset_mismatch"])
        self.assertEqual(report["languages"]["python"]["client_factory_hits"], 0)

    def test_one_bad_discovery_chunk_discards_all_bucket_bindings(self):
        for name in ("a.py", "b.py"):
            self.write("repo/" + name, "import requests as r\n")
        obj = self.match("a.py", "import requests as r", "import.python", captured=False)
        report = {"reasons": []}
        with mock.patch.object(mod, "FILES_PER_CALL", 1), mock.patch.object(
                mod, "_run_one", side_effect=[([obj], None), ([], ("timeout", "timed out"))]) as run:
            bindings, stats = mod._discover_bindings(str(self.fake), "python", "python",
                                                    ["a.py", "b.py"], self.repo, report)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(bindings, {"a.py": [], "b.py": []})
        self.assertEqual((stats["files_scanned"], stats["files_with_aliases"], stats["discovery_runs"]),
                         (2, 0, 2))
        self.assertEqual([r["cause"] for r in report["reasons"]], ["alias_discovery_failed"])

    def test_discovery_failure_keeps_plain_hits_and_reports_unknown_alias_coverage(self):
        self.write("repo/c.py", 'import requests as r\nrequests.get("u")\n')
        self.emit(self.match("c.py", 'requests.get("u")', "py.requests"))
        for mode in ("fail", "exit1_stderr", "sleep"):
            with self.subTest(mode):
                self.env(FAKE_DISCOVERY_MODE=mode, PEACH_AST_GREP_TIMEOUT="1")
                hits, report = self.scan(["c.py"])
                self.assertEqual([h.callee_text for h in hits], ["requests.get"])
                self.assertEqual([r["cause"] for r in report["reasons"]], ["alias_discovery_failed"])
                self.assertIn("alias calls may be missed", report["reasons"][0]["detail"])
                self.assertEqual(report["languages"]["python"]["files_with_aliases"], 0)
                self.assertEqual(report["languages"]["python"]["runs"], 1)
                self.assertEqual(report["languages"]["python"]["discovery_runs"], 1)

    def test_discovery_malformed_output_does_not_hide_plain_results(self):
        self.write("repo/c.py", 'requests.get("u")\n')
        self.emit(self.match("c.py", 'requests.get("u")', "py.requests"))
        self.env(FAKE_DISCOVERY_STDOUT=self.write("bad-discovery.jsonl", "not json\n"))
        hits, report = self.scan(["c.py"])
        self.assertEqual([h.callee_text for h in hits], ["requests.get"])
        self.assertEqual([r["cause"] for r in report["reasons"]], ["alias_discovery_failed"])
        self.assertIn("unreadable_output", report["reasons"][0]["detail"])

    # ---- positions
    def test_non_ascii_before_call_gives_byte_columns(self):
        src = (FIX / "unicode.py").read_text(encoding="utf-8")
        self.write("repo/unicode.py", src)
        call = 'requests.get("https://unicode.example.com")'
        self.emit(self.match("unicode.py", call, "py.requests", callee="requests.get"))
        hits, report = self.scan(["unicode.py"])
        self.assertEqual(report["reasons"], [])
        self.assertEqual(len(hits), 1)
        line = src.splitlines()[2]
        byte_col = len(line[:line.index("requests.get")].encode("utf-8"))
        self.assertNotEqual(byte_col, line.index("requests.get"), "fixture must have non-ASCII before the call")
        h = hits[0]
        self.assertEqual((h.file, h.start_line, h.start_col), ("unicode.py", 3, byte_col))
        self.assertEqual((h.end_line, h.end_col), (3, byte_col + len(call.encode("utf-8"))))
        self.assertEqual((h.pattern_id, h.language, h.callee_text), ("py.requests", "python", "requests.get"))

    def test_crlf_file(self):
        self.write("repo/crlf.py", b'import requests\r\n\r\nx = requests.get("u")\r\n')
        self.emit(self.match("crlf.py", 'requests.get("u")', "py.requests"))
        hits, report = self.scan(["crlf.py"])
        self.assertEqual(report["reasons"], [])
        self.assertEqual([(h.start_line, h.start_col, h.end_line) for h in hits], [(3, 4, 3)])

    def test_multiline_call_has_distinct_end_line(self):
        self.write("repo/multi.py", 'import requests\nx = requests.get(\n    "u",\n)\n')
        call = 'requests.get(\n    "u",\n)'
        self.emit(self.match("multi.py", call, "py.requests", callee="requests.get"))
        hits, _ = self.scan(["multi.py"])
        self.assertEqual([(h.start_line, h.start_col, h.end_line, h.end_col) for h in hits], [(2, 4, 4, 1)])

    def test_callee_split_over_lines_is_normalised(self):
        self.write("repo/c.js", 'axios\n  .get("u")\n  .then(f);\n')
        self.emit(self.match("c.js", 'axios\n  .get("u")', "js.axios.method", callee="axios\n  .get"))
        hits, report = self.scan(["c.js"], "javascript")
        self.assertEqual(report["reasons"], [])
        self.assertEqual([(h.callee_text, h.start_line, h.end_line) for h in hits], [("axios.get", 1, 2)])

    def test_type_arguments_do_not_leak_into_the_callee(self):
        self.write("repo/a.ts", 'const r = await axios.get<User>("u");\n')
        self.emit(self.match("a.ts", 'axios.get<User>("u")', "js.axios.method", callee="axios.get"))
        hits, report = self.scan(["a.ts"], "typescript")
        self.assertEqual(report["reasons"], [])
        self.assertEqual([(h.pattern_id, h.callee_text, h.language) for h in hits],
                         [("js.axios.method", "axios.get", "typescript")])

    def test_absolute_and_dot_slash_paths_are_normalised(self):
        self.write("repo/pkg/c.py", 'import requests\nx = requests.get("u")\n')
        for printed in (str(self.repo / "pkg" / "c.py"), "./pkg/c.py", "pkg\\c.py"):
            with self.subTest(printed=printed):
                self.emit(self.match("pkg/c.py", 'requests.get("u")', "py.requests", file=printed))
                hits, _ = self.scan(["pkg/c.py"])
                self.assertEqual([h.file for h in hits], ["pkg/c.py"])

    # ---- unreadable output: never guessed
    def test_missing_byte_offset_is_unreadable_output_not_a_guess(self):
        src = (FIX / "unicode.py").read_text(encoding="utf-8")
        self.write("repo/unicode.py", src)
        call = 'requests.get("https://unicode.example.com")'
        obj = self.match("unicode.py", call, "py.requests", byte_offset=False)   # line/column ARE present
        self.emit(obj, obj)
        hits, report = self.scan(["unicode.py"])
        self.assertEqual(hits, [])
        (r,) = report["reasons"]
        check_reason_shape(self, r)
        self.assertEqual((r["kind"], r["cause"], r["language"]), ("ast_grep_language_failed", "unreadable_output", "python"))
        self.assertEqual(r["detail"], "missing range.byteOffset")
        self.assertEqual((r["count"], r["samples"]), (2, ["unicode.py"]))

    def test_missing_captured_callee_is_unreadable_output(self):
        self.write("repo/c.py", 'import requests\nx = requests.get("u")\n')
        self.emit(self.match("c.py", 'requests.get("u")', "py.requests", captured=False))
        hits, report = self.scan(["c.py"])
        self.assertEqual(hits, [])
        self.assertEqual(report["reasons"][0]["detail"], "missing metaVariables.single.F.text")

    def test_missing_rule_id_and_unknown_rule_id(self):
        self.write("repo/c.py", 'import requests\nx = requests.get("u")\n')
        self.emit(self.match("c.py", 'requests.get("u")', None, with_rule_id=False),
                  self.match("c.py", 'requests.get("u")', "py.not.in.catalogue"))
        hits, report = self.scan(["c.py"])
        self.assertEqual(hits, [])
        self.assertEqual([r["cause"] for r in report["reasons"]], ["unknown_rule_id", "unknown_rule_id"])
        self.assertEqual(sum(r["count"] for r in report["reasons"]), 2)
        self.assertIn("None", report["reasons"][0]["detail"])
        self.assertIn("py.not.in.catalogue", report["reasons"][1]["detail"])

    def test_json_array_output_is_not_accepted(self):
        self.write("repo/c.py", 'import requests\nx = requests.get("u")\n')
        self.emit(raw=json.dumps([self.match("c.py", 'requests.get("u")', "py.requests")]) + "\n")
        hits, report = self.scan(["c.py"])
        self.assertEqual(hits, [])
        self.assertEqual([r["cause"] for r in report["reasons"]], ["unreadable_output"])

    def test_unparseable_output_is_a_reason(self):
        self.write("repo/c.py", "x = 1\n")
        self.emit(raw="this is not json\n")
        hits, report = self.scan(["c.py"])
        self.assertEqual(hits, [])
        r = report["reasons"][0]
        self.assertEqual(r["cause"], "unreadable_output")
        self.assertIn("stdout is not JSON lines", r["detail"])
        self.assertEqual(r["samples"], ["c.py"])

    def test_wrong_offsets_are_dropped_and_counted(self):
        self.write("repo/c.py", 'import requests\nx = requests.get("u")\n')
        self.emit(self.match("c.py", 'requests.get("u")', "py.requests", shift=1))
        hits, report = self.scan(["c.py"])
        self.assertEqual(hits, [])
        r = self.reasons(report, "offset_mismatch")[0]
        self.assertEqual((r["count"], r["samples"]), (1, ["c.py"]))

    def test_lone_cr_and_invalid_utf8_files_are_dropped_and_counted(self):
        cases = {"cr.py": ("lone_cr", b'import requests\rx = requests.get("u")\n'),
                 "bad.py": ("invalid_utf8", b'# \xff\nx = requests.get("u")\n')}
        for name, (cause, data) in cases.items():
            with self.subTest(name):
                self.write(f"repo/{name}", data)
                obj = self.match(name, 'requests.get("u")', "py.requests")
                self.emit(obj, obj)
                hits, report = self.scan([name])
                self.assertEqual(hits, [])
                (r,) = self.reasons(report, cause)
                self.assertEqual((r["count"], r["samples"]), (2, [name]))

    # ---- exit codes and process failures
    def test_exit_1_with_no_output_at_all_means_no_matches(self):
        self.write("repo/c.py", "x = 1\n")
        self.env(FAKE_MODE="exit1_silent")
        hits, report = self.scan(["c.py"])
        self.assertEqual((hits, report["reasons"]), ([], []))
        self.assertEqual(report["languages"]["python"]["hits"], 0)

    def test_exit_1_with_stderr_is_a_failure_carrying_the_stderr(self):
        self.write("repo/c.py", "x = 1\n")
        self.env(FAKE_MODE="exit1_stderr")
        hits, report = self.scan(["c.py"])
        self.assertEqual(hits, [])
        (r,) = report["reasons"]
        check_reason_shape(self, r)
        self.assertEqual(r["cause"], "exit_nonzero")
        self.assertIn("exit 1", r["detail"])
        self.assertIn("bad rule", r["detail"])
        self.assertEqual(r["samples"], ["c.py"])

    def test_exit_1_with_stdout_is_a_failure(self):
        self.write("repo/c.py", 'import requests\nx = requests.get("u")\n')
        self.emit(self.match("c.py", 'requests.get("u")', "py.requests"))
        self.env(FAKE_MODE="exit1_with_output")
        hits, report = self.scan(["c.py"])
        self.assertEqual(hits, [])
        self.assertEqual(report["reasons"][0]["cause"], "exit_nonzero")
        self.assertIn("stdout was not empty", report["reasons"][0]["detail"])

    def test_other_nonzero_exit_is_a_failure_with_stderr(self):
        self.write("repo/c.py", "x = 1\n")
        self.env(FAKE_MODE="fail")
        hits, report = self.scan(["c.py"])
        self.assertEqual(hits, [])
        r = report["reasons"][0]
        self.assertEqual(r["cause"], "exit_nonzero")
        self.assertIn("exit 2", r["detail"])
        self.assertIn("boom", r["detail"])

    def test_timeout_is_a_countable_reason(self):
        self.write("repo/c.py", "x = 1\n")
        self.env(FAKE_MODE="sleep", PEACH_AST_GREP_TIMEOUT="1")
        hits, report = self.scan(["c.py"])
        self.assertEqual(hits, [])
        (r,) = report["reasons"]
        self.assertEqual((r["cause"], r["count"], r["samples"]), ("timeout", 1, ["c.py"]))
        self.assertIn("timed out after 1s", r["detail"])

    def test_failures_in_every_chunk_add_up(self):
        names = [f"f{i}.py" for i in range(5)]
        for n in names:
            self.write(f"repo/{n}", "\n")
        self.env(FAKE_MODE="fail")
        with mock.patch.object(mod, "FILES_PER_CALL", 2):
            _, report = self.scan(names)
        (r,) = report["reasons"]
        self.assertEqual((r["cause"], r["count"]), ("exit_nonzero", 3))
        self.assertEqual(len(r["samples"]), mod.MAX_SAMPLES)

    # ---- stage-level problems
    def test_bogus_override_gives_empty_hits_and_a_binary_reason(self):
        self.env(PEACH_AST_GREP_BIN=self.tmp / "definitely-not-here")
        self.write("repo/c.py", "x = 1\n")
        hits, report = self.scan(["c.py"])
        self.assertEqual(hits, [])
        self.assertIsNone(report["binary"])
        (r,) = report["reasons"]
        check_reason_shape(self, r)
        self.assertEqual((r["kind"], r["cause"], r["language"]), ("ast_grep_unavailable", "binary_missing", ""))
        self.assertIn("definitely-not-here", r["detail"])

    def test_never_raises_on_nonsense_input(self):
        hits, report = mod.run_patterns(None, None)
        self.assertEqual(hits, [])
        (r,) = report["reasons"]
        self.assertEqual((r["kind"], r["cause"]), ("stage_failed", "exception"))

    def test_missing_files_and_unknown_language_are_counted_with_samples(self):
        hits, report = mod.run_patterns(self.repo, {"python": ["nope.py", "nada.py"], "ruby": ["a.rb"]})
        self.assertEqual(hits, [])
        (missing,) = self.reasons(report, "missing_file")
        self.assertEqual((missing["language"], missing["count"], missing["samples"]), ("python", 2, ["nada.py", "nope.py"]))
        (unknown,) = self.reasons(report, "unknown_language")
        self.assertEqual((unknown["language"], unknown["count"], unknown["samples"]), ("ruby", 1, ["a.rb"]))
        for r in report["reasons"]:
            check_reason_shape(self, r)

    def test_same_reason_merges_and_samples_are_capped(self):
        names = [f"m{i}.py" for i in range(9)]
        _, report = mod.run_patterns(self.repo, {"python": names})
        (r,) = report["reasons"]
        self.assertEqual((r["cause"], r["count"]), ("missing_file", 9))
        self.assertEqual(len(r["samples"]), mod.MAX_SAMPLES)

    # ---- command line and process count
    def test_one_process_per_bucket_with_all_rules_in_it(self):
        for name in ("a.ts", "b.tsx", "c.js", "d.py"):
            self.write(f"repo/{name}", "\n")
        log = self.tmp / "argv.jsonl"
        self.env(FAKE_ARGV_LOG=log)
        mod.run_patterns(self.repo, {"typescript": ["b.tsx", "a.ts"], "javascript": ["c.js"], "python": ["d.py"]})
        runs = [json.loads(l) for l in log.read_text().splitlines()]
        plain_runs = [a for a in runs if not rules_of(a)[0]["id"].startswith("import.")]
        discovery_runs = [a for a in runs if rules_of(a)[0]["id"].startswith("import.")]
        self.assertEqual(len(plain_runs), 4, "expected exactly one plain process per language bucket")
        self.assertEqual(len(discovery_runs), 4, "expected exactly one discovery process per language bucket")
        self.assertEqual(len(runs), 8)
        self.assertEqual(sum("--rule" in a for a in runs), 0, "no bindings means no alias pass")
        seen = {}
        discovery_seen = {}
        for args in runs:
            self.assertEqual(args[0], "scan")
            self.assertIn("--json=stream", args)
            for forbidden in ("run", "--pattern", "--lang"):
                self.assertNotIn(forbidden, args)
            docs = rules_of(args)
            lang = docs[0]["language"]
            self.assertEqual({d["language"] for d in docs}, {lang})
            parser_lang = "python" if lang == "python" else "javascript"
            if docs[0]["id"].startswith("import."):
                self.assertEqual(docs, mod._import_rule_docs(lang))
                discovery_seen[lang] = tuple(args[args.index("--") + 1:])
                continue
            self.assertEqual([d["id"] for d in docs], [e.id for e in family(parser_lang)])
            self.assertEqual(docs, [mod._rule_doc(e, lang) for e in family(parser_lang)])
            seen[lang] = tuple(args[args.index("--") + 1:])
        self.assertEqual(seen, {"typescript": ("a.ts",), "tsx": ("b.tsx",), "javascript": ("c.js",), "python": ("d.py",)})
        self.assertEqual(discovery_seen, seen)

    def test_files_are_chunked(self):
        names = [f"f{i}.py" for i in range(5)]
        for n in names:
            self.write(f"repo/{n}", "\n")
        log = self.tmp / "argv.jsonl"
        self.env(FAKE_ARGV_LOG=log)
        with mock.patch.object(mod, "FILES_PER_CALL", 2):
            _, report = self.scan(names)
        runs = [json.loads(l) for l in log.read_text().splitlines()]
        plain_runs = [a for a in runs if not rules_of(a)[0]["id"].startswith("import.")]
        discovery_runs = [a for a in runs if rules_of(a)[0]["id"].startswith("import.")]
        self.assertEqual(sorted(len(a[a.index("--") + 1:]) for a in plain_runs), [1, 2, 2])
        self.assertEqual(sorted(len(a[a.index("--") + 1:]) for a in discovery_runs), [1, 2, 2])
        self.assertEqual([a[a.index("--") + 1:] for a in discovery_runs],
                         [a[a.index("--") + 1:] for a in plain_runs])
        self.assertEqual(len(runs), 6)
        self.assertEqual(report["languages"]["python"]["runs"], 3)
        self.assertEqual(report["languages"]["python"]["discovery_runs"], 3)
        self.assertEqual(report["languages"]["python"]["alias_pass_launches"], 0)

    def test_ctx_counters_and_report(self):
        self.write("repo/c.py", 'import requests\nx = requests.get("u")\ny = requests.get("u")\n')
        self.emit(self.match("c.py", 'requests.get("u")', "py.requests"))
        self.ctx = Ctx()
        hits, report = self.scan(["c.py"])
        self.assertEqual(len(hits), 1)
        self.assertEqual(self.ctx.counts["ast_grep.hits"], 1)
        self.assertEqual(self.ctx.counts["ast_grep.files"], 1)
        self.assertEqual(self.ctx.counts["ast_grep.runs"], 1)
        self.assertEqual(report["version"], "ast-grep 9.9.9-fake")
        self.assertEqual(set(report), {"binary", "version", "reasons", "languages", "client_instances"})
        self.assertEqual(report["client_instances"], [])
        st = report["languages"]["python"]
        self.assertEqual((st["files"], st["runs"], st["hits"], st["dropped"]), (1, 1, 1, 0))


# -------------------------------------------------- the real ast-grep (skippable) ----

def _line_of(path, needle):
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if needle in line:
            return i
    raise AssertionError(f"{needle!r} not in {path}")


@unittest.skipIf(REAL_BIN is None, NO_REAL)
class BindingDiscoveryTests(TmpCase):
    def test_alias_rule_file_transport_in_all_real_grammars(self):
        for lang, rel, callee in (("python", "alias.py", "r.get"), ("javascript", "alias.js", "ax.get"),
                                  ("typescript", "alias.ts", "nestedAx.get"), ("tsx", "alias.tsx", "nestedAx.get")):
            with self.subTest(lang):
                pl = "typescript" if lang == "tsx" else lang
                report = {"reasons": []}
                bindings, _ = mod._discover_bindings(REAL_BIN, pl, lang, [rel], FIX, report)
                objs, stats = mod._run_alias_pass(REAL_BIN, pl, lang, bindings, FIX, report)
                self.assertEqual(report["reasons"], [])
                self.assertEqual(stats["alias_pass_launches"], 1)
                self.assertEqual(stats["alias_files"], 1)
                self.assertIn(callee, [o["metaVariables"]["single"]["F"]["text"] for o in objs])
                self.assertTrue(all(o["ruleId"].endswith("#alias") for o in objs))

    def test_every_import_form_and_exact_per_file_bindings(self):
        expected_py = [("g", "requests", "get"), ("get", "requests", "get"),
                       ("inner", "httpx", None), ("p", "requests", "post"),
                       ("put", "requests", "put"), ("r", "requests", None),
                       ("r2", "requests", None), ("request", "urllib.request", None),
                       ("ur", "urllib.request", None), ("urlopen", "urllib.request", "urlopen")]
        expected_js = [("AxiosError", "axios", "AxiosError"), ("ax", "axios", None),
                       ("g", "axios", "get"), ("get", "axios", "get"), ("h", "http", None),
                       ("hs", "https", None), ("nestedAx", "axios", None), ("ns", "axios", None),
                       ("p", "axios", "post"), ("put", "axios", "put"),
                       ("required", "axios", None), ("rg", "axios", "get")]
        for lang, rel in (("python", "alias.py"), ("javascript", "alias.js"),
                          ("typescript", "alias.ts"), ("tsx", "alias.tsx")):
            with self.subTest(lang):
                report = {"reasons": []}
                bindings, stats = mod._discover_bindings(
                    REAL_BIN, "typescript" if lang == "tsx" else lang, lang, [rel], FIX, report)
                expected = expected_py if lang == "python" else expected_js
                if lang in ("typescript", "tsx"):
                    expected = sorted(expected + [("axEquals", "axios", None), ("headAlias", "axios", "head")])
                self.assertEqual(report["reasons"], [])
                self.assertEqual(bindings, {rel: expected})
                self.assertEqual((stats["files_scanned"], stats["files_with_aliases"],
                                  stats["alias_names"], stats["discovery_runs"], stats["discovery_skipped_files"]),
                                 (1, 1, len(expected), 1, 0))

    def test_bindings_do_not_leak_between_files_or_from_other_modules(self):
        files = ["alias_cross/requests_a.py", "alias_cross/httpx_b.py",
                 "alias_cross/no_import.py", "alias_cross/other.py"]
        report = {"reasons": []}
        bindings, stats = mod._discover_bindings(REAL_BIN, "python", "python", files, FIX, report)
        self.assertEqual(report["reasons"], [])
        self.assertEqual(bindings, {files[0]: [("r", "requests", None)],
                                    files[1]: [("r", "httpx", None)], files[2]: [], files[3]: []})
        self.assertEqual((stats["files_scanned"], stats["files_with_aliases"], stats["alias_names"]), (4, 2, 1))

    def test_typescript_and_tsx_stats_count_distinct_names_together(self):
        hits, report = mod.run_patterns(FIX, {"typescript": ["alias.ts", "alias.tsx"]})
        self.assertEqual(report["reasons"], [])
        stats = report["languages"]["typescript"]
        self.assertEqual((stats["files_scanned"], stats["files_with_aliases"], stats["alias_names"],
                          stats["runs"], stats["discovery_runs"]), (2, 2, 14, 2, 2))
        # Plain results remain exact; alias results are separately required in full.
        self.assertEqual([h.callee_text for h in hits if not h.via_alias], ["axios.get", "axios.get"])
        expected_aliases = ["nestedAx.get", "ax.get", "ns.post", "g", "get", "rg", "required",
                            "p", "put", "h.get", "hs.request", "axEquals.get", "headAlias"]
        self.assertEqual([h.callee_text for h in hits if h.via_alias], expected_aliases * 2)
        self.assertEqual((stats["alias_pass_launches"], stats["alias_hits_raw"],
                          stats["alias_hits_kept"], stats["alias_hits_dropped"]), (2, 26, 26, 0))


@unittest.skipIf(REAL_BIN is None, NO_REAL)
class RealBinaryTests(TmpCase):

    def test_awaited_generic_alias_callee_positions(self):
        for rel in ("awaited_alias.ts", "awaited_alias.tsx"):
            with self.subTest(rel):
                hits, _ = self.run_fixture("typescript", rel)
                self.assert_alias_positions(hits, rel, [
                    (3, "ax.get", "js.axios.method"), (4, "get", "js.axios.method")])
                self.assertEqual([h.start_col for h in hits], [6, 6])
                self.assertTrue(all(not h.matched_text.startswith("await") for h in hits))

    def test_persisted_aliased_factories_only_enter_client_instances(self):
        for language, rel, count, entry, samples in [
            ("python", "aliased_factories.py", 2, "py.requests.Session", ["3:4 r.Session", "4:0 S"]),
            ("javascript", "aliased_factories.js", 1, "js.axios.create", ["2:0 ax.create"]),
        ]:
            with self.subTest(rel):
                hits, report = self.run_fixture(language, rel)
                self.assertEqual(hits, [])
                self.assertEqual(report["client_instances"], [{
                    "file": rel, "language": language, "count": count, "via_alias_count": count,
                    "factories": [entry], "samples": samples, "kind": "client_instance_untracked",
                    "impact": "calls made through this client object are not detected"}])
                self.assertEqual(report["languages"][language]["client_factory_hits"], count)
                print("REAL FACTORY", report["client_instances"])

    def assert_alias_positions(self, hits, rel, expected):
        self.assertEqual([(h.start_line, h.callee_text, h.pattern_id, h.via_alias) for h in hits],
                         [(line, callee, entry, True) for line, callee, entry in expected])
        raw = (FIX / rel).read_bytes()
        lines = raw.splitlines(keepends=True)
        for hit in hits:
            line = lines[hit.start_line - 1]
            start = line.index(hit.callee_text.encode())
            end = line.rindex(b")") + 1
            self.assertEqual((hit.start_col, hit.end_line, hit.end_col), (start, hit.start_line, end))
            start_byte = sum(map(len, lines[:hit.start_line - 1])) + start
            end_byte = sum(map(len, lines[:hit.start_line - 1])) + end
            self.assertEqual(raw[start_byte:end_byte], hit.matched_text.encode())
            print("REAL POSITION", rel, hit.callee_text, hit.pattern_id,
                  (hit.start_line, hit.start_col, hit.end_line, hit.end_col),
                  "bytes", (start_byte, end_byte), "via_alias", hit.via_alias)

    def test_callable_got_and_ky_aliases(self):
        for rel in ("callable_alias.ts", "callable_alias.tsx"):
            with self.subTest(rel):
                hits, _ = self.run_fixture("typescript", rel)
                self.assert_alias_positions(hits, rel, [
                    (3, "g", "js.got"), (4, "k", "js.ky"),
                    (5, "g.get", "js.got.method"), (6, "k.post", "js.ky.method")])

    def test_real_filter_drop_examples(self):
        cases = [
            ("python", ["filter1_a.py", "filter1_b.py"],
             [("filter1_a.py", "r.get", "py.requests")],
             ("filter1_b.py", "r.get", "py.requests", "check 1")),
            ("javascript", ["filter3_a.js", "filter3_b.js"],
             [("filter3_a.js", "ax", "js.axios"),
              ("filter3_b.js", "ax", "js.axios.method")],
             ("filter3_b.js", "ax", "js.axios", "check 3")),
            ("python", ["filter4_a.py", "filter4_b.py"],
             [("filter4_a.py", "x", "py.requests.request"),
              ("filter4_b.py", "x", "py.requests")],
             ("filter4_a.py", "x", "py.requests", "check 4")),
        ]
        original = mod._alias_filter_reason
        for language, files, expected, dropped in cases:
            with self.subTest(dropped[-1]):
                decisions = []
                def audit(hit, entry, bindings):
                    reason = original(hit, entry, bindings)
                    decisions.append((hit.file, hit.callee_text, entry.id, reason))
                    return reason
                with mock.patch.object(mod, "_alias_filter_reason", side_effect=audit):
                    hits, report = self.run_fixture(language, *files)
                self.assertEqual([(h.file, h.callee_text, h.pattern_id) for h in hits], expected)
                self.assertTrue(all(h.via_alias for h in hits))
                self.assertTrue(any(d[:3] == dropped[:3] and d[3] and d[3].startswith(dropped[3])
                                    for d in decisions), decisions)
                self.assertGreaterEqual(report["languages"][language]["alias_hits_dropped"], 1)
                print("REAL FILTER", decisions)
    def test_dollar_identifier_is_literal_in_real_rust_regex(self):
        for ext in ("js", "ts", "tsx"):
            rel = "dollar." + ext
            self.write(rel, 'import $ax from "axios";\n$ax.get("x");\nax.get("x");\n')
            hits, report = self.run_fixture("javascript" if ext == "js" else "typescript", rel, root=self.tmp)
            self.assertEqual([(h.callee_text, h.pattern_id, h.via_alias) for h in hits],
                             [("$ax.get", "js.axios.method", True)])

    def test_alias_factories_and_callable_collision_are_reported_only_as_factories(self):
        self.write("factories.py", 'import requests as r\nfrom requests import Session\n'
                   'r.Session()\nSession()\nrequests.Session()\n')
        hits, report = self.run_fixture("python", "factories.py", root=self.tmp)
        self.assertEqual(hits, [])
        self.assertEqual(report["client_instances"], [{
            "file": "factories.py", "language": "python", "count": 3, "via_alias_count": 2,
            "factories": ["py.requests.Session"],
            "samples": ["3:0 r.Session", "4:0 Session", "5:0 requests.Session"],
            "kind": "client_instance_untracked", "impact": "calls made through this client object are not detected"}])
        for ext in ("js", "ts", "tsx"):
            # This real import shadows the plain axios callable with a factory.
            self.write("collision." + ext, 'import { create as axios } from "axios";\n'
                       'import ax from "axios";\naxios({});\nax.create({});\n')
            lang = "javascript" if ext == "js" else "typescript"
            hits, report = self.run_fixture(lang, "collision." + ext, root=self.tmp)
            self.assertEqual(hits, [])
            instance, = report["client_instances"]
            self.assertEqual((instance["count"], instance["via_alias_count"], instance["factories"]),
                             (2, 2, ["js.axios.create"]))
            self.assertEqual(report["languages"][lang]["hits"], 0)
            self.assertEqual(report["languages"][lang]["client_factory_hits"], 2)

    def test_no_fixture_call_is_both_an_external_hit_and_a_factory(self):
        files = {}
        for p in sorted(FIX.rglob("*")):
            if p.suffix in (".py", ".js", ".ts", ".tsx"):
                lang = {".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "typescript"}[p.suffix]
                files.setdefault(lang, []).append(p.relative_to(FIX).as_posix())
        factory_hits = []
        original = mod._record_factory
        def record(hit, report):
            factory_hits.append(hit)
            original(hit, report)
        with mock.patch.object(mod, "_record_factory", side_effect=record):
            hits, report = mod.run_patterns(FIX, files)
        self.assertEqual(report["reasons"], [])
        def byte_key(hit):
            raw = (FIX / hit.file).read_bytes()
            lines = raw.splitlines(keepends=True)
            start = sum(map(len, lines[:hit.start_line - 1])) + hit.start_col
            end = sum(map(len, lines[:hit.end_line - 1])) + hit.end_col
            self.assertEqual(raw[start:end], hit.matched_text.encode())
            return hit.file, start, end
        self.assertFalse({byte_key(h) for h in hits} & {byte_key(h) for h in factory_hits})
        self.assertEqual(sum(r["count"] for r in report["client_instances"]), len(factory_hits))
        for record in report["client_instances"]:
            self.assertEqual(set(record), {"file", "language", "count", "via_alias_count", "factories",
                                           "samples", "kind", "impact"})
            self.assertEqual(record["via_alias_count"], sum(h.via_alias for h in factory_hits if h.file == record["file"]))

    def test_cross_file_filter_has_exact_canonical_results(self):
        files = ["alias_cross/requests_a.py", "alias_cross/httpx_b.py",
                 "alias_cross/no_import.py", "alias_cross/other.py"]
        hits, report = self.run_fixture("python", *files)
        self.assertEqual([(h.file, h.callee_text, h.pattern_id, h.via_alias) for h in hits], [
            ("alias_cross/httpx_b.py", "r.get", "py.httpx", True),
            ("alias_cross/requests_a.py", "r.get", "py.requests", True)])
        stats = report["languages"]["python"]
        self.assertEqual((stats["alias_files"], stats["alias_hits_raw"], stats["alias_hits_kept"],
                          stats["alias_hits_dropped"], stats["runs"], stats["alias_pass_launches"]), (2, 4, 2, 2, 1, 1))

    def test_factories_are_gaps_only_and_instance_calls_are_untracked(self):
        expected_py = ["py.httpx.AsyncClient", "py.httpx.Client", "py.requests.Session", "py.requests.session"]
        expected_js = ["js.axios.create", "js.got.extend", "js.ky.create", "js.ky.extend", "js.superagent.agent"]
        samples = {
            "factories.py": ["6:4 requests.Session", "8:5 requests.Session", "10:8 requests.session"],
            "factories.js": ["1:10 axios.create", "3:10 got.extend", "5:10 ky.create"],
            "factories.ts": ["3:18 axios.create", "5:12 got.extend", "6:18 ky.create"],
            "factories.tsx": ["3:18 axios.create", "5:12 got.extend", "6:18 ky.create"],
        }
        for lang, rel in (("python", "factories.py"), ("javascript", "factories.js"),
                          ("typescript", "factories.ts"), ("typescript", "factories.tsx")):
            with self.subTest(rel):
                hits, report = self.run_fixture(lang, rel)
                # Includes s.get/client.get/agent.get: none is a catalogue call.
                self.assertEqual(hits, [])
                self.assertEqual(report["client_instances"], [{
                    "file": rel, "language": lang, "count": 5, "via_alias_count": 0,
                    "factories": expected_py if lang == "python" else expected_js,
                    "samples": samples[rel], "kind": "client_instance_untracked",
                    "impact": "calls made through this client object are not detected"}])
                self.assertEqual(report["languages"][lang]["client_factory_hits"], 5)
                self.assertEqual(report["languages"][lang]["hits"], 0)
                self.assertEqual(report["languages"][lang]["runs"], 1)

    def test_unqualified_factories_from_other_modules_are_not_reported(self):
        hits, report = self.run_fixture("python", "factory_other_session.py")
        self.assertEqual(hits, [])
        self.assertEqual(report["client_instances"], [])
        self.assertEqual(report["languages"]["python"]["client_factory_hits"], 0)

    def test_real_smoke_succeeds_with_factory_gaps(self):
        import subprocess
        proc = subprocess.run([sys.executable, "-m", "externalcalls.ast_grep_smoke", str(FIX), "--examples", "0"],
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("reasons (0)", proc.stdout)

    def run_fixture(self, language, *files, root=FIX):
        hits, report = mod.run_patterns(root, {language: list(files)})
        self.assertEqual(report["reasons"], [], json.dumps(report, indent=1))
        return hits, report

    def assert_not_on_lines(self, hits, path, *needles):
        bad = {_line_of(path, n) for n in needles}
        self.assertFalse({h.start_line for h in hits} & bad,
                         f"matched inside a comment/string/definition: {[h for h in hits if h.start_line in bad]}")

    def test_python_positives_and_negatives(self):
        hits, _ = self.run_fixture("python", "client.py")
        self.assertEqual(sorted(h.callee_text for h in hits),
                         sorted(["httpx.get", "httpx.stream", "requests.get", "requests.options",
                                 "requests.post", "requests.request", "urllib.request.urlopen"]))
        by = {h.callee_text: h.pattern_id for h in hits}
        self.assertEqual(by["requests.request"], "py.requests.request")
        self.assertEqual(by["requests.options"], "py.requests")
        self.assertEqual(by["httpx.stream"], "py.httpx.stream")
        self.assertEqual(by["httpx.get"], "py.httpx")
        # repo-defined get(), requests.Session(), comment, strings
        self.assert_not_on_lines(hits, FIX / "client.py", "def get", "= get(", "Session",
                                 "# requests.get", "instring", "docstring")

    def test_javascript_positives_and_negatives(self):
        hits, _ = self.run_fixture("javascript", "client.js")
        self.assertEqual(sorted(h.callee_text for h in hits),
                         sorted(["axios", "axios.get", "fetch", "globalThis.fetch", "got", "http.get",
                                 "http.request", "https.get", "https.request", "ky.post",
                                 "superagent.get", "window.fetch"]))
        by = {h.callee_text: h.pattern_id for h in hits}
        self.assertEqual(by["window.fetch"], "js.window.fetch")
        self.assertEqual(by["globalThis.fetch"], "js.globalThis.fetch")
        self.assertEqual(by["fetch"], "js.fetch")
        self.assert_not_on_lines(hits, FIX / "client.js", "function get", "= get(", "commented",
                                 "blockcomment", "instring", "template.example")

    def test_typescript_call_with_type_arguments(self):
        hits, _ = self.run_fixture("typescript", "client.ts")
        # client.ts contains one awaited generic axios.get and one plain fetch.
        self.assertEqual(sorted(h.callee_text for h in hits),
                         ["axios.get", "fetch"])
        generic = next(h for h in hits if h.callee_text == "axios.get")
        self.assertEqual(generic.pattern_id, "js.axios.method")
        self.assertIn("<User>", generic.matched_text)
        line = (FIX / "client.ts").read_bytes().splitlines()[5]
        self.assertEqual((generic.start_line, generic.start_col, generic.end_line, generic.end_col),
                         (6, line.index(b"axios.get"), 6, line.rindex(b")") + 1))
        self.assertEqual(generic.matched_text.encode(), line[generic.start_col:generic.end_col])

    def test_generic_and_await_variants_have_exact_positions_and_no_duplicates(self):
        for rel in ("generics.ts", "generics.tsx"):
            with self.subTest(rel):
                hits, _ = self.run_fixture("typescript", rel)
                # Non-await generic axios.get, awaited generic fetch (identifier),
                # and plain awaited axios.get, respectively, are all catalogue calls.
                expected = [(3, "axios.get", "js.axios.method"),
                            (4, "fetch", "js.fetch"),
                            (5, "axios.get", "js.axios.method")]
                self.assertEqual([(h.start_line, h.callee_text, h.pattern_id) for h in hits], expected)
                lines = (FIX / rel).read_bytes().splitlines()
                for h in hits:
                    line = lines[h.start_line - 1]
                    start = line.index(h.callee_text.encode())
                    end = line.rindex(b")") + 1
                    self.assertEqual((h.start_col, h.end_line, h.end_col), (start, h.start_line, end))
                    self.assertEqual(h.matched_text.encode(), line[start:end])
                self.assertEqual(len({(h.file, h.start_line, h.start_col, h.end_line,
                                       h.end_col, h.pattern_id) for h in hits}), 3)

    def test_tsx_is_parsed_with_the_tsx_grammar(self):
        hits, _ = self.run_fixture("typescript", "client.tsx")
        self.assertEqual(sorted(h.callee_text for h in hits), ["axios.post", "fetch"])

    def test_callee_split_over_lines(self):
        hits, _ = self.run_fixture("javascript", "chain.js")
        self.assertEqual([(h.callee_text, h.pattern_id, h.start_line, h.end_line) for h in hits],
                         [("axios.get", "js.axios.method", 3, 4)])

    def test_non_ascii_before_call_columns_are_bytes(self):
        hits, _ = self.run_fixture("python", "unicode.py")
        self.assertEqual(len(hits), 1)
        line = (FIX / "unicode.py").read_text(encoding="utf-8").splitlines()[2]
        byte_col = len(line[:line.index("requests.get")].encode("utf-8"))
        self.assertNotEqual(byte_col, line.index("requests.get"))
        self.assertEqual((hits[0].start_line, hits[0].start_col), (3, byte_col))

    def test_two_identical_calls_on_one_line(self):
        hits, _ = self.run_fixture("python", "twice.py")
        line = (FIX / "twice.py").read_text(encoding="utf-8").splitlines()[2]
        first = line.index("requests.get")
        second = line.index("requests.get", first + 1)
        self.assertEqual([(h.start_line, h.start_col) for h in hits], [(3, first), (3, second)])

    def test_nested_fetch_reports_both_calls(self):
        hits, _ = self.run_fixture("javascript", "nested.js")
        self.assertEqual(sorted(h.matched_text for h in hits), ["fetch(fetch(x))", "fetch(x)"])
        outer = next(h for h in hits if h.matched_text == "fetch(fetch(x))")
        inner = next(h for h in hits if h.matched_text == "fetch(x)")
        self.assertLess(outer.start_col, inner.start_col)
        self.assertGreater(outer.end_col, inner.end_col)

    def test_crlf_file(self):
        self.write("crlf.py", b'import requests\r\n\r\nx = "\xc3\xa9"; y = requests.get("u")\r\n')
        hits, _ = self.run_fixture("python", "crlf.py", root=self.tmp)
        self.assertEqual([(h.start_line, h.start_col) for h in hits],
                         [(3, len('x = "é"; y = '.encode("utf-8")))])

    def test_a_file_with_no_matches_is_not_a_failure(self):
        hits, report = self.run_fixture("python", "nocalls.py")
        self.assertEqual(hits, [])
        self.assertEqual(report["languages"]["python"]["hits"], 0)

    def test_only_given_files_are_searched(self):
        hits, _ = self.run_fixture("python", "twice.py")
        self.assertEqual({h.file for h in hits}, {"twice.py"})

    def test_every_rule_in_the_process_reports_under_its_catalogue_id(self):
        hits_py, _ = self.run_fixture("python", "client.py")
        hits_js, _ = self.run_fixture("javascript", "client.js")
        by_id = {e.id: e for e in mod.CATALOGUE}
        self.assertGreater(len({h.pattern_id for h in hits_py}), 3, "several rules of one process must report")
        for h in hits_py + hits_js:
            self.assertIn(h.pattern_id, by_id)
            self.assertTrue(re.search(by_id[h.pattern_id].callee_regex, h.callee_text),
                            f"{h.callee_text} does not match the regex of {h.pattern_id}")

    def test_a_broken_rule_is_a_failure_not_silence(self):
        objs, failure = mod._run_one(REAL_BIN, "{this is not a rule", ["twice.py"], FIX)
        self.assertEqual(objs, [])
        self.assertIsNotNone(failure)
        self.assertEqual(failure[0], "exit_nonzero")


@unittest.skipIf(REAL_BIN is None, NO_REAL)
@unittest.skipUnless(HAVE_TS, "tree_sitter_language_pack is not installed")
class ParserAgreementTests(unittest.TestCase):
    """Every real hit must contain exactly the position parser.py records for
    the same call (callee_line/callee_col) and spell the callee the same way:
    the join the next step relies on."""

    def test_alias_hits_contain_parser_callee_positions(self):
        cases = [("python", "filter1_a.py", "r.get"),
                 ("typescript", "callable_alias.ts", "g"),
                 ("typescript", "awaited_alias.ts", "ax.get"),
                 ("typescript", "awaited_alias.ts", "get"),
                 ("typescript", "awaited_alias.tsx", "ax.get"),
                 ("typescript", "awaited_alias.tsx", "get")]
        for language, rel, callee in cases:
            with self.subTest(rel=rel, callee=callee):
                rec = parser_module.parse_file(FIX, FIX / rel)
                self.assertEqual(rec.parse_mode, "treesitter")
                hits, report = mod.run_patterns(FIX, {language: [rel]})
                self.assertEqual(report["reasons"], [])
                hit, = [h for h in hits if h.callee_text == callee]
                self.assertTrue(hit.via_alias)
                call, = [c for c in rec.calls if c.callee_expr == callee]
                self.assertLessEqual((hit.start_line, hit.start_col), (call.callee_line, call.callee_col))
                self.assertLess((call.callee_line, call.callee_col), (hit.end_line, hit.end_col))
                print("REAL AGREEMENT", rel, callee, "parser", (call.callee_line, call.callee_col),
                      "hit", (hit.start_line, hit.start_col, hit.end_line, hit.end_col))

    def test_nonawait_generic_agrees_with_parser(self):
        # Awaited generics are a known parser disagreement (<expr>, col -1).
        # The existing all-hit agreement assertion is retained to expose it.
        for rel in ("generics.ts", "generics.tsx"):
            with self.subTest(rel):
                rec = parser_module.parse_file(FIX, FIX / rel)
                self.assertEqual(rec.parse_mode, "treesitter")
                hits, report = mod.run_patterns(FIX, {"typescript": [rel]})
                self.assertEqual(report["reasons"], [])
                hit, = [h for h in hits if h.start_line == 3]
                call, = [c for c in rec.calls if c.callee_line == 3]
                self.assertEqual(call.callee_expr, hit.callee_text)
                line = (FIX / rel).read_bytes().splitlines()[2]
                self.assertEqual(call.callee_col, line.index(b"get"))
                self.assertLessEqual((hit.start_line, hit.start_col), (call.callee_line, call.callee_col))
                self.assertLess((call.callee_line, call.callee_col), (hit.end_line, hit.end_col))

    def test_hits_contain_the_parsers_callee_position(self):
        parser = parser_module
        for lang, rel in (("python", "client.py"), ("python", "unicode.py"), ("python", "twice.py"),
                          ("javascript", "client.js"), ("javascript", "nested.js"),
                          ("javascript", "chain.js"), ("typescript", "client.ts"),
                          ("typescript", "client.tsx")):
            with self.subTest(rel):
                rec = parser.parse_file(FIX, FIX / rel)
                if rec.parse_mode != "treesitter":
                    self.skipTest(f"parser.py used its line-based fallback for {rel}")
                hits, report = mod.run_patterns(FIX, {lang: [rel]})
                self.assertEqual(report["reasons"], [])
                claimed = set()
                for h in hits:
                    inside = [i for i, c in enumerate(rec.calls)
                              if c.callee_expr == h.callee_text
                              and (h.start_line, h.start_col) <= (c.callee_line, c.callee_col)
                              < (h.end_line, h.end_col)]
                    self.assertTrue(inside, f"no CallSite for {h}")
                    if rel != "nested.js":              # nested: the outer hit contains both callees
                        self.assertEqual(len(inside), 1, f"{h} contains several CallSites")
                    claimed.update(inside)
                self.assertEqual(len(claimed), len(hits), "two hits claimed the same CallSite")


if __name__ == "__main__":
    unittest.main()
