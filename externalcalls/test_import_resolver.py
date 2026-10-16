from pathlib import Path

from analyzer_new.parser import parse_repo
from externalcalls.import_resolver import resolve_import


def _write(root, relative, text="pass\n"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_python_import_forms_and_parser_raw_values(tmp_path):
    root = tmp_path
    source = "\n".join([
        "import a.b",
        "import a.b as c",
        "from a.b import X",
        "from a.b import X as Y",
        "from . import m",
        "from .sib import X",
        "from ..up import X",
    ]) + "\n"
    _write(root, "pkg/sub/forms.py", source)
    for path in ("a/b.py", "a/b/X.py", "pkg/sub/m.py", "pkg/sub/sib/X.py", "pkg/up/X.py"):
        _write(root, path)

    records = parse_repo(root)
    record = next(record for record in records if record.path == "pkg/sub/forms.py")
    raw = [item.raw for item in record.imports]
    print(f"python raw imports: {raw!r}")
    assert raw == ["a.b", "a.b", "a.b.X", "a.b.X", "m", "sib.X", "up.X"]
    repo_files = {record.path for record in records}
    expected = [
        "a/b.py", "a/b.py", "a/b/X.py", "a/b/X.py",
        "pkg/sub/m.py", "pkg/sub/sib/X.py", "pkg/up/X.py",
    ]
    resolved = [resolve_import(record.path, item, repo_files, "python", root)[0]
                for item in record.imports]
    assert resolved == expected


def test_python_imported_submodule_src_layout_and_external(tmp_path):
    root = tmp_path
    _write(root, "pkg/app.py", "from pkg import consts\nfrom lib import URL\nimport requests\n")
    _write(root, "pkg/consts.py")
    _write(root, "src/lib.py")
    records = parse_repo(root)
    app = next(record for record in records if record.path == "pkg/app.py")
    files = {record.path for record in records}

    print(f"python submodule raw={app.imports[0].raw!r}; src raw={app.imports[1].raw!r}")
    assert resolve_import(app.path, app.imports[0], files, "python", root) == ("pkg/consts.py", None)
    assert resolve_import(app.path, app.imports[1], files, "python", root) == ("src/lib.py", None)
    assert resolve_import(app.path, app.imports[2], files, "python", root) == (None, "external")


def test_unresolved_python_relative_import(tmp_path):
    root = tmp_path
    _write(root, "pkg/app.py", "from .missing import Thing\n")
    record = next(item for item in parse_repo(root) if item.path == "pkg/app.py")
    assert resolve_import(record.path, record.imports[0], {record.path}, "python", root) == (
        None, "unresolved")


def test_javascript_typescript_import_forms_and_raw_values(tmp_path):
    root = tmp_path
    source = "\n".join([
        'import X from "./m";',
        'import {A, B as C} from "./m";',
        'import * as n from "./m";',
        'const {A} = require("./m");',
        'import axios from "axios";',
    ]) + "\n"
    _write(root, "pkg/forms.ts", source)
    _write(root, "pkg/m.ts")
    records = parse_repo(root)
    record = next(item for item in records if item.path == "pkg/forms.ts")
    raw = [item.raw for item in record.imports]
    print(f"typescript raw imports: {raw!r}")
    assert raw == ["./m", "./m", "./m", "./m", "axios"]
    files = {item.path for item in records}
    resolved = [resolve_import(record.path, item, files, "typescript", root)
                for item in record.imports]
    assert resolved == [("pkg/m.ts", None)] * 4 + [(None, "external")]


def test_javascript_extension_fallback_and_alias_classification(tmp_path):
    files = {"src/app.ts", "src/module.ts", "app/routes.ts"}
    assert resolve_import("src/app.ts", "./module.js", files, "typescript") == (
        "src/module.ts", None)
    assert resolve_import("src/app.ts", "@/module", files, "typescript") == (
        None, "alias_unresolved")
    assert resolve_import("src/app.ts", "~/module", files, "javascript") == (
        None, "alias_unresolved")
    assert resolve_import("src/app.ts", "app/routes", files, "typescript") == (
        None, "alias_unresolved")
    assert resolve_import("src/app.ts", "@scope/pkg", files, "typescript") == (
        None, "external")
    assert resolve_import("src/app.ts", "./absent", files, "typescript") == (
        None, "unresolved")

