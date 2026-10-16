"""Tree-sitter access helpers shared by URL inspection stages."""

from pathlib import Path

from analyzer_new.parser import _get_ts_parser


def read_source(repo_root, rel_path):
    text = (Path(repo_root) / rel_path).read_text(encoding="utf-8", errors="ignore")
    return text, text.encode("utf-8", "ignore")


def load_tree(repo_root, rel_path, cache):
    """Use a caller-owned, per-run cache; return (tree, source, error)."""
    key = str(rel_path)
    if key not in cache:
        try:
            _, source = read_source(repo_root, rel_path)
        except Exception:
            cache[key] = (None, None, "file_unreadable")
        else:
            grammar = {".py": "python", ".js": "javascript", ".jsx": "javascript",
                       ".mjs": "javascript", ".cjs": "javascript", ".ts": "typescript",
                       ".tsx": "tsx"}.get(Path(rel_path).suffix.lower())
            try:
                parser = _get_ts_parser(grammar) if grammar else None
                if parser is None:
                    raise ValueError("grammar unavailable")
                cache[key] = (parser.parse(source), source, None)
            except Exception:
                cache[key] = (None, source, "grammar_unavailable")
            else:
                tree, source, error = cache[key]
                cache[key] = (_IndexedTree(tree), source, error)
    return cache[key]


def _walk(node):
    yield node
    for child in node.named_children:
        yield from _walk(child)


def unwrap(node):
    while node is not None and node.type in (
            "parenthesized_expression", "as_expression", "satisfies_expression",
            "non_null_expression", "await_expression"):
        node = node.named_children[0] if node.named_children else None
    return node


def _build_call_index(root):
    index = {}
    for node in _walk(root):
        if node.type not in ("call", "call_expression"):
            continue
        callee = unwrap(node.child_by_field_name("function"))
        while callee is not None and callee.type in ("attribute", "member_expression"):
            callee = unwrap(callee.child_by_field_name(
                "attribute" if callee.type == "attribute" else "property"))
        if callee is not None and callee.type in ("identifier", "property_identifier"):
            row, col = callee.start_point
            index.setdefault((row + 1, col), []).append(node)
    return index


class _IndexedTree:
    """Keep the tree and its call index together in the caller-owned run cache."""

    def __init__(self, tree):
        self._tree = tree
        self._call_index = _build_call_index(tree.root_node)

    def __getattr__(self, name):
        return getattr(self._tree, name)


def find_call_node(tree, callee_line, callee_col):
    index = (tree._call_index if isinstance(tree, _IndexedTree)
             else _build_call_index(tree.root_node))
    found = index.get((callee_line, callee_col), [])
    return found[0] if len(found) == 1 else (
        "ambiguous_node" if found else "node_not_found")


def find_node_at(tree, line, col, node_type):
    found = [n for n in _walk(tree.root_node) if n.type == node_type
             and tuple(n.start_point) == (line - 1, col)]
    return found[0] if len(found) == 1 else None


def get_url_argument(call_node, entry, language):
    args_node = call_node.child_by_field_name("arguments")
    args = [n for n in args_node.named_children if n.type != "comment"] if args_node else []
    index = entry.url_arg_index
    positionals = []
    for arg in args:
        if arg.type in ("list_splat", "dictionary_splat", "spread_element"):
            if index is None or len(positionals) <= index:
                return None, "spread_before_url"
        if language == "python" and arg.type == "keyword_argument":
            continue
        positionals.append(arg)
    if language != "python" and entry.url_arg_property and positionals:
        first = unwrap(positionals[0])
        if first.type == "object":
            for prop in first.named_children:
                if prop.type == "pair":
                    key = prop.child_by_field_name("key")
                    spelling = key.text.decode("utf-8").strip("\"'")
                    if spelling == entry.url_arg_property:
                        return unwrap(prop.child_by_field_name("value")), None
                elif prop.type == "shorthand_property_identifier":
                    if prop.text.decode("utf-8") == entry.url_arg_property:
                        return prop, None
    if index is not None and len(positionals) > index:
        return unwrap(positionals[index]), None
    if language == "python" and entry.url_arg_keyword:
        for arg in args:
            if arg.type == "keyword_argument":
                name = arg.child_by_field_name("name")
                if name.text.decode("utf-8") == entry.url_arg_keyword:
                    return unwrap(arg.child_by_field_name("value")), None
    return None, "no_url_argument"
