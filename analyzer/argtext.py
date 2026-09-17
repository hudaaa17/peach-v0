"""
Call-argument expression helpers, shared by the literal / const-prop /
Joern stages.

These exist because every one of those stages used to do the same two
things by hand, both of them subtly wrong on anything more complex than a
bare name:

    first_arg = edge.arg_text.split(",")[0].strip()

...which splits inside nested calls and inside string literals, and

    _STRING_LITERAL.search(first_arg)

...which reports a "literal" for any quoted fragment *anywhere* in the
expression — so `BASE + "?key=" + token` resolves to the host `?key=`.

Keeping the logic here means a fix lands once, and the stages can log a
precise reason when they decline to handle an expression.
"""
import re

_OPENERS = {"(": ")", "[": "]", "{": "}"}
_CLOSERS = {")", "]", "}"}
_QUOTES = ("'", '"', "`")

_BARE_IDENTIFIER = re.compile(r"^[A-Za-z_$][\w$]*$")

# Identifiers we should never treat as a value to trace backwards.
_IDENT_STOPWORDS = {
    "f", "rb", "br", "r", "b", "u",           # string prefixes left by unparse
    "True", "False", "None", "null", "undefined",
    "str", "int", "format", "join", "encode", "decode",
}

_IDENT_IN_EXPR = re.compile(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*")

# A whole-expression string literal: optional prefix, then one quoted run
# that consumes the entire expression.
_WHOLE_STRING = re.compile(
    r"""^(?P<prefix>[fFrRbBuU]{0,2})(?P<q>['"])(?P<body>(?:\\.|(?!(?P=q))[^\\])*)(?P=q)$""",
    re.DOTALL,
)
_WHOLE_TEMPLATE = re.compile(r"^`(?P<body>[^`]*)`$", re.DOTALL)


def split_top_level(text: str):
    """Split an argument list on commas that are *not* inside brackets,
    quotes, or a template-literal substitution. Returns [] for empty text."""
    if not text:
        return []
    parts = []
    buf = []
    depth = []
    quote = None
    escaped = False
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            buf.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            i += 1
            continue
        if ch in _QUOTES:
            quote = ch
            buf.append(ch)
        elif ch in _OPENERS:
            depth.append(_OPENERS[ch])
            buf.append(ch)
        elif ch in _CLOSERS:
            if depth and depth[-1] == ch:
                depth.pop()
            buf.append(ch)
        elif ch == "," and not depth:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail or parts:
        parts.append(tail)
    return parts


def first_argument(arg_text: str) -> str:
    """The first positional argument, as source text."""
    parts = split_top_level(arg_text)
    return parts[0] if parts else ""


def is_bare_identifier(expr: str) -> bool:
    return bool(_BARE_IDENTIFIER.match((expr or "").strip()))


def string_literal_value(expr: str):
    """If `expr` *is* a string literal (or an f-string / template literal),
    return its body text. Otherwise None.

    Note the difference from a substring search: `"a" + b` returns None
    here, because the expression as a whole is not a literal — it's a
    concatenation, and guessing from its first fragment is how you end up
    with a host of `?key=`."""
    if not expr:
        return None
    expr = expr.strip()
    m = _WHOLE_STRING.match(expr)
    if m:
        return m.group("body")
    m = _WHOLE_TEMPLATE.match(expr)
    if m:
        return m.group("body")
    return None


def identifier_roots(expr: str, limit: int = 6):
    """Candidate variable names appearing in a compound expression, most
    promising first, for a backward trace to try one at a time.

    `BASE_URL + "/v1/" + path` -> ["BASE_URL", "path"]
    `f"{scheme}://{host}/x"`   -> ["scheme", "host"]
    `self.base + suffix`       -> ["self.base", "suffix"]

    Dotted names are kept whole so `self.attr` traces still work. Names
    that are obviously not values (string prefixes, builtins, literals)
    are dropped. Ordering prefers endpoint-ish names, then longer names,
    since `BASE_URL` is a much better trace target than `i`."""
    if not expr:
        return []
    seen = []
    for m in _IDENT_IN_EXPR.finditer(expr):
        name = m.group(0)
        if name in _IDENT_STOPWORDS or name.split(".")[0] in _IDENT_STOPWORDS:
            continue
        # skip a name immediately followed by '(' — that's a call, not a value
        after = expr[m.end():m.end() + 1]
        if after == "(":
            continue
        if name not in seen:
            seen.append(name)
    hints = ("url", "host", "endpoint", "service", "base", "addr", "uri", "target", "api")

    def rank(name):
        lowered = name.lower()
        return (0 if any(h in lowered for h in hints) else 1, -len(name))

    return sorted(seen, key=rank)[:limit]


def classify(expr: str) -> str:
    """Coarse shape of an argument expression, used as a logging /
    bail-reason label so a stage can report *what kind* of thing it
    couldn't handle rather than just 'unresolved'."""
    if not expr:
        return "empty"
    expr = expr.strip()
    if expr == "<expr>":
        return "opaque_placeholder"
    if string_literal_value(expr) is not None:
        return "string_literal"
    if is_bare_identifier(expr):
        return "bare_identifier"
    if re.match(r"^self\.\w+$", expr):
        return "self_attribute"
    if re.match(r"^[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+$", expr):
        return "dotted_name"
    if "+" in expr or "%" in expr or ".format(" in expr:
        return "concatenation"
    if expr[:1] in ("f", "F") and expr[1:2] in ("'", '"'):
        return "fstring"
    if expr.startswith("`"):
        return "template_literal"
    if "(" in expr:
        return "call_expression"
    return "other_expression"
