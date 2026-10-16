"""Convert call positions between the parser and Joern conventions."""

from pathlib import Path


def _column_base(language):
    """Return Joern's measured column base for a supported source language."""
    normalized = str(language or "python").lower()
    if normalized in {"python", "py", "pythonsrc"}:
        return 1
    if normalized in {"javascript", "js", "jssrc", "typescript", "ts"}:
        return 0
    return None


def parser_to_joern_start(line_text, callee_expr, callee_col_bytes,
                          language="python"):
    """Return the Joern column where a dotted callee starts.

    Python columns are 1-based; JavaScript and TypeScript columns are
    0-based. JS/TS UTF-16 conversion has not been verified against Joern.
    """
    base = _column_base(language)
    if base is None:
        return None, "unsupported_language"
    if not isinstance(line_text, str) or not isinstance(callee_expr, str):
        return None, "callee_text_mismatch"
    if not isinstance(callee_col_bytes, int) or isinstance(callee_col_bytes, bool):
        return None, "col_out_of_range"
    if callee_col_bytes < 0:
        return None, "callee_col_unknown"

    try:
        encoded = line_text.encode("utf-8")
        if callee_col_bytes > len(encoded):
            return None, "col_out_of_range"
        prefix = encoded[:callee_col_bytes].decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None, "col_out_of_range"

    leaf = callee_expr.rsplit(".", 1)[-1]
    start = len(prefix) - (len(callee_expr) - len(leaf))
    if start < 0 or line_text[start:start + len(callee_expr)] != callee_expr:
        return None, "callee_text_mismatch"
    try:
        # Joern's columns count UTF-16 code units. JS/TS use a 0-based column
        # base, but UTF-16 behavior for those frontends remains unverified.
        return len(line_text[:start].encode("utf-16-le")) // 2 + base, None
    except UnicodeEncodeError:
        return None, "col_out_of_range"


def joern_to_parser_col(line_text, joern_col, language="python"):
    """Convert a Joern column to a 0-based UTF-8 byte column.

    JS/TS columns use a 0-based base; their UTF-16 behavior is unverified.
    """
    base = _column_base(language)
    if base is None:
        return None, "unsupported_language"
    if not isinstance(line_text, str):
        return None, "col_out_of_range"
    if (not isinstance(joern_col, int)
            or isinstance(joern_col, bool)
            or joern_col < base):
        return None, "col_out_of_range"
    try:
        target_units = joern_col - base
        total_units = len(line_text.encode("utf-16-le")) // 2
        if target_units > total_units:
            return None, "col_out_of_range"
        units = 0
        byte_col = 0
        for character in line_text:
            if units == target_units:
                return byte_col, None
            width = 2 if ord(character) > 0xFFFF else 1
            if target_units < units + width:
                return None, "col_out_of_range"
            units += width
            byte_col += len(character.encode("utf-8"))
        if units == target_units:
            return byte_col, None
        return None, "col_out_of_range"
    except UnicodeEncodeError:
        return None, "col_out_of_range"


def read_line(repo_root, rel_path, line_1based):
    """Read one line using the parser's UTF-8 decoding and newline rules."""
    if (not isinstance(line_1based, int) or isinstance(line_1based, bool)
            or line_1based < 1):
        return None, "line_out_of_range"
    try:
        text = (Path(repo_root) / rel_path).read_text(
            encoding="utf-8", errors="ignore")
        lines = text.split("\n")
    except (OSError, TypeError, ValueError):
        return None, "read_error"
    if line_1based > len(lines):
        return None, "line_out_of_range"
    return lines[line_1based - 1], None
