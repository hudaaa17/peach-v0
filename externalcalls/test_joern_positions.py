from pathlib import Path

from analyzer_new.parser import parse_file
from externalcalls.joern_positions import (
    joern_to_parser_col,
    parser_to_joern_start,
    read_line,
)


def test_ascii_call_start():
    assert parser_to_joern_start("r = requests.get(URL)", "requests.get", 13) == (5, None)


def test_non_ascii_positions():
    line = 'x = "é"; requests.get(URL)'
    assert parser_to_joern_start(line, "requests.get", 19) == (10, None)
    assert joern_to_parser_col(line, 10) == (10, None)
    assert joern_to_parser_col(line, 23) == (23, None)


def test_utf16_emoji_positions():
    line = 'x = "\U0001F600"; requests.get(URL)'
    print("emoji UTF-16 assertions:",
          parser_to_joern_start(line, "requests.get", 21),
          joern_to_parser_col(line, 24),
          joern_to_parser_col(line, 7))
    assert parser_to_joern_start(line, "requests.get", 21) == (11, None)
    assert joern_to_parser_col(line, 24) == (25, None)
    assert joern_to_parser_col(line, 7) == (None, "col_out_of_range")


def test_language_dependent_column_base():
    line = "x = requests.get(URL)"
    assert parser_to_joern_start(line, "requests.get", 13,
                                 language="python") == (5, None)
    assert parser_to_joern_start(line, "requests.get", 13,
                                 language="javascript") == (4, None)
    assert parser_to_joern_start(line, "requests.get", 13,
                                 language="typescript") == (4, None)
    assert joern_to_parser_col(line, 4, language="javascript") == (4, None)
    assert joern_to_parser_col(line, 5, language="python") == (4, None)


def test_unknown_position_language():
    assert parser_to_joern_start("requests.get(URL)", "requests.get", 9,
                                 language="ruby") == (None, "unsupported_language")
    assert joern_to_parser_col("requests.get(URL)", 1,
                               language="ruby") == (None, "unsupported_language")


def test_several_calls_on_one_line():
    line = "requests.post(build(data))"
    assert parser_to_joern_start(line, "requests.post", 9) == (1, None)
    assert parser_to_joern_start(line, "build", 14) == (15, None)


def test_conversion_failures():
    line = 'x = "é"; requests.get(URL)'
    assert parser_to_joern_start(line, "requests.get", 18) == (None, "callee_text_mismatch")
    assert parser_to_joern_start(line, "requests.get", -1) == (None, "callee_col_unknown")
    assert parser_to_joern_start(line, "requests.get", 100) == (None, "col_out_of_range")
    assert parser_to_joern_start("requests.get(URL)", "requests . get", 9) == (
        None, "callee_text_mismatch")
    assert parser_to_joern_start(line, "requests.get", 6) == (None, "col_out_of_range")
    assert joern_to_parser_col(line, 0) == (None, "col_out_of_range")
    assert joern_to_parser_col(line, 100) == (None, "col_out_of_range")


def test_real_parser_callsite_and_conversion(tmp_path):
    line = 'x = "é"; requests.get(URL)'
    source = tmp_path / "m.py"
    source.write_text(line + "\n", encoding="utf-8")

    record = parse_file(tmp_path, source)
    call = next(call for call in record.calls if call.callee_expr == "requests.get")
    print(f"real parser: callee_expr={call.callee_expr!r}, callee_line={call.callee_line}, "
          f"callee_col={call.callee_col}")
    assert call.callee_col == 19
    line_text, error = read_line(tmp_path, Path("m.py"), call.callee_line)
    assert error is None
    assert parser_to_joern_start(line_text, call.callee_expr, call.callee_col) == (10, None)


def test_crlf_file_has_same_positions(tmp_path):
    line = 'x = "é"; requests.get(URL)'
    source = tmp_path / "crlf.py"
    source.write_bytes((line + "\r\n").encode("utf-8"))

    text, error = read_line(tmp_path, "crlf.py", 1)
    assert error is None
    assert parser_to_joern_start(text, "requests.get", 19) == (10, None)
    assert joern_to_parser_col(text, 10) == (10, None)
