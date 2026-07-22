"""Headless net for the geometry-IO cluster extracted from GUI.py.

Pins the on-disk format read/serialize behavior that was previously buried in
MainWindow.openGeo/parseOldGeo/saveGeo — the parse of legacy Xamine/dispwind
``.win`` files, the native dict-literal round-trip, the format sniff, and the
edge cases (empty / unrecognized / no-Geometry-line). Qt-free, so it runs in the
system python3 with no PyQt5.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

import pytest
from services import geometry_io


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# ---------------------------------------------------------------- native format

def test_serialize_round_trips_through_read(tmp_path):
    properties = {
        0: {"name": "h1", "x": [0.0, 10.0], "y": [1.0, 2.0], "scale": True},
        1: {"name": "h2", "x": None, "y": None, "scale": False},
    }
    text = geometry_io.serialize_geometry(2, 3, properties)
    path = _write(tmp_path, "native.win", text)
    got = geometry_io.read_geometry(path)
    assert got == {"row": 2, "col": 3, "geo": properties}


def test_serialize_coerces_row_col_to_int():
    # saveGeo passes combobox currentText() (strings); serialize must int() them.
    text = geometry_io.serialize_geometry("4", "5", {})
    assert text == str({"row": 4, "col": 5, "geo": {}})


def test_read_native_empty_geo(tmp_path):
    path = _write(tmp_path, "n.win", str({"row": 1, "col": 1, "geo": {}}))
    assert geometry_io.read_geometry(path) == {"row": 1, "col": 1, "geo": {}}


# ------------------------------------------------------------- legacy .win format

LEGACY = """\
Geometry 2 2
Window "raw00"
  COUNTSAXIS
  Expanded 10 200 5 60
EndWindow
Window "raw01"
EndWindow
"""


def test_parse_legacy_win(tmp_path):
    path = _write(tmp_path, "old.win", LEGACY)
    got = geometry_io.read_geometry(path)
    assert got["row"] == 2 and got["col"] == 2
    # window 0: log on (COUNTSAXIS), x/y from Expanded
    assert got["geo"][0] == {"name": "raw00", "x": [10.0, 200.0],
                             "y": [5.0, 60.0], "scale": True}
    # window 1: no COUNTSAXIS/Expanded -> natural range, linear
    assert got["geo"][1] == {"name": "raw01", "x": None,
                             "y": None, "scale": False}


def test_parse_legacy_expanded_x_only(tmp_path):
    text = ('Geometry 1 1\n'
            'Window "s"\n  Expanded 3 40\nEndWindow\n')
    got = geometry_io.read_geometry(_write(tmp_path, "x.win", text))
    assert got["geo"][0]["x"] == [3.0, 40.0]
    assert got["geo"][0]["y"] is None


def test_parse_legacy_skips_unnamed_window_but_advances_index(tmp_path):
    # A Window with no quoted name is not stored, but the flat index still advances
    # (matches the original loader), so the following named window keeps its slot.
    text = ('Geometry 3 1\n'
            'Window\nEndWindow\n'
            'Window "kept"\nEndWindow\n')
    got = geometry_io.read_geometry(_write(tmp_path, "u.win", text))
    assert 0 not in got["geo"]
    assert got["geo"][1] == {"name": "kept", "x": None, "y": None, "scale": False}


def test_parse_legacy_ignores_comments_and_blank_lines(tmp_path):
    text = ('# a comment\n\n'
            'Geometry 1 1\n'
            '# another\n'
            'Window "z"\nEndWindow\n')
    got = geometry_io.read_geometry(_write(tmp_path, "c.win", text))
    assert got == {"row": 1, "col": 1,
                   "geo": {0: {"name": "z", "x": None, "y": None, "scale": False}}}


# --------------------------------------------------------------- format sniff / edges

def test_empty_file_returns_none(tmp_path):
    assert geometry_io.read_geometry(_write(tmp_path, "e.win", "")) is None


def test_unrecognized_format_returns_none(tmp_path):
    assert geometry_io.read_geometry(_write(tmp_path, "bad.win", "hello world\n")) is None


def test_legacy_without_geometry_line_returns_none(tmp_path):
    # First meaningful line starts with "geometry" (sniffed to legacy) but the
    # keyword scan finds no valid "Geometry R C" -> None.
    text = 'geometry-ish header\nWindow "x"\nEndWindow\n'
    assert geometry_io.read_geometry(_write(tmp_path, "ng.win", text)) is None


def test_sniff_is_case_insensitive_and_comment_tolerant(tmp_path):
    text = '# header\nGEOMETRY 1 1\nWindow "c"\nEndWindow\n'
    got = geometry_io.read_geometry(_write(tmp_path, "ci.win", text))
    assert got["row"] == 1 and got["geo"][0]["name"] == "c"


def test_read_geometry_rejects_code_execution(tmp_path):
    # eval() executed arbitrary code from a hostile .win file.
    p = tmp_path / "evil.win"
    p.write_text('{"row": __import__("os").getpid()}')
    assert geometry_io.read_geometry(str(p)) is None


def test_read_geometry_native_roundtrip_survives_literal_eval(tmp_path):
    props = {0: {"name": "h1", "x": [0.0, 1.0], "y": None, "scale": False}}
    p = tmp_path / "geo.win"
    p.write_text(geometry_io.serialize_geometry(2, 2, props))
    got = geometry_io.read_geometry(str(p))
    assert got == {"row": 2, "col": 2, "geo": props}


# --------------------------------------------------- multi-tab session format

def _session_tabs():
    return [
        {"name": "alpha run 42", "row": 2, "col": 2,
         "geo": {0: {"name": "specA", "x": [0.0, 512.0], "y": [0.0, 1000.0], "scale": False},
                 1: {"name": "spec B", "x": None, "y": None, "scale": True}}},
        {"name": "Tab 2", "row": 1, "col": 1,
         "geo": {0: {"name": "", "x": None, "y": None, "scale": False}}},
    ]


def test_session_roundtrip_preserves_tabs_names_order(tmp_path):
    p = tmp_path / "s.win"
    p.write_text(geometry_io.serialize_session(_session_tabs()))
    kind, payload = geometry_io.read_geometry_any(str(p))
    assert kind == "session"
    assert [t["name"] for t in payload["tabs"]] == ["alpha run 42", "Tab 2"]
    assert payload["tabs"][0]["geo"][0]["x"] == [0.0, 512.0]
    assert payload["tabs"][0]["geo"][1]["scale"] is True


def test_read_any_tags_v1_single_and_matches_read_geometry(tmp_path):
    props = {0: {"name": "h1", "x": [0.0, 1.0], "y": None, "scale": False}}
    p = tmp_path / "v1.win"
    p.write_text(geometry_io.serialize_geometry(2, 2, props))
    kind, payload = geometry_io.read_geometry_any(str(p))
    assert kind == "single"
    assert payload == geometry_io.read_geometry(str(p))


def test_read_any_tags_legacy_as_single(tmp_path):
    p = tmp_path / "old.win"
    p.write_text('Geometry 1,2\nWindow "spec one"\nEndwindow\n')
    kind, payload = geometry_io.read_geometry_any(str(p))
    assert kind == "single"
    assert payload["row"] == 1 and payload["col"] == 2
    assert payload["geo"][0]["name"] == "spec one"


def test_read_any_rejects_session_missing_name(tmp_path):
    p = tmp_path / "bad.win"
    p.write_text(str({"version": 2, "tabs": [{"row": 1, "col": 1, "geo": {}}]}))
    assert geometry_io.read_geometry_any(str(p)) is None


def test_read_any_rejects_empty_tabs(tmp_path):
    p = tmp_path / "bad.win"
    p.write_text(str({"version": 2, "tabs": []}))
    assert geometry_io.read_geometry_any(str(p)) is None


def test_read_any_rejects_bad_rowcol(tmp_path):
    p = tmp_path / "bad.win"
    p.write_text(str({"version": 2, "tabs": [{"name": "t", "row": 0, "col": 1, "geo": {}}]}))
    assert geometry_io.read_geometry_any(str(p)) is None


def test_read_any_rejects_code_execution(tmp_path):
    p = tmp_path / "evil.win"
    p.write_text('{"tabs": __import__("os").getpid()}')
    assert geometry_io.read_geometry_any(str(p)) is None


def test_validate_session_happy_path():
    obj = {"version": 2, "kind": "cutiepie-session", "tabs": _session_tabs()}
    assert geometry_io.validate_session(obj) == []


# --------------------------- single-geometry validation / mis-load safety

def test_validate_single_geometry_happy_path():
    obj = {"row": 2, "col": 2, "geo": {0: {"name": "h1"}}}
    assert geometry_io.validate_single_geometry(obj) == []


def test_validate_single_geometry_rejects_session_dict():
    # a v2 session file is NOT a valid single geometry (this is the mis-load bug)
    obj = {"version": 2, "kind": "cutiepie-session", "tabs": _session_tabs()}
    assert geometry_io.validate_single_geometry(obj) != []


def test_validate_single_geometry_rejects_junk_and_bad_rowcol():
    assert geometry_io.validate_single_geometry("not a dict") != []
    assert geometry_io.validate_single_geometry({"row": 0, "col": 1, "geo": {}}) != []
    assert geometry_io.validate_single_geometry({"row": 2, "col": 2}) != []          # no geo
    assert geometry_io.validate_single_geometry({"row": 2, "col": 2, "geo": []}) != []


def test_read_any_rejects_session_shaped_dict_as_single(tmp_path):
    # the reported crash: a session file must never come back tagged "single"
    p = tmp_path / "sess.win"
    p.write_text(geometry_io.serialize_session(_session_tabs()))
    kind, _ = geometry_io.read_geometry_any(str(p))
    assert kind == "session"


def test_read_any_rejects_malformed_single_dict(tmp_path):
    # a native dict that is neither a session nor a valid single geometry
    p = tmp_path / "junk.win"
    p.write_text(str({"foo": 1, "bar": 2}))
    assert geometry_io.read_geometry_any(str(p)) is None
