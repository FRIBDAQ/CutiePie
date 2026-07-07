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
