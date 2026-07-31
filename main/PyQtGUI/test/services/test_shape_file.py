"""Tests for the shape-file format guard (services.shape_file).

The AlphaEMGMulti/AlphaEMGMultiSigma loader (_load_shapes) tolerantly skips
malformed rows, so a wrong file parses to zero shapes and the fit silently runs
with no components. validate_shape_file is the up-front guard that answers "is
this even a shape file?" so the caller can abort with a clear message. It is
Qt-free and lmfit-free so it runs in this headless environment (the creator
module itself is not importable here — it needs lmfit).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

from services import shape_file


_GOOD = (
    "# isotope, half_life, energy_keV, intensity, sigma, tau1, tau2, eta, flag, chain\n"
    "Bi211, 2.14m, 6623, 0.836, 6.5, 8.0, 40.0, 0.7, s, A227\n"
    "Bi211, 2.14m, 6278, 0.164, 6.5, 8.0, 40.0, 0.7, -, A227\n"
    "Po215, 1.78ms, 7386, 1.0, 6.5, 8.0, 40.0, 0.7, *, A227\n"
)


def _w(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_good_shape_file_is_valid(tmp_path):
    assert shape_file.validate_shape_file(_w(tmp_path, "s.txt", _GOOD)) == []


def test_count_shape_rows_counts_only_valid_rows(tmp_path):
    # 3 data rows; the comment header does not count
    assert shape_file.count_shape_rows(_w(tmp_path, "s.txt", _GOOD)) == 3


def test_empty_file_is_rejected(tmp_path):
    assert shape_file.validate_shape_file(_w(tmp_path, "e.txt", "")) != []


def test_missing_file_is_rejected(tmp_path):
    assert shape_file.validate_shape_file(str(tmp_path / "nope.txt")) != []


def test_json_calibration_is_named_as_calibration(tmp_path):
    p = _w(tmp_path, "cal.json", '{"a": 2.5, "b": -3.0}')
    problems = shape_file.validate_shape_file(p)
    assert problems and "calibration" in problems[0].lower()


def test_geometry_dict_is_rejected(tmp_path):
    p = _w(tmp_path, "g.win", str({"row": 2, "col": 2, "geo": {}}))
    assert shape_file.validate_shape_file(p) != []


def test_session_dict_is_rejected(tmp_path):
    p = _w(tmp_path, "sess.win",
           str({"version": 2, "kind": "cutiepie-session", "tabs": []}))
    assert shape_file.validate_shape_file(p) != []


def test_calibration_text_has_no_shape_rows(tmp_path):
    # the swap hazard: a calibration text file must NOT read as a shape file
    p = _w(tmp_path, "cal.txt", "slope 3.5 offset 7\n")
    assert shape_file.validate_shape_file(p) != []
    assert shape_file.count_shape_rows(p) == 0


def test_looks_like_shape_file_discriminates(tmp_path):
    good = _w(tmp_path, "s.txt", _GOOD)
    cal = _w(tmp_path, "cal.txt", "a = 4.0\nb = -1e-2\n")
    assert shape_file.looks_like_shape_file(good) is True
    assert shape_file.looks_like_shape_file(cal) is False


# ---------------------------------------------------------------------------
# Undecodable input (AUDIT L12). Both guards documented "Never raises" and
# both let UnicodeDecodeError straight through on any non-UTF-8 file, because
# they caught only OSError. Contained by luck — UnicodeDecodeError subclasses
# ValueError, which the two callers happen to swallow — so what the user
# actually saw was a raw codec error in a "Fit cancelled" box instead of the
# clear message this module exists to produce.
#
# The reads now decode with errors="replace", which also means a shape file
# saved in latin-1 (an accent in an isotope label) loads instead of being
# refused: everything the parser reads is numeric except the isotope and chain
# names.
# ---------------------------------------------------------------------------

def _wb(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


_LATIN1_ROW = "Bi211, 2.14m, 6623, 0.836, 6.5, 8.0, 40.0, 0.7, s, cha\xeene A227\n"

UNDECODABLE = {
    "binary_high_bytes": bytes(range(128, 256)),
    "nul_bearing":       b"a,b,c\x00,d,e,f,g\n",
    "utf16":             "Bi211,1,6623,0.8,6.5,8.0,40.0,0.7,s,A\n".encode("utf-16"),
    "latin1":            (_LATIN1_ROW * 2).encode("latin-1"),
}


@pytest.mark.parametrize("name", sorted(UNDECODABLE))
def test_guards_never_raise_on_undecodable_input(tmp_path, name):
    """Each of these raised UnicodeDecodeError out of all three functions."""
    p = _wb(tmp_path, name, UNDECODABLE[name])
    shape_file.count_shape_rows(p)
    shape_file.looks_like_shape_file(p)
    shape_file.validate_shape_file(p)


def test_binary_file_is_named_as_binary(tmp_path):
    p = _wb(tmp_path, "hist.root", b"root\x00\x01\x02binary junk here" * 20)
    problems = shape_file.validate_shape_file(p)
    assert problems and "binary" in problems[0].lower()


def test_binary_without_nul_is_still_named_as_binary(tmp_path):
    """High-byte content with no NUL: caught by the replacement-char ratio."""
    p = _wb(tmp_path, "img.dat", bytes(range(128, 256)) * 8)
    problems = shape_file.validate_shape_file(p)
    assert problems and "binary" in problems[0].lower()


def test_latin1_shape_file_still_loads(tmp_path):
    """The sympathetic case: a real shape file with one accented name."""
    p = _wb(tmp_path, "s.txt", (_LATIN1_ROW * 2).encode("latin-1"))
    assert shape_file.count_shape_rows(p) == 2
    assert shape_file.looks_like_shape_file(p) is True
    assert shape_file.validate_shape_file(p) == []


def test_a_few_accents_are_not_mistaken_for_binary(tmp_path):
    """Guards the ratio threshold from the other side."""
    p = _wb(tmp_path, "s.txt", (_LATIN1_ROW * 40).encode("latin-1"))
    assert shape_file.validate_shape_file(p) == []


def test_directory_is_rejected_without_raising(tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    assert shape_file.validate_shape_file(str(d)) != []
    assert shape_file.looks_like_shape_file(str(d)) is False


def test_utf8_shape_file_is_unaffected(tmp_path):
    """errors="replace" must not change what a normal file parses to."""
    p = _w(tmp_path, "s.txt", _GOOD)
    assert shape_file.count_shape_rows(p) == 3
    assert shape_file.validate_shape_file(p) == []
