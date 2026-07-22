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
