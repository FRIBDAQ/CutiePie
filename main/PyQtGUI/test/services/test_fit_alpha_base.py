import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

import math

import fit_alpha_base as base


# M9: _as_float was hoisted from fit_alpha{12,22,32}_creator into
# fit_alpha_base (the three copies were AST-identical). fit_alpha_base is
# Qt-free and importable headless, so this net locks the single-source
# behavior — the creators now import it and add no coverage here (they pull
# in PyQt5 and skip in this environment).

def _isnan(v):
    return isinstance(v, float) and math.isnan(v)


def test_as_float_numeric_passthrough():
    assert base._as_float(3.5) == 3.5
    assert base._as_float(0) == 0.0
    assert base._as_float(-2) == -2.0
    assert isinstance(base._as_float(0), float)


def test_as_float_numeric_strings_are_parsed():
    assert base._as_float("3.5") == 3.5
    assert base._as_float("  12 ") == 12.0        # whitespace tolerated
    assert base._as_float("1e-3") == 0.001


def test_as_float_none_and_blank_are_nan():
    assert _isnan(base._as_float(None))
    assert _isnan(base._as_float(""))
    assert _isnan(base._as_float("   "))


def test_as_float_none_nan_words_are_nan_case_insensitive():
    assert _isnan(base._as_float("none"))
    assert _isnan(base._as_float(" None "))
    assert _isnan(base._as_float("NaN"))
    assert _isnan(base._as_float("nan"))


def test_as_float_non_numeric_is_nan_never_raises():
    assert _isnan(base._as_float("abc"))
    assert _isnan(base._as_float([1]))
    assert _isnan(base._as_float({}))


def test_as_float_preserves_infinity():
    assert base._as_float("inf") == float("inf")
    assert base._as_float(float("inf")) == float("inf")
