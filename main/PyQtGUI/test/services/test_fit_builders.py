"""Fit/algo builders hand back a fresh object, and numpy's invalid-value
suppression belongs to a fit rather than to the process. Both halves used to
be latent rather than broken: the cached builders only ever received static
config, and the process-wide `np.seterr` only ever hid warnings nobody was
reading."""

import ast
import importlib
import pathlib
import sys
import warnings

import numpy as np
import pytest

GUI = pathlib.Path(__file__).resolve().parents[2] / "gui"
sys.path.insert(0, str(GUI))

CREATORS = sorted(GUI.glob("*_creator.py"))


def builder_classes(path):
    tree = ast.parse(path.read_text())
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name.endswith("Builder")]


def importable(stem):
    try:
        return importlib.import_module(stem)
    except ImportError:
        return None


@pytest.mark.parametrize("path", CREATORS, ids=lambda p: p.stem)
def test_no_builder_caches_its_instance(path):
    """Source-level, so the lmfit/PyQt5/cv2 creators are covered too."""
    for cls in builder_classes(path):
        assert "_instance" not in ast.unparse(cls), \
            "%s.%s still caches; a later config change would be discarded" % (path.name, cls.name)


def test_the_excluded_creator_is_still_covered_by_the_source_check():
    for stem in NEEDS_DATA_FILE:
        path = next(p for p in CREATORS if p.stem == stem)
        assert builder_classes(path), "%s has no Builder class to check" % stem


def test_every_creator_defines_exactly_one_builder():
    """Guards the check above against a renamed class quietly dropping coverage."""
    counts = {p.name: len(builder_classes(p)) for p in CREATORS}
    assert all(n == 1 for n in counts.values()), counts
    assert len(counts) >= 17


IMPORTABLE = [p.stem for p in CREATORS if importable(p.stem) is not None]

# AlphaEMGLinear's default construction reads `shapes_Chand.txt`, a data file
# this checkout does not ship, so it raises FileNotFoundError before the builder
# ever returns. Excluded from the runtime check rather than skipped, so the
# suite's skip count stays at the 7 environment gates; the source-level check
# above still covers its builder.
NEEDS_DATA_FILE = {"fit_alpha_linear_creator"}
RUNNABLE = [s for s in IMPORTABLE if s not in NEEDS_DATA_FILE]


@pytest.mark.parametrize("stem", RUNNABLE)
def test_builder_returns_a_new_object_each_call(stem):
    mod = importlib.import_module(stem)
    cls = next(getattr(mod, n) for n in dir(mod) if n.endswith("Builder"))
    build = cls()
    first, second = build(), build()
    assert first is not second


def test_builder_honours_config_given_on_a_later_call():
    """The defect itself: the second fit's parameters must reach the second fit."""
    from fit_gaus_creator import GausFitBuilder
    build = GausFitBuilder()
    first = build(amplitude=1000, mean=100, standard_deviation=10)
    second = build(amplitude=55, mean=7, standard_deviation=3)
    assert list(first.p_init) == [1000.0, 100.0, 10.0]
    assert list(second.p_init) == [55.0, 7.0, 3.0]     # was first.p_init when cached


def test_importing_fit_function_leaves_numpy_alone():
    """`np.seterr(invalid='ignore')` at import muted every numpy op in the GUI."""
    np.seterr(invalid="warn")
    importlib.reload(importlib.import_module("fit_function"))
    assert np.geterr()["invalid"] == "warn"


def test_start_suppresses_invalid_only_for_the_duration_of_the_fit():
    from fit_function import FitFunction

    seen = []

    class Probe(FitFunction):
        def model(self, x, params):
            seen.append(np.geterr()["invalid"])
            return np.abs(params[0]) * np.ones_like(np.asarray(x, dtype=float))

    class Ax:
        def plot(self, *a, **k):
            class Line:
                pass
            return (Line(),)

    x = np.linspace(1, 10, 10)
    y = np.full_like(x, 5.0)
    old = np.geterr()["invalid"]
    np.seterr(invalid="warn")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Probe(np.array([5.0])).start(x, y, 1, 10, [5.0], Ax(), [])
        assert seen and set(seen) == {"ignore"}          # suppressed inside the fit
        assert np.geterr()["invalid"] == "warn"          # and restored on the way out
    finally:
        np.seterr(invalid=old)
