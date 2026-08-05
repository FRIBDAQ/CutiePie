import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

import math
import numpy as np
import pytest

# fit_function (base Poisson-MLE fitter) and the simple GausFit creator are
# Qt-free and importable headless. This net locks the chi2 reporting added so
# Save Fit can write goodness-of-fit for the simple (non-Alpha) fit types,
# which previously computed no chi2 at all.
import fit_function as ff
from fit_gaus_creator import GausFit


class _FakeLine:
    """A stand-in for the matplotlib Line2D returned by axis.plot — accepts
    arbitrary attribute assignment (chi2/redchi/ndof)."""
    def __init__(self, x, y):
        self._x, self._y = np.asarray(x), np.asarray(y)

    def get_xdata(self):
        return self._x

    def get_ydata(self):
        return self._y


class _FakeAxis:
    def plot(self, x, y, *a, **k):
        return (_FakeLine(x, y),)


class _FakeResults:
    def __init__(self):
        self.lines = []

    def append(self, s):
        self.lines.append(str(s))

    def text(self):
        return "\n".join(self.lines)


def _gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def test_base_start_attaches_pearson_chi2():
    rng = np.random.default_rng(0)
    x = np.linspace(4150.0, 4250.0, 200)
    truth = _gauss(x, 1000.0, 4200.0, 6.0)
    y = np.maximum(truth, 0.0)  # noiseless: fit should be near-perfect

    fit = GausFit(1000.0, 4200.0, 6.0)
    axis, results = _FakeAxis(), _FakeResults()
    fitln = fit.start(x, y, x[0], x[-1], [1000.0, 4200.0, 6.0], axis, results)

    assert fitln is not None
    assert isinstance(fitln.chi2, float) and math.isfinite(fitln.chi2)
    assert isinstance(fitln.redchi, float) and math.isfinite(fitln.redchi)
    assert isinstance(fitln.ndof, int)
    # 200 points, 3 params
    assert fitln.ndof == 200 - 3
    # redchi is chi2 / ndof
    assert fitln.redchi == fit_redchi(fitln.chi2, fitln.ndof)
    # a near-perfect fit has a tiny Pearson chi2 per dof
    assert fitln.redchi < 1e-3


def fit_redchi(chi2, ndof):
    return chi2 / max(ndof, 1)


def test_base_start_appends_stats_line():
    x = np.linspace(0.0, 100.0, 120)
    y = np.maximum(_gauss(x, 500.0, 50.0, 8.0), 0.0)

    fit = GausFit(500.0, 50.0, 8.0)
    results = _FakeResults()
    fit.start(x, y, x[0], x[-1], [500.0, 50.0, 8.0], _FakeAxis(), results)

    joined = results.text().lower()
    assert "[stats]" in joined
    assert "chi-square" in joined


def test_base_start_pearson_chi2_matches_formula():
    # A deliberately imperfect fit: data has an offset the model can't match,
    # so chi2 is non-trivial and we can check the Pearson formula exactly
    # against the model evaluated at the returned parameters.
    x = np.linspace(0.0, 40.0, 60)
    y = np.maximum(_gauss(x, 300.0, 20.0, 4.0) + 5.0, 0.0)

    fit = GausFit(300.0, 20.0, 4.0)
    fitln = fit.start(x, y, x[0], x[-1], [300.0, 20.0, 4.0], _FakeAxis(),
                      _FakeResults())

    pred = np.maximum(fit.model(x, fit._last_params), 1e-10)
    expected = float(np.sum((y - pred) ** 2 / pred))
    assert abs(fitln.chi2 - expected) < 1e-6 * max(expected, 1.0)


# ------------------------------------------------------------- the typed seam

def _fitted():
    x = np.linspace(4150.0, 4250.0, 200)
    y = np.maximum(_gauss(x, 1000.0, 4200.0, 6.0), 0.0)
    return GausFit(1000.0, 4200.0, 6.0), x, y


SEED = [1000.0, 4200.0, 6.0]


def test_run_returns_the_fit_without_touching_a_widget():
    """A fit that reports a value instead of painting one. `run` takes data and
    returns data, so a plugin is testable without Qt and without a matplotlib
    axis."""
    fit, x, y = _fitted()
    res = fit.run(ff.FitRequest(x=x, y=y, xmin=x[0], xmax=x[-1],
                                params=SEED))
    assert res.converged
    assert len(res.params) == 3
    assert res.params[1] == pytest.approx(4200.0, abs=1.0)     # the centroid


def test_run_carries_the_goodness_of_fit():
    fit, x, y = _fitted()
    res = fit.run(ff.FitRequest(x=x, y=y, xmin=x[0], xmax=x[-1],
                                params=SEED))
    assert math.isfinite(res.chi2) and math.isfinite(res.redchi)
    assert res.ndof == len(x) - 3
    assert res.redchi == pytest.approx(res.chi2 / res.ndof)


def test_run_carries_the_curve_to_draw():
    fit, x, y = _fitted()
    res = fit.run(ff.FitRequest(x=x, y=y, xmin=x[0], xmax=x[-1],
                                params=SEED))
    xf, yf = res.curve
    assert len(xf) == len(yf) == 10000
    assert xf[0] == pytest.approx(x[0]) and xf[-1] == pytest.approx(x[-1])


def test_run_carries_the_report_lines_rather_than_writing_them():
    fit, x, y = _fitted()
    res = fit.run(ff.FitRequest(x=x, y=y, xmin=x[0], xmax=x[-1],
                                params=SEED))
    assert res.stats_line and res.stats_line.startswith("[stats]")
    assert len(res.param_lines) == 3
    assert all(s.startswith("Par[") for s in res.param_lines)


def test_start_still_takes_the_old_positional_signature():
    """The shim is mandatory: user plugins outside this repo call `start` with
    seven positionals, and they must keep working unchanged."""
    fit, x, y = _fitted()
    axis, results = _FakeAxis(), _FakeResults()
    line = fit.start(x, y, x[0], x[-1], SEED, axis, results)
    assert line is not None
    assert math.isfinite(line.chi2)


def test_start_writes_the_same_lines_in_the_same_order():
    """Stats first, then one line per parameter — the order the results box
    has always shown."""
    fit, x, y = _fitted()
    results = _FakeResults()
    fit.start(x, y, x[0], x[-1], SEED, _FakeAxis(), results)
    assert results.lines[0].startswith("[stats]")
    assert [s[:4] for s in results.lines[1:]] == ["Par[", "Par[", "Par["]


def test_a_plot_that_fails_suppresses_the_parameter_lines():
    """They share a try in the original, so a pad that cannot be drawn on
    reports the stats and nothing else. Preserved deliberately."""
    class _Explodes:
        def plot(self, *a, **k):
            raise RuntimeError("no canvas")
    fit, x, y = _fitted()
    results = _FakeResults()
    line = fit.start(x, y, x[0], x[-1], SEED, _Explodes(), results)
    assert line is None
    assert len(results.lines) == 1 and results.lines[0].startswith("[stats]")


def test_run_remembers_the_fitted_parameters():
    """`_last_params` is what Save Fit re-evaluates the model from."""
    fit, x, y = _fitted()
    res = fit.run(ff.FitRequest(x=x, y=y, xmin=x[0], xmax=x[-1],
                                params=SEED))
    assert list(fit._last_params) == list(res.params)
