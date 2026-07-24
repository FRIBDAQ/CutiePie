import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

import math
import numpy as np

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
