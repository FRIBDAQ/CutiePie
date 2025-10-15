# alpha_emg22_simple.py  — minimal lmfit version

import numpy as np
from scipy.special import erfc
from lmfit import Model, Parameters, fit_report
import fit_factory  # keep for factory discovery

_INV_SQRT2 = 1.0 / np.sqrt(2.0)

def emg_one_tail(x, A, mu, sigma, tau):
    # Plain left-tailed EMG (Gaussian convolved with exponential)
    return (A/(2.0*tau)) * np.exp((sigma**2)/(2.0*tau**2) - (x - mu)/tau) * \
           erfc(((sigma/tau) - (x - mu)/sigma) * _INV_SQRT2)

def sum_two_emg_two_tail(x,
                         mu1, mu2, sigma1, sigma2,
                         tau11, tau12, eta1,
                         tau21, tau22, eta2,
                         A1, A2):
    y1 = (1.0 - eta1) * emg_one_tail(x, A1, mu1, sigma1, tau11) \
       + (        eta1) * emg_one_tail(x, A1, mu1, sigma1, tau12)
    y2 = (1.0 - eta2) * emg_one_tail(x, A2, mu2, sigma2, tau21) \
       + (        eta2) * emg_one_tail(x, A2, mu2, sigma2, tau22)
    return y1 + y2

# Order of seeds expected in fitpar:
SEED_ORDER = [
    "mu1", "mu2", "sigma1", "sigma2",
    "tau11", "tau12", "eta1",
    "tau21", "tau22", "eta2",
    "A1"
    # (A2 is tied: A2 = ratio*A1)
]

def _default_bounds():
    eps = 1e-9
    return {
        "mu1":    (0, np.inf),
        "mu2":    (0, np.inf),
        "sigma1": (eps, np.inf),
        "sigma2": (eps, np.inf),
        "tau11":  (eps, np.inf),
        "tau12":  (eps, np.inf),
        "eta1":   (0.0, 1.0),
        "tau21":  (eps, np.inf),
        "tau22":  (eps, np.inf),
        "eta2":   (0.0, 1.0),
        "A1":     (0.0, np.inf),
        # A2 is an expression -> bounds not needed
    }

class AlphaEMG22FitVanilla:
    """
    Minimal lmfit-based fitter.
    - Pass initial guesses as `fitpar` in SEED_ORDER.
    - Set A2 = ratio * A1 (ratio fixed).
    - Optional bounds dict via builder/config (same keys as SEED_ORDER).
    """
    def __init__(self, param_1=1.0, param_2=None, param_3=None, *,
                 ratio=None, bounds=None):
        # Allow ratio by keyword or param_1; fall back to a sensible constant if missing.
        self.ratio = float(ratio if ratio is not None else param_1 or 1.0)
        self.user_bounds = bounds or {}

        # Build the lmfit Model once
        self._model = Model(sum_two_emg_two_tail, independent_vars=["x"])

    def _make_params(self, seeds):
        # seeds: list/tuple in SEED_ORDER
        pars = Parameters()

        # merge user bounds with defaults
        b = _default_bounds()
        b.update(self.user_bounds or {})

        # add A2 as an expression of A1
        pars.add('ratio', value=self.ratio, vary=False)

        # Set seed values and bounds
        for name, val in zip(SEED_ORDER, seeds):
            lo, hi = b.get(name, (-np.inf, np.inf))
            pars.add(name, value=float(val), min=lo, max=hi)

        # Add A2 as tied parameter
        pars.add('A2', expr='ratio*A1')

        return pars

    def start(self, x, y, xmin, xmax, fitpar, axis, fit_results):
        # --- subrange & sanity ---
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if np.isfinite(xmin) and np.isfinite(xmax):
            m = (x >= xmin) & (x <= xmax)
            x, y = x[m], y[m]
        if x.size < 5:
            fit_results.setPlainText("Not enough data to fit.")
            return None

        # --- seeds ---
        if fitpar is None or len(fitpar) < len(SEED_ORDER):
            fit_results.setPlainText(f"Need {len(SEED_ORDER)} seeds in order: {SEED_ORDER}")
            return None
        seeds = [float(fitpar[i]) for i in range(len(SEED_ORDER))]

        # --- build params & fit ---
        pars = self._make_params(seeds)
        res = self._model.fit(y, params=pars, x=x, method='least_squares')

        # --- quick report ---
        try:
            rpt = []
            rpt.append(f"[notes] method=least_squares ; A2 tied: A2 = {self.ratio:.6g} * A1")
            rpt.append(f"[stats] chi-square={res.chisqr:.3g} ; red-chi={res.redchi:.3g} ; dof={res.nfree}")
            for name in (SEED_ORDER + ["A2"]):
                p = res.params[name]
                if p.vary:
                    err = (p.stderr if (p.stderr is not None and np.isfinite(p.stderr)) else float('nan'))
                    rpt.append(f"{name:>6} = {p.value:.6g} ± {err:.3g}")
                else:
                    rpt.append(f"{name:>6} = {p.value:.6g} (fixed/tied)")
            rpt.append("")  # spacing
            rpt.append(fit_report(res, show_correl=False))
            fit_results.setPlainText("\n".join(rpt))
        except Exception:
            fit_results.setPlainText(str(res))

        # --- draw fitted curve ---
        xx = np.linspace(x.min(), x.max(), 1200)
        yy = res.eval(x=xx)
        (fitln,) = axis.plot(xx, yy, lw=2)
        return fitln


class AlphaEMG22FitVanillaBuilder:

    def __init__(self):
        self._instance = None

    # Accept ratio/bounds via fit_factory configs OR legacy param_1
    def __call__(self, param_1=1.0, param_2=None, param_3=None, **config):
        if not self._instance:
            self._instance = AlphaEMG22FitVanilla(param_1=param_1,
                                           ratio=config.get("ratio", None),
                                           bounds=config.get("bounds", None))
        return self._instance
