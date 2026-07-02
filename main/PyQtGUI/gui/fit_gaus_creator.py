#!/usr/bin/env python

from fit_function import FitFunction
import numpy as np

try:
    from scipy.signal import find_peaks, peak_widths
    _HAVE_SCIPY_SIGNAL = True
except Exception:
    _HAVE_SCIPY_SIGNAL = False

# Maximum number of Gaussians supported by the GUI.
MAX_GAUSS = 10


class GausFit(FitFunction):
    """Sum of ``n_gauss`` un-normalized Gaussians.

    Each Gaussian contributes three parameters laid out contiguously::

        params = [A0, mu0, sigma0, A1, mu1, sigma1, ...]

    Initial parameters may be partially seeded from the fit panel; any value
    left blank is estimated automatically by peak detection over the fit range.
    """

    def __init__(self, n_gauss=1, amplitude=1000, mean=100, standard_deviation=10):
        self.n_gauss = max(1, min(int(n_gauss), MAX_GAUSS))
        # A sensible default seed; the real seeds are built in
        # set_initial_parameters() once the data is known.
        params = np.tile(
            np.array([amplitude, mean, standard_deviation], dtype=np.float64),
            self.n_gauss,
        )
        super().__init__(params)

    @staticmethod
    def _single(x, a, m, s):
        sigma = float(abs(s))
        if sigma <= 1e-12:
            sigma = 1e-12
        return a * np.exp(-(x - m)**2 / (2 * sigma**2))

    def model(self, x, params):
        """Sum of ``n_gauss`` un-normalized Gaussians."""
        x = np.asarray(x, dtype=np.float64)
        total = np.zeros_like(x, dtype=np.float64)
        for i in range(self.n_gauss):
            a, m, s = params[3*i], params[3*i + 1], params[3*i + 2]
            total = total + self._single(x, a, m, s)
        return total

    # -------------------------------------------------------------------------
    def _auto_peaks(self, x, y, n_needed):
        """Return up to ``n_needed`` (mean, amplitude, sigma) guesses found in
        the data, ordered from most to least prominent.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        guesses = []

        if _HAVE_SCIPY_SIGNAL and len(y) >= 3:
            # Require a little prominence so noise spikes are ignored.
            prom = max((np.nanmax(y) - np.nanmin(y)) * 0.05, 1e-9)
            idx, props = find_peaks(y, prominence=prom)
            if len(idx) > 0:
                order = np.argsort(props["prominences"])[::-1]
                idx = idx[order]
                # Width (in bins) -> sigma estimate via FWHM.
                try:
                    widths = peak_widths(y, idx, rel_height=0.5)[0]
                except Exception:
                    widths = np.full(len(idx), np.nan)
                dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
                for k, pi in enumerate(idx):
                    w = widths[k] if k < len(widths) else np.nan
                    if np.isfinite(w) and w > 0:
                        sigma = (w * dx) / 2.3548
                    else:
                        sigma = float(np.std(x)) / max(n_needed, 1)
                    guesses.append((float(x[pi]), float(y[pi]), float(sigma)))

        # Fall back to evenly spaced seeds if detection came up short.
        if len(guesses) < n_needed:
            xmin, xmax = float(x[0]), float(x[-1])
            span = (xmax - xmin) if xmax > xmin else 1.0
            amax = float(np.nanmax(y)) if len(y) else 1.0
            sig = span / (4.0 * max(n_needed, 1))
            for j in range(len(guesses), n_needed):
                frac = (j + 1) / (n_needed + 1)
                guesses.append((xmin + frac * span, amax, sig))

        return guesses

    def set_initial_parameters(self, x, y, params):
        """Build a 3*n_gauss seed vector.

        ``params`` is the raw fit-panel vector (entries are None when blank).
        For each Gaussian, blanks are filled from automatic peak detection.
        """
        p = list(params)

        def _get(i):
            return float(p[i]) if (i < len(p) and p[i] is not None) else None

        # Means the user explicitly provided (so auto-detection can avoid them).
        auto = self._auto_peaks(x, y, self.n_gauss)

        clean = []
        for i in range(self.n_gauss):
            a0 = _get(3*i)
            m0 = _get(3*i + 1)
            s0 = _get(3*i + 2)
            am, aa, asig = auto[i] if i < len(auto) else (float(np.mean(x)),
                                                          float(np.nanmax(y)),
                                                          float(np.std(x)))
            if m0 is None:
                m0 = am
            if a0 is None:
                a0 = aa
            if s0 is None:
                s0 = asig
            clean.extend([a0, m0, s0])

        clean_params = np.array(clean, dtype=np.float64)
        super().set_initial_parameters(x, y, clean_params)

    def start(self, x, y, xmin, xmax, params, axis, fit_results):
        """Fit, then plot the total and (for n_gauss > 1) dashed sub-peaks with
        labeled per-Gaussian results.
        """
        from scipy.optimize import minimize

        self.set_initial_parameters(x, y, params)
        result = minimize(self.neg_log_likelihood_p,
                          x0=self.p_init,
                          args=(x, y),
                          method='bfgs',
                          jac='3-point',
                          options={"gtol": 1e-3})
        if not result.success:
            print(f"WARNING: fit did not terminate successfully:\n{result}")

        fitln = None
        try:
            x_fit = np.linspace(x[0], x[-1], 10000)
            y_fit = self.model(x_fit, result.x)
            fitln, = axis.plot(x_fit, y_fit, 'r-')

            # Dashed individual Gaussians when there is more than one.
            if self.n_gauss > 1:
                for i in range(self.n_gauss):
                    a, m, s = result.x[3*i], result.x[3*i + 1], result.x[3*i + 2]
                    axis.plot(x_fit, self._single(x_fit, a, m, s), 'r--',
                              linewidth=0.8)

            err = np.sqrt(np.abs(np.diag(result.hess_inv)))
            labels = ['A', 'mu', 'sigma']
            for i in range(self.n_gauss):
                for j in range(3):
                    k = 3*i + j
                    s = (f'{labels[j]}[{i}]: {round(result.x[k], 6)}'
                         f'+/-{round(err[k], 6)}')
                    fit_results.append(s)
        except Exception:
            pass

        return fitln


class GausFitBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, n_gauss=1, amplitude=1000, mean=100, standard_deviation=10):
        n = max(1, min(int(n_gauss), MAX_GAUSS))
        # Rebuild when the requested number of Gaussians changes.
        if self._instance is None or getattr(self._instance, "n_gauss", None) != n:
            self._instance = GausFit(n, amplitude, mean, standard_deviation)
        return self._instance
