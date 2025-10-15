#!/usr/bin/env python
# fit_alpha1_creator.py
#
# Single-tail EMG (AlphaEMG1) with optional bin-width integration and Poisson weighting.
# Works with your fit_factory/GUI:
#   - class name: AlphaEMG1Fit
#   - builder:    AlphaEMG1FitBuilder
#   - entry:      AlphaEMG1Fit.start(x, y, xmin, xmax, fitpar, axis, fit_results)
#
# fitpar layouts supported (backward compatible):
#   OLD (10 fields): [A, mu, sigma, tau1, tau2, eta, b0, b1, bw, wmode]
#   NEW ( 6 fields): [A, mu, sigma, tau1, bw, wmode]
#   MIN ( 4 fields): [A, mu, sigma, tau1]   -> bw defaults to median dx; wmode defaults to 1

import sys, os
sys.path.append(os.getcwd())

import numpy as np
from lmfit import Model, Parameters, fit_report
from scipy.special import erfcx

import fit_factory  # keep import so the factory can discover this module

# 7-point Gauss–Legendre nodes/weights on [-1, 1]
_GL7_T = np.array([0.0,
                   -0.4058451513773972,  0.4058451513773972,
                   -0.7415311855993945,  0.7415311855993945,
                   -0.9491079123427585,  0.9491079123427585], dtype=float)
_GL7_W = np.array([0.4179591836734694,
                   0.3818300505051189,  0.3818300505051189,
                   0.2797053914892766,  0.2797053914892766,
                   0.1294849661688697,  0.1294849661688697], dtype=float)

def _emg1(x, A, mu, sigma, tau1):
    """
    Single-tail EMG (no background). Returns y=f(x) (height/density units).
    """
    sigma = max(float(sigma), 1e-9)
    tau1  = max(float(tau1),  1e-9)
    beta1 = ((x - mu)/sigma + (sigma/tau1))/np.sqrt(2.0)
    g     = np.exp(-((x - mu)**2)/(2.0*sigma**2))
    return 0.5 * A * (1.0 / tau1) * g * erfcx(beta1)

def _bin_integral(fun, x, bw, *args):
    """
    Integral over [x - bw/2, x + bw/2] using 7-pt Gauss–Legendre.
    If bw<=0, returns fun(x, *args).
    """
    if bw <= 0.0:
        return fun(x, *args)
    acc = 0.0
    half = 0.5 * bw
    for wi, ti in zip(_GL7_W, _GL7_T):
        acc += wi * fun(x + half * ti, *args)
    return half * acc  # integral (counts per bin)

def _emg1_binned(x, A, mu, sigma, tau1, bw):
    """
    If bw>0, return counts per bin by integrating _emg1 over each bin;
    otherwise return the height.
    """
    bw = float(bw)
    return _bin_integral(_emg1, x, bw, A, mu, sigma, tau1)

class AlphaEMG1Fit:
    def __init__(self, param_1=1, param_2=2, param_3=10):
        self.param_1 = param_1
        self.param_2 = param_2
        self.param_3 = param_3

    def start(self, x, y, xmin, xmax, fitpar, axis, fit_results):
        """
        fitpar (len-tolerant):
          OLD: [A, mu, sigma, tau1, tau2, eta, b0, b1, bw, wmode]
          NEW: [A, mu, sigma, tau1, bw, wmode]
          MIN: [A, mu, sigma, tau1]  -> bw=median(dx), wmode=1

        wmode: 0=unweighted, 1=Poisson(data), 2=Poisson(model IRLS)
        """
        # finite data only
        m = np.isfinite(x) & np.isfinite(y)
        x = np.asarray(x)[m]
        y = np.asarray(y)[m]
        if x.size < 5:
            fit_results.setPlainText("Not enough data to fit.")
            return None

        # --- Backward-compatible fitpar parsing ---
        fp = list(fitpar) if fitpar is not None else []
        A0_ui = mu0_ui = sig0_ui = t10_ui = np.nan
        bw_ui = wmode_ui = np.nan

        if len(fp) >= 4:
            A0_ui, mu0_ui, sig0_ui, t10_ui = fp[:4]

        # Prefer legacy positions if 10+ fields (old double-EMG GUI)
        if len(fp) >= 10:
            bw_ui, wmode_ui = fp[8], fp[9]
        elif len(fp) >= 6:
            # New compact layout: [A, mu, sigma, tau1, bw, wmode]
            bw_ui, wmode_ui = fp[4], fp[5]

        def parse_wmode(val):
            try:
                k = int(round(float(val)))
                return k if k in (0, 1, 2) else 1
            except Exception:
                return 1  # default Poisson(data)

        # Data-driven seeds
        dx     = float(np.median(np.diff(x))) if x.size > 1 else 1.0
        span   = float(x.max() - x.min()) or 1.0
        mu0    = float(x[np.argmax(y)]) if np.all(np.isfinite(y)) else float((x.min()+x.max())/2.0)
        sigma0 = max(dx, span/50.0)
        tau10  = 0.5*sigma0
        A0     = float(np.trapz(np.clip(y, 0, None), x))

        bw     = float(bw_ui) if np.isfinite(bw_ui) and float(bw_ui) != 0 else dx
        wmode  = parse_wmode(wmode_ui)

        def pick(ui, auto):
            return float(ui) if np.isfinite(ui) and ui != 0 else float(auto)

        pars = Parameters()
        pars.add('A',     value=max(pick(A0_ui, A0), 1.0), min=0)
        # tighten mu to a local window to improve identifiability
        mu_lo = mu0 - 3.0*sigma0
        mu_hi = mu0 + 3.0*sigma0
        pars.add('mu',    value=pick(mu0_ui, mu0),
                          min=float(max(x.min(), mu_lo)),
                          max=float(min(x.max(), mu_hi)))
        pars.add('sigma', value=max(pick(sig0_ui, sigma0), dx/4.0), min=dx/4.0, max=span)
        pars.add('tau1',  value=max(pick(t10_ui, tau10),  dx/10.0),  min=dx/10.0, max=10.0*sigma0)
        pars.add('bw',    value=max(bw, 0.0), vary=False, min=0.0)

        model = Model(_emg1_binned, independent_vars=['x'])

        # Preflight sanity: ensure finite initial curve; if not, broaden
        ok = False
        for _ in range(5):
            y0 = model.eval(pars, x=x)
            if np.all(np.isfinite(y0)):
                ok = True
                break
            pars['sigma'].set(value=max(pars['sigma'].value*2.0, dx/2.0))
            pars['tau1'].set(value=max(pars['tau1'].value*2.0, dx/5.0))
        if not ok:
            fit_results.setPlainText("Model non-finite at initial guess; aborting.")
            return None

        # Weights
        def w_from_data(y_arr):
            return 1.0 / np.sqrt(np.clip(y_arr, 1.0, None))

        if wmode == 0:
            weights = None
        elif wmode == 1:
            weights = w_from_data(y)
        else:  # wmode == 2 (IRLS)
            weights = w_from_data(y)

        # First fit
        res = model.fit(y, params=pars, x=x, method='least_squares', weights=weights)

        # IRLS refinement (one step) if requested
        if wmode == 2:
            try:
                yhat = model.eval(res.params, x=x)
                w = 1.0 / np.sqrt(np.clip(yhat, 1.0, None))
                res = model.fit(y, params=res.params.copy(), x=x, method='least_squares', weights=w)
            except Exception:
                pass


        # Report (named parameters first, then full lmfit report)
        def _format_params(params):
            lines = []
            for name, par in params.items():
                if name == 'bw':      # usually fixed; skip in the short list
                    continue
                val = par.value
                err = par.stderr
                if err is not None and np.isfinite(err):
                    lines.append(f"{name:>8} = {val:.6g}  ± {err:.3g}")
                else:
                    lines.append(f"{name:>8} = {val:.6g}")
            return "\n".join(lines)

        stats_line = (f"[stats] chi-square={res.chisqr:.3f} ; "
                      f"reduced chi-square={res.redchi:.3f} ; "
                      f"dof={res.nfree} (N={res.ndata}, k={res.nvarys})")

        try:
            wtxt   = {0: "none", 1: "Poisson(data)", 2: "Poisson(model, IRLS)"}.get(wmode, "Poisson(data)")
            header = f"Notes: bandwidth = {res.params['bw'].value:.6g} ; weighting = {wtxt}"
            named  = _format_params(res.params)

            fit_results.setPlainText(
                header + "\n" +
                stats_line + "\n\n" +
                "Parameters (named):\n" + named + "\n\n" +
                fit_report(res, show_correl=True)
            )
        except Exception:
            fit_results.setPlainText(str(res))


        # Draw fitted curve (counts if bw>0)
        xx = np.linspace(x.min(), x.max(), 2000)
        yy = model.eval(res.params, x=xx)
        (fitln,) = axis.plot(xx, yy, lw=2)
        return fitln

class AlphaEMG1FitBuilder:
    def __init__(self):
        self._instance = None
    def __call__(self, param_1=1, param_2=2, param_3=10, **_ignored):
        if not self._instance:
            self._instance = AlphaEMG1Fit(param_1, param_2, param_3)
        return self._instance
