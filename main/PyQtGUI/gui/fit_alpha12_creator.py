#!/usr/bin/env python
# fit_alpha1_creator.py
#
# AlphaEMG1: Single-peak Exponentially-Modified Gaussian with optional two-tail mixture,
# bin-width integration, and Poisson weighting.
#
# Works with your fit_factory/GUI:
#   - class name: AlphaEMG1Fit
#   - builder:    AlphaEMG1FitBuilder
#   - entry:      AlphaEMG1Fit.start(x, y, xmin, xmax, fitpar, axis, fit_results)
#
# fitpar layouts supported (backward compatible):
#   OLD (10 fields): [A, mu, sigma, tau1, tau2, eta, b0, b1, bw, wmode]
#   NEW ( 6 fields): [A, mu, sigma, tau1, bw, wmode]      (defaults: eta=0, tau2=tau1)
#   MIN ( 4 fields): [A, mu, sigma, tau1]                 (bw=median dx, wmode=1)
#
# wmode: 0=unweighted, 1=Poisson(data), 2=Poisson(model, one-step IRLS)
# mu = 6.9621 e - 4761.15
'''
Pu-239:
5156 → 31135.44, 5144 → 31051.89, 5106 → 30787.33

Am-241:
5486 → 33432.93, 5443 → 33133.56

Cm-244:
5805 → 35653.84, 5763 → 35361.43
'''

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

# At module top:
_INV_SQRT2 = 1.0 / np.sqrt(2.0)

def _emg_tail_mixture(x, A, mu, sigma, tau1, tau2, eta):
    x = np.asarray(x)  # preserve dtype of input

    sigma = max(float(sigma), 1e-9)
    tau1  = max(float(tau1),  1e-9)
    tau2  = max(float(tau2),  1e-9)
    eta   = float(np.clip(eta, 0.0, 1.0))

    inv_sigma = 1.0 / sigma
    dx = (x - mu) * inv_sigma
    g = np.exp(-0.5 * dx * dx)

    z1 = (dx + sigma / tau1) * _INV_SQRT2
    z2 = (dx + sigma / tau2) * _INV_SQRT2

    tail = (1.0 - eta) * erfcx(z1) / tau1 + eta * erfcx(z2) / tau2
    return 0.5 * A * g * tail

    # erfcx(z) = exp(z^2) * erfc(z) is the scaled complementary error function


def _bin_integral(fun, x, bw, *args):
    """
    Integral over [x - bw/2, x + bw/2] using 7-pt Gauss–Legendre.
    If bw<=0, returns fun(x, *args).
    """
    bw = float(bw)
    if bw <= 0.0:
        return fun(x, *args)
    acc = 0.0
    half = 0.5 * bw
    for wi, ti in zip(_GL7_W, _GL7_T):
        acc += wi * fun(x + half * ti, *args)
    return half * acc  # integral (counts per bin)

def _alphaemg1_binned(x, A, mu, sigma, tau1, tau2, eta, bw):
    """
    Bin-aware version: integrate the tail-mixed EMG over bin width if bw>0.
    """
    return _bin_integral(_emg_tail_mixture, x, bw, A, mu, sigma, tau1, tau2, eta)

class AlphaEMG12Fit:
    def __init__(self, param_1=1, param_2=2, param_3=10,
                 eta_vary=True, eta_value=0.0):
        self.param_1 = param_1
        self.param_2 = param_2
        self.param_3 = param_3
        self.eta_vary = bool(eta_vary)
        self.eta_value = float(eta_value)


    def start(self, x, y, xmin, xmax, fitpar, axis, fit_results):
        """
        fitpar (len-tolerant):
          OLD: [A, mu, sigma, tau1, tau2, eta, b0, b1, bw, wmode]
          NEW: [A, mu, sigma, tau1, bw, wmode]    -> eta=0, tau2=tau1
          MIN: [A, mu, sigma, tau1]               -> bw=median(dx), wmode=1

        wmode: 0=unweighted, 1=Poisson(data), 2=Poisson(model IRLS)
        """
        # finite data only
        m = np.isfinite(x) & np.isfinite(y)
        x = np.asarray(x)[m]
        y = np.asarray(y)[m]
        if x.size < 5:
            fit_results.setPlainText("Not enough data to fit.")
            return None

        # --- Parse fitpar (backward-compatible) ---
        fp = list(fitpar) if fitpar is not None else []
        A_ui = mu_ui = sig_ui = tau1_ui = tau2_ui = eta_ui = np.nan
        bw_ui = wmode_ui = np.nan

        if len(fp) >= 4:
            A_ui, mu_ui, sig_ui, tau1_ui = fp[:4]

        if len(fp) >= 10:
            # OLD layout
            tau2_ui, eta_ui = fp[4], fp[5]
            bw_ui, wmode_ui = fp[8], fp[9]
        elif len(fp) >= 6:
            # NEW compact layout
            bw_ui, wmode_ui = fp[4], fp[5]
        # else: MIN layout → tau2, eta remain NaN; bw/wmode NaN (handled below)

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
        tau10  = 0.5 * sigma0
        tau20  = 1.0 * sigma0  # a slower tail as a reasonable default
        A0     = float(np.trapz(np.clip(y, 0, None), x))
        eta0   = 0.0  # default to single-tail unless user asks

        bw     = float(bw_ui) if np.isfinite(bw_ui) and float(bw_ui) != 0 else dx
        wmode  = parse_wmode(wmode_ui)

        def pick(ui, auto):
            return float(ui) if np.isfinite(ui) and ui != 0 else float(auto)

        # Build parameters
        pars = Parameters()
        pars.add('A',     value=max(pick(A_ui, A0), 1.0), min=0)

        # tighten mu to improve identifiability around the peak
        mu_lo = mu0 - 3.0 * sigma0
        mu_hi = mu0 + 3.0 * sigma0
        pars.add('mu',    value=pick(mu_ui, mu0),
                          min=float(max(x.min(), mu_lo)),
                          max=float(min(x.max(), mu_hi)))

        pars.add('sigma', value=max(pick(sig_ui,  sigma0), dx/4.0), min=dx/4.0, max=span)
        pars.add('tau1',  value=max(pick(tau1_ui, tau10), dx/10.0), min=dx/10.0, max=10.0*sigma0)

        # tau2 / eta: if not provided, default to tau2=tau1 and eta=0 (single-tail)
        t2_seed  = pick(tau2_ui, tau20)
        eta_seed = np.clip(pick(eta_ui, eta0), 0.0, 1.0)

        # If the builder/factory wants to force eta, override the seed.
        if not self.eta_vary:
            if np.isfinite(eta_ui):
                eta_seed = np.clip(float(eta_ui), 0.0, 1.0)
            else:
                eta_seed = np.clip(self.eta_value, 0.0, 1.0)
        pars.add('eta', value=float(eta_seed), min=0.0, max=1.0, vary=self.eta_vary)

        # Enforce tau2 > tau1 via positive offset dtau:
        dtau0 = max(t2_seed - pars['tau1'].value, max(dx/10.0, 0.05*sigma0))

        # ---- Identifiability guards ----
        # If eta == 0, the slow tail never contributes -> freeze tau2 (dtau) to avoid NaNs/huge errors
        # If eta == 1, the fast tail does not contribute -> it's often best to freeze tau1.
        freeze_dtau = (not self.eta_vary) and (eta_seed <= 1e-8)
        freeze_tau1 = (not self.eta_vary) and (eta_seed >= 1.0 - 1e-8)

        pars.add('dtau', value=dtau0, min=max(dx/10.0, 0.05*sigma0), max=20.0*sigma0, vary=not freeze_dtau)
        pars.add('tau2', expr='tau1 + dtau')

        if freeze_tau1:
            pars['tau1'].set(vary=False)



        # Fixed bin width for the model
        pars.add('bw', value=max(bw, 0.0), vary=False, min=0.0)

        # Model
        model = Model(_alphaemg1_binned, independent_vars=['x'])

        # Preflight sanity: ensure finite initial curve; if not, broaden
        ok = False
        for _ in range(6):
            y0 = model.eval(pars, x=x)
            if np.all(np.isfinite(y0)):
                ok = True
                break
            pars['sigma'].set(value=max(pars['sigma'].value*2.0, dx/2.0))
            pars['tau1'].set(value=max(pars['tau1'].value*2.0, dx/5.0))
            pars['tau2'].set(value=max(pars['tau2'].value*2.0, dx/5.0))
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
        else:  # wmode == 2 (IRLS) – seed with data weights
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

class AlphaEMG12FitBuilder:
    def __init__(self):
        self._instance = None
    def __call__(self, param_1=1, param_2=2, param_3=10, eta_vary=True, eta_value=0.0, **_ignored):
        if not self._instance:
            self._instance = AlphaEMG12Fit(param_1, param_2, param_3, eta_vary=eta_vary, eta_value=eta_value)
        return self._instance
