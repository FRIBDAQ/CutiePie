#!/usr/bin/env python
# fit_alpha2_creator.py
#
# Two-peak Alpha EMG (sum of two single-tail EMGs) with bin-width integration and Poisson weighting.
# Discovery: keep this in the same folder as fit_factory so the factory can import it.
#
# Class:   AlphaEMG2Fit
# Builder: AlphaEMG2FitBuilder
#
# fitpar layouts supported:
#   8 seeds (GUI boxes): [A1, mu1, sig1, tau11,  A2, mu2, sig2, tau12]
#   + appended by GUI:   [bw, wmode]  -> total length 10 (backward-compatible)
#   Minimal:             you may leave p4..p7 as 0; the second peak is auto-guessed.
#
# wmode: 0=unweighted, 1=Poisson(data), 2=Poisson(model IRLS)

import sys, os
sys.path.append(os.getcwd())

import numpy as np
from lmfit import Model, Parameters, fit_report
from scipy.special import erfcx

import fit_factory  # so the factory discovers this module

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
    sigma = max(float(sigma), 1e-9)
    tau1  = max(float(tau1),  1e-9)
    beta1 = ((x - mu)/sigma + (sigma/tau1))/np.sqrt(2.0)
    g     = np.exp(-((x - mu)**2)/(2.0*sigma**2))
    return 0.5 * A * (1.0 / tau1) * g * erfcx(beta1)

def _bin_integral(fun, x, bw, *args):
    if bw <= 0.0:
        return fun(x, *args)
    acc = 0.0
    half = 0.5 * float(bw)
    for wi, ti in zip(_GL7_W, _GL7_T):
        acc += wi * fun(x + half * ti, *args)
    return half * acc

def _emg2sum(x, A1, mu1, sig1, tau11, A2, mu2, sig2, tau12):
    return _emg1(x, A1, mu1, sig1, tau11) + _emg1(x, A2, mu2, sig2, tau12)

def _emg2sum_binned(x, A1, mu1, sig1, tau11, A2, mu2, sig2, tau12, bw):
    return _bin_integral(_emg2sum, x, bw, A1, mu1, sig1, tau11, A2, mu2, sig2, tau12)

class AlphaEMG2Fit:
    def __init__(self, link_tau=False, param_1=1, param_2=2, param_3=10):
        """
        link_tau: if True, both peaks share the same tau1 (reduces 1 DoF).
        Other params kept for factory signature compatibility.
        """
        self.link_tau = bool(link_tau)
        self.param_1 = param_1
        self.param_2 = param_2
        self.param_3 = param_3

    def start(self, x, y, xmin, xmax, fitpar, axis, fit_results):
        # finite data only
        m = np.isfinite(x) & np.isfinite(y)
        x = np.asarray(x)[m]
        y = np.asarray(y)[m]
        if x.size < 5:
            fit_results.setPlainText("Not enough data to fit.")
            return None

        # ---- Parse seeds (8 GUI boxes) + bw/wmode at [8],[9] if present ----
        fp = list(fitpar) if fitpar is not None else []
        vals = fp + [np.nan]*10  # pad
        A1_ui, mu1_ui, s1_ui, t11_ui, A2_ui, mu2_ui, s2_ui, t12_ui, bw_ui, wmode_ui = vals[:10]

        def parse_wmode(v):
            try:
                k = int(round(float(v)))
                return k if k in (0,1,2) else 1
            except Exception:
                return 1

        # Data-driven seeds
        dx     = float(np.median(np.diff(x))) if x.size > 1 else 1.0
        span   = float(x.max() - x.min()) or 1.0
        sigma0 = max(dx, span/50.0)
        tau10  = 0.5 * sigma0
        A_tot  = float(np.trapz(np.clip(y,0,None), x))

        min_sep = max(3.0 * sigma0, 6.0 * dx)   # require peaks to start/finalize at least this far apart

        # Find two separated peaks (smooth a bit, then pick best two ≥ min_sep apart)
        w_bins = max(3, int(round(min_sep / dx)))           # smoothing width in bins
        if w_bins % 2 == 0:
            w_bins += 1
        ys = np.convolve(y, np.ones(w_bins) / w_bins, mode='same') if y.size >= w_bins else y.copy()

        # simple local maxima detection (>= right neighbor, > left neighbor)
        left  = np.r_[True, ys[1:] > ys[:-1]]
        right = np.r_[ys[:-1] >= ys[1:], True]
        cand = np.where(left & right)[0]

        # sort candidates by height descending
        cand = cand[np.argsort(ys[cand])[::-1]]

        mu_candidates = []
        for idx in cand:
            xi = float(x[idx])
            if not mu_candidates:
                mu_candidates.append(xi)
            elif abs(xi - mu_candidates[0]) >= min_sep:
                mu_candidates.append(xi)
                break

        # fallback if only one found
        if len(mu_candidates) < 2:
            peak = float(x[np.argmax(y)])
            mu_candidates = [peak, peak + min_sep]

        mu_candidates.sort()
        mu1_0, mu2_0 = mu_candidates[0], mu_candidates[1]

        # UI overrides if provided (non-zero/non-NaN)
        def pick(ui, auto):
            return float(ui) if np.isfinite(ui) and ui != 0 else float(auto)

        A1_0   = pick(A1_ui, 0.5*A_tot)
        mu1_0  = pick(mu1_ui, mu1_0)
        s1_0   = max(pick(s1_ui, sigma0), dx/4.0)

        A2_0   = pick(A2_ui, 0.5*A_tot)
        mu2_0  = pick(mu2_ui, mu2_0)
        s2_0   = max(pick(s2_ui, sigma0), dx/4.0)

        t11_0  = max(pick(t11_ui, tau10), dx/10.0)
        t12_0  = max(pick(t12_ui, tau10), dx/10.0)

        # bw & weighting
        bw     = float(bw_ui) if np.isfinite(bw_ui) and float(bw_ui) != 0 else dx
        wmode  = parse_wmode(wmode_ui)

        # Parameters
        pars = Parameters()
        pars.add('A1',   value=max(A1_0, 1.0), min=0)
        pars.add('mu1',  value=mu1_0, min=float(x.min()), max=float(x.max()))
        pars.add('sig1', value=s1_0,  min=dx/4.0, max=span)
        pars.add('tau11',value=t11_0, min=dx/10.0, max=10.0*sigma0)

        # Enforce mu2 > mu1 via expression and require a minimum separation
        dmu_min = float(min_sep)
        dmu0 = max(mu2_0 - mu1_0, dmu_min)
        pars.add('dmu', value=dmu0, min=dmu_min, max=span)
        pars.add('mu2', expr='mu1 + dmu')

        pars.add('A2',   value=max(A2_0, 1.0), min=0)
        pars.add('sig2', value=s2_0,  min=dx/4.0, max=span)

        if self.link_tau:
            pars.add('tau12', expr='tau11')
        else:
            pars.add('tau12', value=t12_0, min=dx/10.0, max=10.0*sigma0)

        pars.add('bw', value=max(bw, 0.0), vary=False, min=0.0)

        model = Model(_emg2sum_binned, independent_vars=['x'])

        # Preflight: ensure finiteness; if not, broaden widths
        ok = False
        for _ in range(6):
            y0 = model.eval(pars, x=x)
            if np.all(np.isfinite(y0)):
                ok = True
                break
            for nm in ('sig1','sig2'):
                pars[nm].set(value=max(pars[nm].value*2.0, dx/2.0))
            for nm in ('tau11','tau12'):
                if pars[nm].vary:
                    pars[nm].set(value=max(pars[nm].value*2.0, dx/5.0))
        if not ok:
            fit_results.setPlainText("Model non-finite at initial guess; aborting.")
            return None

        # Weights
        def w_from_data(y_arr): return 1.0/np.sqrt(np.clip(y_arr, 1.0, None))
        if wmode == 0:
            weights = None
        elif wmode == 1:
            weights = w_from_data(y)
        else:
            weights = w_from_data(y)

        # First fit
        res = model.fit(y, params=pars, x=x, method='least_squares', weights=weights)

        # IRLS refinement (one step)
        if wmode == 2:
            try:
                yhat = model.eval(res.params, x=x)
                w = 1.0 / np.sqrt(np.clip(yhat, 1.0, None))
                res = model.fit(y, params=res.params.copy(), x=x, method='least_squares', weights=w)
            except Exception:
                pass

        # Report
        try:
            wtxt = {0: "none", 1: "Poisson(data)", 2: "Poisson(model, IRLS)"}[wmode]
        except KeyError:
            wtxt = "Poisson(data)"
        header = f"Notes: bandwidth = {res.params['bw'].value:.6g} ; weighting = {wtxt}"
        stats = (f"[stats] chi-square={res.chisqr:.3f} ; reduced chi-square={res.redchi:.3f} ; "
                 f"dof={res.nfree} (N={res.ndata}, k={res.nvarys})")
        try:
            fit_results.setPlainText(header + "\n" + stats + "\n" + fit_report(res, show_correl=True))
        except Exception:
            fit_results.setPlainText(str(res))

        # Plot sum curve
        xx = np.linspace(x.min(), x.max(), 2000)
        yy = model.eval(res.params, x=xx)
        (fitln,) = axis.plot(xx, yy, lw=2)
        return fitln

class AlphaEMG2FitBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, link_tau=False, param_1=1, param_2=2, param_3=10, **_ignored):
        if not self._instance:
            self._instance = AlphaEMG2Fit(link_tau=link_tau, param_1=param_1, param_2=param_2, param_3=param_3)
        return self._instance

