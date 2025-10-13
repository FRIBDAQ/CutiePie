#!/usr/bin/env python
# fit_alpha3_creator.py
#
# Three-peak Alpha EMG (sum of three single-tail EMGs) with bin-width integration and Poisson weighting.
# Keep this beside fit_factory so the factory can import/discover it.
#
# Class:   AlphaEMG3Fit
# Builder: AlphaEMG3FitBuilder
#
# fitpar layouts supported (len-tolerant):
#   12 seeds: [A1,mu1,sig1,tau11,  A2,mu2,sig2,tau12,  A3,mu3,sig3,tau13]
#   + GUI appends: [bw, wmode]  -> total length up to 14 (back-compat with your GUI)
#   If fewer than 12 seeds are provided, missing ones are auto-guessed from the data.
#
# wmode: 0=unweighted, 1=Poisson(data), 2=Poisson(model IRLS)

''' old
Pu-239:
5156 → 31135.44, 5144 → 31051.89, 5106 → 30787.33

Am-241:
5486 → 33432.93, 5443 → 33133.56

Cm-244:
5805 → 35653.84, 5763 → 35361.43
'''

''' new: 10/2/2023

1: mu = 6.9621 e - 4761.15

Pu-239:
5106 → , 12%
5144 → , 17%
5156 → 30299.3, 71%

Am-241:
5443 → , 13%
5486 → 32572.0, 85%

Cm-244:
5763 → , 23%
5805 → 34766.5, 77%
-----------------------------------------
2: mu = 5.972 e

5443 → 32504
5486 → 32761
diff = 257

'''

import sys, os
sys.path.append(os.getcwd())

import numpy as np
from lmfit import Model, Parameters, fit_report
from scipy.special import erfcx

import fit_factory  # factory discovery

# 7-point Gauss–Legendre nodes/weights on [-1, 1]
_GL7_T = np.array([0.0,
                   -0.4058451513773972,  0.4058451513773972,
                   -0.7415311855993945,  0.7415311855993945,
                   -0.9491079123427585,  0.9491079123427585], dtype=float)
_GL7_W = np.array([0.4179591836734694,
                   0.3818300505051189,  0.3818300505051189,
                   0.2797053914892766,  0.2797053914892766,
                   0.1294849661688697,  0.1294849661688697], dtype=float)

from scipy.special import erfc, erfcx  # add erfc import

def _emg1(x, A, mu, sigma, tau1):
    sigma = max(float(sigma), 1e-9)
    tau1  = max(float(tau1),  1e-9)

    # z per stable EMG identity
    z = ((sigma/tau1) - ((x - mu)/sigma)) / np.sqrt(2.0)

    # Prefactor
    pref = 0.5 * A * (1.0 / tau1)

    # Two numerically-stable branches:
    # - For z >= 0: use g * erfcx(z)   (no huge numbers)
    # - For z < 0 : use exp((σ/τ)^2/2 - (x-μ)/τ) * erfc(z)  (avoids 0*inf)
    out = np.empty_like(x, dtype=float)

    # branch 1: z >= 0
    m = (z >= 0)
    if np.any(m):
        g = np.exp(-((x[m] - mu)**2) / (2.0 * sigma**2))
        out[m] = pref * g * erfcx(z[m])

    # branch 2: z < 0
    if np.any(~m):
        expfac = np.exp(0.5 * (sigma/tau1)**2 - (x[~m] - mu)/tau1)
        out[~m] = pref * expfac * erfc(z[~m])

    # guard against any lingering NaNs/Infs from extreme params
    out = np.where(np.isfinite(out), out, 0.0)
    return out


def _bin_integral(fun, x, bw, *args):
    if bw <= 0.0:
        return fun(x, *args)
    acc = 0.0
    half = 0.5 * float(bw)
    for wi, ti in zip(_GL7_W, _GL7_T):
        acc += wi * fun(x + half * ti, *args)
    return half * acc

def _emg3sum(x, A1, mu1, s1, t11, A2, mu2, s2, t12, A3, mu3, s3, t13):
    return (_emg1(x, A1, mu1, s1, t11)
          + _emg1(x, A2, mu2, s2, t12)
          + _emg1(x, A3, mu3, s3, t13))

def _emg3sum_binned(x, A1, mu1, s1, t11, A2, mu2, s2, t12, A3, mu3, s3, t13, bw):
    return _bin_integral(_emg3sum, x, bw, A1, mu1, s1, t11, A2, mu2, s2, t12, A3, mu3, s3, t13)

class AlphaEMG3Fit:
    def __init__(self, link_tau=False, param_1=1, param_2=2, param_3=10):
        """
        link_tau: if True, all peaks share the same tau (reduces 2 DoF).
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

        # ---- Parse seeds: up to 14 fields (12 seeds + bw + wmode) ----
        fp = list(fitpar) if fitpar is not None else []
        vals = fp + [np.nan]*14
        (A1_ui, mu1_ui, s1_ui, t11_ui,
         A2_ui, mu2_ui, s2_ui, t12_ui,
         A3_ui, mu3_ui, s3_ui, t13_ui,
         bw_ui, wmode_ui) = vals[:14]

        def parse_wmode(v):
            try:
                k = int(round(float(v)))
                return k if k in (0,1,2) else 2
            except Exception:
                return 2


        # Data-driven seeds
        dx     = float(np.median(np.diff(x))) if x.size > 1 else 1.0
        span   = float(x.max() - x.min()) or 1.0
        sigma0 = max(dx, span/50.0)
        tau10  = 0.5 * sigma0
        A_tot  = float(np.trapz(np.clip(y,0,None), x))

        # Require minimum separation so means don't collapse
        min_sep = max(3.0 * sigma0, 6.0 * dx)
        sig_max = 3.0 * sigma0                  # cap Gaussian widths
        t_min   = max(dx, 0.2 * sigma0)         # tails not absurdly small
        t_max   = 5.0 * sigma0                  # and not enormous either

        # Smooth & pick up to 3 separated local maxima as mu seeds
        w_bins = max(3, int(round(min_sep / dx)))
        if w_bins % 2 == 0:
            w_bins += 1
        ys = np.convolve(y, np.ones(w_bins)/w_bins, mode='same') if y.size >= w_bins else y.copy()
        left  = np.r_[True, ys[1:] > ys[:-1]]
        right = np.r_[ys[:-1] >= ys[1:], True]
        cand = np.where(left & right)[0]
        cand = cand[np.argsort(ys[cand])[::-1]]

        mu_candidates = []
        for idx in cand:
            xi = float(x[idx])
            if not mu_candidates:
                mu_candidates.append(xi)
            else:
                # only accept if far from all previous picks
                if all(abs(xi - mj) >= min_sep for mj in mu_candidates):
                    mu_candidates.append(xi)
            if len(mu_candidates) >= 3:
                break
        if len(mu_candidates) == 0:
            peak = float(x[np.argmax(y)])
            mu_candidates = [peak, peak + min_sep, peak + 2*min_sep]
        elif len(mu_candidates) == 1:
            peak = mu_candidates[0]
            mu_candidates = [peak - min_sep, peak, peak + min_sep]
        elif len(mu_candidates) == 2:
            mu_candidates.append(mu_candidates[-1] + min_sep)

        mu_candidates = sorted(mu_candidates)
        mu1_0, mu2_0, mu3_0 = mu_candidates[:3]

        # UI overrides if provided (non-zero/non-NaN)
        def pick(ui, auto): return float(ui) if np.isfinite(ui) and ui != 0 else float(auto)

        A1_0  = pick(A1_ui, A_tot/3.0);  mu1_0 = pick(mu1_ui, mu1_0);  s1_0 = max(pick(s1_ui, sigma0), dx/4.0)
        A2_0  = pick(A2_ui, A_tot/3.0);  mu2_0 = pick(mu2_ui, mu2_0);  s2_0 = max(pick(s2_ui, sigma0), dx/4.0)
        A3_0  = pick(A3_ui, A_tot/3.0);  mu3_0 = pick(mu3_ui, mu3_0);  s3_0 = max(pick(s3_ui, sigma0), dx/4.0)
        t11_0 = max(pick(t11_ui, tau10), dx/10.0)
        t12_0 = max(pick(t12_ui, tau10), dx/10.0)
        t13_0 = max(pick(t13_ui, tau10), dx/10.0)

        # bw & weighting
        bw    = float(bw_ui) if np.isfinite(bw_ui) and float(bw_ui) != 0 else dx
        wmode = parse_wmode(wmode_ui)

        # Parameters
        pars = Parameters()
        pars.add('A1',   value=max(A1_0, 1.0), min=0)
        xmin, xmax = float(x.min()), float(x.max())
        # leave room for 2 separations to the right of mu1
        pars.add('mu1',
            value=float(np.clip(mu1_0, xmin + 0.5*min_sep, xmax - 2*min_sep)),
            min=xmin + 0.5*min_sep, max=xmax - 2*min_sep)

        pars.add('s1', value=s1_0, min=dx/4.0, max=sig_max)
        # pars.add('t11',  value=t11_0, min=dx/10.0, max=10.0*sigma0)

        # Ordered means with enforced minimum separations
        d12_0 = max(mu2_0 - mu1_0, min_sep)
        d23_0 = max(mu3_0 - mu2_0, min_sep)
        step_max = max(min_sep, 0.5*span)
        pars.add('dmu12', value=d12_0, min=min_sep, max=step_max)
        pars.add('mu2', expr='mu1 + dmu12')

        pars.add('A2',   value=max(A2_0, 1.0), min=0)
        pars.add('s2', value=s2_0, min=dx/4.0, max=sig_max)
        # pars.add('t12',  value=t12_0, min=dx/10.0, max=10.0*sigma0)

        pars.add('dmu23', value=d23_0, min=min_sep, max=step_max)
        pars.add('mu3', expr='mu2 + dmu23')

        pars.add('A3',   value=max(A3_0, 1.0), min=0)
        pars.add('s3', value=s3_0, min=dx/4.0, max=sig_max)

        if self.link_tau:
            pars.add('t11', value=t11_0, min=t_min, max=t_max)
            pars.add('t12', expr='t11')
            pars.add('t13', expr='t11')
        else:
            pars.add('t11', value=t11_0, min=t_min, max=t_max)
            pars.add('t12', value=t12_0, min=t_min, max=t_max)
            pars.add('t13', value=t13_0, min=t_min, max=t_max)

        pars.add('bw', value=max(bw, 0.0), vary=False, min=0.0)

        model = Model(_emg3sum_binned, independent_vars=['x'])

        # Preflight: ensure finiteness; if not, broaden widths
        ok = False
        for _ in range(6):
            y0 = model.eval(pars, x=x)
            if np.all(np.isfinite(y0)):
                ok = True
                break
            for nm in ('s1','s2','s3'):
                pars[nm].set(value=max(pars[nm].value*2.0, dx/2.0))
            for nm in ('t11','t12','t13'):
                if nm in pars and pars[nm].vary:
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
        stats  = (f"[stats] chi-square={res.chisqr:.3f} ; reduced chi-square={res.redchi:.3f} ; "
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

class AlphaEMG3FitBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, link_tau=False, param_1=1, param_2=2, param_3=10, **_ignored):
        if not self._instance:
            self._instance = AlphaEMG3Fit(link_tau=link_tau, param_1=param_1, param_2=param_2, param_3=param_3)
        return self._instance
