#!/usr/bin/env python
# fit_alpha32_creator.py
#
# AlphaEMG32: Three-peak EMG (each peak has a two-tail mixture).
#   - Fitted: A1, (A2=ratio2*A1), (A3=ratio3*A1),
#             mu1 (free), dmu12, dmu23  -> mu2=mu1+dmu12, mu3=mu2+dmu23
#             s1,s2,s3,
#             t11,t12, (t21=t11, t22=t21+dtau2= t11+dtau1), (t31=t11, t32=t31+dtau3= t11+dtau1)
#             eta1, eta2, eta3  (constrained to (0,1))
#   - Bin-width integration (GL7), Poisson weights, optional one-step IRLS.
#
# Seeds / popup order (len-tolerant; 0-based labels shown):
#   [A1(p0), mu1(p1), s1(p2),  t11(p3), t12(p4), eta1(p5),
#    A2(p6), mu2(p7), s2(p8),  t21(p9), t22(p10), eta2(p11),
#    A3(p12), mu3(p13), s3(p14), t31(p15), t32(p16), eta3(p17),
#    bw(p18), wmode(p19)]
#
# Notes:
# - As in your AlphaEMG22, A2 and A3 seeds are interpreted as *amplitude ratios*
#   to A1 if provided (positive finite). Otherwise defaults are used (1.0).
# - If you want A2/A3 to be free (not ratio-tied), replace those expr lines
#   where A2/A3 are defined and add them as normal free parameters.

import sys, os, csv
from datetime import datetime 
sys.path.append(os.getcwd())

import numpy as np
from lmfit import Model, Parameters, fit_report
from scipy.special import erfcx, erfc

# Keep import so the factory can discover this module
import fit_factory  # noqa: F401
try:
    from PyQt5.QtWidgets import QApplication
except Exception:
    QApplication = None


# 7-point Gauss–Legendre on [-1, 1]
_GL7_T = np.array(
    [0.0,
     -0.4058451513773972,  0.4058451513773972,
     -0.7415311855993945,  0.7415311855993945,
     -0.9491079123427585,  0.9491079123427585], dtype=float)
_GL7_W = np.array(
    [0.4179591836734694,
     0.3818300505051189,  0.3818300505051189,
     0.2797053914892766,  0.2797053914892766,
     0.1294849661688697,  0.1294849661688697], dtype=float)

_INV_SQRT2 = 1.0 / np.sqrt(2.0)


def _emg_one_tail_stable(x, A, mu, sigma, tau):
    """
    Numerically stable EMG (left tail) for all x.
    Preserves the required 1/tau factor and avoids 0*inf.
    """
    x = np.asarray(x, dtype=float)
    sigma = max(float(sigma), 1e-9)
    tau   = max(float(tau),   1e-9)

    pref = 0.5 * A / tau
    inv_sigma = 1.0 / sigma

    # Two equivalent forms with different stability regions
    # u >= 0 : use g * erfcx(u)
    # u <  0 : use exp( (σ/τ)^2/2 - (x-μ)/τ ) * erfc(u)
    u = (_INV_SQRT2) * ((sigma / tau) + ((x - mu) * inv_sigma))

    out = np.empty_like(x)

    m = (u >= 0.0)
    if np.any(m):
        g = np.exp(-0.5 * ((x[m] - mu) * inv_sigma)**2)
        out[m] = pref * g * erfcx(u[m])

    if np.any(~m):
        expfac = np.exp(0.5 * (sigma / tau)**2 + (x[~m] - mu) / tau)
        out[~m] = pref * expfac * erfc(u[~m])

    # Replace any residual non-finites by 0 (far-out tails)
    out = np.where(np.isfinite(out), out, 0.0)
    return out


def _emg_two_tail_stable(x, A, mu, sigma, tau_fast, tau_slow, eta):
    eta = float(np.clip(eta, 0.0, 1.0))
    return ((1.0 - eta) * _emg_one_tail_stable(x, A, mu, sigma, tau_fast)
          + (      eta) * _emg_one_tail_stable(x, A, mu, sigma, tau_slow))


def _bin_integral(fun, x, bw, *args):
    bw = float(bw)
    if bw <= 0.0:
        return fun(x, *args)
    x = np.asarray(x, dtype=float)
    half = 0.5 * bw
    acc = np.zeros_like(x, dtype=float)
    for wi, ti in zip(_GL7_W, _GL7_T):
        acc += wi * fun(x + half * ti, *args)
    return half * acc


def _sum3(x,
          A1, mu1, s1, t11, t12, eta1,
          A2, mu2, s2, t21, t22, eta2,
          A3, mu3, s3, t31, t32, eta3):
    return (_emg_two_tail_stable(x, A1, mu1, s1, t11, t12, eta1)
          + _emg_two_tail_stable(x, A2, mu2, s2, t21, t22, eta2)
          + _emg_two_tail_stable(x, A3, mu3, s3, t31, t32, eta3)
          )

def _sum3_binned_ratios(x,
                        A1, ratio2, ratio3,
                        mu1, sigma1, tau11, tau12, eta1,
                        mu2, sigma2, tau21, tau22, eta2,
                        mu3, sigma3, tau31, tau32, eta3,
                        bw):
    # derive A2/A3 from A1 via fixed ratios
    A2 = ratio2 * A1
    A3 = ratio3 * A1
    return _bin_integral(
        _sum3, x, bw,
        A1, mu1, sigma1, tau11, tau12, eta1,
        A2,  mu2, sigma2, tau21, tau22, eta2,
        A3,  mu3, sigma3, tau31, tau32, eta3
    )

'''
Pu-239:
5106 → 30258, 12%
5144 → 30520, 17%
5156 → 30602, 71%
'''

def _sum3_binned(x,
                 A1, mu1, sigma1, tau11, tau12, eta1,
                 A2, mu2, sigma2, tau21, tau22, eta2,
                 A3, mu3, sigma3, tau31, tau32, eta3,
                 bw):
    # Map sigma* -> s*, tau** keep same meaning
    return _bin_integral(
        _sum3, x, bw,
        A1, mu1, sigma1, tau11, tau12, eta1,
        A2, mu2, sigma2, tau21, tau22, eta2,
        A3, mu3, sigma3, tau31, tau32, eta3
    )


def _as_float(x):
    """Return float(x) or np.nan if x is None/blank/non-numeric."""
    try:
        if x is None:
            return np.nan
        if isinstance(x, str):
            s = x.strip()
            if not s or s.lower() in {"none", "nan"}:
                return np.nan
            return float(s)
        return float(x)
    except Exception:
        return np.nan

def pick(ui, auto, *, zero_means_blank=True):
    f = _as_float(ui)
    if not np.isfinite(f):
        return float(auto)
    if zero_means_blank and f == 0.0:
        return float(auto)
    return float(f)

def parse_wmode(v, default=2):
    f = _as_float(v)
    if not np.isfinite(f):
        return int(default)
    k = int(round(f))
    return k if k in (0, 1, 2) else int(default)

def _inside01(v):
    f = _as_float(v)
    if not np.isfinite(f):
        f = 0.5
    return float(np.clip(f, 0.0, 1.0))


class AlphaEMG32Fit:
    def __init__(self, param_1=1, param_2=2, param_3=10):
        self.param_1 = param_1
        self.param_2 = param_2
        self.param_3 = param_3

    def start(self, x, y, xmin, xmax, fitpar, axis, fit_results):
        # Finite data only
        mfin = np.isfinite(x) & np.isfinite(y)
        x = np.asarray(x)[mfin]
        y = np.asarray(y)[mfin]

        if x.size < 7:
            fit_results.setPlainText("Not enough data to fit.")
            return None

        # ---- Parse seeds (len-tolerant) ----
        # Expect up to 20 values:
        # [A1, mu1, s1, t11, t12, eta1,
        #  A2, mu2, s2, t21, t22, eta2,
        #  A3, mu3, s3, t31, t32, eta3,
        #  bw, wmode]
        fp = list(fitpar) if fitpar is not None else []
        vals = fp + [np.nan] * 20
        (A1_ui, mu1_ui, s1_ui, t11_ui, t12_ui, eta1_ui,
         A2_ui, mu2_ui, s2_ui, t21_ui, t22_ui, eta2_ui,
         A3_ui, mu3_ui, s3_ui, t31_ui, t32_ui, eta3_ui,
         bw_ui, wmode_ui) = vals[:20]

        # Data-driven auto seeds
        dx     = float(np.median(np.diff(x))) if x.size > 1 else 1.0
        span   = float(x.max() - x.min()) or 1.0
        sigma0 = max(dx, span / 50.0)
        tau0   = 0.5 * sigma0
        A_tot  = float(np.trapz(np.clip(y, 0, None), x))

        idx_max = int(np.argmax(y))
        mu_auto = float(x[idx_max])

        # Seeds (UI wins if provided)
        A1_0  = pick(A1_ui, A_tot / 3.0)   # <- always define A1_0 first
        mu1_0 = pick(mu1_ui, mu_auto)

        # If the user effectively didn't provide A1, refresh with a safe default
        A1_val = _as_float(A1_ui)
        if not np.isfinite(A1_val) or A1_val == 0.0:
            A1_0 = max(A_tot / 3.0, 1.0)


        mu1_0 = pick(mu1_ui, mu_auto)

        # provisional μ2, μ3 estimates to size the window
        step_est = max(6.0 * dx, 3.0 * sigma0)
        mu2_est = mu1_0 + step_est
        mu3_est = mu2_est + step_est

        # Window
        if np.isfinite(xmin) and np.isfinite(xmax):
            m = (x >= xmin) & (x <= xmax)
        else:
            W = 6.0 * max(sigma0, tau0)
            L = mu1_0  - W
            R = mu3_est + W
            m = (x >= L) & (x <= R)
        x, y = x[m], y[m]

        # Refresh basics post-window
        dx     = float(np.median(np.diff(x))) if x.size > 1 else dx
        span   = float(x.max() - x.min()) or span
        sigma0 = max(dx, span / 50.0)
        tau0   = 0.5 * sigma0
        A_tot  = float(np.trapz(np.clip(y, 0, None), x))

        '''
        # Adjust A1 if not provided
        if not np.isfinite(A1_ui) or float(A1_ui) == 0.0:
            A1_0 = max(A_tot / 3.0, 1.0)
        '''
        # Peak-1
        s1_0   = max(pick(s1_ui, sigma0), dx / 4.0)
        t11_0  = max(pick(t11_ui, tau0), dx / 10.0)
        t12_0  = max(pick(t12_ui, t11_0 + max(dx / 10.0, 0.05 * sigma0)), dx / 10.0)
        eta1_0 = np.clip(pick(eta1_ui, 0.0), 0.0, 1.0)

        # Peak-2
        mu2_0  = pick(mu2_ui, mu1_0 + step_est)
        s2_0   = max(pick(s2_ui, sigma0), dx / 4.0)
        t21_0  = max(pick(t21_ui, tau0), dx / 10.0)
        t22_0  = max(pick(t22_ui, t21_0 + max(dx / 10.0, 0.05 * sigma0)), dx / 10.0)
        eta2_0 = np.clip(pick(eta2_ui, 0), 0.0, 1.0)

        # Peak-3
        mu3_0  = pick(mu3_ui, mu2_0 + step_est)
        s3_0   = max(pick(s3_ui, sigma0), dx / 4.0)
        t31_0  = max(pick(t31_ui, tau0), dx / 10.0)
        t32_0  = max(pick(t32_ui, t31_0 + max(dx / 10.0, 0.05 * sigma0)), dx / 10.0)
        eta3_0 = np.clip(pick(eta3_ui, 0.0), 0.0, 1.0)

        # bw    = float(bw_ui) if np.isfinite(bw_ui) and float(bw_ui) != 0 else dx

        _bw = _as_float(bw_ui)
        bw  = float(_bw) if np.isfinite(_bw) and _bw != 0.0 else dx
        wmode = parse_wmode(wmode_ui)

        # ----------------
        # Build parameters
        # ----------------
        pars = Parameters()

        # Amplitudes: A2, A3 are *not* Params; use fixed ratios to A1

        pars.add('A1', value=max(A1_0, 1.0), min=0)

        A2_val = _as_float(A2_ui)
        A3_val = _as_float(A3_ui)

        use_ratio2 = np.isfinite(A2_val) and (A2_val > 0.0)
        use_ratio3 = np.isfinite(A3_val) and (A3_val > 0.0)

        if use_ratio2:
            pars.add('ratio2', value=float(A2_val), vary=False)
            pars.add('A2', expr='ratio2*A1')
        else:
            A2_0 = pick(A2_ui, A_tot / 3.0)
            pars.add('A2', value=max(A2_0, 1.0), min=0)

        if use_ratio3:
            pars.add('ratio3', value=float(A3_val), vary=False)
            pars.add('A3', expr='ratio3*A1')
        else:
            A3_0 = pick(A3_ui, A_tot / 3.0)
            pars.add('A3', value=max(A3_0, 1.0), min=0)



        # Centers: mu2 = mu1 + dmu12 ; mu3 = mu2 + dmu23
        mu1_val = pick(mu1_ui, mu1_0, zero_means_blank=False)
        mu2_val = pick(mu2_ui, mu1_0 + step_est, zero_means_blank=False)
        mu3_val = pick(mu3_ui, mu2_val + step_est, zero_means_blank=False)

        dmu12_0 = mu2_val - mu1_val
        dmu23_0 = mu3_val - mu2_val

        pars.add('mu1', value=float(mu1_0))
        sep_max = 7.0
        pars.add('dmu12', value=dmu12_0, min=dmu12_0 - sep_max, max=dmu12_0 + sep_max)
        pars.add('dmu23', value=dmu23_0, min=dmu23_0 - sep_max, max=dmu23_0 + sep_max)
        pars.add('mu2', expr='mu1 + dmu12')
        pars.add('mu3', expr='mu2 + dmu23')


        mu_wiggle = 2000.0
        pars['mu1'].set(vary=True, min=mu1_0 - mu_wiggle, max=mu1_0 + mu_wiggle)
        # mu2/mu3 are derived; no explicit bounds on them

        # Sigmas
        sig_cap = 10.0 * sigma0
        pars.add('sigma1', value=s1_0, min=max(dx/10.0, 1e-6), max=sig_cap)
        '''
        pars.add('s2', value=s2_0, min=max(dx/10.0, 1e-6), max=sig_cap)
        pars.add('s3', value=s3_0, min=max(dx/10.0, 1e-6), max=sig_cap)
        '''
        wiggle_s = 1.0  
        pars.add('ds12', min=-wiggle_s, max=+wiggle_s)
        pars.add('sigma2', expr='sigma1 + ds12')
        pars.add('ds23', min=-wiggle_s, max=+wiggle_s)
        pars.add('sigma3', expr='sigma2 + ds23')

        
        # Shared multipliers (fast tail and slow gap)
        pars.add('ct',  value=0.5,  min=0.1, max=1.0)   # τ_fast ≈ 0.1–1.0 of σ
        pars.add('cdt', value=0.15, min=0.05, max=0.5)  # (τ_slow - τ_fast) ≈ 0.05–0.5 of σ

        # Peak-specific τ via σ
        pars.add('tau11',   expr='ct*sigma1')
        pars.add('dtau1', expr='cdt*sigma1')
        pars.add('tau12',   expr='tau11 + dtau1')

        # pars.add('tau21',   expr='ct*sigma2')
        # pars.add('dtau2', expr='cdt*sigma2')
        # pars.add('tau22',   expr='tau21 + dtau2')
        wiggle_t = 1.0

        pars.add('dt11', min=-wiggle_t, max=+wiggle_t)
        pars.add('tau21', expr='tau11 + dt11')

        pars.add('dt12', min=-wiggle_t, max=+wiggle_t)
        pars.add('tau22', expr='tau12 + dt12')

        # pars.add('tau31',   expr='ct*sigma3')
        # pars.add('dtau3', expr='cdt*sigma3')
        # pars.add('tau32',   expr='tau31 + dtau3')
        pars.add('dt21', min=-wiggle_t, max=+wiggle_t)
        pars.add('tau31', expr='tau21 + dt21')
        pars.add('dt22', min=-wiggle_t, max=+wiggle_t)
        pars.add('tau32', expr='tau22 + dt22')

        '''
        # per-peak independent taus
        # Peak 1
        pars.add('tau11', value=t11_0, min=t_min, max=t_max)
        pars.add('tau12', value=t12_0, min=max(t11_0 + floor, t_min), max=t_max)

        # Peak 2
        pars.add('tau21', value=t21_0, min=t_min, max=t_max)
        pars.add('tau22', value=t22_0, min=max(t21_0 + floor, t_min), max=t_max)

        # Peak 3
        pars.add('tau31', value=t31_0, min=t_min, max=t_max)
        pars.add('tau32', value=t32_0, min=max(t31_0 + floor, t_min), max=t_max)
        ''' 


        '''
        pars.add('eta1', value=_inside01(eta1_0), min=1e-6, max=1-1e-6, vary=True)
        pars.add('eta2', value=_inside01(eta2_0), min=1e-6, max=1-1e-6, vary=True)
        pars.add('eta3', value=_inside01(eta3_0), min=1e-6, max=1-1e-6, vary=True)
        '''
        pars.add('eta1', value=_inside01(eta1_ui), vary=False)
        pars.add('eta2', value=_inside01(eta2_ui), vary=False)
        pars.add('eta3', value=_inside01(eta3_ui), vary=False)

        # Fixed bin width
        pars.add('bw', value=max(bw, 0.0), vary=False, min=0.0)

        # Model
        # model = Model(_sum3_binned_ratios, independent_vars=['x'])
        model = Model(_sum3_binned, independent_vars=['x'])


        # Preflight: ensure finite; broaden if needed
        for _ in range(6):
            y0 = model.eval(pars, x=x)
            if np.all(np.isfinite(y0)):
                break
            for nm in ('sigma1', 'sigma2', 'sigma3'):
                pars[nm].set(value=max(pars[nm].value * 2.0, dx/2.0))
            pars['tau11'].set(value=max(pars['tau11'].value * 2.0, dx/5.0))
            pars['dtau1'].set(value=max(pars['dtau1'].value * 2.0, dx/5.0))
        else:
            fit_results.setPlainText("Model non-finite at initial guess; aborting.")
            return None

        # Weights
        def w_from_data(y_arr):
            return 1.0 / np.sqrt(np.clip(y_arr, 1.0, None))
        if wmode == 0:
            weights = None
        elif wmode == 1:
            weights = w_from_data(y)
        else:
            weights = w_from_data(y)

        fit_kws = dict(loss='soft_l1', f_scale=1.0)  # for method='least_squares'

        # ---- Bashir's Abort callback ----
        _should_abort = getattr(self, "_should_abort", None)

        def _iter_cb(params=None, iter=None, resid=None, *args, **kwargs):
            # let the GUI process clicks so Abort can be pressed
            if QApplication is not None:
                QApplication.processEvents()
            # returning True tells lmfit to stop
            return bool(_should_abort and _should_abort())
        ###########################################################

        # First fit
        res = model.fit(y, params=pars, x=x, method='least_squares',
                        weights=weights, fit_kws=fit_kws, max_nfev=2000,
                        iter_cb=_iter_cb # Bashir added for aborting fit
                        )


        # for nm in ('eta1','eta2','eta3'):
        #     if res.params[nm].value < 0.02:
        #         p = res.params.copy()
        #         p['cdt'].set(value=max(p['cdt'].min, 0.05), vary=False)  # effectively one-tail-ish
        #         res = model.fit(y, params=p, x=x, method='least_squares', weights=weights, fit_kws=fit_kws)
        #         break


        print(f"[AlphaEMG32] mu1={res.params['mu1'].value:.2f}, "
              f"mu2={res.params['mu2'].value:.2f}, "
              f"mu3={res.params['mu3'].value:.2f}, "
              f"sigma1={res.params['sigma1'].value:.2g}, sigma2={res.params['sigma2'].value:.2g}, sigma3={res.params['sigma3'].value:.2g}, "
              f"tau11={res.params['tau11'].value:.2g}")

        # Optional one-step IRLS
        if wmode == 2:
            try:
                for _ in range(6):
                    yhat = model.eval(res.params, x=x)
                    w = 1.0 / np.sqrt(np.clip(yhat, 1.0, None))
                    new = model.fit(y, params=res.params.copy(), x=x,
                                    method='least_squares', weights=w, fit_kws=fit_kws, max_nfev=1500,
                                    iter_cb=_iter_cb # Bashir added for aborting fit
                                    )
                    if abs(new.chisqr - res.chisqr)/max(res.chisqr,1) < 1e-3:
                        res = new; break
                    res = new
            except Exception:
                pass

        R2_plain = getattr(res, "rsquared", None)
        if R2_plain is None:
            yhat = model.eval(res.params, x=x)
            ss_res = float(np.sum((y - yhat)**2))
            ss_tot = float(np.sum((y - y.mean())**2))
            R2_plain = 1.0 - ss_res / (ss_tot + 1e-16)

        if _iter_cb():
            return None

        # Writing to a CSV file
        # --- Save CSV (main values only) --------------------------------------
        p = res.params  # shorthand

        def _get(name):
            try:
                return float(p[name].value)
            except Exception:
                return np.nan

        redchi = float(getattr(res, "redchi", np.nan)) 
        timestamp = datetime.now().isoformat(timespec="seconds") # e.g. '2024-06-30T14:23:05'
        fieldnames = [
            "timestamp",
            "chi-square",
            "R2",
            "A1","mu1","sigma1","tau11","tau12","eta1",
            "A2","mu2","sigma2","tau21","tau22","eta2",
            "A3","mu3","sigma3","tau31","tau32","eta3",
        ]
        row = {
            "timestamp": timestamp,
            "chi-square": redchi,
            "R2": R2_plain,
            "A1": _get("A1"),        "mu1": _get("mu1"),        "sigma1": _get("sigma1"),
            "tau11": _get("tau11"),    "tau12": _get("tau12"),      "eta1": _get("eta1"),
            "A2": _get("A2"),        "mu2": _get("mu2"),        "sigma2": _get("sigma2"),
            "tau21": _get("tau21"),    "tau22": _get("tau22"),      "eta2": _get("eta2"),
            "A3": _get("A3"),        "mu3": _get("mu3"),        "sigma3": _get("sigma3"),
            "tau31": _get("tau31"),    "tau32": _get("tau32"),      "eta3": _get("eta3"),
        }

        csv_path = os.path.join(os.getcwd(), "fit_EMG32_results.csv")
        write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)

        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow(row)
        # ----------------------------------------------------------------------

        

        # Report
        wtxt = {0: "none", 1: "Poisson(data)", 2: "Poisson(model, IRLS)"}\
               .get(wmode, "Poisson(data)")
        header = f"Notes: bandwidth = {res.params['bw'].value:.6g} ; weighting = {wtxt}"
        stats  = (f"[stats] chi-square={res.chisqr:.3f} ; reduced chi-square={res.redchi:.3f} ; "
                  f"dof={res.nfree} (N={res.ndata}, k={res.nvarys})")

        def _fmt_with_pct(params):
            """
            Lines like:
                name = value ± stderr   [cv=..%, var%=..%]
            Fixed or missing stderr handled gracefully.
            """
            lines = []
            for name, par in params.items():
                if name == 'bw':
                    continue
                val = par.value
                err = par.stderr

                if not par.vary:
                    lines.append(f"{name:>8} = {val:.6g}  (fixed)")
                    continue

                if err is None or not np.isfinite(err):
                    lines.append(f"{name:>8} = {val:.6g}  ± n/a   [cv=n/a, var%=n/a]")
                    continue

                if val != 0 and np.isfinite(val):
                    cv   = abs(err/val) * 100.0
                    pvar = (err/val)**2 * 100.0
                    lines.append(f"{name:>8} = {val:.6g}  ± {err:.3g}   [cv={cv:.2f}%, var%={pvar:.2f}%]")
                else:
                    lines.append(f"{name:>8} = {val:.6g}  ± {err:.3g}   [cv=n/a, var%=n/a]")
            return "\n".join(lines)

        try:
            named = _fmt_with_pct(res.params)
            fit_results.setPlainText(
                header + "\n" + stats + "\n\n"
                + "Parameters (named + %CV, %variance from 1σ):\n" + named + "\n\n"
                + fit_report(res, show_correl=True)
            )
        except Exception:
            fit_results.setPlainText(str(res))

        # Draw fitted curve
        '''
        xx = np.linspace(x.min(), x.max(), 1600)
        yy = model.eval(res.params, x=xx)
        (fitln,) = axis.plot(xx, yy, lw=2)
        return fitln
        '''
        # Draw fitted curve + components
        xx = np.linspace(x.min(), x.max(), 1600)
        p  = res.params

        # Amplitudes from A1 and fixed ratios
        A1 = p['A1'].value
        A2 = p['A2'].value
        A3 = p['A3'].value
        bw = p['bw'].value


        # Individual peak contributions (with the same bin integration)
        y1 = _bin_integral(_emg_two_tail_stable, xx, bw,
                        A1, p['mu1'].value, p['sigma1'].value, p['tau11'].value, p['tau12'].value, p['eta1'].value)
        y2 = _bin_integral(_emg_two_tail_stable, xx, bw,
                        A2, p['mu2'].value, p['sigma2'].value, p['tau21'].value, p['tau22'].value, p['eta2'].value)
        y3 = _bin_integral(_emg_two_tail_stable, xx, bw,
                        A3, p['mu3'].value, p['sigma3'].value, p['tau31'].value, p['tau32'].value, p['eta3'].value)
        ytot = y1 + y2 + y3

        # Plot total + components
        (line_tot,) = axis.plot(xx, ytot, lw=2, label='fit')
        axis.plot(xx, y1, '--', lw=3, label='peak 1')
        axis.plot(xx, y2, '--', lw=2.5, label='peak 2')
        axis.plot(xx, y3, '--', lw=2.5, label='peak 3')
        axis.legend()

        # Safe ratios for inspection/UI, regardless of whether ratio params existed
        ratio2 = (p['A2'].value / max(p['A1'].value, 1e-12))
        ratio3 = (p['A3'].value / max(p['A1'].value, 1e-12))
        line_tot._ratio2 = ratio2
        line_tot._ratio3 = ratio3


        return line_tot


class AlphaEMG32FitBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, param_1=1, param_2=2, param_3=10, **_ignored):
        if not self._instance:
            self._instance = AlphaEMG32Fit(param_1=param_1, param_2=param_2, param_3=param_3)
        return self._instance
