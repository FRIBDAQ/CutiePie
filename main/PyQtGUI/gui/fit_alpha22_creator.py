#!/usr/bin/env python
# fit_alpha22_creator.py
#
# AlphaEMG22: Two-peak EMG (each peak has a two-tail mixture).
#   - User FIXED: mu1, mu2, eta1, eta2
#   - Fitted: A1, A2, s1, s2, tau11, tau12, tau21, tau22
#   - Bin-width integration (GL7), Poisson weights, optional one-step IRLS.
#
# Seeds / popup order (len-tolerant; 0-based labels shown):
#   [A1(p0), mu1(p1), s1(p2), tau11(p3), tau12(p4), eta1(p5),
#    A2(p6), mu2(p7), s2(p8), tau21(p9), tau22(p10), eta2(p11),
#    bw(p12), wmode(p13)]
# You can omit tau12/tau22; they’ll auto-seed > tau11/tau21 via dtau guards.
#
# wmode: 0=unweighted, 1=Poisson(data), 2=Poisson(model, 1-step IRLS)

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

def _peak_binned(x, A, mu, s, t1, t2, eta, bw):
    """Single EMG (two-tail) with the same GL7 bin integration used by the total."""
    return _bin_integral(_emg_two_tail_stable, x, bw, A, mu, s, t1, t2, eta)


def _emg_two_tail(x, A, mu, sigma, tau_fast, tau_slow, eta):
    """
    Single EMG peak with two-tail mixture; left-tailed; erfcx-stable.
    eta in [0,1] is FIXED per peak by the GUI/user (we still pass it as a param with vary=False).
    """
    x = np.asarray(x, dtype=float)
    sigma    = max(float(sigma),    1e-9)
    tau_fast = max(float(tau_fast), 1e-9)
    tau_slow = max(float(tau_slow), 1e-9)
    eta      = float(np.clip(eta, 0.0, 1.0))

    inv_sigma = 1.0 / sigma
    dx = (x - mu) * inv_sigma
    g  = np.exp(-0.5 * dx * dx)

    zf = (dx + sigma / tau_fast) * _INV_SQRT2
    zs = (dx + sigma / tau_slow) * _INV_SQRT2

    tail = (1.0 - eta) * erfcx(zf) / tau_fast + eta * erfcx(zs) / tau_slow
    return 0.5 * A * g * tail

    # erfcx(z) = exp(z^2) * erfc(z) is the scaled complementary error function

################################## Stable two-tail ###################################################
def _emg_one_tail_stable(x, A, mu, sigma, tau):
    """
    Numerically stable EMG (left tail) for all x.
    This preserves the required 1/tau factor and avoids 0*inf.
    """
    x = np.asarray(x, dtype=float)
    sigma = max(float(sigma), 1e-9)
    tau   = max(float(tau),   1e-9)

    # Prefactors
    pref = 0.5 * A / tau
    inv_sigma = 1.0 / sigma

    # Two equivalent forms with different stability regions
    # z >= 0 : use g * erfcx(z)
    # z <  0 : use exp( (σ/τ)^2/2 - (x-μ)/τ ) * erfc(u)
    # with u = ( (σ/τ) - (x-μ)/σ ) / sqrt(2)
    u = (_INV_SQRT2) * ((sigma / tau) - ((x - mu) * inv_sigma))

    out = np.empty_like(x)

    m = (u >= 0.0)
    if np.any(m):
        g = np.exp(-0.5 * ((x[m] - mu) * inv_sigma)**2)
        out[m] = pref * g * erfcx(u[m])

    if np.any(~m):
        expfac = np.exp(0.5 * (sigma / tau)**2 - (x[~m] - mu) / tau)
        out[~m] = pref * expfac * erfc(u[~m])

    # Replace any residual non-finites by 0 (far-out tails)
    out = np.where(np.isfinite(out), out, 0.0)
    return out

def _emg_two_tail_stable(x, A, mu, sigma, tau_fast, tau_slow, eta):
    eta = float(np.clip(eta, 0.0, 1.0))
    return ((1.0 - eta) * _emg_one_tail_stable(x, A, mu, sigma, tau_fast)
          + (      eta) * _emg_one_tail_stable(x, A, mu, sigma, tau_slow))
#####################################################################################

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


def _sum2(x,
          A1, mu1, s1, t11, t12, eta1,
          A2, mu2, s2, t21, t22, eta2):
        
    
    return (_emg_two_tail_stable(x, A1, mu1, s1, t11, t12, eta1)
          + _emg_two_tail_stable(x, A2, mu2, s2, t21, t22, eta2)
          )



def _sum2_binned(x,
                 A1, mu1, sigma1, tau11, tau12, eta1,
                 A2, mu2, sigma2, tau21, tau22, eta2,
                 bw):
    return _bin_integral(
                        _sum2, x, bw,
                        A1, mu1, # 0, 1
                        sigma1, tau11, # 2, 3
                        tau12, eta1, # 4, 5
                        A2, mu2, # 6, 7
                        sigma2, tau21, # 8, 9
                        tau22, eta2 # 10, 11
                        )


class AlphaEMG22Fit:
    def __init__(self, param_1=1, param_2=2, param_3=10):
        self.param_1 = param_1
        self.param_2 = param_2
        self.param_3 = param_3

    def start(self, x, y, xmin, xmax, fitpar, axis, fit_results):
        # Finite data only
        mfin = np.isfinite(x) & np.isfinite(y)
        x = np.asarray(x)[mfin]
        y = np.asarray(y)[mfin]

        if x.size < 5:
            fit_results.setPlainText("Not enough data to fit.")
            return None

        # ---- Parse seeds (len-tolerant) ----
        # Expect up to 14 values:
        # [A1, mu1, s1, t11, t12, eta1,  A2, mu2, s2, t21, t22, eta2,  bw, wmode]
        fp = list(fitpar) if fitpar is not None else []
        vals = fp + [np.nan] * 14
        (A1_ui, mu1_ui, s1_ui, t11_ui, t12_ui, eta1_ui,
         A2_ui, mu2_ui, s2_ui, t21_ui, t22_ui, eta2_ui,
         bw_ui, wmode_ui) = vals[:14]

        def pick(ui, auto):
            try:
                v = float(ui)
                return v if np.isfinite(v) and v != 0.0 else float(auto)
            except Exception:
                return float(auto)

        def parse_wmode(v):
            try:
                k = int(round(float(v)))
                return k if k in (0, 1, 2) else 2
            except Exception:
                return 2

        # Data-driven auto seeds
        dx     = float(np.median(np.diff(x))) if x.size > 1 else 1.0
        span   = float(x.max() - x.min()) or 1.0
        sigma0 = max(dx, span / 80.0)
        tau0   = 0.5 * sigma0
        A_tot  = float(np.trapz(np.clip(y, 0, None), x))

        # Initial guesses if μ not given (but we will freeze μ later)
        idx_max = int(np.argmax(y))
        mu_auto = float(x[idx_max])
        # Seeds (UI wins if provided)
        A1_0  = pick(A1_ui, A_tot / 2.0)
        mu1_0 = pick(mu1_ui, mu_auto)

        # provisional μ2 estimate to size the window
        mu2_est = mu1_0 + max(6.0*dx, 3.0*sigma0)

        if np.isfinite(xmin) and np.isfinite(xmax):
            m = (x >= xmin) & (x <= xmax)
        else:
            W = 6.0 * max(sigma0, tau0)
            L = mu1_0  - W
            R = mu2_est + W   # <-- use mu2_est here
            m = (x >= L) & (x <= R)
        x, y = x[m], y[m]


        dx     = float(np.median(np.diff(x))) if x.size > 1 else dx
        span   = float(x.max() - x.min()) or span
        sigma0 = max(dx, span / 50.0)
        tau0   = 0.5 * sigma0
        A_tot  = float(np.trapz(np.clip(y, 0, None), x))

        # if A1_ui was not provided, refresh A1_0 from windowed area
        if not np.isfinite(A1_ui) or float(A1_ui) == 0.0:
            A1_0 = max(A_tot / 2.0, 1.0)

        s1_0  = max(pick(s1_ui, sigma0), dx / 4.0)
        t11_0 = max(pick(t11_ui, tau0), dx / 10.0)
        t12_0 = max(pick(t12_ui, t11_0 + max(dx / 10.0, 0.05 * sigma0)), dx / 10.0)
        eta1_0 = np.clip(pick(eta1_ui, 0.0), 0.0, 1.0)

        # A2_0  = pick(A2_ui, A_tot / 2.0)
        mu2_0 = pick(mu2_ui, mu1_0 + max(6.0 * dx, 3.0 * sigma0))
        s2_0  = max(pick(s2_ui, sigma0), dx / 4.0)
        t21_0 = max(pick(t21_ui, tau0), dx / 10.0)
        t22_0 = max(pick(t22_ui, t21_0 + max(dx / 10.0, 0.05 * sigma0)), dx / 10.0)
        eta2_0 = np.clip(pick(eta2_ui, 0.0), 0.0, 1.0)

        bw    = float(bw_ui) if np.isfinite(bw_ui) and float(bw_ui) != 0 else dx
        wmode = parse_wmode(wmode_ui)



        # ----------------
        # Build parameters
        # ----------------
        pars = Parameters()
        pars.add('A1', value=max(A1_0, 1.0), min=0)
        # default_ratio = 848.0 / 131.0
        default_ratio = 85.0 / 13.0
        ratio_ui = float(A2_ui) if np.isfinite(A2_ui) else np.nan
        use_ratio = np.isfinite(ratio_ui) and (ratio_ui > 0.0)
        ratio = ratio_ui if use_ratio else default_ratio
        pars.add('ratio', value=ratio, vary=False)
        pars.add('A2', expr='ratio*A1')

        # centers: keep mu2 as expression; vary mu1 and dmu only
        pars.add('mu1', value=float(mu1_0))
        sep_min = max(dx, 0.5*sigma0)
        sep_max = 7.0          # pick something generous for your setup
        # dmu0    = np.clip(mu2_0 - mu1_0, -sep_max, sep_max)
        # pars.add('dmu', value=dmu0, min=sep_min, max=sep_max)
        pars.add('dmu', value=mu2_ui-mu1_ui, min=mu2_ui-mu1_ui-sep_max, max=mu2_ui-mu1_ui+sep_max)
        pars.add('mu2', expr='mu1 + dmu')

        mu_wiggle = 2000.0
        pars['mu1'].set(vary=True, min=mu1_0 - mu_wiggle, max=mu1_0 + mu_wiggle)

        # DO NOT set bounds on mu2; it's derived from mu1,dmu
        # (remove the line that called pars['mu2'].set(...))

        
        # σ’s
        '''
        sig_cap = 10.0*sigma0
        pars.add('sigma1', value=s1_0, min=max(dx/10.0, 1e-6), max=sig_cap)
        pars.add('sigma2', value=s2_0, min=max(dx/10.0, 1e-6), max=sig_cap)
        '''

        # σ’s — make sigma2 ~ sigma1 with a small wiggle
        sig_cap = 8.0 * sigma0
        pars.add('sigma1', value=s1_0, min=max(dx/10.0, 1e-6), max=sig_cap)

        # sigma2 = sigma1 + ds, with tight bounds so it stays close to sigma1
        wiggle_s = 1.0          
        # ds0 = float(np.clip((s2_0 - s1_0), -wiggle, +wiggle))
        pars.add('ds', min=-wiggle_s, max=+wiggle_s)
        pars.add('sigma2', expr='sigma1 + ds')


        # Per-peak fast tails
        t_min = max(dx/10.0, 1e-6)
        t_max = 10.0*sigma0
        floor = max(dx/20.0, 0.02*sigma0)

        '''
        pars.add('tau11',  value=t11_0, min=t_min, max=t_max)
        pars.add('dtau1', value=max(t12_0 - t11_0, floor), min=floor, max=50.0*sigma0)
        pars.add('tau12', expr='tau11 + dtau1')

        wiggle_t = 20.0
        pars.add('dt1', min=-wiggle_t, max=+wiggle_t)
        pars.add('tau21', expr='tau11 + dt1')
        pars.add('dt2', min=-wiggle_t, max=+wiggle_t)
        pars.add('tau22',  expr='tau12 + dt2')
        '''

        # Peak 1
        pars.add('ct',  value=0.35,  min=0.2, max=0.8)   # t_fast ≈ 0.1–1.0 of σ
        pars.add('cdt', value=0.18, min=0.1, max=0.35)  # delta_slow ≈ 0.05–0.5 of σ
        pars.add('tau11',  expr='ct*sigma1')
        # pars.add('tau11',  value=t11_0, min=t_min, max=t_max)
        pars.add('dtau1',expr='cdt*sigma1')
        pars.add('tau12',  expr='tau11 + dtau1')

        # Peak 2 (share same ratios so both peaks move coherently)
        wiggle_t = 1.0
        pars.add('dt1', min=-wiggle_t, max=+wiggle_t)
        pars.add('tau21', expr='tau11 + dt1')
        pars.add('dt2', min=-wiggle_t, max=+wiggle_t)
        pars.add('tau22',  expr='tau12 + dt2')


        # η’s:
        
        # FIXED (user-provided)
        def _inside01(v): 
            v = 0.5 if (not np.isfinite(v)) else v
            return float(np.clip(v, 0.0, 1.0))

        '''
        pars.add('eta1', value=_inside01(eta1_0), min=0.0, max=1.0, vary=True)
        pars.add('eta2', value=_inside01(eta2_0), min=0.0, max=1.0, vary=True)
        '''

        
        eta1_val = _inside01(eta1_ui)
        eta2_val = _inside01(eta2_ui)

        pars.add('eta1', value=eta1_val, vary=False)
        pars.add('eta2', value=eta2_val, vary=False)
        

        # Fixed bin width
        pars.add('bw', value=max(bw, 0.0), vary=False, min=0.0)


        # Model
        model = Model(_sum2_binned, independent_vars=['x'])

        # Preflight: ensure finite; if not, broaden widths/tails a bit
        for _ in range(6):
            y0 = model.eval(pars, x=x)
            if np.all(np.isfinite(y0)):
                break
            # widen only free, non-expr params
            for nm in ('sigma1', 'sigma2'):
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

        fit_kws = dict(loss='soft_l1', f_scale=1.0)  # only works with method='least_squares'

        # ---- Bashir's Abort callback ----
        _should_abort = getattr(self, "_should_abort", None)

        def _iter_cb(params=None, iter=None, resid=None, *args, **kwargs):
            # let the GUI process clicks so Abort can be pressed
            if QApplication is not None:
                QApplication.processEvents()
            # returning True tells lmfit to stop
            return bool(_should_abort and _should_abort())
        #########################################################

        # First fit
        # res = model.fit(y, params=pars, x=x, method='least_squares', weights=weights)
        res = model.fit(y, params=pars, x=x, method='least_squares',
                weights=weights, fit_kws=fit_kws, max_nfev=2000,
                iter_cb=_iter_cb # Bashir added for aborting fit
                )

        if res.params['eta1'].value < 0.02:
            # freeze two-tail for peak 1 into one-tail:
            pars['cdt'].set(value=0.10, vary=False)  # or remove slow-tail for that peak specifically
            # or add a boolean switch in your code path to use one-tail EMG for peak 1

        # Debug summary for quick glance
        print(f"[AlphaEMG22] mu1={res.params['mu1'].value:.2f}, "
            f"mu2={res.params['mu2'].value:.2f}, "
            f"sigma1={res.params['sigma1'].value:.2g}, sigma2={res.params['sigma2'].value:.2g}, "
            f"tau11={res.params['tau11'].value:.2g}, tau21={res.params['tau21'].value:.2g}")


        # Optional one-step IRLS
        if wmode == 2:
            try:
                for _ in range(6):
                    yhat = model.eval(res.params, x=x)
                    w = 1.0 / np.sqrt(np.clip(yhat, 1.0, None))
                    new = model.fit(y, params=res.params.copy(), x=x,
                                    method='least_squares', weights=w, fit_kws=fit_kws,
                                    max_nfev=1500, iter_cb=_iter_cb # Bashir added for aborting fit
                                    )
                    if abs(new.chisqr - res.chisqr)/max(res.chisqr,1) < 1e-3:
                        res = new; break
                    res = new
            except Exception:
                pass

        if _iter_cb():
            return None

        R2_plain = getattr(res, "rsquared", None)
        if R2_plain is None:
            yhat = model.eval(res.params, x=x)
            ss_res = float(np.sum((y - yhat)**2))
            ss_tot = float(np.sum((y - y.mean())**2))
            R2_plain = 1.0 - ss_res / (ss_tot + 1e-16)

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
        ]
        row = {
            "timestamp": timestamp,
            "chi-square": redchi,
            "R2": R2_plain,
            "A1": _get("A1"),        "mu1": _get("mu1"),        "sigma1": _get("sigma1"),
            "tau11": _get("tau11"),    "tau12": _get("tau12"),      "eta1": _get("eta1"),
            "A2": _get("A2"),        "mu2": _get("mu2"),        "sigma2": _get("sigma2"),
            "tau21": _get("tau21"),    "tau22": _get("tau22"),      "eta2": _get("eta2"),

        }

        csv_path = os.path.join(os.getcwd(), "fit_EMG22_results.csv")
        write_header = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)

        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow(row)
        # ----------------------------------------------------------------------

        
        # Report
        wtxt = {0: "none", 1: "Poisson(data)", 2: "Poisson(model, IRLS)"}.get(wmode, "Poisson(data)")
        header = f"Notes: bandwidth = {res.params['bw'].value:.6g} ; weighting = {wtxt}"
        stats  = (f"[stats] chi-square={res.chisqr:.3f} ; reduced chi-square={res.redchi:.3f} ; "
                  f"dof={res.nfree} (N={res.ndata}, k={res.nvarys})")

        def _fmt_with_pct(params):
            """
            Lines like:
                name = value ± stderr   [cv=..%, var%=..%]
            Fixed or missing stderr are handled gracefully.
            cv%   = 100 * stderr / |value|
            var%  = 100 * (stderr/value)^2
            """
            lines = []
            for name, par in params.items():
                if name == 'bw':
                    continue
                val = par.value
                err = par.stderr

                # Fixed parameters (no uncertainties)
                if not par.vary:
                    lines.append(f"{name:>8} = {val:.6g}  (fixed)")
                    continue

                # No stderr available
                if err is None or not np.isfinite(err):
                    lines.append(f"{name:>8} = {val:.6g}  ± n/a   [cv=n/a, var%=n/a]")
                    continue

                # Percent metrics
                if val != 0 and np.isfinite(val):
                    cv  = abs(err/val) * 100.0
                    pvar = (err/val)**2 * 100.0
                    lines.append(f"{name:>8} = {val:.6g}  ± {err:.3g}   [cv={cv:.2f}%, var%={pvar:.2f}%]")
                else:
                    # value == 0 → avoid division-by-zero
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
        xx = np.linspace(x.min(), x.max(), 1400)
        yy = model.eval(res.params, x=xx)
        (fitln,) = axis.plot(xx, yy, lw=2)
        return fitln
        '''

        # Draw fitted curve(s) — total + separate components
        xx = np.linspace(x.min(), x.max(), 1400)
        p = res.params
        bwv = p['bw'].value

        y1 = _peak_binned(xx, p['A1'].value, p['mu1'].value, p['sigma1'].value,
                        p['tau11'].value, p['tau12'].value, p['eta1'].value, bwv)
        y2 = _peak_binned(xx, p['A2'].value, p['mu2'].value, p['sigma2'].value,
                        p['tau21'].value, p['tau22'].value, p['eta2'].value, bwv)
        ytot = y1 + y2

        fitln_total, = axis.plot(xx, ytot, lw=3, label='fit total')
        fitln_p1,    = axis.plot(xx, y1,  lw=2.5, ls='--', alpha=0.9, label='peak 1')
        fitln_p2,    = axis.plot(xx, y2,  lw=2.5, ls='--', alpha=0.9, label='peak 2')

        # keep the legend tidy if the user adds many fits
        if axis.legend_ is None:
            axis.legend(loc='best', frameon=False)

        # (optional) attach component handles & data to the main line for easy deletion later
        fitln_total.components = [fitln_p1, fitln_p2]
        fitln_total.component_data = {'x': xx, 'y1': y1, 'y2': y2, 'ytot': ytot}

        fitln_total._ratio = res.params['ratio'].value  # or res.params['A2'].value / res.params['A1'].value

        return fitln_total



class AlphaEMG22FitBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, param_1=1, param_2=2, param_3=10, **_ignored):
        if not self._instance:
            self._instance = AlphaEMG22Fit(param_1=param_1, param_2=param_2, param_3=param_3)
        return self._instance
