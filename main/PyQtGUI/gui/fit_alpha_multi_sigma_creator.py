#!/usr/bin/env python
# fit_alpha_multi_sigma_creator.py
#
# AlphaMultiEMG: Sum of many EMG sub-peaks grouped by isotope.
#   - Loads shapes from a text/CSV file with rows:
#     isotope, half_life, energy_keV, percent, sigma, tau1, tau2, eta, flag
#
#     where group_flag is one of: s (start), e (end), - (middle)
#   - Known per-subpeak amplitude ratios from Percent within each isotope.
#   - User FITTED: one amplitude A_<isotope> per isotope (>=0).
#   - Optional FITTED: small per-isotope shift dm_<isotope> added to every μ in that isotope.
#   - FIXED (from file): μ (via linear calibration), σ, τ1, τ2, η.
#   - Bin-width integration (GL7), Poisson weights, optional one-step IRLS.
#
# Seeds / popup order (len-tolerant):
#   [... you can ignore most; we only care (optionally) about bw(pN), wmode(pN+1) if GUI passes them ...]
#
# wmode: 0=unweighted, 1=Poisson(data), 2=Poisson(model, 1-step IRLS)
#
# To pass the shape file & calibration, set them in fit_factory config:
#   fit_factory.register("AlphaMultiEMG", AlphaMultiEMGFitBuilder(),
#                        shape_file="/path/to/shapes.txt",
#                        calib_a=6.8941013584, calib_b=-4943.2400523,
#                        allow_shift=True, shift_bound=300.0)
#
# If GUI doesn’t inject bw/wmode, we auto-pick: bw=median(dx), wmode=2.

import sys, os, csv, re
from datetime import datetime
sys.path.append(os.getcwd())

import numpy as np
from lmfit import Model, Parameters, fit_report
from scipy.special import erfcx, erfc

# Keep import so the factory can discover this module
import fit_factory  # noqa: F401

# Bashir added for aborting fit
try:
    from PyQt5.QtWidgets import QApplication
except Exception:
    QApplication = None


# ---- Gauss–Legendre (GL7) for bin integration -----------------------------------
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

_GL3_T = np.array([0.0, -0.7745966692, 0.7745966692], dtype=float)
_GL3_W = np.array([0.8888888889, 0.5555555556, 0.5555555556], dtype=float)

USE_GL3 = True  # set True for speed

_INV_SQRT2 = 1.0 / np.sqrt(2.0)

IRLS_MAX_ITERS = 6      # was 6
IRLS_IMPROVE   = 1e-3 # was 1e-3 (early stop threshold)

def _bin_integral(fun, x, bw, *args):
    bw = float(bw)
    if bw <= 0.0:
        return fun(x, *args)
    x = np.asarray(x, dtype=float)
    half = 0.5 * bw
    acc = np.zeros_like(x, dtype=float)
    T = _GL3_T if USE_GL3 else _GL7_T
    W = _GL3_W if USE_GL3 else _GL7_W
    for wi, ti in zip(W, T):
        acc += wi * fun(x + half * ti, *args)
    return half * acc

# ---- EMG (two-tail) pieces -------------------------------------------------------
def _emg_one_tail_stable(x, A, mu, sigma, tau):
    x = np.asarray(x, dtype=float)
    sigma = max(float(sigma), 1e-9)
    tau   = max(float(tau),   1e-9)
    pref = 0.5 * A / tau
    inv_sigma = 1.0 / sigma
    u = (_INV_SQRT2) * ((sigma / tau) + ((x - mu) * inv_sigma))
    out = np.empty_like(x)
    m = (u >= 0.0)
    if np.any(m):
        g = np.exp(-0.5 * ((x[m] - mu) * inv_sigma)**2)
        out[m] = pref * g * erfcx(u[m])
    if np.any(~m):
        expfac = np.exp(0.5 * (sigma / tau)**2 + (x[~m] - mu) / tau)
        out[~m] = pref * expfac * erfc(u[~m])
    return np.where(np.isfinite(out), out, 0.0)

def _emg_two_tail_stable(x, A, mu, sigma, tau_fast, tau_slow, eta):
    eta = float(np.clip(eta, 0.0, 1.0))
    return ((1.0 - eta) * _emg_one_tail_stable(x, A, mu, sigma, tau_fast)
          + (      eta) * _emg_one_tail_stable(x, A, mu, sigma, tau_slow))

def _peak_binned(x, A, mu, sigma, t1, t2, eta, bw):
    # same GL7 bin-integration you use for the total
    return _bin_integral(_emg_two_tail_stable, x, bw, A, mu, sigma, t1, t2, eta)

# ---- Utility: parse shapes file --------------------------------------------------
def _safe_name(s):
    """Make a safe parameter stem from an isotope name."""
    return re.sub(r'[^A-Za-z0-9_]+', '_', str(s).strip())

def _parse_percent(p):
    """Accept '13%' or '13' -> float fraction 0..1."""
    if isinstance(p, str):
        p = p.strip()
        if p.endswith('%'):
            p = p[:-1]
    try:
        v = float(p)
    except Exception:
        v = 0.0
    return max(v, 0.0) / 100.0

def _load_shapes(shape_file, a, b):
    """
    NEW TXT format (per line):
      isotope, half_life, energy_keV, percent, sigma, tau1, tau2, eta, flag
    Notes:
      • Empty isotope (',,') inherits the previous isotope.
      • flag ∈ {'s','e','-','*'}; '*' = single-line isotope.
      • Separator lines like ',,,,,,,,' are ignored.
      • 'percent' may be '31.6' or '31.6%'; both OK (parsed to fraction 0..1).
    Returns:
      isotopes: [{name, safe, start_idx, end_idx}]
      pulses  : [{iso_idx, name, safe, E, mu0, pct, ratio, sigma, tau1, tau2, eta}]
    """
    rows = []
    with open(shape_file, 'r', newline='') as f:
        rdr = csv.reader(f, skipinitialspace=True)
        last_iso = ""
        for raw in rdr:
            # Skip blank or comment lines
            if not raw or all(not str(x).strip() for x in raw):
                continue
            first = str(raw[0]).strip()
            if first.startswith("#"):
                continue

            # Pad to at least 9 fields
            raw = [str(x).strip() for x in raw]
            if len(raw) < 9:
                raw += [""] * (9 - len(raw))

            # Inherit isotope if empty on continuation lines
            iso = raw[0] or last_iso
            if not iso:
                continue
            last_iso = iso

            # Columns: 0 iso | 1 half-life(ignored) | 2 E | 3 percent | 4 sigma | 5 tau1 | 6 tau2 | 7 eta | 8 flag
            try:
                E = float(raw[2])
            except Exception:
                # e.g., separator line ',,,,,,,,' or malformed energy
                continue

            pct   = _parse_percent(raw[3])  # defined above; returns fraction 0..1
            try:
                sigma = float(raw[4]); tau1 = float(raw[5]); tau2 = float(raw[6])
            except Exception:
                continue

            try:
                eta = float(raw[7]) if raw[7] != "" else 0.2
            except Exception:
                eta = 0.2

            flag = (raw[8].lower()[:1] if raw[8] else '-')
            if flag not in ('s', 'e', '-', '*'):
                flag = '-'

            mu0 = a * E + b
            rows.append(dict(iso=iso, E=E, pct=pct, flag=flag,
                             sigma=sigma, tau1=tau1, tau2=tau2, eta=eta, mu0=mu0))

    # ---- Group into isotopes by contiguous blocks (respect '*' singletons) ----
    isotopes, pulses = [], []
    i = 0
    while i < len(rows):
        iso_name = rows[i]['iso']
        safe     = _safe_name(iso_name)

        # Single-line isotope
        if rows[i]['flag'] == '*':
            start = end = i
        else:
            start = i
            j = i
            while j + 1 < len(rows) and rows[j+1]['iso'] == iso_name and rows[j+1]['flag'] != 's':
                j += 1
            end = j

        isotopes.append(dict(name=iso_name, safe=safe, start_idx=len(pulses), end_idx=None))

        # Amplitude ratios within the block: relative to the first row's percent
        first_pct = rows[start]['pct'] if rows[start]['pct'] > 0 else 1.0
        for k in range(start, end + 1):
            r = rows[k]
            ratio = 1.0 if (start == end) else ((r['pct'] / first_pct) if first_pct > 0 else 1.0)
            pulses.append(dict(
                iso_idx=len(isotopes) - 1,
                name=iso_name,
                safe=safe,
                E=r['E'],
                mu0=r['mu0'],
                pct=r['pct'],
                ratio=ratio,
                sigma=r['sigma'],
                tau1=r['tau1'],
                tau2=r['tau2'],
                eta=r['eta'],
            ))

        isotopes[-1]['end_idx'] = len(pulses) - 1
        i = end + 1

    return isotopes, pulses


# ---- The fitter ------------------------------------------------------------------

class AlphaMultiEMGSigmaFit:
    def __init__(self,
                 shape_file,
                 calib_a=7.1195126, calib_b=-7029.0,
                 calibration_file=None,
                #  calib_a=5.1787509, calib_b=-5436.5,
                 allow_shift=True, shift_bound=0.0,
                 fix_ratios=True, wmode_default=2,
                 fit_global_shapes=True,
                 fit_iso_shape_scales=True,      
                 iso_scale_bounds=None,     
                 shape_bounds=None):
        self.shape_file    = shape_file

        if isinstance(calibration_file, (tuple, list)) and len(calibration_file) == 2:
            a, b = calibration_file
            self.calib_a = float(a)
            self.calib_b = float(b)
            self.calibration_file = None
        else:
            self.calib_a = float(calib_a)
            self.calib_b = float(calib_b)
            # keep the path only for info/reporting (optional)
            self.calibration_file = calibration_file if isinstance(calibration_file, str) else None


        self.allow_shift   = bool(allow_shift)
        self.shift_bound   = float(shift_bound)
        self.fix_ratios    = bool(fix_ratios)
        self.wmode_default = int(wmode_default)
        self.fit_global_shapes = bool(fit_global_shapes)
        self.fit_iso_shape_scales = bool(fit_iso_shape_scales)

        if self.fit_iso_shape_scales and not self.fit_global_shapes:
            self.fit_iso_shape_scales = False  # silently coerce (or raise if you prefer)

        self.shape_bounds = shape_bounds or {
            "sigma": (1.0, 2000.0),          # ADC
            "tau1":  (1.0, 2000.0),          # ADC
            "tau2":  (1.0, 2000.0),          # ADC
            "eta":   (0.0, 1.0),
        }

        self.iso_scale_bounds = iso_scale_bounds or {             # <--- NEW
            "s_sigma": (0.5, 1.5),
            "s_tau1":  (0.5, 1.5),
            "s_tau2":  (0.5, 1.5),
        }

        if not os.path.isfile(self.shape_file):
            raise FileNotFoundError(f"shape_file not found: {self.shape_file}")

        # Use the TOP-LEVEL loader defined earlier in the file:
        # self._isotopes, self._pulses = _load_shapes(self.shape_file, self.calib_a, self.calib_b)


    # ---- helpers -------------------------------------------------------------


    # ---- public API used by GUI ----
    def start(self, x, y, xmin, xmax, fitpar, axis, fit_results):
        self._isotopes, self._pulses = _load_shapes(self.shape_file, self.calib_a, self.calib_b)
        # Basic hygiene
        mfin = np.isfinite(x) & np.isfinite(y)
        x = np.asarray(x)[mfin]
        y = np.asarray(y)[mfin]
        if x.size < 5:
            fit_results.setPlainText("Not enough data to fit.")
            return None

        # Window
        dx   = float(np.median(np.diff(x))) if x.size > 1 else 1.0
        if np.isfinite(xmin) and np.isfinite(xmax):
            m = (x >= xmin) & (x <= xmax)
            x, y = x[m], y[m]
            if x.size < 5:
                fit_results.setPlainText("Not enough data in selected window.")
                return None
            dx = float(np.median(np.diff(x))) if x.size > 1 else dx

        # --- Parse bw, wmode, and bounds from GUI tail; blank/0 => defaults ---
        fp = list(fitpar) if fitpar is not None else []
        dx_med = float(dx)

        D0_DEF = 1000.0
        DG_DEF = 2e-2
        DM_DEF = self.shift_bound

        def _to_float(v):
            try:
                z = float(v)
                return z if np.isfinite(z) else None
            except Exception:
                return None

        def _abs_or_default(v, default):
            z = _to_float(v)
            # blank/None/NaN or exactly 0.0 => use default
            if z is None or z == 0.0:
                return float(default)
            return abs(float(z))

        # defaults
        bw    = dx_med
        wmode = int(getattr(self, "wmode_default", 2))
        # use current shape upper-bounds as defaults
        sig_def = self.shape_bounds["sigma"][1]
        t1_def  = self.shape_bounds["tau1"][1]
        t2_def  = self.shape_bounds["tau2"][1]

        # user tail payload: [bw, wmode, sigma_bound, tau1_bound, tau2_bound, d0_abs, dg_abs, dm_abs]
        tail = fp[-8:] if len(fp) >= 8 else []

        if tail:
            # bw / wmode
            bw_in = _to_float(tail[0]);   bw    = bw_in if (bw_in and bw_in > 0) else bw
            wm_in = _to_float(tail[1]);   wmode = int(round(wm_in)) if (wm_in is not None and wm_in in (0.0,1.0,2.0)) else wmode

            # bounds: blank or 0 => keep defaults
            sigma_bound = _abs_or_default(tail[2], sig_def)
            tau1_bound  = _abs_or_default(tail[3], t1_def)
            tau2_bound  = _abs_or_default(tail[4], t2_def)

            # globals: blank or 0 => default magnitudes
            d0_abs = _abs_or_default(tail[5], D0_DEF)
            dg_abs = _abs_or_default(tail[6], DG_DEF)

            # dm: blank => keep DM_DEF; exactly 0 => lock shifts
            dm_in = _to_float(tail[7])
            dm_abs = 0.0 if dm_in == 0.0 else (DM_DEF if dm_in is None else float(dm_in))
        else:
            sigma_bound, tau1_bound, tau2_bound = sig_def, t1_def, t2_def
            d0_abs, dg_abs, dm_abs = D0_DEF, DG_DEF, DM_DEF

        # final sanitize
        bw    = bw if (np.isfinite(bw) and bw > 0) else dx_med
        wmode = wmode if wmode in (0, 1, 2) else int(getattr(self, "wmode_default", 2))

        # apply caps only on uppers; keep existing lowers
        def _cap(key, ub):
            if ub > 0.0:
                lo, _ = self.shape_bounds.get(key, (1.0, 2000.0))
                self.shape_bounds[key] = (max(1.0, float(lo)), float(ub))

        _cap("sigma", sigma_bound)
        _cap("tau1",  tau1_bound)
        _cap("tau2",  tau2_bound)

        # dm=0 => lock per-isotope shifts
        self.allow_shift = bool(self.allow_shift and dm_abs != 0.0)

        # always vary d0/dg later when adding Parameters:
        # pars.add('d0', value=0.0, min=-d0_abs, max=+d0_abs, vary=True)
        # pars.add('dg', value=0.0, min=-dg_abs, max=+dg_abs, vary=True)


        ######################################################
        # Build parameters
        pars = Parameters()

        # Seeds from file medians (robust)
        sig_seed = float(np.median([p['sigma'] for p in self._pulses])) or 200.0
        t1_seed  = float(np.median([p['tau1']  for p in self._pulses])) or 80.0
        t2_seed  = float(np.median([p['tau2']  for p in self._pulses])) or 300.0
        eta_seed = float(np.median([p['eta']   for p in self._pulses])) or 0.7

        if self.fit_global_shapes:
            lo, hi = self.shape_bounds["sigma"]
            pars.add('sigma', value=sig_seed, min=lo, max=hi, vary=True)
            lo, hi = self.shape_bounds["tau1"]
            pars.add('tau1',  value=t1_seed,  min=lo, max=hi, vary=True)
            lo, hi = self.shape_bounds["tau2"]
            pars.add('tau2',  value=t2_seed,  min=lo, max=hi, vary=True)
            lo, hi = self.shape_bounds["eta"]
            pars.add('eta',   value=eta_seed, min=lo, max=hi, vary=True)

        # vary_d0 = (d0_abs > 0)
        # vary_dg = (dg_abs > 0)
        pars.add('d0', value=0.0,
                min=(-float(d0_abs)),
                max=(+float(d0_abs)),
                vary=True)
        pars.add('dg', value=0.0,
                min=(-float(dg_abs)),
                max=(+float(dg_abs)),
                vary=True)

        # background
        # pars.add('b0', value=max(np.percentile(y, 1), 0.0), min=0.0)
        # pars.add('b1', value=0.0)  # optional slope
        
        # one amplitude per isotope, plus a tightly-bounded dm with a good seed
        for k, iso in enumerate(self._isotopes):
            stem = iso['safe']

            # window around the median mu of this isotope’s peaks
            mu0s  = [q['mu0'] for q in self._pulses if q['iso_idx'] == k]
            sig0s = [q['sigma'] for q in self._pulses if q['iso_idx'] == k]
            mu0i  = float(np.median(mu0s))
            sig0  = float(np.median(sig0s)) if sig0s else dx
            W     = max(8.0*sig0, 400.0)     # generous window
            mwin  = (x >= mu0i - W) & (x <= mu0i + W)
            Aseed = float(np.trapz(np.clip(y[mwin], 0, None), x[mwin])) if np.any(mwin) else 1.0

            pars.add(f"A_{stem}", value=max(Aseed, 1.0), min=0.0)

            if self.allow_shift:
                dm0 = 0.0
                if np.count_nonzero(mwin) > 5:
                    xxw, yyw = x[mwin], y[mwin]
                    dm0 = float(xxw[np.argmax(yyw)] - mu0i)
                dm_lo = -float(dm_abs)
                dm_hi = +float(dm_abs)
                pars.add(
                    f"dm_{stem}",
                    value=float(np.clip(dm0, dm_lo, dm_hi)),
                    min=dm_lo, max=dm_hi, vary=True
                )
            
            # --- NEW: optional per-isotope scale factors for σ, τ1, τ2
            if self.fit_iso_shape_scales:
                lo, hi = self.iso_scale_bounds["s_sigma"]
                pars.add(f"s_sigma_{stem}", value=1.0, min=lo, max=hi, vary=True)
                lo, hi = self.iso_scale_bounds["s_tau1"]
                pars.add(f"s_tau1_{stem}",  value=1.0, min=lo, max=hi, vary=True)
                lo, hi = self.iso_scale_bounds["s_tau2"]
                pars.add(f"s_tau2_{stem}",  value=1.0, min=lo, max=hi, vary=True)


        # fixed bin width
        pars.add('bw', value=max(float(bw), 0.0), vary=False, min=0.0)


        # Closure over pulses/isotopes
        pulses = self._pulses
        allow_shift = self.allow_shift

        # ---- the model (NESTED inside start!) ----
        def _sum_multi_binned(x, **kw):
            bw_loc = float(kw.get('bw', dx))

            def _sum_multi(x):
                out = np.zeros_like(x, dtype=float)
                d0 = float(kw.get('d0', 0.0))
                dg = float(kw.get('dg', 0.0))

                # cache global shapes once if enabled
                if self.fit_global_shapes:
                    g_sigma = float(kw.get('sigma', 0.0))
                    g_t1    = float(kw.get('tau1',  0.0))
                    g_t2    = float(kw.get('tau2',  0.0))
                    g_eta   = float(kw.get('eta',   0.0))

                for p in pulses:
                    stem   = self._isotopes[p['iso_idx']]['safe']
                    A_iso  = float(kw.get(f"A_{stem}", 0.0))
                    if A_iso <= 0.0:
                        continue

                    dm = float(kw.get(f"dm_{stem}", 0.0)) if allow_shift else 0.0
                    A  = A_iso * p['ratio']
                    mu = (1.0 + dg) * p['mu0'] + d0 + dm

                    # (1) choose baseline shapes
                    if self.fit_global_shapes:
                        base_sigma, base_t1, base_t2, eta = g_sigma, g_t1, g_t2, g_eta
                    else:
                        base_sigma, base_t1, base_t2, eta = p['sigma'], p['tau1'], p['tau2'], p['eta']

                    # (2) optionally apply per-isotope scale factors
                    if getattr(self, 'fit_iso_shape_scales', False):
                        sS = float(kw.get(f"s_sigma_{stem}", 1.0))
                        s1 = float(kw.get(f"s_tau1_{stem}",  1.0))
                        s2 = float(kw.get(f"s_tau2_{stem}",  1.0))
                        sigma = base_sigma * sS
                        t1    = base_t1    * s1
                        t2    = base_t2    * s2
                    else:
                        sigma, t1, t2 = base_sigma, base_t1, base_t2

                    out += _emg_two_tail_stable(x, A, mu, sigma, t1, t2, eta)

                return out

            return _bin_integral(_sum_multi, x, bw_loc)

        model = Model(_sum_multi_binned, independent_vars=['x'])

        # Fit
        # res = model.fit(y, params=pars, x=x, method='least_squares', weights=weights, fit_kws=fit_kws, max_nfev=2000)

        # ----- optional decimation for speed -----
        stride = 1               # set 2–4 for speed if you want
        xf, yf = x[::stride], y[::stride]

        def weights_from_data(y_arr):
            return 1.0 / np.sqrt(np.clip(y_arr, 1.0, None))

        # build initial weights
        if wmode == 0:
            w_fit = None
        else:
            w_fit = weights_from_data(yf)        # bootstrap with data

        # Use linear loss to get interpretable χ²
        fit_kws = dict(loss='linear')

        # ---- Bashir's Abort callback ----
        _should_abort = getattr(self, "_should_abort", None)

        def _iter_cb(params=None, iter=None, resid=None, *args, **kwargs):
            # let the GUI process clicks so Abort can be pressed
            if QApplication is not None:
                QApplication.processEvents()
            # returning True tells lmfit to stop
            return bool(_should_abort and _should_abort())
        ###########################################################

        # single initial fit
        res = model.fit(yf, params=pars, x=xf, method='least_squares',
                        weights=w_fit, fit_kws=fit_kws, max_nfev=2000,
                        iter_cb=_iter_cb,   # Bashir added for aborting fit
                        )

        # IRLS: switch weights to model
        if wmode == 2:
            try:
                last = res
                for _ in range(IRLS_MAX_ITERS):
                    yhat = model.eval(last.params, x=xf)
                    w_fit = 1.0 / np.sqrt(np.clip(yhat, 1.0, None))
                    new = model.fit(yf, params=last.params.copy(), x=xf,
                                    method='least_squares', weights=w_fit,
                                    fit_kws=fit_kws, max_nfev=1500,
                                    iter_cb=_iter_cb,   # Bashir added for aborting fit
                                    )
                    if _should_abort and _should_abort():
                        fit_results.append("[abort] Stopped during IRLS.")
                        return None

                    rel = abs(new.chisqr - last.chisqr) / max(last.chisqr, 1.0)
                    last = new
                    if rel < IRLS_IMPROVE:
                        break
                res = last
            except Exception:
                pass

        '''
        # ---------------- Reporting ----------------

        # Compact params: print A_<iso> and dm_<iso> (if enabled)
        lines = []
        for iso in self._isotopes:
            stem = iso['safe']
            name = iso['name']
            A   = res.params[f"A_{stem}"].value
            if self.allow_shift:
                dm = res.params[f"dm_{stem}"].value
                lines.append(f"{name:>10}:  A={A:.6g}   dm={dm:.6g}")
            else:
                lines.append(f"{name:>10}:  A={A:.6g}")

        # Classic R^2 on raw counts (not weighted)
        yhat_full = model.eval(res.params, x=x)
        ss_res = float(np.sum((y - yhat_full)**2))
        ss_tot = float(np.sum((y - y.mean())**2))
        R2_plain = 1.0 - ss_res / (ss_tot + 1e-16)

        wtxt = {0: "none", 1: "Poisson(data)", 2: "Poisson(model, IRLS)"}[wmode]
        header = f"Notes: bandwidth = {res.params['bw'].value:.6g} ; weighting = {wtxt}"
        stats  = (f"[stats] chi-square={res.chisqr:.3f} ; reduced chi-square={res.redchi:.3f} ; "
          f"dof={res.nfree} (N={res.ndata}, k={res.nvarys}) ; R^2 (plain)={R2_plain:.4f}")

        fit_results.setPlainText(
            header + "\n" + stats + "\n\n"
            + "Per-isotope parameters:\n" + "\n".join(lines) + "\n\n"
            + fit_report(res, show_correl=True)
        )
        '''
        if _iter_cb():
            return None

        # ---------- Reporting ----------
        def _fmt_val_err(v, e, digs=6):
            try:
                v = float(v)
            except Exception:
                return str(v)
            ok = (e is not None) and np.isfinite(e)
            if not ok: return f"{v:.{digs}g}"
            e = float(e)
            if v != 0.0 and np.isfinite(v):
                pct = abs(e / v) * 100.0
                return f"{v:.{digs}g} ± {e:.3g} ({pct:.2f}%)"
            return f"{v:.{digs}g} ± {e:.3g}"

        yhat_full = model.eval(res.params, x=x)
        ss_res = float(np.sum((y - yhat_full)**2))
        ss_tot = float(np.sum((y - y.mean())**2))
        R2_plain = 1.0 - ss_res / (ss_tot + 1e-16)

        wtxt = {0: "none", 1: "Poisson(data)", 2: "Poisson(model, IRLS)"}[wmode]
        bw_str = _fmt_val_err(res.params['bw'].value, res.params['bw'].stderr)

        # numeric d0/dg (needed for μ_eff); also formatted strings for header
        d0p = res.params.get('d0'); dgp = res.params.get('dg')
        d0   = float(d0p.value) if d0p is not None else 0.0
        dg   = float(dgp.value) if dgp is not None else 0.0
        d0_str = _fmt_val_err(d0p.value, getattr(d0p, "stderr", None)) if d0p else "n/a"
        dg_str = _fmt_val_err(dgp.value, getattr(dgp, "stderr", None)) if dgp else "n/a"

        header = (f"Notes: bandwidth = {bw_str} ; weighting = {wtxt}\n"
                f"Global shift d0 = {d0_str} ; Global gain dg = {dg_str}")

        # --- Shapes source + mode header line ---
        mode = ("global + per-isotope scales" if (self.fit_global_shapes and self.fit_iso_shape_scales)
                else "global" if self.fit_global_shapes
                else "frozen (file σ,τ₁,τ₂,η)")
        shape_src = os.path.abspath(self.shape_file)
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(self.shape_file)).isoformat(timespec="seconds")
            shape_head = f"[shapes] mode = {mode} ; source = {shape_src} ; mtime = {mtime}"
        except Exception:
            shape_head = f"[shapes] mode = {mode} ; source = {shape_src}"

        # put shapes line before the existing header
        header = shape_head + "\n" + header


        header += (f"\nBounds: |d0|≤{d0_abs:g}, |dg|≤{dg_abs:g}, |dm_*|≤{dm_abs:g}")


        if self.fit_global_shapes:
            def fmt(pn):
                pp = res.params.get(pn)
                return _fmt_val_err(pp.value, getattr(pp, "stderr", None))
            header += ("\nGlobal shapes (fitted): "
                    f"sigma={fmt('sigma')}, tau1={fmt('tau1')}, tau2={fmt('tau2')}, eta={fmt('eta')}")

        iso_lines = []
        iso_lines.append("Per-isotope parameters (A per isotope; ratios fixed):")
        iso_lines.append(f"  allow_shift={self.allow_shift} ; fix_ratios={self.fix_ratios}")

        if self.fit_iso_shape_scales:
            iso_lines.append("\nPer-isotope shape scales (multipliers on baseline σ, τ1, τ2):")
            for iso in self._isotopes:
                stem = iso['safe']; name = iso['name']
                def fmt_s(pn):
                    pp = res.params.get(pn)
                    return _fmt_val_err(pp.value, getattr(pp, "stderr", None)) if pp else "1"
                iso_lines.append(
                    f"{name:>10}: sσ={fmt_s(f's_sigma_{stem}')}, "
                    f"sτ1={fmt_s(f's_tau1_{stem}')}, sτ2={fmt_s(f's_tau2_{stem}')}"
                )

        stats = (f"[stats] chi-square={res.chisqr:.3f} ; reduced chi-square={res.redchi:.3f} ; "
                f"dof={res.nfree} (N={res.ndata}, k={res.nvarys}) ; R^2 (plain)={R2_plain:.4f}")

        for iso in self._isotopes:
            stem = iso['safe']; name = iso['name']
            pA   = res.params.get(f"A_{stem}")
            pDM  = res.params.get(f"dm_{stem}") if self.allow_shift else None
            A_str  = _fmt_val_err(pA.value, getattr(pA, "stderr", None)) if pA else "n/a"
            dm_str = _fmt_val_err(pDM.value, getattr(pDM, "stderr", None)) if pDM else None

            idx0, idx1 = iso['start_idx'], iso['end_idx']
            blockE  = [f"{self._pulses[k]['E']:.1f}" for k in range(idx0, idx1+1)]
            blockRt = [float(self._pulses[k]['ratio']) for k in range(idx0, idx1+1)]
            E_str   = ", ".join(blockE) + " keV"
            rat_str = ", ".join(f"{r:.3f}" for r in blockRt)

            base = f"{name:>10}:  A={A_str}"
            if dm_str is not None: base += f"   dm={dm_str}"
            base += f"   (subpeaks={len(blockE)}; ratios=[{rat_str}]; E=[{E_str}])"
            iso_lines.append(base)

        # ---- build per-isotope effective values ONCE (rows), then use for report + CSV
        p = res.params
        use_scales = getattr(self, 'fit_iso_shape_scales', False)
        now = datetime.now().isoformat(timespec="seconds")
        rows = []

        for iso in self._isotopes:
            stem = iso['safe']; name = iso['name']

            A_iso = float(p[f"A_{stem}"].value)
            dm    = float(p[f"dm_{stem}"].value) if (self.allow_shift and f"dm_{stem}" in p) else 0.0

            sS = float(p.get(f"s_sigma_{stem}", 1.0).value) if use_scales and p.get(f"s_sigma_{stem}") else 1.0
            s1 = float(p.get(f"s_tau1_{stem}",  1.0).value) if use_scales and p.get(f"s_tau1_{stem}")  else 1.0
            s2 = float(p.get(f"s_tau2_{stem}",  1.0).value) if use_scales and p.get(f"s_tau2_{stem}")  else 1.0

            idx0, idx1 = iso['start_idx'], iso['end_idx']
            q_best = max(self._pulses[idx0:idx1+1], key=lambda q: float(q.get('ratio', 0.0)))

            if self.fit_global_shapes:
                sigma0, t10, t20, eta0 = float(p['sigma'].value), float(p['tau1'].value), float(p['tau2'].value), float(p['eta'].value)
            else:
                sigma0, t10, t20, eta0 = float(q_best['sigma']), float(q_best['tau1']), float(q_best['tau2']), float(q_best['eta'])

            mu_eff   = (1.0 + dg) * float(q_best['mu0']) + d0 + dm
            sigma_eff = sigma0 * sS
            tau1_eff  = t10    * s1
            tau2_eff  = t20    * s2

            rows.append({
                "timestamp": now,
                "chi-square": float(getattr(res, "redchi", np.nan)),   # store redchi here; change if you want raw chisqr
                "R2": float(R2_plain),
                "A": float(A_iso),
                "mu": float(mu_eff),
                "sigma": float(sigma_eff),
                "tau11": float(tau1_eff),
                "tau12": float(tau2_eff),
                "eta": float(eta0),
                # optionally: "isotope": name, "E_keV": q_best['E']
            })


        # --- NEW: Per-subpeak (component) printout: A, σ, τ1, τ2, η ---
        subpeak_lines = []
        subpeak_lines.append("\nPer-subpeak parameters:")

        use_scales = getattr(self, 'fit_iso_shape_scales', False)

        for iso in self._isotopes:
            stem = iso['safe']; name = iso['name']

            A_iso = float(p[f"A_{stem}"].value)
            dm    = float(p[f"dm_{stem}"].value) if (self.allow_shift and f"dm_{stem}" in p) else 0.0

            sS = float(p.get(f"s_sigma_{stem}", 1.0).value) if use_scales and p.get(f"s_sigma_{stem}") else 1.0
            s1 = float(p.get(f"s_tau1_{stem}",  1.0).value) if use_scales and p.get(f"s_tau1_{stem}")  else 1.0
            s2 = float(p.get(f"s_tau2_{stem}",  1.0).value) if use_scales and p.get(f"s_tau2_{stem}")  else 1.0

            if self.fit_global_shapes:
                sigma0 = float(p['sigma'].value)
                t10    = float(p['tau1'].value)
                t20    = float(p['tau2'].value)
                eta0   = float(p['eta'].value)
            else:
                sigma0 = t10 = t20 = eta0 = None

            for kk in range(iso['start_idx'], iso['end_idx'] + 1):
                q = self._pulses[kk]

                A_sub  = A_iso * float(q['ratio'])
                mu_sub = (1.0 + dg) * float(q['mu0']) + d0 + dm

                if self.fit_global_shapes:
                    sigma_eff = sigma0 * sS
                    tau1_eff  = t10    * s1
                    tau2_eff  = t20    * s2
                    eta_eff   = eta0
                else:
                    sigma_eff = float(q['sigma']) * sS
                    tau1_eff  = float(q['tau1'])  * s1
                    tau2_eff  = float(q['tau2'])  * s2
                    eta_eff   = float(q['eta'])

                subpeak_lines.append(
                    f"{name:>10}  E={q['E']:.1f} keV : "
                    f"A={A_sub:.6g}, μ={mu_sub:.3f}, σ={sigma_eff:.3g}, "
                    f"τ1={tau1_eff:.3g}, τ2={tau2_eff:.3g}, η={eta_eff:.3g}"
                )


        # ---- include in the report text ----
        hits = []
        for iso in self._isotopes:
            pDM = res.params.get(f"dm_{iso['safe']}")
            if pDM and (abs(pDM.value - pDM.min) < 1e-12 or abs(pDM.value - pDM.max) < 1e-12):
                hits.append(iso['name'])

        more = []
        for nm in ('d0','dg'):
            pG = res.params.get(nm)
            if pG and (abs(pG.value - pG.min) < 1e-12 or abs(pG.value - pG.max) < 1e-12):
                more.append(nm)

        notice = ""
        if hits or more:
            who = f"dm_* at bounds: {', '.join(hits)}" if hits else ""
            gbl = f"globals at bounds: {', '.join(more)}" if more else ""
            join = " ; " if (who and gbl) else ""
            notice = f"\n[notice] {who}{join}{gbl}"

        fit_results.setPlainText(
            header + "\n" + stats + "\n\n" +
            "\n".join(iso_lines) + "\n" +             # your existing per-isotope lines
            "\n".join(subpeak_lines) +                # NEW per-subpeak lines
            notice + "\n\n" +
            fit_report(res, show_correl=True)
        )


        # ---- CSV: per-subpeak components ----
        subrows = []
        now = datetime.now().isoformat(timespec="seconds")
        use_scales = getattr(self, 'fit_iso_shape_scales', False)
        p = res.params

        for iso in self._isotopes:
            stem = iso['safe']; name = iso['name']
            A_iso = float(p[f"A_{stem}"].value)
            dm    = float(p[f"dm_{stem}"].value) if (self.allow_shift and f"dm_{stem}" in p) else 0.0

            sS = float(p.get(f"s_sigma_{stem}", 1.0).value) if use_scales and p.get(f"s_sigma_{stem}") else 1.0
            s1 = float(p.get(f"s_tau1_{stem}",  1.0).value) if use_scales and p.get(f"s_tau1_{stem}")  else 1.0
            s2 = float(p.get(f"s_tau2_{stem}",  1.0).value) if use_scales and p.get(f"s_tau2_{stem}")  else 1.0

            # global or local baselines
            if self.fit_global_shapes:
                sigma0 = float(p['sigma'].value); t10 = float(p['tau1'].value)
                t20 = float(p['tau2'].value);     eta0 = float(p['eta'].value)
            # else: use each q's own shapes below

            for kk in range(iso['start_idx'], iso['end_idx'] + 1):
                q = self._pulses[kk]
                A_sub = A_iso * float(q['ratio'])
                mu_sub = (1.0 + dg) * float(q['mu0']) + d0 + dm

                if self.fit_global_shapes:
                    sigma_eff = sigma0 * sS; tau1_eff = t10 * s1; tau2_eff = t20 * s2; eta_eff = eta0
                else:
                    sigma_eff = float(q['sigma']) * sS
                    tau1_eff  = float(q['tau1'])  * s1
                    tau2_eff  = float(q['tau2'])  * s2
                    eta_eff   = float(q['eta'])

                subrows.append({
                    "timestamp": now,
                    "isotope": name,
                    "E_keV": float(q['E']),
                    "A_sub": float(A_sub),
                    "mu": float(mu_sub),
                    "sigma": float(sigma_eff),
                    "tau1": float(tau1_eff),
                    "tau2": float(tau2_eff),
                    "eta": float(eta_eff),
                    "chi2": float(getattr(res, "redchi", np.nan)),
                    "R2": float(R2_plain),
                    "d0_bound": d0_abs,
                    "dg_bound": dg_abs,
                    "dm_bound": dm_abs,
                })

        subcsv = os.path.join(os.getcwd(), "fit_EMGMultiSigma_subpeaks.csv")
        subfields = ["timestamp","isotope","E_keV","A_sub","mu","sigma","tau1","tau2","eta","chi2","R2",
             "d0_bound","dg_bound","dm_bound"]
        write_header = (not os.path.exists(subcsv)) or (os.path.getsize(subcsv) == 0)
        with open(subcsv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=subfields)
            if write_header:
                w.writeheader()
            w.writerows(subrows)

                        
        
        # ---------------- plotting ----------------
        xx  = np.linspace(x.min(), x.max(), 1400)
        p   = res.params
        bwv = p['bw'].value if 'bw' in p else float(np.median(np.diff(xx)))
        ytot = model.eval(res.params, x=xx)

        # total fit (solid line)
        (fitln_total,) = axis.plot(xx, ytot, lw=2.5, color='tab:orange', label='fit total')

        # convenience
        d0 = float(p['d0'].value) if 'd0' in p else 0.0
        dg = float(p['dg'].value) if 'dg' in p else 0.0

        # cache global shapes if we’re fitting them
        if self.fit_global_shapes:
            sig_v = float(p['sigma'].value)
            t1_v  = float(p['tau1'].value)
            t2_v  = float(p['tau2'].value)
            eta_v = float(p['eta'].value)

        # --- draw one dashed line per sub-peak ---
        # --- draw one dashed line per sub-peak ---
        use_scales = getattr(self, 'fit_iso_shape_scales', False)
        sub_lines = []

        for iso in self._isotopes:
            stem = iso['safe']; name = iso['name']
            A_iso = float(p[f"A_{stem}"].value)
            dm    = float(p[f"dm_{stem}"].value) if (self.allow_shift and f"dm_{stem}" in p) else 0.0

            # scale factors (default 1 if not fitted)
            sS = float(p.get(f"s_sigma_{stem}", 1.0).value) if use_scales and p.get(f"s_sigma_{stem}") else 1.0
            s1 = float(p.get(f"s_tau1_{stem}",  1.0).value) if use_scales and p.get(f"s_tau1_{stem}")  else 1.0
            s2 = float(p.get(f"s_tau2_{stem}",  1.0).value) if use_scales and p.get(f"s_tau2_{stem}")  else 1.0

            for kk in range(iso['start_idx'], iso['end_idx'] + 1):
                q   = self._pulses[kk]
                A   = A_iso * float(q['ratio'])
                mu  = (1.0 + dg) * float(q['mu0']) + d0 + dm

                # choose baselines (global if fitted, else from file)
                if self.fit_global_shapes:
                    sigma0, t10, t20, eta0 = sig_v, t1_v, t2_v, eta_v
                else:
                    sigma0, t10, t20, eta0 = q['sigma'], q['tau1'], q['tau2'], q['eta']

                # apply per-isotope scales
                sigma = sigma0 * sS
                t1    = t10    * s1
                t2    = t20    * s2

                yj = _peak_binned(xx, A, mu, sigma, t1, t2, eta0, bwv)

                label = f"{name} {q['E']:.1f} keV"
                (ln,) = axis.plot(xx, yj, lw=1.8, ls='--', alpha=0.9, label=label)
                sub_lines.append(ln)


        # --- labels after all lines are drawn ---
        axis.relim(); axis.autoscale_view()
        ymin, ymax = axis.get_ylim(); yspan = max(ymax - ymin, 1.0)

        for iso in self._isotopes:
            stem = iso['safe']; name = iso['name']
            dm   = float(p[f"dm_{stem}"].value) if (self.allow_shift and f"dm_{stem}" in p) else 0.0
            start_idx, end_idx = iso['start_idx'], iso['end_idx']
            block = self._pulses[start_idx:end_idx+1]
            if not block:
                continue
            q_best = max(block, key=lambda q: float(q.get('ratio', 0.0)))
            mu_lbl = (1.0 + dg) * float(q_best['mu0']) + d0 + dm
            if xx[0] <= mu_lbl <= xx[-1]:
                y_at  = float(np.interp(mu_lbl, xx, ytot))
                axis.text(mu_lbl, y_at + 0.04 * yspan, name,
                        ha='center', va='bottom', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6))

        # legend once, at the end
        axis.legend(loc='best', frameon=False)

        # stash for GUI and return
        fitln_total.components = sub_lines
        fitln_total.component_data = {'x': xx, 'ytot': ytot}
        fitln_total._isotopes = [iso['name'] for iso in self._isotopes]
        return fitln_total


class AlphaMultiEMGSigmaFitBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, **cfg):
        # cfg can include: shape_file, calib_a, calib_b, allow_shift, shift_bound
        # if not self._instance:
            # self._instance = AlphaMultiEMGSigmaFit(**cfg)
        # return self._instance
        return AlphaMultiEMGSigmaFit(**cfg)
