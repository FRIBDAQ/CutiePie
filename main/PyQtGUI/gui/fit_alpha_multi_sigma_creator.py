#!/usr/bin/env python
# fit_alpha_multi_sigma_creator.py
#
# AlphaMultiEMG: Sum of many EMG sub-peaks grouped by isotope.
#   - Loads shapes from a text/CSV file with rows:
#     isotope, half_life, energy_keV, percent, sigma, tau1, tau2, eta, flag
#   - Ratios fixed within each isotope from "percent"
#   - Fit: A_<isotope> (>=0), optional dm_<isotope>, global (sigma,tau1,tau2,eta),
#          optional per-isotope shape scales (s_sigma,s_tau1,s_tau2)
#   - Bin integration (GL3/GL7), Poisson weights, optional IRLS.
#
# wmode: 0=unweighted, 1=Poisson(data), 2=Poisson(model, IRLS)
# Defaults (when GUI tail omits): bw=median(dx), wmode=1

import sys, os, csv, re
from datetime import datetime
sys.path.append(os.getcwd())

import numpy as np
from lmfit import Model, Parameters, fit_report
from scipy.special import erfcx, erfc

# Keep import so the factory can discover this module
import fit_factory  # noqa: F401

# Optional GUI-abort
try:
    from PyQt5.QtWidgets import QApplication
except Exception:
    QApplication = None

# ---- Gauss–Legendre quadrature (bin integration) ------------------------------
_GL7_T = np.array([0.0, -0.4058451513773972, 0.4058451513773972,
                   -0.7415311855993945, 0.7415311855993945,
                   -0.9491079123427585, 0.9491079123427585], dtype=float)
_GL7_W = np.array([0.4179591836734694,
                   0.3818300505051189, 0.3818300505051189,
                   0.2797053914892766, 0.2797053914892766,
                   0.1294849661688697, 0.1294849661688697], dtype=float)

_GL3_T = np.array([0.0, -0.7745966692, 0.7745966692], dtype=float)
_GL3_W = np.array([0.8888888889, 0.5555555556, 0.5555555556], dtype=float)

USE_GL3 = True  # True for speed, switch to False for GL7 accuracy
_INV_SQRT2 = 1.0 / np.sqrt(2.0)

IRLS_MAX_ITERS = 6
IRLS_IMPROVE   = 1e-3

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

# ---- EMG pieces ----------------------------------------------------------------
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

def _peak_binned(x, A, mu, s, t1, t2, eta, bw):
    return _bin_integral(_emg_two_tail_stable, x, bw, A, mu, s, t1, t2, eta)

# ---- Utilities -----------------------------------------------------------------
def _safe_name(s):
    return re.sub(r'[^A-Za-z0-9_]+', '_', str(s).strip())

def _parse_percent(p):
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
    TXT rows:
      isotope, half_life, energy_keV, percent, sigma, tau1, tau2, eta, flag
    flag in {'s','e','-','*'}; '*' => single line isotope
    Empty isotope cell inherits previous isotope. Lines like ',,,,,,,,' ignored.
    """
    rows = []
    with open(shape_file, 'r', newline='') as f:
        rdr = csv.reader(f, skipinitialspace=True)
        last_iso = ""
        for raw in rdr:
            if not raw or all(not str(x).strip() for x in raw):  # blank
                continue
            if str(raw[0]).strip().startswith("#"):              # comment
                continue
            raw = [str(x).strip() for x in raw]
            if len(raw) < 9:
                raw += [""] * (9 - len(raw))

            iso = raw[0] or last_iso
            if not iso:
                continue
            last_iso = iso

            try:
                E = float(raw[2])
            except Exception:
                continue  # malformed / separator line

            pct   = _parse_percent(raw[3])
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

    # Group contiguous rows by isotope (respect '*' singletons)
    isotopes, pulses = [], []
    i = 0
    while i < len(rows):
        iso_name = rows[i]['iso']
        safe     = _safe_name(iso_name)
        if rows[i]['flag'] == '*':
            start = end = i
        else:
            start = i
            j = i
            while j + 1 < len(rows) and rows[j+1]['iso'] == iso_name and rows[j+1]['flag'] != 's':
                j += 1
            end = j

        isotopes.append(dict(name=iso_name, safe=safe, start_idx=len(pulses), end_idx=None))

        first_pct = rows[start]['pct'] if rows[start]['pct'] > 0 else 1.0
        for k in range(start, end + 1):
            r = rows[k]
            ratio = 1.0 if (start == end) else ((r['pct'] / first_pct) if first_pct > 0 else 1.0)
            pulses.append(dict(
                iso_idx=len(isotopes) - 1, name=iso_name, safe=safe,
                E=r['E'], mu0=r['mu0'], pct=r['pct'], ratio=ratio,
                sigma=r['sigma'], tau1=r['tau1'], tau2=r['tau2'], eta=r['eta'],
            ))

        isotopes[-1]['end_idx'] = len(pulses) - 1
        i = end + 1

    return isotopes, pulses

# ---- Helpers that remove duplication -------------------------------------------
def _fmt_val_err(v, e, digs=6):
    try:
        v = float(v)
    except Exception:
        return str(v)
    ok = (e is not None) and np.isfinite(e)
    if not ok:
        return f"{v:.{digs}g}"
    e = float(e)
    if v != 0.0 and np.isfinite(v):
        pct = abs(e / v) * 100.0
        return f"{v:.{digs}g} ± {e:.3g} ({pct:.2f}%)"
    return f"{v:.{digs}g} ± {e:.3g}"

def _parse_tail_bounds(shape_bounds, tail):
    """Cap upper bounds for sigma/tau1/tau2 if given in tail[2:5]."""
    def cap(key, idx):
        try:
            z = None if tail[idx] is None else float(tail[idx])
        except Exception:
            z = None
        if (z is not None) and np.isfinite(z) and (z > 0.0):
            lo, _ = shape_bounds.get(key, (1.0, 2000.0))
            shape_bounds[key] = (max(1.0, float(lo)), float(z))
    cap("sigma", 2); cap("tau1", 3); cap("tau2", 4)

def _extract_window(x, y, xmin, xmax):
    if np.isfinite(xmin) and np.isfinite(xmax):
        m = (x >= xmin) & (x <= xmax)
        return x[m], y[m]
    return x, y

def _weights_from(y):
    return 1.0 / np.sqrt(np.clip(y, 1.0, None))

def _get(params, key, default=0.0):
    """Works with lmfit.Parameters or a plain dict of floats."""
    try:
        v = params.get(key, default)
    except Exception:
        v = default
    try:
        return float(getattr(v, 'value', v))
    except Exception:
        return float(default)

# Given (params, iso, q, globals, scales) → effective (A, mu, sigma, tau1, tau2, eta)
def _effective_subpeak(params, iso_name, iso_stem, q, allow_shift, use_global):
    A_iso = _get(params, f"A_{iso_stem}", 0.0)
    if A_iso <= 0.0:
        return None

    dm = _get(params, f"dm_{iso_stem}", 0.0) if allow_shift else 0.0
    d0 = _get(params, 'd0', 0.0)
    dg = _get(params, 'dg', 0.0)

    mu = (1.0 + dg) * float(q['mu0']) + d0 + dm
    A  = A_iso * float(q['ratio'])

    # per-isotope scales (default 1)
    sS = _get(params, f"s_sigma_{iso_stem}", 1.0)
    s1 = _get(params, f"s_tau1_{iso_stem}",  1.0)
    s2 = _get(params, f"s_tau2_{iso_stem}",  1.0)

    if use_global:
        sigma0 = _get(params, 'sigma')
        t10    = _get(params, 'tau1')
        t20    = _get(params, 'tau2')
        eta    = _get(params, 'eta')
        sigma, t1, t2 = sigma0 * sS, t10 * s1, t20 * s2
    else:
        eta   = float(q['eta'])
        sigma = float(q['sigma']) * sS
        t1    = float(q['tau1'])  * s1
        t2    = float(q['tau2'])  * s2

    return dict(isotope=iso_name, A=A, mu=mu, sigma=sigma, tau1=t1, tau2=t2, eta=eta, E=q['E'])

# ---- The fitter -----------------------------------------------------------------
class AlphaMultiEMGSigmaFit:
    def __init__(self,
                 shape_file,
                 calib_a=7.1195126, calib_b=-7029.0,
                 calibration_file=None,
                 allow_shift=True, shift_bound=0.0,
                 fix_ratios=True,
                 fit_global_shapes=True,
                 fit_iso_shape_scales=True,
                 iso_scale_bounds=None,
                 shape_bounds=None):

        self.shape_file = shape_file

        if isinstance(calibration_file, (tuple, list)) and len(calibration_file) == 2:
            self.calib_a, self.calib_b = map(float, calibration_file)
            self.calibration_file = None
        else:
            self.calib_a = float(calib_a)
            self.calib_b = float(calib_b)
            self.calibration_file = calibration_file if isinstance(calibration_file, str) else None

        self.allow_shift = bool(allow_shift)
        self.shift_bound = float(shift_bound)
        self.fix_ratios  = bool(fix_ratios)

        self.fit_global_shapes      = bool(fit_global_shapes)
        self.fit_iso_shape_scales   = bool(fit_iso_shape_scales)
        if self.fit_iso_shape_scales and not self.fit_global_shapes:
            self.fit_iso_shape_scales = False  # scales only make sense with global shapes

        self.shape_bounds = shape_bounds or {
            "sigma": (1.0, 2000.0),
            "tau1":  (1.0, 2000.0),
            "tau2":  (1.0, 2000.0),
            "eta":   (0.0, 1.0),
        }
        self.iso_scale_bounds = iso_scale_bounds or {
            "s_sigma": (0.5, 1.5),
            "s_tau1":  (0.5, 1.5),
            "s_tau2":  (0.5, 1.5),
        }

        if not os.path.isfile(self.shape_file):
            raise FileNotFoundError(f"shape_file not found: {self.shape_file}")

    # ---- public API used by GUI ----
    def start(self, x, y, xmin, xmax, fitpar, axis, fit_results):
        self._isotopes, self._pulses = _load_shapes(self.shape_file, self.calib_a, self.calib_b)

        # Clean & window
        mfin = np.isfinite(x) & np.isfinite(y)
        x = np.asarray(x)[mfin]
        y = np.asarray(y)[mfin]
        if x.size < 5:
            fit_results.setPlainText("Not enough data to fit.")
            return None

        x, y = _extract_window(x, y, xmin, xmax)
        if x.size < 5:
            fit_results.setPlainText("Not enough data in selected window.")
            return None

        dx_med = float(np.median(np.diff(x))) if x.size > 1 else 1.0

        # Defaults (single source of truth), with GUI tail override
        D0_DEF, DG_DEF, DM_DEF = 1000.0, 2e-2, self.shift_bound
        bw, wmode = dx_med, 1

        tail = (list(fitpar)[-8:] + [None]*8)[:8] if fitpar is not None else [None]*8
        # indices: [bw, wmode, sigma_ub, tau1_ub, tau2_ub, |d0|, |dg|, |dm*|]

        # Override bw
        try:
            bw_in = None if tail[0] is None else float(tail[0])
        except Exception:
            bw_in = None
        if (bw_in is not None) and np.isfinite(bw_in) and (bw_in >= 0.0):
            bw = bw_in

        # Override wmode
        try:
            wm_in = None if tail[1] is None else int(round(float(tail[1])))
        except Exception:
            wm_in = None
        if wm_in in (0, 1, 2):
            wmode = wm_in

        # Bounds caps for global shapes
        _parse_tail_bounds(self.shape_bounds, tail)

        # |d0|, |dg|, |dm*|
        def _get_abs(idx, default):
            try:
                z = None if tail[idx] is None else float(tail[idx])
            except Exception:
                z = None
            if z is None:
                return float(default)
            return 0.0 if z == 0.0 else abs(float(z))

        d0_abs = _get_abs(5, D0_DEF)
        dg_abs = _get_abs(6, DG_DEF)
        dm_abs = (float(self.shift_bound) if tail[7] is None else float(tail[7]))
        self.allow_shift = bool(self.allow_shift and dm_abs != 0.0)

        # ---------------- Parameters ----------------
        pars = Parameters()

        # Global shapes (if enabled), robust seeds from file medians
        if self.fit_global_shapes:
            med = lambda k: float(np.median([p[k] for p in self._pulses]))
            pars.add('sigma', value=med('sigma') or 200.0, **dict(min=self.shape_bounds["sigma"][0], max=self.shape_bounds["sigma"][1], vary=True))
            pars.add('tau1',  value=med('tau1')  or 80.0, **dict(min=self.shape_bounds["tau1"][0],  max=self.shape_bounds["tau1"][1],  vary=True))
            pars.add('tau2',  value=med('tau2')  or 300.0,**dict(min=self.shape_bounds["tau2"][0],  max=self.shape_bounds["tau2"][1],  vary=True))
            pars.add('eta',   value=med('eta')   or 0.7,  **dict(min=self.shape_bounds["eta"][0],   max=self.shape_bounds["eta"][1],   vary=True))

        # d0, dg
        pars.add('d0', value=0.0, vary=(d0_abs > 0.0), min=-d0_abs, max=+d0_abs)
        pars.add('dg', value=0.0, vary=(dg_abs > 0.0), min=-dg_abs, max=+dg_abs)

        # Per-isotope A and dm (+ optional per-isotope scales)
        for k, iso in enumerate(self._isotopes):
            stem = iso['safe']
            mu0s  = [q['mu0']  for q in self._pulses if q['iso_idx'] == k]
            sig0s = [q['sigma'] for q in self._pulses if q['iso_idx'] == k]
            mu0i  = float(np.median(mu0s))
            sig0  = float(np.median(sig0s)) if sig0s else dx_med
            W     = max(8.0*sig0, 400.0)
            mwin  = (x >= mu0i - W) & (x <= mu0i + W)
            Aseed = float(np.trapz(np.clip(y[mwin], 0, None), x[mwin])) if np.any(mwin) else 1.0

            pars.add(f"A_{stem}", value=max(Aseed, 1.0), min=0.0)

            if self.allow_shift:
                dm0 = 0.0
                if np.count_nonzero(mwin) > 5:
                    xxw, yyw = x[mwin], y[mwin]
                    dm0 = float(xxw[np.argmax(yyw)] - mu0i)
                pars.add(f"dm_{stem}", value=float(np.clip(dm0, -dm_abs, +dm_abs)),
                         min=-dm_abs, max=+dm_abs, vary=True)

            if self.fit_iso_shape_scales:
                pars.add(f"s_sigma_{stem}", value=1.0, min=self.iso_scale_bounds["s_sigma"][0], max=self.iso_scale_bounds["s_sigma"][1], vary=True)
                pars.add(f"s_tau1_{stem}",  value=1.0, min=self.iso_scale_bounds["s_tau1"][0],  max=self.iso_scale_bounds["s_tau1"][1],  vary=True)
                pars.add(f"s_tau2_{stem}",  value=1.0, min=self.iso_scale_bounds["s_tau2"][0],  max=self.iso_scale_bounds["s_tau2"][1],  vary=True)

        pars.add('bw', value=max(float(bw), 0.0), vary=False, min=0.0)

        # ---------------- Model ----------------
        pulses = self._pulses
        allow_shift = self.allow_shift
        use_global  = self.fit_global_shapes

        def _sum_multi_binned(x, **kw):
            bw_loc = float(kw.get('bw', dx_med))
            def _sum_multi(x):
                out = np.zeros_like(x, dtype=float)
                for p in pulses:
                    iso = self._isotopes[p['iso_idx']]
                    sp = _effective_subpeak(kw, iso['name'], iso['safe'], p, allow_shift, use_global)
                    if sp is None:
                        continue
                    out += _emg_two_tail_stable(x, sp['A'], sp['mu'], sp['sigma'], sp['tau1'], sp['tau2'], sp['eta'])
                return out
            return _bin_integral(_sum_multi, x, bw_loc)

        model = Model(_sum_multi_binned, independent_vars=['x'])

        # Data (optional decimation)
        stride = 1
        xf, yf = x[::stride], y[::stride]

        # Initial weights
        w_fit = None if wmode == 0 else _weights_from(yf)

        # Abort callback
        _should_abort = getattr(self, "_should_abort", None)
        def _iter_cb(params=None, iter=None, resid=None, *args, **kwargs):
            if QApplication is not None:
                QApplication.processEvents()
            return bool(_should_abort and _should_abort())

        # First fit
        res = model.fit(yf, params=pars, x=xf, method='least_squares',
                        weights=w_fit, fit_kws=dict(loss='linear'),
                        max_nfev=2000, iter_cb=_iter_cb)

        # IRLS if requested
        if wmode == 2:
            try:
                last = res
                for _ in range(IRLS_MAX_ITERS):
                    yhat = model.eval(last.params, x=xf)
                    w_fit = _weights_from(yhat)
                    new = model.fit(yf, params=last.params.copy(), x=xf,
                                    method='least_squares', weights=w_fit,
                                    fit_kws=dict(loss='linear'), max_nfev=1500,
                                    iter_cb=_iter_cb)
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

        if _iter_cb():
            return None

        # ---------------- Reporting ----------------
        yhat_full = model.eval(res.params, x=x)
        ss_res = float(np.sum((y - yhat_full)**2))
        ss_tot = float(np.sum((y - y.mean())**2))
        R2_plain = 1.0 - ss_res / (ss_tot + 1e-16)

        wtxt = {0: "none", 1: "Poisson(data)", 2: "Poisson(model, IRLS)"}[wmode]
        bw_str = _fmt_val_err(res.params['bw'].value, res.params['bw'].stderr)

        d0p = res.params.get('d0'); dgp = res.params.get('dg')
        d0, dg = (float(d0p.value) if d0p else 0.0), (float(dgp.value) if dgp else 0.0)
        d0_str = _fmt_val_err(d0p.value, getattr(d0p, "stderr", None)) if d0p else "n/a"
        dg_str = _fmt_val_err(dgp.value, getattr(dgp, "stderr", None)) if dgp else "n/a"

        # Shapes header
        mode = ("global + per-isotope scales" if (self.fit_global_shapes and self.fit_iso_shape_scales)
                else "global" if self.fit_global_shapes
                else "frozen (file σ,τ₁,τ₂,η)")
        shape_src = os.path.abspath(self.shape_file)
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(self.shape_file)).isoformat(timespec="seconds")
            shape_head = f"[shapes] mode = {mode} ; source = {shape_src} ; mtime = {mtime}"
        except Exception:
            shape_head = f"[shapes] mode = {mode} ; source = {shape_src}"

        header = (shape_head +
                  f"\nNotes: bandwidth = {bw_str} ; weighting = {wtxt}\n"
                  f"Global shift d0 = {d0_str} ; Global gain dg = {dg_str}\n"
                  f"Bounds: |d0|≤{d0_abs:g}, |dg|≤{dg_abs:g}, |dm_*|≤{dm_abs:g}")

        if self.fit_global_shapes:
            fmt = lambda pn: _fmt_val_err(res.params[pn].value, getattr(res.params[pn], "stderr", None))
            header += ("\nGlobal shapes (fitted): "
                       f"sigma={fmt('sigma')}, tau1={fmt('tau1')}, tau2={fmt('tau2')}, eta={fmt('eta')}")

        stats = (f"[stats] chi-square={res.chisqr:.3f} ; reduced chi-square={res.redchi:.3f} ; "
                 f"dof={res.nfree} (N={res.ndata}, k={res.nvarys}) ; R^2 (plain)={R2_plain:.4f}")

        # Per-isotope block summary
        iso_lines = ["Per-isotope parameters (A per isotope; ratios fixed):",
                     f"  allow_shift={self.allow_shift} ; fix_ratios={self.fix_ratios}"]

        if self.fit_iso_shape_scales:
            iso_lines.append("\nPer-isotope shape scales (multipliers on baseline σ, τ1, τ2):")
            for iso in self._isotopes:
                stem = iso['safe']; name = iso['name']
                def fmt_s(pn):
                    pp = res.params.get(pn)
                    return _fmt_val_err(pp.value, getattr(pp, "stderr", None)) if pp else "1"
                iso_lines.append(f"{name:>10}: sσ={fmt_s(f's_sigma_{stem}')}, "
                                 f"sτ1={fmt_s(f's_tau1_{stem}')}, sτ2={fmt_s(f's_tau2_{stem}')}")

        for iso in self._isotopes:
            stem = iso['safe']; name = iso['name']
            pA   = res.params.get(f"A_{stem}")
            pDM  = res.params.get(f"dm_{stem}") if self.allow_shift else None
            A_str  = _fmt_val_err(pA.value, getattr(pA, "stderr", None)) if pA else "n/a"
            dm_str = _fmt_val_err(pDM.value, getattr(pDM, "stderr", None)) if pDM else None

            idx0, idx1 = iso['start_idx'], iso['end_idx']
            blockE  = [f"{self._pulses[k]['E']:.1f}" for k in range(idx0, idx1+1)]
            blockRt = [float(self._pulses[k]['ratio']) for k in range(idx0, idx1+1)]
            base = f"{name:>10}:  A={A_str}"
            if dm_str is not None: base += f"   dm={dm_str}"
            base += f"   (subpeaks={len(blockE)}; ratios=[{', '.join(f'{r:.3f}' for r in blockRt)}]; E=[{', '.join(blockE)} keV])"
            iso_lines.append(base)

        # Per-subpeak detailed list (single pass using helper)
        subpeak_lines = ["\nPer-subpeak parameters:"]
        for iso in self._isotopes:
            for kk in range(iso['start_idx'], iso['end_idx'] + 1):
                sp = _effective_subpeak(res.params, iso['name'], iso['safe'], self._pulses[kk], self.allow_shift, self.fit_global_shapes)
                if sp is None: continue
                subpeak_lines.append(
                    f"{iso['name']:>10}  E={sp['E']:.1f} keV : "
                    f"A={sp['A']:.6g}, μ={sp['mu']:.3f}, σ={sp['sigma']:.3g}, "
                    f"τ1={sp['tau1']:.3g}, τ2={sp['tau2']:.3g}, η={sp['eta']:.3g}"
                )

        # Notice if anything pegged at bounds
        def _at_bound(p):
            return p and (abs(p.value - p.min) < 1e-12 or abs(p.value - p.max) < 1e-12)

        pegged_dm = [iso['name'] for iso in self._isotopes if _at_bound(res.params.get(f"dm_{iso['safe']}"))]
        pegged_g  = [nm for nm in ('d0','dg') if _at_bound(res.params.get(nm))]
        notice = ""
        if pegged_dm or pegged_g:
            who = f"dm_* at bounds: {', '.join(pegged_dm)}" if pegged_dm else ""
            gbl = f"globals at bounds: {', '.join(pegged_g)}" if pegged_g else ""
            join = " ; " if (who and gbl) else ""
            notice = f"\n[notice] {who}{join}{gbl}"

        fit_results.setPlainText(
            header + "\n" + stats + "\n\n" +
            "\n".join(iso_lines) + "\n" +
            "\n".join(subpeak_lines) +
            notice + "\n\n" +
            fit_report(res, show_correl=True)
        )

        # ---------------- CSV (per-subpeak) ----------------
        now = datetime.now().isoformat(timespec="seconds")
        subrows = []
        for iso in self._isotopes:
            for kk in range(iso['start_idx'], iso['end_idx'] + 1):
                sp = _effective_subpeak(res.params, iso['name'], iso['safe'], self._pulses[kk], self.allow_shift, self.fit_global_shapes)
                if sp is None: continue
                subrows.append({
                    "timestamp": now,
                    "isotope": iso['name'],
                    "E_keV": float(sp['E']),
                    "A_sub": float(sp['A']),
                    "mu": float(sp['mu']),
                    "sigma": float(sp['sigma']),
                    "tau1": float(sp['tau1']),
                    "tau2": float(sp['tau2']),
                    "eta": float(sp['eta']),
                    "chi2": float(getattr(res, "redchi", np.nan)),
                    "R2": float(R2_plain),
                    "d0_bound": d0_abs, "dg_bound": dg_abs, "dm_bound": dm_abs,
                })

        subcsv = os.path.join(os.getcwd(), "fit_EMGMultiSigma_subpeaks.csv")
        subfields = ["timestamp","isotope","E_keV","A_sub","mu","sigma","tau1","tau2","eta",
                     "chi2","R2","d0_bound","dg_bound","dm_bound"]
        write_header = (not os.path.exists(subcsv)) or (os.path.getsize(subcsv) == 0)
        with open(subcsv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=subfields)
            if write_header: w.writeheader()
            w.writerows(subrows)

        # ---------------- Plotting ----------------
        xx  = np.linspace(x.min(), x.max(), 1400)
        p   = res.params
        bwv = p['bw'].value if 'bw' in p else float(np.median(np.diff(xx)))
        ytot = model.eval(res.params, x=xx)

        (fitln_total,) = axis.plot(xx, ytot, lw=2.5, color='tab:orange', label='fit total')

        sub_lines = []
        for iso in self._isotopes:
            for kk in range(iso['start_idx'], iso['end_idx'] + 1):
                sp = _effective_subpeak(res.params, iso['name'], iso['safe'], self._pulses[kk], self.allow_shift, self.fit_global_shapes)
                if sp is None: continue
                yj = _peak_binned(xx, sp['A'], sp['mu'], sp['sigma'], sp['tau1'], sp['tau2'], sp['eta'], bwv)
                (ln,) = axis.plot(xx, yj, lw=1.8, ls='--', alpha=0.9, label=f"{iso['name']} {sp['E']:.1f} keV")
                sub_lines.append(ln)

        axis.relim(); axis.autoscale_view()
        ymin, ymax = axis.get_ylim(); yspan = max(ymax - ymin, 1.0)
        dy = 0.035 * yspan

        for iso in self._isotopes:
            dm_p = res.params.get(f"dm_{iso['safe']}"); dm_val = float(dm_p.value) if dm_p else 0.0
            bump = 0
            for kk in range(iso['start_idx'], iso['end_idx'] + 1):
                q = self._pulses[kk]
                mu = (1.0 + dg) * float(q['mu0']) + d0 + dm_val
                if xx[0] <= mu <= xx[-1]:
                    y_at = float(np.interp(mu, xx, ytot))
                    axis.text(mu, y_at + dy + bump*0.6*dy,
                              f"{iso['name']} {q['E']:.0f}",
                              ha='center', va='bottom', fontsize=9,
                              bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6))
                    bump += 1

        # Sort legend entries by energy, keep total first
        handles, labels = axis.get_legend_handles_labels()
        def _extract_E(lbl: str) -> float:
            m = re.search(r'(\d+(?:\.\d+)?)\s*keV', lbl)
            return float(m.group(1)) if m else float('inf')
        triples, total = [], []
        for h, lbl in zip(handles, labels):
            if lbl.strip().lower() == 'fit total':
                total.append((-float('inf'), h, lbl))
            else:
                triples.append((_extract_E(lbl), h, lbl))
        triples.sort(key=lambda t: t[0])
        axis.legend([h for _, h, _ in total + triples],
                    [lbl for _, _, lbl in total + triples],
                    loc='best', frameon=False)

        # Stash for GUI
        fitln_total.components = sub_lines
        fitln_total.component_data = {'x': xx, 'ytot': ytot}
        fitln_total._isotopes = [iso['name'] for iso in self._isotopes]
        return fitln_total


class AlphaMultiEMGSigmaFitBuilder:
    def __init__(self):
        self._instance = None
    def __call__(self, **cfg):
        return AlphaMultiEMGSigmaFit(**cfg)
