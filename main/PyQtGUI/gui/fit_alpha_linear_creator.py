#!/usr/bin/env python
# fit_alpha_linear.py
#
# AlphaEMGLinear: Freeze μ (from calibration), fix σ/τ1/τ2/η, and solve ONLY for
# nonnegative per-isotope amplitudes A_i by (weighted) least squares.
#
# Shapes file format (CSV/TXT, commas OK, blanks allowed):
#   isotope, half_life, energy_keV, percent, sigma, tau1, tau2, eta, flag
# Where:
#   - Empty isotope inherits previous (",,")
#   - flag ∈ {'s','e','-','*'}; '*' = single-line isotope
#   - 'percent' can be "31.6" or "31.6%" (used to set ratios within each isotope)
#
# Registration example (in your main launcher):
#   fitfactory.register_builder('AlphaEMGLinear',
#       fit_alpha_linear.AlphaEMGLinearFitBuilder(),
#       {
#           'shape_file': os.path.join(os.getcwd(), "shapes.txt"),
#           'calib_a': 6.81714733542319,
#           'calib_b': -4702.0,
#           'wmode_default': 1,       # 0=unweighted, 1=Poisson(data), 2=Poisson(model IRLS)
#           'allow_baseline': False,  # if True, fits a nonnegative constant b0 as well
#       })
#
# GUI parameters:
#   - Append [bw, wmode] at the END of fitpar (if provided):
#       bw    = bin width in channels (fallback: median Δx)
#       wmode = 0/1/2 as above
#
# Output:
#   - Text report with chi-square, reduced-chi-square, R^2(plain), and A_i values
#   - Plot with ONE solid total line (tab:orange) and ONE dashed line per sub-peak (11)
#
# Notes:
#   - This module does not use lmfit; it solves a linear NNLS system.
#   - If SciPy is available, uses scipy.optimize.nnls; otherwise falls back to a simple
#     projected-gradient NNLS (good enough for our sizes).

import os, csv, sys
import numpy as np

# Keep import so the factory can discover this module
import fit_factory  # noqa: F401

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from fit_alpha_base import (
    _GL7_T, _GL7_W, _GL3_T, _GL3_W, _INV_SQRT2,
    _safe_name, _parse_percent,
    _emg_one_tail_stable, _emg_two_tail_stable,
)

USE_GL3 = False  # set True for speed

IRLS_MAX_ITERS = 5
IRLS_IMPROVE   = 1e-3  # relative chi2 improvement

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

def _peak_binned(x, A, mu, sigma, t1, t2, eta, bw):
    return _bin_integral(_emg_two_tail_stable, x, bw, A, mu, sigma, t1, t2, eta)

def _load_shapes(shape_file, a, b):
    """
    Returns:
      isotopes: [{name, safe, start_idx, end_idx}]
      pulses  : [{iso_idx, name, safe, E, mu0, pct, ratio, sigma, tau1, tau2, eta}]
    """
    rows = []
    with open(shape_file, 'r', newline='') as f:
        rdr = csv.reader(f, skipinitialspace=True)
        last_iso = ""
        for raw in rdr:
            if not raw or all(not str(x).strip() for x in raw):
                continue
            first = str(raw[0]).strip()
            if first.startswith("#"):
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
                continue

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
            if flag not in ('s','e','-','*'):
                flag = '-'

            mu0 = a * E + b
            rows.append(dict(iso=iso, E=E, pct=pct, flag=flag,
                             sigma=sigma, tau1=tau1, tau2=tau2, eta=eta, mu0=mu0))

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
                iso_idx=len(isotopes) - 1,
                name=iso_name, safe=safe,
                E=r['E'], mu0=r['mu0'], pct=r['pct'], ratio=ratio,
                sigma=r['sigma'], tau1=r['tau1'], tau2=r['tau2'], eta=r['eta'],
            ))

        isotopes[-1]['end_idx'] = len(pulses) - 1
        i = end + 1

    return isotopes, pulses

# ---------------- Weighted NNLS helpers ----------------
def _nnls(X, y):
    """NNLS with SciPy if available, else projected-gradient fallback."""
    try:
        from scipy.optimize import nnls as sp_nnls
        beta, _ = sp_nnls(X, y)
        return beta
    except Exception:
        # Simple projected gradient descent
        beta = np.zeros(X.shape[1], dtype=float)
        Xt = X.T
        t = 1.0 / (np.linalg.norm(X, ord=2)**2 + 1e-12)
        for _ in range(2000):
            grad = Xt @ (X @ beta - y)
            beta_new = beta - t * grad
            beta_new = np.maximum(beta_new, 0.0)
            if np.linalg.norm(beta_new - beta) <= 1e-6 * (np.linalg.norm(beta) + 1e-12):
                beta = beta_new
                break
            beta = beta_new
        return beta

def _weighted_nnls(X, y, w):
    """Solve argmin || sqrt(w)*(X b - y) ||_2 with b>=0."""
    if w is None:
        return _nnls(X, y)
    s = np.sqrt(np.clip(w, 0.0, None))
    Xw = X * s[:, None]
    yw = y * s
    return _nnls(Xw, yw)

# ---------------- The linear fitter ----------------
class AlphaEMGLinearFit:
    def __init__(self,
                 shape_file=os.path.join(os.getcwd(), "shapes_Chand.txt"),
                 calib_a=7.1195126, calib_b=-7029.0,
                 wmode_default=1,
                 allow_baseline=False):
        self.shape_file    = shape_file
        self.calib_a       = float(calib_a)
        self.calib_b       = float(calib_b)
        self.wmode_default = int(wmode_default)
        self.allow_baseline= bool(allow_baseline)

        if not os.path.isfile(self.shape_file):
            raise FileNotFoundError(f"shape_file not found: {self.shape_file}")

        self._isotopes, self._pulses = _load_shapes(self.shape_file, self.calib_a, self.calib_b)

    def start(self, x, y, xmin, xmax, fitpar, axis, fit_results):
        # --- basic hygiene / window ---
        mfin = np.isfinite(x) & np.isfinite(y)
        x = np.asarray(x)[mfin]
        y = np.asarray(y)[mfin]
        if x.size < 5:
            fit_results.setPlainText("Not enough data to fit.")
            return None

        dx = float(np.median(np.diff(x))) if x.size > 1 else 1.0
        if np.isfinite(xmin) and np.isfinite(xmax):
            m = (x >= xmin) & (x <= xmax)
            x, y = x[m], y[m]
            if x.size < 5:
                fit_results.setPlainText("Not enough data in selected window.")
                return None
            dx = float(np.median(np.diff(x))) if x.size > 1 else dx

        # --- options from fitpar: [ ... , bw, wmode ] at the end ---
        bw    = dx
        wmode = self.wmode_default
        if fitpar is not None and len(fitpar) >= 2:
            try:
                bw_cand = float(fitpar[-2])
                if np.isfinite(bw_cand) and bw_cand > 0:
                    bw = bw_cand
            except Exception:
                pass
            try:
                w_cand = int(round(float(fitpar[-1])))
                if w_cand in (0,1,2):
                    wmode = w_cand
            except Exception:
                pass

        # --- build design matrix: one column per isotope template (+ optional baseline) ---
        n = x.size
        mcols = len(self._isotopes) + (1 if self.allow_baseline else 0)
        X = np.zeros((n, mcols), dtype=float)

        # Precompute per-subpeak template at A=1, then sum by isotope with ratios
        # Column ordering: [iso0, iso1, ..., isoK-1, [baseline]]
        for iso_idx, iso in enumerate(self._isotopes):
            col = np.zeros(n, dtype=float)
            for kk in range(iso['start_idx'], iso['end_idx'] + 1):
                q = self._pulses[kk]
                # amplitude = 1 * ratio
                col += _peak_binned(x, 1.0 * q['ratio'], q['mu0'],
                                    q['sigma'], q['tau1'], q['tau2'], q['eta'], bw)
            X[:, iso_idx] = col

        if self.allow_baseline:
            X[:, -1] = 1.0  # nonnegative constant b0

        # --- weights ---
        if wmode == 0:
            w = None
        elif wmode == 1:
            w = 1.0 / np.clip(y, 1.0, None)  # Poisson(data)
        else:
            # start with data; IRLS will update
            w = 1.0 / np.clip(y, 1.0, None)

        # --- solve NNLS (and IRLS if requested) ---
        beta = _weighted_nnls(X, y, w)

        if wmode == 2:
            chisq_prev = np.inf
            for _ in range(IRLS_MAX_ITERS):
                yhat = X @ beta
                w = 1.0 / np.clip(yhat, 1.0, None)  # Poisson(model)
                beta = _weighted_nnls(X, y, w)
                # check relative improvement in χ²
                resid = y - (X @ beta)
                chisq = float(np.sum((w if w is not None else 1.0) * resid**2))
                if abs(chisq - chisq_prev) / max(chisq_prev, 1.0) < IRLS_IMPROVE:
                    break
                chisq_prev = chisq

        # --- stats ---
        yhat = X @ beta
        resid = y - yhat
        if wmode == 0:
            chisq = float(np.sum(resid**2))
        else:
            chisq = float(np.sum(np.clip(w,0,None) * resid**2))
        k = mcols
        dof = max(n - k, 1)
        redchi = chisq / dof
        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - y.mean())**2))
        R2_plain = 1.0 - ss_res / ss_tot

        # --- report ---
        wtxt = {0: "none", 1: "Poisson(data)", 2: "Poisson(model, IRLS)"}[wmode]
        header = f"Notes: bandwidth = {bw:.6g} ; weighting = {wtxt}"
        stats  = (f"[stats] chi-square={chisq:.3f} ; reduced chi-square={redchi:.3f} ; "
                  f"dof={dof} (N={n}, k={k}) ; R^2 (plain)={R2_plain:.4f}")

        lines = []
        for iso_idx, iso in enumerate(self._isotopes):
            Ai = beta[iso_idx]
            lines.append(f"{iso['name']:>10}:  A={Ai:.6g}")
        if self.allow_baseline:
            lines.append(f"{'baseline':>10}:  b0={beta[-1]:.6g}")

        fit_results.setPlainText(header + "\n" + stats + "\n\n"
                                 + "Per-isotope amplitudes (μ,σ,τ fixed):\n"
                                 + "\n".join(lines))

        # ---------------- plotting ----------------
        xx  = np.linspace(x.min(), x.max(), 1400)
        # remake design columns on xx for clean curves
        cols_xx = []
        for iso_idx, iso in enumerate(self._isotopes):
            col = np.zeros_like(xx)
            for kk in range(iso['start_idx'], iso['end_idx'] + 1):
                q = self._pulses[kk]
                col += _peak_binned(xx, 1.0 * q['ratio'], q['mu0'],
                                    q['sigma'], q['tau1'], q['tau2'], q['eta'], bw)
            cols_xx.append(col)

        ytot = np.zeros_like(xx)
        for iso_idx, col in enumerate(cols_xx):
            ytot += beta[iso_idx] * col
        if self.allow_baseline:
            ytot += beta[-1]

        # Solid total with a fixed, recognizable color
        (fitln_total,) = axis.plot(xx, ytot, lw=2.5, color='tab:orange', label='fit total')

        # One dashed curve per SUB-PEAK (11 total for your file)
        sub_lines = []
        for iso_idx, iso in enumerate(self._isotopes):
            A_iso = beta[iso_idx]
            for kk in range(iso['start_idx'], iso['end_idx'] + 1):
                q = self._pulses[kk]
                yj = _peak_binned(xx, A_iso * q['ratio'], q['mu0'],
                                  q['sigma'], q['tau1'], q['tau2'], q['eta'], bw)
                label = f"{iso['name']} {q['E']:.1f} keV"
                (ln,) = axis.plot(xx, yj, lw=1.8, ls='--', alpha=0.9, label=label)
                sub_lines.append(ln)

        axis.legend(loc='best', frameon=False)

        # Attach for GUI bookkeeping
        fitln_total.components = sub_lines
        fitln_total.component_data = {'x': xx, 'ytot': ytot}
        fitln_total._isotopes = [iso['name'] for iso in self._isotopes]
        fitln_total.chi2 = float(chisq)
        fitln_total.redchi = float(redchi)
        fitln_total.ndof = int(dof)

        return fitln_total


class AlphaEMGLinearFitBuilder:
    def __init__(self):
        self._instance = None

    def __call__(self, **cfg):
        if not self._instance:
            self._instance = AlphaEMGLinearFit(**cfg)
        return self._instance
