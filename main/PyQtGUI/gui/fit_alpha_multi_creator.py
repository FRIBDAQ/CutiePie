#!/usr/bin/env python
# fit_alpha_multi_creator.py
#
# AlphaMultiEMG: Sum of many EMG sub-peaks grouped by isotope.
#   - Loads shapes from a text/CSV file with rows:
#     isotope, half_life, energy_keV, percent, sigma, tau1, tau2, eta, flag
# where group_flag is one of: s (start), e (end), - (middle) - Known
# per-subpeak amplitude ratios from Percent within each isotope. - User
# FITTED: one amplitude A_<isotope> per isotope (>=0).
#   fit_factory.register("AlphaMultiEMG", AlphaMultiEMGFitBuilder(),
#                        shape_file="/path/to/shapes.txt",
#                        calib_a=6.8941013584, calib_b=-4943.2400523,
#                        allow_shift=True, shift_bound=300.0)
#
# If GUI doesn’t inject bw/wmode, we auto-pick: bw=median(dx), wmode=2.

import sys, os, csv
sys.path.append(os.getcwd())

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import numpy as np
from lmfit import Model, Parameters, fit_report

# Keep import so the factory can discover this module
import fit_factory  # noqa: F401

from fit_alpha_base import (
    _GL7_T, _GL7_W, _GL3_T, _GL3_W, _safe_name,
    _parse_percent, _emg_two_tail_stable,
)

USE_GL3 = True  # set True for speed

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

def _peak_binned(x, A, mu, sigma, t1, t2, eta, bw):
    return _bin_integral(_emg_two_tail_stable, x, bw, A, mu, sigma, t1, t2, eta)

def _load_shapes(shape_file, a, b):
    """NEW TXT format (per line): isotope, half_life, energy_keV, percent,
    sigma, tau1, tau2, eta, flag Notes: • Empty isotope (',,') inherits the
    previous isotope. • flag ∈ {'s','e','-','*'}; '*' = single-line isotope."""
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

class AlphaMultiEMGFit:
    def __init__(self,
                 shape_file,
                 calib_a=7.1195126, calib_b=-7029.0,
                 allow_shift=True, shift_bound=100.0,
                 fix_ratios=True,
                 wmode_default=2):
        self.shape_file    = shape_file
        self.calib_a       = float(calib_a)
        self.calib_b       = float(calib_b)
        self.allow_shift   = bool(allow_shift)
        self.shift_bound   = float(shift_bound)
        self.fix_ratios    = bool(fix_ratios)
        self.wmode_default = int(wmode_default)

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

        # Pull bw/wmode from fitpar if present; else default to IRLS
        fp = list(fitpar) if fitpar is not None else []
        bw    = dx
        wmode = getattr(self, "wmode_default", 2)  # default = 2

        if len(fp) >= 2:
            try:
                cand_bw = float(fp[-2])
                if np.isfinite(cand_bw) and cand_bw > 0:
                    bw = cand_bw
            except Exception:
                pass
            try:
                cand_wm = int(round(float(fp[-1])))
                if cand_wm in (0, 1, 2):
                    wmode = cand_wm
            except Exception:
                pass

        if not (isinstance(bw, float) and np.isfinite(bw) and bw > 0):
            bw = dx
        if wmode not in (0, 1, 2):
            wmode = 2

        # Build parameters
        pars = Parameters()
        pars.add('d0', value=0.0, min=-5000, max=+5000)      # global μ offset
        pars.add('dg', value=0.0, min=-2e-2, max=+2e-2)      # tiny global gain (±0.5%)

        
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
                dm_lo = -self.shift_bound; dm_hi = +self.shift_bound
                pars.add(f"dm_{stem}", value=float(np.clip(dm0, dm_lo, dm_hi)),
                        min=dm_lo, max=dm_hi, vary=True)


        # fixed bin width
        pars.add('bw', value=max(float(bw), 0.0), vary=False, min=0.0)


        # Closure over pulses/isotopes
        pulses = self._pulses
        allow_shift = self.allow_shift

        def _sum_multi_binned(x, **kw):
            bw_loc = kw.get('bw', dx)

            def _sum_multi(x):
                out = np.zeros_like(x, dtype=float)
                d0 = float(kw.get('d0', 0.0))
                dg = float(kw.get('dg', 0.0))
                for p in pulses:
                    stem = self._isotopes[p['iso_idx']]['safe']
                    A_iso = float(kw.get(f"A_{stem}", 0.0))
                    dm    = float(kw.get(f"dm_{stem}", 0.0)) if allow_shift else 0.0
                    A  = A_iso * p['ratio']
                    mu = (1.0 + dg) * p['mu0'] + d0 + dm
                    out += _emg_two_tail_stable(x, A, mu, p['sigma'], p['tau1'], p['tau2'], p['eta'])
                return out

            # b0 = float(kw.get('b0', 0.0))
            # b1 = float(kw.get('b1', 0.0))

            # return _bin_integral(_sum_multi, x, bw_loc) + (b0 + b1 * (x - x.mean()))
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

        # single initial fit
        res = model.fit(yf, params=pars, x=xf, method='least_squares',
                        weights=w_fit, fit_kws=fit_kws, max_nfev=2000)

        # IRLS: switch weights to model
        if wmode == 2:
            try:
                last = res
                for _ in range(IRLS_MAX_ITERS):
                    yhat = model.eval(last.params, x=xf)
                    w_fit = 1.0 / np.sqrt(np.clip(yhat, 1.0, None))
                    new = model.fit(yf, params=last.params.copy(), x=xf,
                                    method='least_squares', weights=w_fit,
                                    fit_kws=fit_kws, max_nfev=1500)
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
        # ---------- Reporting (compact, with uncertainties) ----------
        def _fmt_val_err(v, e, digs=6):
            try: v = float(v)
            except Exception: return str(v)
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
        d0p = res.params.get('d0'); dgp = res.params.get('dg')
        d0_str = _fmt_val_err(d0p.value, getattr(d0p, "stderr", None)) if d0p else "n/a"
        dg_str = _fmt_val_err(dgp.value, getattr(dgp, "stderr", None)) if dgp else "n/a"

        header = (f"Notes: bandwidth = {bw_str} ; weighting = {wtxt}\n"
                  f"Global shift d0 = {d0_str} ; Global gain dg = {dg_str}")
        stats  = (f"[stats] chi-square={res.chisqr:.3f} ; reduced chi-square={res.redchi:.3f} ; "
                  f"dof={res.nfree} (N={res.ndata}, k={res.nvarys}) ; R^2 (plain)={R2_plain:.4f}")

        iso_lines = []
        iso_lines.append("Per-isotope parameters (A per isotope; ratios fixed):")
        iso_lines.append(f"  allow_shift={self.allow_shift} ; fix_ratios={self.fix_ratios}")
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
            if dm_str is not None:
                base += f"   dm={dm_str}"
            base += f"   (subpeaks={len(blockE)}; ratios=[{rat_str}]; E=[{E_str}])"
            iso_lines.append(base)
        hits = []
        for iso in self._isotopes:
            stem = iso['safe']
            pDM = res.params.get(f"dm_{stem}")
            if pDM and (abs(pDM.value - pDM.min) < 1e-9 or abs(pDM.value - pDM.max) < 1e-9):
                hits.append(iso['name'])
        if hits:
            fit_results.append(f"\n[notice] These dm’s hit the bound ({self.shift_bound}): " + ", ".join(hits))

        fit_results.setPlainText(
            header + "\n" + stats + "\n\n" + "\n".join(iso_lines) + "\n\n" +
            fit_report(res, show_correl=True)
        )

        # ---------------- Plotting (total + each sub-peak dashed + labels) ----------------
        xx  = np.linspace(x.min(), x.max(), 1400)
        p   = res.params
        bwv = p['bw'].value if 'bw' in p else float(np.median(np.diff(xx)))
        ytot = model.eval(res.params, x=xx)

        # TOTAL: fixed color so it doesn't change with the color cycle
        (fitln_total,) = axis.plot(xx, ytot, lw=2.5, color='tab:orange', label='fit total')

        # convenience for centroid shifts
        d0 = float(p['d0'].value) if 'd0' in p else 0.0
        dg = float(p['dg'].value) if 'dg' in p else 0.0

        # --- one dashed line per SUB-PEAK (11 total for your file) ---
        sub_lines = []
        for iso in self._isotopes:
            stem = iso['safe']
            name = iso['name']
            A_iso = float(p[f"A_{stem}"].value)
            dm    = float(p[f"dm_{stem}"].value) if self.allow_shift and f"dm_{stem}" in p else 0.0

            for kk in range(iso['start_idx'], iso['end_idx'] + 1):
                q   = self._pulses[kk]
                A   = A_iso * float(q['ratio'])
                mu  = (1.0 + dg) * float(q['mu0']) + d0 + dm
                yj  = _peak_binned(xx, A, mu, q['sigma'], q['tau1'], q['tau2'], q['eta'], bwv)

                # label shows isotope and the line energy for clarity
                label = f"{name} {q['E']:.1f} keV"
                (ln,) = axis.plot(xx, yj, lw=1.8, ls='--', alpha=0.9, label=label)
                sub_lines.append(ln)

        # --- labels (same as before) ---
        axis.relim(); axis.autoscale_view()
        ymin, ymax = axis.get_ylim(); yspan = max(ymax - ymin, 1.0)
        for iso in self._isotopes:
            stem = iso['safe']; name = iso['name']
            dm   = float(p[f"dm_{stem}"].value) if self.allow_shift and f"dm_{stem}" in p else 0.0
            start, end = iso['start_idx'], iso['end_idx']
            block = self._pulses[start:end+1]
            if not block:
                continue
            q_best = max(block, key=lambda q: float(q.get('ratio', 0.0)))
            mu_lbl = (1.0 + dg) * float(q_best['mu0']) + d0 + dm
            if xx[0] <= mu_lbl <= xx[-1]:
                y_at  = float(np.interp(mu_lbl, xx, ytot))
                y_lab = y_at + 0.04 * yspan
                axis.text(mu_lbl, y_lab, name,
                        ha='center', va='bottom', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6))

        # full legend (1 solid + 11 dashed entries)
        axis.legend(loc='best', frameon=False)

        # stash
        fitln_total.components = sub_lines
        fitln_total.component_data = {'x': xx, 'ytot': ytot}
        fitln_total._isotopes = [iso['name'] for iso in self._isotopes]
        fitln_total.chi2 = float(getattr(res, "chisqr", np.nan))
        fitln_total.redchi = float(getattr(res, "redchi", np.nan))
        fitln_total.ndof = int(getattr(res, "nfree", 0))
        return fitln_total



        '''
        # ---------------- Plotting (total only + isotope labels) ----------------
        xx  = np.linspace(x.min(), x.max(), 1400)
        p   = res.params
        ytot = model.eval(res.params, x=xx)
        (fitln_total,) = axis.plot(xx, ytot, lw=2.5, label='fit total')

        # prepare y-offset for labels
        axis.relim(); axis.autoscale_view()
        ymin, ymax = axis.get_ylim()
        yspan = max(ymax - ymin, 1.0)

        # use fitted global offset/gain for label positions
        d0 = float(p['d0'].value) if 'd0' in p else 0.0
        dg = float(p['dg'].value) if 'dg' in p else 0.0

        for iso in self._isotopes:
            stem = iso['safe']
            name = iso['name']
            dm   = float(p[f"dm_{stem}"].value) if self.allow_shift and f"dm_{stem}" in p else 0.0

            # strongest sub-peak in this isotope (by ratio)
            start, end = iso['start_idx'], iso['end_idx']
            block = self._pulses[start:end+1]
            if not block:
                continue
            q_best = max(block, key=lambda q: float(q.get('ratio', 0.0)))

            # label μ must include global gain/offset + isotope shift
            mu_lbl = (1.0 + dg) * float(q_best['mu0']) + d0 + dm
            if mu_lbl < xx[0] or mu_lbl > xx[-1]:
                continue

            y_at  = float(np.interp(mu_lbl, xx, ytot))
            y_lab = y_at + 0.04 * yspan
            axis.text(mu_lbl, y_lab, name,
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6))

        if axis.legend_ is None:
            axis.legend(loc='best', frameon=False)

        fitln_total.components = []
        fitln_total.component_data = {'x': xx, 'ytot': ytot}
        fitln_total._isotopes = [iso['name'] for iso in self._isotopes]
        return fitln_total
        '''


class AlphaMultiEMGFitBuilder:
    def __call__(self, **cfg):
        # cfg can include: shape_file, calib_a, calib_b, allow_shift, shift_bound
        # if not self._instance:
            # self._instance = AlphaMultiEMGFit(**cfg)
        # return self._instance
        return AlphaMultiEMGFit(**cfg)
