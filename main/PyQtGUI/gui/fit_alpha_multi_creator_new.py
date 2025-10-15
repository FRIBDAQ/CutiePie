#!/usr/bin/env python
# fit_alpha_multi_creator.py  —  cleaned & vectorized

import os, csv, re, sys
from datetime import datetime
sys.path.append(os.getcwd())

import numpy as np
from lmfit import Model, Parameters, fit_report
from scipy.special import erfcx, erfc

# discoverable by your factory
import fit_factory  # noqa: F401

# ---------- Config: quadrature choice (speed vs accuracy) ----------
USE_GL3 = True  # True = faster (good enough for 1–4 bins); False = GL7 (more accurate)

# ---------- Gauss–Legendre nodes/weights ----------
_GL7_T = np.array([0.0, -0.4058451513773972,  0.4058451513773972,
                          -0.7415311855993945,  0.7415311855993945,
                          -0.9491079123427585,  0.9491079123427585], dtype=float)
_GL7_W = np.array([0.4179591836734694,
                   0.3818300505051189,  0.3818300505051189,
                   0.2797053914892766,  0.2797053914892766,
                   0.1294849661688697,  0.1294849661688697], dtype=float)
_GL3_T = np.array([0.0, -0.7745966692, 0.7745966692], dtype=float)
_GL3_W = np.array([0.8888888889, 0.5555555556, 0.5555555556], dtype=float)

_T = _GL3_T if USE_GL3 else _GL7_T
_W = _GL3_W if USE_GL3 else _GL7_W

_INV_SQRT2 = 1.0 / np.sqrt(2.0)
IRLS_MAX_ITERS = 6
IRLS_IMPROVE   = 1e-3


# ---------- utils ----------
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


# ---------- loader (reads each time) ----------
def _load_shapes(shape_file, a, b):
    """
    TXT/CSV rows:
      isotope, half_life, energy_keV, percent, sigma, tau1, tau2, eta, flag
    flag ∈ {'s','e','-','*'}; '*' = single-line isotope
    Returns (isotopes, pulses, arrays) where arrays is a dict of numpy arrays for speed.
    """
    rows = []
    with open(shape_file, 'r', newline='') as f:
        rdr = csv.reader(f, skipinitialspace=True)
        last_iso = ""
        for raw in rdr:
            if not raw or all(not str(x).strip() for x in raw):
                continue
            if str(raw[0]).strip().startswith("#"):
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

    # group contiguous blocks into isotopes
    isotopes, pulses = [], []
    i = 0
    while i < len(rows):
        iso_name = rows[i]['iso']
        safe     = _safe_name(iso_name)
        start = i
        if rows[i]['flag'] == '*':
            end = i
        else:
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
                iso_idx=len(isotopes)-1, name=iso_name, safe=safe,
                E=r['E'], mu0=r['mu0'], pct=r['pct'], ratio=ratio,
                sigma=r['sigma'], tau1=r['tau1'], tau2=r['tau2'], eta=r['eta'],
            ))
        isotopes[-1]['end_idx'] = len(pulses) - 1
        i = end + 1

    # pack to arrays for speed
    iso_idx = np.array([p['iso_idx'] for p in pulses], dtype=int)
    mu0     = np.array([p['mu0']    for p in pulses], dtype=float)
    sigma   = np.array([p['sigma']  for p in pulses], dtype=float)
    tau1    = np.array([p['tau1']   for p in pulses], dtype=float)
    tau2    = np.array([p['tau2']   for p in pulses], dtype=float)
    eta     = np.array([p['eta']    for p in pulses], dtype=float)
    ratio   = np.array([p['ratio']  for p in pulses], dtype=float)
    energy  = np.array([p['E']      for p in pulses], dtype=float)

    arrays = dict(iso_idx=iso_idx, mu0=mu0, sigma=sigma, tau1=tau1, tau2=tau2, eta=eta,
                  ratio=ratio, energy=energy)
    return isotopes, pulses, arrays


# ---------- EMG (vectorized, numerically stable) ----------

def _emg_one_tail_stable_vec(x, A, mu, sigma, tau):
    """
    x: (N,1) or (N,) -> treated as (N,1)
    A, mu, sigma, tau: (1,M)
    returns (N,M)
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]         # (N,1)

    sigma = np.clip(sigma, 1e-9, None)
    tau   = np.clip(tau,   1e-9, None)

    pref = 0.5 * A / tau                 # (1,M)
    inv_sigma = 1.0 / sigma              # (1,M)

    # Broadcasted arrays: (N,M)
    dx_over_sig = (x - mu) * inv_sigma
    u = (_INV_SQRT2) * ((sigma / tau) - dx_over_sig)

    # Two numerically-stable forms
    term_pos = pref * np.exp(-0.5 * dx_over_sig**2) * erfcx(u)
    term_neg = pref * np.exp(0.5 * (sigma / tau)**2 - (x - mu) / tau) * erfc(u)

    out = np.where(u >= 0.0, term_pos, term_neg)
    out[~np.isfinite(out)] = 0.0
    return out

def _emg_two_tail_stable_vec(x, A, mu, sigma, tau_fast, tau_slow, eta):
    # All inputs broadcast to (N,M)
    eta = np.clip(eta, 0.0, 1.0)
    return ((1.0 - eta) * _emg_one_tail_stable_vec(x, A, mu, sigma, tau_fast)
          + (      eta) * _emg_one_tail_stable_vec(x, A, mu, sigma, tau_slow))

def _bin_integral_vec(fun, x, bw, *parms):
    """
    fun(x, *parms) returns (N,M). We integrate across each bin via GL nodes.
    """
    bw = float(bw)
    if bw <= 0.0:
        return fun(x[:, None], *parms)
    half = 0.5 * bw
    acc = 0.0
    for w, t in zip(_W, _T):
        acc += w * fun((x + half * t)[:, None], *parms)
    return half * acc

# ---------- Fitter ----------
class AlphaMultiEMGFit:
    def __init__(self,
                 shape_file,
                 calib_a=7.1787509, calib_b=-7436.5,
                 allow_shift=True, shift_bound=100.0,
                 fix_ratios=True, wmode_default=2):
        if not os.path.isfile(shape_file):
            raise FileNotFoundError(f"shape_file not found: {shape_file}")
        self.shape_file    = shape_file
        self.calib_a       = float(calib_a)
        self.calib_b       = float(calib_b)
        self.allow_shift   = bool(allow_shift)
        self.shift_bound   = float(shift_bound)
        self.fix_ratios    = bool(fix_ratios)
        self.wmode_default = int(wmode_default)

    def start(self, x, y, xmin, xmax, fitpar, axis, fit_results):
        # fresh load each fit (so a new shapes file is picked up immediately)
        self._isotopes, self._pulses, arr = _load_shapes(self.shape_file, self.calib_a, self.calib_b)

        # clean data / window
        mfin = np.isfinite(x) & np.isfinite(y)
        x = np.asarray(x)[mfin]; y = np.asarray(y)[mfin]
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

        # params to estimate
        pars = Parameters()
        pars.add('d0', value=0.0, min=-5000, max=+5000)
        pars.add('dg', value=0.0, min=-2e-2, max=+2e-2)

        # one amplitude per isotope, optional small dm per isotope
        n_iso = len(self._isotopes)
        total_area = max(np.trapz(np.clip(y, 0, None), x), 1.0)
        seed_A = total_area / max(n_iso, 1)
        for k, iso in enumerate(self._isotopes):
            stem = iso['safe']
            pars.add(f"A_{stem}", value=seed_A, min=0.0)
            if self.allow_shift:
                # seed dm by local mode near median mu0 of that isotope
                mu0s = [p['mu0'] for p in self._pulses[iso['start_idx']:iso['end_idx']+1]]
                mu0i = float(np.median(mu0s))
                # ~8σ or 400 samples window
                sig0s = [p['sigma'] for p in self._pulses[iso['start_idx']:iso['end_idx']+1]]
                sig0  = float(np.median(sig0s)) if sig0s else dx
                W     = max(8.0*sig0, 400.0)
                mwin  = (x >= mu0i - W) & (x <= mu0i + W)
                dm0   = float(x[np.argmax(y[mwin])] - mu0i) if np.count_nonzero(mwin) > 5 else 0.0
                pars.add(f"dm_{stem}", value=np.clip(dm0, -self.shift_bound, +self.shift_bound),
                         min=-self.shift_bound, max=+self.shift_bound, vary=True)

        pars.add('bw', value=max(dx, 0.0), vary=False, min=0.0)

        # pull bw/wmode from GUI tail or default
        fp = list(fitpar) if fitpar is not None else []
        if len(fp) >= 2:
            try:
                bw_cand = float(fp[-2])
                if np.isfinite(bw_cand) and bw_cand > 0:
                    pars['bw'].set(value=bw_cand)
            except Exception:
                pass
        wmode = self.wmode_default
        if len(fp) >= 1:
            try:
                wm = int(round(float(fp[-1])))
                if wm in (0,1,2): wmode = wm
            except Exception:
                pass

        # capture arrays locally for the model closure
        iso_idx = arr['iso_idx']; mu0 = arr['mu0']; sigma = arr['sigma']
        tau1    = arr['tau1'];   tau2 = arr['tau2']; eta   = arr['eta']
        ratio   = arr['ratio']

        n_pulses = len(mu0)

        def _model_binned(x, **kw):
            d0 = float(kw.get('d0', 0.0))
            dg = float(kw.get('dg', 0.0))
            bw = float(kw.get('bw', dx))

            # A for each pulse = A_iso * ratio
            A_iso = np.zeros(len(self._isotopes), dtype=float)
            dm_iso = np.zeros(len(self._isotopes), dtype=float)

            for k, iso in enumerate(self._isotopes):
                stem = iso['safe']
                A_iso[k]  = float(kw.get(f"A_{stem}", 0.0))
                if self.allow_shift:
                    dm_iso[k] = float(kw.get(f"dm_{stem}", 0.0))

            A  = A_iso[iso_idx] * ratio
            mu = (1.0 + dg) * mu0 + d0 + dm_iso[iso_idx]

            # broadcast to (N,M)
            Arow   = A[None, :]
            murow  = mu[None, :]
            srow   = sigma[None, :]
            t1row  = tau1[None, :]
            t2row  = tau2[None, :]
            etarow = eta[None, :]

            y_nm = _bin_integral_vec(_emg_two_tail_stable_vec, x, bw,
                                     Arow, murow, srow, t1row, t2row, etarow)
            return np.sum(y_nm, axis=1)  # sum over M

        model = Model(_model_binned, independent_vars=['x'])

        # initial weights (Poisson)
        def w_data(yv): return 1.0 / np.sqrt(np.clip(yv, 1.0, None))
        weights = None if wmode == 0 else w_data(y)

        # first fit
        res = model.fit(y, params=pars, x=x, method='least_squares',
                        weights=weights, fit_kws=dict(loss='linear'), max_nfev=2000)

        # IRLS: switch to model-based Poisson weights
        if wmode == 2:
            last = res
            for _ in range(IRLS_MAX_ITERS):
                yhat = model.eval(last.params, x=x)
                w = 1.0 / np.sqrt(np.clip(yhat, 1.0, None))
                new = model.fit(y, params=last.params.copy(), x=x,
                                method='least_squares', weights=w,
                                fit_kws=dict(loss='linear'), max_nfev=1500)
                rel = abs(new.chisqr - last.chisqr) / max(last.chisqr, 1.0)
                last = new
                if rel < IRLS_IMPROVE:
                    break
            res = last

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

        fit_results.setPlainText(
            header + "\n" + stats + "\n\n" + "\n".join(iso_lines) + "\n\n" +
            fit_report(res, show_correl=True)
        )

        # ---------- Plotting ----------
        xx = np.linspace(x.min(), x.max(), 1400)
        ytot = model.eval(res.params, x=xx)
        (fitln_total,) = axis.plot(xx, ytot, lw=2.5, color='tab:orange', label='fit total')


        # Evaluate model on the actual data x for residuals
        yhat_data = model.eval(res.params, x=x)

        # Pulls = (data - model) / sqrt(model)
        pull = (y - yhat_data) / np.sqrt(np.clip(yhat_data, 1.0, None))

        ax_pull = axis.twinx()
        ax_pull.plot(x, pull, lw=0.8, alpha=0.6)
        ax_pull.set_ylabel("pull (σ)")
        ax_pull.set_ylim(-5, 5)        # typical visual band
        ax_pull.grid(False)
        # Optional: lighten the pull axis spines
        for sp in ("right", "top"):
            ax_pull.spines[sp].set_alpha(0.3)


        d0 = float(res.params['d0'].value); dg = float(res.params['dg'].value)
        bwv = float(res.params['bw'].value)

        # dashed per sub-peak (11 total in your shapes)
        sub_lines = []
        for iso in self._isotopes:
            stem = iso['safe']; name = iso['name']
            A_iso = float(res.params[f"A_{stem}"].value)
            dm    = float(res.params.get(f"dm_{stem}", 0.0).value) if self.allow_shift else 0.0

            for kk in range(iso['start_idx'], iso['end_idx'] + 1):
                q = self._pulses[kk]
                A   = A_iso * float(q['ratio'])
                mu  = (1.0 + dg) * float(q['mu0']) + d0 + dm
                # tiny 1-peak eval using the same quadrature for consistency
                def _one(xv):
                    return _bin_integral_vec(_emg_two_tail_stable_vec, xv, bwv,
                                             A*np.ones((1,1)), mu*np.ones((1,1)),
                                             np.array([[q['sigma']]]),
                                             np.array([[q['tau1']]]),
                                             np.array([[q['tau2']]]),
                                             np.array([[q['eta']]]) ).ravel()
                yj = _one(xx)
                (ln,) = axis.plot(xx, yj, lw=1.8, ls='--', alpha=0.9, label=f"{name} {q['E']:.1f} keV")
                sub_lines.append(ln)

        # label each isotope at its strongest sub-peak
        axis.relim(); axis.autoscale_view()
        ymin, ymax = axis.get_ylim(); yspan = max(ymax - ymin, 1.0)
        for iso in self._isotopes:
            stem = iso['safe']; name = iso['name']
            dm   = float(res.params.get(f"dm_{stem}", 0.0).value) if self.allow_shift else 0.0
            block = self._pulses[iso['start_idx']:iso['end_idx']+1]
            if not block: continue
            q_best = max(block, key=lambda q: float(q.get('ratio', 0.0)))
            mu_lbl = (1.0 + dg) * float(q_best['mu0']) + d0 + dm
            if xx[0] <= mu_lbl <= xx[-1]:
                y_at  = float(np.interp(mu_lbl, xx, ytot))
                axis.text(mu_lbl, y_at + 0.04*yspan, name,
                          ha='center', va='bottom', fontsize=9,
                          bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6))

        axis.legend(loc='best', frameon=False)

        fitln_total.components = sub_lines
        fitln_total.component_data = {'x': xx, 'ytot': ytot}
        fitln_total._isotopes = [iso['name'] for iso in self._isotopes]
        return fitln_total


class AlphaMultiEMGFitBuilder:
    def __call__(self, **cfg):
        # return a NEW fitter every time, so a new shapes file can be chosen each click
        return AlphaMultiEMGFit(**cfg)
