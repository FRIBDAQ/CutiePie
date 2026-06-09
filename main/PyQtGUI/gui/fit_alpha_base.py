"""Shared constants and pure functions for the alpha-fit family.

Consumers:
  fit_alpha_linear_creator    — imports everything listed below
  fit_alpha_multi_creator     — imports everything listed below
  fit_alpha_multi_sigma_creator — imports _GL* / _INV_SQRT2 / _safe_name / _parse_percent only
                                  (its _emg_one_tail_stable uses the opposite sign)
  fit_alpha{12,22,32}_creator — left untouched (different _bin_integral and EMG physics)

NOT in this module:
  USE_GL3        — differs between files (linear=False, multi/multi_sigma=True)
  _bin_integral  — reads module-level USE_GL3 so it stays per-file
  _peak_binned   — calls _bin_integral
  _load_shapes   — different signatures across files
"""

import re
import numpy as np
from scipy.special import erfcx, erfc

# ── Gauss-Legendre quadrature nodes and weights ──────────────────────────────

_GL7_T = np.array([
    0.0, -0.4058451513773972,  0.4058451513773972,
         -0.7415311855993945,  0.7415311855993945,
         -0.9491079123427585,  0.9491079123427585], dtype=float)
_GL7_W = np.array([
    0.4179591836734694,
    0.3818300505051189,  0.3818300505051189,
    0.2797053914892766,  0.2797053914892766,
    0.1294849661688697,  0.1294849661688697], dtype=float)

_GL3_T = np.array([0.0, -0.7745966692, 0.7745966692], dtype=float)
_GL3_W = np.array([0.8888888889, 0.5555555556, 0.5555555556], dtype=float)

_INV_SQRT2 = 1.0 / np.sqrt(2.0)

# ── EMG (left-tail, MINUS sign) ───────────────────────────────────────────────
# Used by fit_alpha_linear_creator and fit_alpha_multi_creator.
# fit_alpha_multi_sigma_creator uses the right-tail (PLUS sign) variant instead.

def _emg_one_tail_stable(x, A, mu, sigma, tau):
    x = np.asarray(x, dtype=float)
    sigma = max(float(sigma), 1e-9)
    tau   = max(float(tau),   1e-9)
    pref = 0.5 * A / tau
    inv_sigma = 1.0 / sigma
    u = _INV_SQRT2 * ((sigma / tau) - ((x - mu) * inv_sigma))
    out = np.empty_like(x)
    m = (u >= 0.0)
    if np.any(m):
        g = np.exp(-0.5 * ((x[m] - mu) * inv_sigma) ** 2)
        out[m] = pref * g * erfcx(u[m])
    if np.any(~m):
        expfac = np.exp(0.5 * (sigma / tau) ** 2 - (x[~m] - mu) / tau)
        out[~m] = pref * expfac * erfc(u[~m])
    return np.where(np.isfinite(out), out, 0.0)


def _emg_two_tail_stable(x, A, mu, sigma, tau_fast, tau_slow, eta):
    eta = float(np.clip(eta, 0.0, 1.0))
    return ((1.0 - eta) * _emg_one_tail_stable(x, A, mu, sigma, tau_fast)
          + (      eta) * _emg_one_tail_stable(x, A, mu, sigma, tau_slow))

# ── Shapes-file utilities ─────────────────────────────────────────────────────

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
