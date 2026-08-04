"""Peak finding — the Qt-free core of the peaks cluster. The peak-analysis
feature runs a peak search over the counts of the selected spectrum,
restricted to the currently-visible x range, and reports the found peaks
(position + FWHM)."""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_prominences, peak_widths, savgol_filter

try:                                    # numpy>=2 renamed trapz -> trapezoid
    from numpy import trapezoid as _trapz
except ImportError:                     # pragma: no cover - older numpy
    from numpy import trapz as _trapz

_FWHM_K = 2.0 * np.sqrt(2.0 * np.log(2.0))   # sigma -> FWHM


def find_peaks_in_range(x_axis, y_data, xmin, xmax, width):
    """Locate peaks in ``y_data`` restricted to the ``[xmin, xmax)`` x window, then
    run ``scipy.signal.find_peaks`` on the clipped counts. Returns
    ``(datax, datay, peaks, properties)``."""
    x = []
    y = []
    # create new tmp list with subrange for fitting
    for i in range(len(x_axis)):
        if x_axis[i] >= xmin and x_axis[i] < xmax:
            x.append(x_axis[i])
            y.append(y_data[i])
    datax = np.array(x)
    datay = np.array(y)
    peaks, properties = find_peaks(datay, prominence=1, width=width)
    return datax, datay, peaks, properties


def _clip_to_window(x_axis, y_data, xmin, xmax):
    """Vectorized equivalent of the original per-bin clip loop.

    Same half-open ``[xmin, xmax)`` window semantics; accepts lists or arrays
    (analyzePeak historically passes the counts as a Python list)."""
    x = np.asarray(x_axis, dtype=float)
    y = np.asarray(y_data, dtype=float)
    mask = (x >= xmin) & (x < xmax)
    return x[mask], y[mask]


def _empty_properties():
    return {key: np.array([]) for key in
            ("prominences", "left_bases", "right_bases",
             "widths", "width_heights", "left_ips", "right_ips")}


def find_peaks_in_range_vectorized(x_axis, y_data, xmin, xmax, width):
    """Drop-in replacement for :func:`find_peaks_in_range`: identical
    ``find_peaks(prominence=1, width=width)`` call on the identically-clipped
    counts, so the results are bit-identical — only the clip is numpy instead
    of a Python loop."""
    datax, datay = _clip_to_window(x_axis, y_data, xmin, xmax)
    peaks, properties = find_peaks(datay, prominence=1, width=width)
    return datax, datay, peaks, properties


def find_peaks_in_range_smoothed(x_axis, y_data, xmin, xmax, width):
    """Savitzky-Golay smoothing + noise-scaled prominence. The counts are
    smoothed with a window ~2x the expected FWHM (``width``), and a peak must
    rise 3 sigma above the Poisson noise of the typical level."""
    datax, datay = _clip_to_window(x_axis, y_data, xmin, xmax)
    n = len(datay)
    if n < 11:
        return datax, datay, np.array([], dtype=int), _empty_properties()
    win = max(5, 2 * int(width) + 1)
    if win >= n:
        win = n - 1 if n % 2 == 0 else n
    if win % 2 == 0:
        win += 1
    smoothed = savgol_filter(datay, window_length=win, polyorder=min(3, win - 1))
    # a real peak must beat ~3 sigma of Poisson noise at the typical level
    prominence = max(1.0, 3.0 * float(np.median(np.sqrt(np.clip(smoothed, 1.0, None)))))
    peaks, properties = find_peaks(smoothed, prominence=prominence,
                                   width=max(1.0, 0.5 * float(width)))
    return datax, datay, peaks, properties


def find_peaks_in_range_second_diff(x_axis, y_data, xmin, xmax, width):
    """Mariscotti-style smoothed second difference. The second difference of
    the counts is smoothed (3 moving-average passes, window ~ the expected
    FWHM) and compared to its Poisson standard deviation; significant negative
    dips mark peaks, so smooth backgrounds cancel out."""
    datax, datay = _clip_to_window(x_axis, y_data, xmin, xmax)
    n = len(datay)
    if n < 11:
        return datax, datay, np.array([], dtype=int), _empty_properties()

    dd = np.zeros(n)
    dd[1:-1] = datay[2:] - 2.0 * datay[1:-1] + datay[:-2]
    var = np.ones(n)
    var[1:-1] = datay[2:] + 4.0 * datay[1:-1] + datay[:-2]
    w = max(3, int(width) | 1)  # odd smoothing window ~ expected FWHM
    kernel = np.ones(w) / w
    for _ in range(3):
        dd = np.convolve(dd, kernel, mode="same")
        var = np.convolve(var, kernel / w, mode="same")
    sigma = np.sqrt(np.clip(var, 1e-12, None))
    significance = dd / sigma
    # threshold calibrated on synthetic Poisson spectra: with this variance
    # normalization, pure-noise dips stay below ~0.4 while real peaks (even on
    # a steep exponential background) score above ~1.7
    dips, _ = find_peaks(-significance, height=1.0)

    if len(dips) == 0:
        return datax, datay, np.array([], dtype=int), _empty_properties()

    # measure marker properties on a well-behaved smoothed curve: snap each
    # dip to the local maximum of the smoothed counts within +-w bins
    win = w if w >= 5 else 5
    if win >= n:
        win = n - 1 if n % 2 == 0 else n
    if win % 2 == 0:
        win += 1
    smoothed = savgol_filter(datay, window_length=win, polyorder=min(3, win - 1))
    snapped = []
    for d in dips:
        lo, hi = max(0, d - w), min(n, d + w + 1)
        idx = lo + int(np.argmax(smoothed[lo:hi]))
        if idx not in snapped:
            snapped.append(idx)
    peaks = np.array(snapped, dtype=int)

    prominences, left_bases, right_bases = peak_prominences(smoothed, peaks)
    # a snapped index with zero prominence cannot be width-measured — drop it
    keep = prominences > 0
    peaks = peaks[keep]
    if len(peaks) == 0:
        return datax, datay, peaks, _empty_properties()
    prom_data = (prominences[keep], left_bases[keep], right_bases[keep])
    widths, width_heights, left_ips, right_ips = peak_widths(
        smoothed, peaks, rel_height=0.5, prominence_data=prom_data)
    properties = {
        "prominences": prom_data[0], "left_bases": prom_data[1],
        "right_bases": prom_data[2], "widths": widths,
        "width_heights": width_heights, "left_ips": left_ips,
        "right_ips": right_ips,
    }
    return datax, datay, peaks, properties


# display name -> finder, in the order the popup's Algorithm combo shows them;
# index 0 is the combo default — Mariscotti, the accuracy-benchmark winner
# (user-chosen 2026-07-15; the legacy raw-counts search is no longer default)
PEAK_ALGORITHMS = {
    "Mariscotti (2nd difference)": find_peaks_in_range_second_diff,
    "Smoothed (Savitzky-Golay)": find_peaks_in_range_smoothed,
    "Raw counts (legacy)": find_peaks_in_range,
    "Raw counts (legacy, fast)": find_peaks_in_range_vectorized,
}


# ---------------------------------------------------------------------------
# Peak Finder 2 — click-to-fit: gaussian + linear background around a click.
# Qt-free core; MainWindow owns the Start/Stop toggle, the mpl click binding,
# and the drawing (fit curve, dashed background, blue net-area fill).
# ---------------------------------------------------------------------------

# flat gaussian+linear param names -> composite suffixed/background names, so
# callers/tests that fix or seed the classic A/mu/sigma/m/b still work through
# fit_composite (an unrecognized name passes through and errors, as before)
_GL_FLAT_TO_COMPOSITE = {"A": "A1", "mu": "mu1", "sigma": "sigma1",
                         "m": "m", "b": "b"}


def _gl_translate(d):
    if not d:
        return d
    return {_GL_FLAT_TO_COMPOSITE.get(k, k): v for k, v in d.items()}


def fit_gaussian_linear_range(x_axis, y_data, lo, hi, fixed=None, seeds=None):
    """Fit ``A*exp(-(x-mu)^2/2sigma^2) + m*x + b`` over the explicit window
    ``[lo, hi]``. ``fixed`` pins parameters by name and ``seeds`` overrides the
    automatic starting values."""
    spec = {"signal": "gaussian", "n_components": 1, "background": "poly1"}
    r = fit_composite(x_axis, y_data, lo, hi, spec,
                      fixed=_gl_translate(fixed), seeds=_gl_translate(seeds))
    if not r.get("ok"):
        return r
    return _composite_gl_flat(r)


# ---------------------------------------------------------------------------
# Peak Finder 2 — composite shape registry. One fit engine (`fit_composite`)
# over a pluggable signal shape (gaussian / crystal ball) times 1..5
# components, plus a background shape (poly1/2/3).
# ---------------------------------------------------------------------------

_CB_ALPHA_SEED, _CB_N_SEED = 1.5, 3.0
_SHARED_BOUNDS = {"alpha": (0.1, 10.0), "n": (1.01, 100.0)}
_SHARED_SEEDS = {"alpha": _CB_ALPHA_SEED, "n": _CB_N_SEED}
_MAX_COMPONENTS = 5


def _gaussian_eval(x, comp, shared, spec):
    A, mu, sigma = comp
    return A * np.exp(-0.5 * ((np.asarray(x, dtype=float) - mu) / sigma) ** 2)


def _crystal_ball_eval(x, comp, shared, spec):
    """Crystal Ball: gaussian core + power-law tail on one side. ``tail_side``
    (``"low"`` default, or ``"high"``) picks which flank carries the tail;
    ``alpha``/``n`` are the (shared) tail onset and steepness."""
    A, mu, sigma = comp
    alpha, n = shared["alpha"], shared["n"]
    z = (np.asarray(x, dtype=float) - mu) / sigma
    if spec.get("tail_side", "low") == "high":
        z = -z
    aa = abs(alpha)
    B = n / aa - aa
    # curve_fit explores alpha->0.1, n->100 where (n/aa)**n overflows; that's a
    # rejected trial point, not a real evaluation, so silence the transient
    with np.errstate(over="ignore", invalid="ignore"):
        D = (n / aa) ** n * np.exp(-0.5 * aa * aa)
        core = A * np.exp(-0.5 * z * z)
        tail = A * D * np.power(np.clip(B - z, 1e-300, None), -n)
    return np.where(z > -aa, core, tail)


def _gaussian_area(comp, bw, xs, shared, spec):
    A, mu, sigma = comp
    return A * sigma * np.sqrt(2.0 * np.pi) / bw


def _crystal_ball_area(comp, bw, xs, shared, spec):
    # net counts = numeric integral of the component curve over the window / bw
    # (window-truncated; the tail outside the window is not extrapolated)
    return float(_trapz(_crystal_ball_eval(xs, comp, shared, spec), xs)) / bw


SIGNAL_SHAPES = {
    "gaussian": {"params": ("A", "mu", "sigma"), "shared": (),
                 "eval": _gaussian_eval, "area": _gaussian_area},
    "crystal_ball": {"params": ("A", "mu", "sigma"), "shared": ("alpha", "n"),
                     "eval": _crystal_ball_eval, "area": _crystal_ball_area},
}


def _poly1_eval(x, coeffs, lo, hi):
    m, b = coeffs
    return m * np.asarray(x, dtype=float) + b


def _mapped_t(x, lo, hi):
    span = (float(hi) - float(lo)) or 1.0
    return 2.0 * (np.asarray(x, dtype=float) - lo) / span - 1.0


def _polyN_eval(x, coeffs, lo, hi):
    t = _mapped_t(x, lo, hi)
    out = np.zeros_like(t)
    for i, c in enumerate(coeffs):
        out = out + c * t ** i
    return out


BACKGROUND_SHAPES = {
    "poly1": {"params": ("m", "b"), "eval": _poly1_eval},
    "poly2": {"params": ("c0", "c1", "c2"), "eval": _polyN_eval},
    "poly3": {"params": ("c0", "c1", "c2", "c3"), "eval": _polyN_eval},
}


def _composite_param_names(spec):
    """Ordered fit-parameter names: per-component signal params suffixed by the
    1-based component index (``A1, mu1, sigma1, A2, …``), shared signal params
    unsuffixed (``alpha``, ``n``), background params last."""
    sh = SIGNAL_SHAPES[spec["signal"]]
    k = int(spec.get("n_components", 1))
    names = [f"{p}{i}" for i in range(1, k + 1) for p in sh["params"]]
    names += list(sh["shared"])
    names += list(BACKGROUND_SHAPES[spec["background"]]["params"])
    return names


def _seed_background(bg, edge_x, edge_y, lo, hi):
    """Seed the background from the window-edge bins: poly1 a raw line, poly2/3 a
    line in the mapped variable with higher coefficients zero."""
    params = BACKGROUND_SHAPES[bg]["params"]
    try:
        if bg == "poly1":
            m0, b0 = np.polyfit(edge_x, edge_y, 1)
            return {"m": float(m0), "b": float(b0)}
        t = _mapped_t(edge_x, lo, hi)
        c1, c0 = np.polyfit(t, edge_y, 1)   # polyfit: highest power first
    except Exception:
        c0, c1 = float(np.min(edge_y)), 0.0
        if bg == "poly1":
            return {"m": 0.0, "b": c0}
    seed = {p: 0.0 for p in params}
    seed["c0"] = float(c0)
    seed["c1"] = float(c1)
    return seed


def _eval_composite(xv, vals, spec, lo, hi):
    sh = SIGNAL_SHAPES[spec["signal"]]
    k = int(spec.get("n_components", 1))
    shared_vals = {p: vals[p] for p in sh["shared"]}
    total = np.zeros_like(np.asarray(xv, dtype=float))
    for i in range(1, k + 1):
        comp = tuple(vals[f"{p}{i}"] for p in sh["params"])
        total = total + sh["eval"](xv, comp, shared_vals, spec)
    bgsh = BACKGROUND_SHAPES[spec["background"]]
    bg = bgsh["eval"](xv, tuple(vals[p] for p in bgsh["params"]), lo, hi)
    return total, bg


def fit_composite(x_axis, y_data, lo, hi, spec, fixed=None, seeds=None):
    """Fit ``sum_i signal_i(x) + background(x)`` over the window ``[lo, hi]``.
    ``spec = {'signal': 'gaussian'|'crystal_ball', 'n_components': k,
    'background': 'poly1'|'poly2'|'poly3', 'tail_side': 'low'|'high'}``."""
    signal = spec.get("signal")
    bg = spec.get("background")
    if signal not in SIGNAL_SHAPES or bg not in BACKGROUND_SHAPES:
        return dict(ok=False, error=f"unknown shape in spec: {spec}")
    k = int(spec.get("n_components", 1))
    if not (1 <= k <= _MAX_COMPONENTS):
        return dict(ok=False, error=f"n_components must be 1..{_MAX_COMPONENTS}")

    x = np.asarray(x_axis, dtype=float)
    y = np.asarray(y_data, dtype=float)
    lo, hi = (float(lo), float(hi)) if lo <= hi else (float(hi), float(lo))
    mask = (x >= lo) & (x <= hi)
    xs, ys = x[mask], y[mask]
    if xs.size < 6:
        return dict(ok=False, error="fit window holds fewer than 6 bins; "
                                    "widen the window or click inside the spectrum")

    bw = float(np.median(np.diff(xs))) if xs.size > 1 else 1.0
    names = _composite_param_names(spec)
    fixed = {k_: float(v) for k_, v in (fixed or {}).items()}
    seeds = {k_: float(v) for k_, v in (seeds or {}).items()}
    unknown = (set(fixed) | set(seeds)) - set(names)
    if unknown:
        return dict(ok=False, error=f"unknown parameter(s): {sorted(unknown)}")

    sh = SIGNAL_SHAPES[signal]
    bgsh = BACKGROUND_SHAPES[bg]

    # seeds: background from the window edges, component centroids from the
    # background-subtracted residual maxima
    n_edge = max(2, xs.size // 10)
    edge_x = np.concatenate([xs[:n_edge], xs[-n_edge:]])
    edge_y = np.concatenate([ys[:n_edge], ys[-n_edge:]])
    bg_seed = _seed_background(bg, edge_x, edge_y, lo, hi)
    resid = ys - bgsh["eval"](xs, tuple(bg_seed[p] for p in bgsh["params"]), lo, hi)
    if k == 1:
        mu_idx = [int(np.argmax(resid))]
    else:
        pk, _ = find_peaks(resid)
        if len(pk) >= k:
            mu_idx = sorted(pk[np.argsort(resid[pk])[::-1][:k]].tolist())
        else:
            mu_idx = [int(round(v)) for v in
                      np.linspace(xs.size * 0.2, xs.size * 0.8, k)]

    p0, lb, ub = {}, {}, {}
    for slot, i in enumerate(range(1, k + 1)):
        j = min(max(mu_idx[slot], 0), xs.size - 1)
        p0[f"A{i}"] = max(float(resid[j]), 1e-3)
        p0[f"mu{i}"] = float(xs[j])
        p0[f"sigma{i}"] = max((hi - lo) / 12.0, bw)
        lb[f"A{i}"], ub[f"A{i}"] = 0.0, np.inf
        lb[f"mu{i}"], ub[f"mu{i}"] = float(xs[0]), float(xs[-1])
        lb[f"sigma{i}"], ub[f"sigma{i}"] = bw * 0.25, float(xs[-1] - xs[0])
    for p in sh["shared"]:
        p0[p] = _SHARED_SEEDS[p]
        lb[p], ub[p] = _SHARED_BOUNDS[p]
    for p in bgsh["params"]:
        p0[p] = bg_seed[p]
        lb[p], ub[p] = -np.inf, np.inf
    p0.update(seeds)

    free = [nm for nm in names if nm not in fixed]
    if not free:
        return dict(ok=False, error="every parameter is fixed; nothing to fit")

    def model(xv, *free_vals):
        vals = dict(fixed)
        vals.update(zip(free, free_vals))
        total, bg_curve = _eval_composite(xv, vals, spec, lo, hi)
        return total + bg_curve

    # crystal ball's shared alpha/n are strongly correlated and go
    # unconstrained on a peak with little tail, so the solver's default 1e-8
    # tolerances grind to the iteration cap (~250 ms vs ~5 ms). Loosen to 1e-6
    # for shapes carrying shared params — ample for counting data — and leave
    # the well-conditioned gaussian path on the tight default.
    tol = 1e-6 if sh["shared"] else 1e-8
    try:
        popt, pcov = curve_fit(
            model, xs, ys, p0=[p0[nm] for nm in free],
            sigma=np.sqrt(np.clip(ys, 1.0, None)), absolute_sigma=True,
            bounds=([lb[nm] for nm in free], [ub[nm] for nm in free]),
            ftol=tol, xtol=tol, gtol=tol, maxfev=5000)
    except Exception as e:
        return dict(ok=False, error=f"fit did not converge: {e}")

    vals = dict(fixed)
    vals.update(zip(free, (float(v) for v in popt)))
    perr_free = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))
    err = {nm: 0.0 for nm in names}
    err.update(zip(free, (float(e) for e in perr_free)))

    shared_vals = {p: vals[p] for p in sh["shared"]}
    components = []
    for i in range(1, k + 1):
        A, mu, sigma = (vals[f"A{i}"], vals[f"mu{i}"], vals[f"sigma{i}"])
        dA, dmu, dsigma = (err[f"A{i}"], err[f"mu{i}"], err[f"sigma{i}"])
        fwhm = _FWHM_K * sigma
        area = sh["area"]((A, mu, sigma), bw, xs, shared_vals, spec)
        darea = area * float(np.hypot(dA / A if A else 0.0,
                                      dsigma / sigma if sigma else 0.0))
        components.append(dict(A=A, dA=dA, mu=mu, dmu=dmu, sigma=sigma,
                               dsigma=dsigma, fwhm=fwhm, dfwhm=_FWHM_K * dsigma,
                               area=area, darea=darea))

    total_s, bg_s = _eval_composite(xs, vals, spec, lo, hi)
    yhat = total_s + bg_s
    dof = max(xs.size - len(free), 1)
    redchi = float(np.sum((ys - yhat) ** 2 / np.clip(yhat, 1.0, None)) / dof)

    xx = np.linspace(xs[0], xs[-1], 400)
    bg_xx = bgsh["eval"](xx, tuple(vals[p] for p in bgsh["params"]), lo, hi)
    y_comp = []
    total_xx = np.zeros_like(xx)
    for i in range(1, k + 1):
        comp = tuple(vals[f"{p}{i}"] for p in sh["params"])
        ci = sh["eval"](xx, comp, shared_vals, spec)
        total_xx = total_xx + ci
        y_comp.append(ci + bg_xx)
    bg_params = {p: vals[p] for p in bgsh["params"]}
    return dict(ok=True, components=components, redchi=redchi,
                win_lo=lo, win_hi=hi, xx=xx, y_fit=total_xx + bg_xx,
                y_bg=bg_xx, y_comp=y_comp, bg_params=bg_params,
                shared_params=dict(shared_vals), spec=dict(spec))


def eval_composite_result(x, result):
    """Evaluate a fitted composite model (a :func:`fit_composite` result) at
    ``x``. Reconstructs signal components (with the shared params, e.g. a CB's
    ``alpha``/``n``) + background from the stored parameters — used to compute
    the residual on the data bins for the auto-add loop."""
    spec = result["spec"]
    sh = SIGNAL_SHAPES[spec["signal"]]
    shared = result.get("shared_params", {})
    total = np.zeros_like(np.asarray(x, dtype=float))
    for c in result["components"]:
        total = total + sh["eval"](x, (c["A"], c["mu"], c["sigma"]), shared, spec)
    bgsh = BACKGROUND_SHAPES[spec["background"]]
    bg = bgsh["eval"](x, tuple(result["bg_params"][p] for p in bgsh["params"]),
                      result["win_lo"], result["win_hi"])
    return total + bg


def autocomponent_refit(x, y, lo, hi, prev_result, max_components=5):
    """Re-fit ``prev_result``'s model over the new window ``[lo, hi]``,
    matching the component set to what the window now covers. (1)
    **shrink-drop**: components of the previous fit whose μ fell outside the
    new window are dropped (renumbered implicitly)."""
    lo, hi = (float(lo), float(hi)) if lo <= hi else (float(hi), float(lo))
    spec = dict(prev_result["spec"])
    kept = [c for c in prev_result["components"] if lo <= c["mu"] <= hi]

    if kept:
        spec["n_components"] = len(kept)
        seeds = {f"mu{i + 1}": c["mu"] for i, c in enumerate(kept)}
    else:
        # window moved off every previous peak — fall back to one fresh auto fit
        spec["n_components"] = 1
        seeds = None
    r = fit_composite(x, y, lo, hi, spec, seeds=seeds)
    if not r["ok"]:
        return r

    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = (xa >= lo) & (xa <= hi)
    xs, ys = xa[mask], ya[mask]
    while len(r["components"]) < max_components:
        ym = eval_composite_result(xs, r)
        mus = [c["mu"] for c in r["components"]]
        sigs = [c["sigma"] for c in r["components"]]
        cand = find_residual_component(xs, ys, ym, mus, sigs)
        if cand is None:
            break
        spec["n_components"] = len(r["components"]) + 1
        seeds = {f"mu{i + 1}": m for i, m in enumerate(mus)}
        seeds[f"mu{len(mus) + 1}"] = cand["mu"]
        r2 = fit_composite(x, y, lo, hi, spec, seeds=seeds)
        if not r2["ok"]:
            break                       # keep the last good fit
        r = r2
    return r


def find_residual_component(x, y, y_model, existing_mus, existing_sigmas,
                            snr=5.0, neighbour_snr=2.0, sep_sigmas=2.0):
    """Seed for the most significant unmodeled peak in the residual, or None.
    Scans ``resid = y - y_model`` scaled by the Poisson noise of the model
    (``sqrt(max(y_model, 1))``)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ym = np.asarray(y_model, dtype=float)
    resid = y - ym
    signif = resid / np.sqrt(np.clip(ym, 1.0, None))
    n = x.size
    existing_sigmas = list(existing_sigmas)
    sigma_seed = (float(np.median(existing_sigmas)) if existing_sigmas
                  else max((float(x[-1]) - float(x[0])) / 12.0, 1.0))
    for i in np.argsort(signif)[::-1]:
        if signif[i] <= snr:
            break                       # sorted desc — nothing else qualifies
        if i == 0 or i == n - 1:
            continue                    # need both neighbours
        if signif[i - 1] <= neighbour_snr or signif[i + 1] <= neighbour_snr:
            continue
        mu = float(x[i])
        if any(abs(mu - float(m)) <= sep_sigmas * float(sg)
               for m, sg in zip(existing_mus, existing_sigmas)):
            continue
        return {"mu": mu, "A": max(float(resid[i]), 1e-3), "sigma": sigma_seed}
    return None


def fit_composite_auto(x_axis, y_data, center, spec, max_half_window=None):
    """Click-to-fit with an automatic window for an arbitrary composite ``spec``: the
    window is estimated from the data then refined once, and ``max_half_window``
    clamps both passes."""
    est = estimate_fit_window(x_axis, y_data, center)
    if not est["ok"]:
        return dict(ok=False, error=est["error"])

    x = np.asarray(x_axis, dtype=float)
    bw = float(np.median(np.diff(x)))

    def _cap(hw):
        return min(hw, float(max_half_window)) if max_half_window is not None else hw

    hw1 = _cap(est["half_window"])
    r1 = fit_composite(x_axis, y_data, est["mu"] - hw1, est["mu"] + hw1, spec)
    if not r1["ok"]:
        return r1

    c0 = r1["components"][0]
    hw2 = _cap(max(4.0 * c0["sigma"], 8.0 * bw))
    r2 = fit_composite(x_axis, y_data, c0["mu"] - hw2, c0["mu"] + hw2, spec)
    return r2 if r2["ok"] else r1


def fit_gaussian_linear(x_axis, y_data, center, half_window):
    """Fit around a clicked position over ``center +- half_window`` (x units).

    Thin delegate of :func:`fit_gaussian_linear_range`; kept for callers that
    think in click + width."""
    return fit_gaussian_linear_range(x_axis, y_data,
                                     center - half_window, center + half_window)


def estimate_fit_window(x_axis, y_data, center):
    """Estimate the fit window around a clicked position; no user width.
    Plan-A heuristic: lightly smooth the counts, hill-climb from the click to
    the local summit, walk down each flank until the descent stops (3
    consecutive non-decreasing bins), take the lower stop level as the local
    background, and measure a crude FWHM at half of (summit - background)."""
    x = np.asarray(x_axis, dtype=float)
    y = np.asarray(y_data, dtype=float)
    n = x.size
    if n < 12:
        return dict(ok=False, error="spectrum window too small")
    bw = float(np.median(np.diff(x)))

    # light smoothing so the hill-climb and walks don't chase Poisson noise
    kernel = np.ones(5) / 5.0
    ys = np.convolve(y, kernel, mode="same")

    # hill-climb from the click to the local summit (adapts to any binning).
    # After each +-3 climb converges, scan +-15 bins: a small noise bump on a
    # wide peak's flank is a genuine local max that stalls the narrow climb;
    # the wider scan hops over it and the climb resumes.
    i = int(np.argmin(np.abs(x - float(center))))
    for _ in range(50):
        for _ in range(200):
            lo, hi = max(0, i - 3), min(n, i + 4)
            j = lo + int(np.argmax(ys[lo:hi]))
            if j == i:
                break
            i = j
        lo, hi = max(0, i - 15), min(n, i + 16)
        j = lo + int(np.argmax(ys[lo:hi]))
        if j == i:
            break
        i = j
    p = i

    def _walk(step):
        """Follow the flank away from the summit until 3 consecutive
        non-decreasing bins (descent over); return the stop index."""
        k, bad, last = p, 0, ys[p]
        while 0 < k + step < n - 1:
            k += step
            if ys[k] < last:
                bad = 0
                last = ys[k]
            else:
                bad += 1
                if bad >= 3:
                    break
        return k

    l_stop, r_stop = _walk(-1), _walk(+1)
    bg = min(float(np.min(ys[l_stop:p + 1])), float(np.min(ys[p:r_stop + 1])))
    amp = float(ys[p]) - bg
    if amp <= 3.0 * np.sqrt(max(bg, 1.0)):
        return dict(ok=False, error="no peak found near click "
                                    "(not significant above local background)")

    # crude FWHM: first half-crossing on each flank (stop index as fallback)
    level = bg + 0.5 * amp
    li = p
    while li > l_stop and ys[li] > level:
        li -= 1
    ri = p
    while ri < r_stop and ys[ri] > level:
        ri += 1
    fwhm = float(x[ri] - x[li])
    if fwhm < 2.0 * bw:
        return dict(ok=False, error="no peak found near click (narrower than 2 bins)")

    half_window = max(3.0 * fwhm, 8.0 * bw)
    return dict(ok=True, mu=float(x[p]), fwhm=fwhm, half_window=half_window)


def fit_gaussian_linear_auto(x_axis, y_data, center, max_half_window=None):
    """Click-to-fit with an automatic window (plan A). Estimate the window
    from the data (:func:`estimate_fit_window`), fit, then refine once over
    ``mu_fit +- 4*sigma_fit`` so the final window adapts to the *fitted*
    width."""
    est = estimate_fit_window(x_axis, y_data, center)
    if not est["ok"]:
        return dict(ok=False, error=est["error"])

    x = np.asarray(x_axis, dtype=float)
    bw = float(np.median(np.diff(x)))

    def _cap(hw):
        return min(hw, float(max_half_window)) if max_half_window is not None else hw

    hw1 = _cap(est["half_window"])
    r1 = fit_gaussian_linear(x_axis, y_data, est["mu"], hw1)
    r1_win = (est["mu"] - hw1, est["mu"] + hw1)
    if not r1["ok"]:
        return r1

    hw2 = _cap(max(4.0 * r1["sigma"], 8.0 * bw))
    r2 = fit_gaussian_linear(x_axis, y_data, r1["mu"], hw2)
    if r2["ok"]:
        r2["win_lo"], r2["win_hi"] = r1["mu"] - hw2, r1["mu"] + hw2
        return r2
    r1["win_lo"], r1["win_hi"] = r1_win
    return r1


_FIX_PEAK_DEFAULT_HALF_WINDOW_BINS = 50


def sigma_to_fwhm(sigma):
    """Gaussian FWHM from sigma (``2*sqrt(2*ln2)*sigma``). The single source of
    the σ↔FWHM factor for the edit popup's linked fields."""
    return _FWHM_K * float(sigma)


def fwhm_to_sigma(fwhm):
    """Gaussian sigma from FWHM (inverse of :func:`sigma_to_fwhm`)."""
    return float(fwhm) / _FWHM_K


def validate_gauss_edit(fixed, lo=None, hi=None):
    """Reject pinned edit values that would make the fit degenerate before
    they reach the fit core, which happily accepts them (``fixed=`` bypasses
    the bounds). σ must be strictly positive."""
    if "sigma" in fixed and fixed["sigma"] <= 0:
        return "σ must be positive"
    if "mu" in fixed and lo is not None and hi is not None:
        if not (lo <= fixed["mu"] <= hi):
            return "μ must lie within the fit window"
    return None


def nearest_window_edge(x, lo, hi, tol):
    """Which fit-window edge the position ``x`` is within ``tol`` of:
    ``"lo"``, ``"hi"``, or ``None``. All four arguments must be in the SAME
    units (the GUI passes display pixels so the pick radius is uniform)."""
    dlo = abs(float(x) - float(lo))
    dhi = abs(float(x) - float(hi))
    if min(dlo, dhi) > float(tol):
        return None
    return "lo" if dlo <= dhi else "hi"


def nearest_component_index(components, x):
    """Index of the component whose ``mu`` is nearest ``x`` (ties → lower index),
    or None for an empty list. Used by the edit popup to pick which component of
    a multi-component fit a right-click targets."""
    if not components:
        return None
    return min(range(len(components)),
               key=lambda i: abs(float(components[i]["mu"]) - float(x)))


def primary_component(r):
    """The component a multi-component fit is reported by: the largest by area
    (ties → lowest index). None for a result with no components."""
    comps = r.get("components") or ()
    if not comps:
        return None
    return max(comps, key=lambda c: c["area"])


def find_duplicate_mu(new_mu, existing_mus, tol):
    """Index of the first entry in ``existing_mus`` within ``tol`` (x units) of
    ``new_mu``, else ``None``.

    Used by auto click-to-fit to suppress a re-fit of an already-fitted peak:
    an off-peak flank click converges to the same centroid, so a new fit whose
    mu lands within ~1 bin of an existing fit's mu is the same peak."""
    for i, mu in enumerate(existing_mus):
        if abs(float(new_mu) - float(mu)) <= float(tol):
            return i
    return None


def fix_peak_window(center, bin_width, cap_bins=None):
    """Fixed-mu fit window for the Fix Peak tool: symmetric about the clicked
    ``center`` (x units), NOT chosen from the data. ``estimate_fit_window`` is
    deliberately avoided here — its hill-climb snaps to the strongest local
    summit, which is the wrong peak in the exact case Fix Peak exists for (a
    small peak beside a big neighbour)."""
    half_bins = (0.5 * cap_bins) if cap_bins else _FIX_PEAK_DEFAULT_HALF_WINDOW_BINS
    hw = half_bins * float(bin_width)
    return center - hw, center + hw


def format_gauss_fit_output(peak_no, r, tag=None):
    """The Peak Finder 2 output block for one fitted peak (or its error).

    ``tag`` (e.g. ``"fixed μ"``) annotates the peak header for the tool that
    produced it; ``None`` leaves the line byte-identical to the auto path."""
    label = f"Peak {peak_no}" + (f" ({tag})" if tag else "")
    if not r.get("ok"):
        return f"{label}: FAILED — {r.get('error', 'unknown error')}"
    text = (f"{label} @ μ = {r['mu']:.6g} ± {r['dmu']:.2g}\n"
            f"   A = {r['A']:.4g} ± {r['dA']:.2g}, "
            f"σ = {r['sigma']:.4g} ± {r['dsigma']:.2g}, "
            f"FWHM = {r['fwhm']:.4g} ± {r['dfwhm']:.2g}\n"
            f"   net area = {r['area']:.4g} ± {r['darea']:.2g} counts, "
            f"bg = {r['m']:.3g}·x + {r['b']:.4g}, "
            f"red-χ² = {r['redchi']:.3g}")
    if "win_lo" in r and "win_hi" in r:
        text += f"\n   window = [{r['win_lo']:.6g}, {r['win_hi']:.6g}] (auto)"
    return text


# Column headers for the Peak Finder 2 results table (one row per fit).
PEAK2_TABLE_COLUMNS = ("#", "μ", "FWHM", "area", "χ²ᵣ")


def format_gauss_fit_row(peak_no, r, tag=None):
    """Compact one-row-per-fit view for the results table. Returns ``cells`` (display
    text plus a raw numeric sort value), the tool ``tag``, and the full ``tooltip``."""
    marker = " *" if tag else ""
    cells = [
        (f"{peak_no}{marker}", float(peak_no)),
        (f"{r['mu']:.6g} ± {r['dmu']:.2g}", float(r['mu'])),
        (f"{r['fwhm']:.4g} ± {r['dfwhm']:.2g}", float(r['fwhm'])),
        (f"{r['area']:.4g} ± {r['darea']:.2g}", float(r['area'])),
        (f"{r['redchi']:.3g}", float(r['redchi'])),
    ]
    return {"cells": cells, "tag": tag,
            "tooltip": format_gauss_fit_output(peak_no, r, tag=tag)}


# ---- composite output --------------------------------------------

_SIGNAL_LABELS = {"gaussian": "gaussian", "crystal_ball": "crystal ball"}
_BACKGROUND_LABELS = {"poly1": "linear", "poly2": "quadratic", "poly3": "cubic"}


def _is_gauss_linear(spec):
    return (spec.get("signal") == "gaussian"
            and int(spec.get("n_components", 1)) == 1
            and spec.get("background") == "poly1")


def _composite_gl_flat(r):
    """Flatten a gaussian x1 + poly1 composite result into the classic flat
    ``A/mu/sigma/m/b/...`` dict the original gaussian formatters/callers expect."""
    c = r["components"][0]
    m, b = r["bg_params"]["m"], r["bg_params"]["b"]
    return dict(ok=True, A=c["A"], dA=c["dA"], mu=c["mu"], dmu=c["dmu"],
                sigma=c["sigma"], dsigma=c["dsigma"], m=m, b=b,
                fwhm=c["fwhm"], dfwhm=c["dfwhm"], area=c["area"], darea=c["darea"],
                redchi=r["redchi"], win_lo=r["win_lo"], win_hi=r["win_hi"],
                xx=r["xx"], y_fit=r["y_fit"], y_bg=r["y_bg"])


def _spec_label(spec):
    sig = _SIGNAL_LABELS.get(spec.get("signal"), spec.get("signal"))
    k = int(spec.get("n_components", 1))
    bg = _BACKGROUND_LABELS.get(spec.get("background"), spec.get("background"))
    return f"{sig}{f' x{k}' if k > 1 else ''} + {bg}"


def format_composite_fit_output(peak_no, r, tag=None):
    """Detailed output block for a composite fit. The classic gaussian x1 +
    poly1 case delegates to :func:`format_gauss_fit_output` (byte-identical,
    including its ``bg = m·x + b`` line); other shapes get a header line naming
    the model + window + red-χ² plus one line per component."""
    label = f"Peak {peak_no}" + (f" ({tag})" if tag else "")
    if not r.get("ok"):
        return f"{label}: FAILED — {r.get('error', 'unknown error')}"
    spec = r.get("spec", {})
    if _is_gauss_linear(spec):
        return format_gauss_fit_output(peak_no, _composite_gl_flat(r), tag=tag)
    header = (f"{label} [{_spec_label(spec)}, "
              f"window {r['win_lo']:.6g}–{r['win_hi']:.6g}, "
              f"red-χ² = {r['redchi']:.3g}]")
    lines = [header]
    for i, c in enumerate(r["components"], 1):
        lines.append(f"   #{i}: μ = {c['mu']:.6g} ± {c['dmu']:.2g}, "
                     f"FWHM = {c['fwhm']:.4g} ± {c['dfwhm']:.2g}, "
                     f"area = {c['area']:.4g} ± {c['darea']:.2g} counts")
    return "\n".join(lines)


def format_composite_fit_row(peak_no, r, tag=None):
    """Compact one-row-per-fit view for the results table (composite fits). The
    gaussian x1 + poly1 case delegates to :func:`format_gauss_fit_row`; a
    multi-component fit shows its strongest component's μ/FWHM, the summed area,
    and marks the ``#`` cell ``×k``."""
    spec = r.get("spec", {})
    if _is_gauss_linear(spec):
        return format_gauss_fit_row(peak_no, _composite_gl_flat(r), tag=tag)
    comps = r["components"]
    k = len(comps)
    primary = primary_component(r)
    total_area = float(sum(c["area"] for c in comps))
    marker = " *" if tag else ""
    mult = f"×{k}" if k > 1 else ""     # only a genuine doublet+ shows the count
    cells = [
        (f"{peak_no}{mult}{marker}", float(peak_no)),
        (f"{primary['mu']:.6g} ± {primary['dmu']:.2g}", float(primary["mu"])),
        (f"{primary['fwhm']:.4g} ± {primary['dfwhm']:.2g}", float(primary["fwhm"])),
        (f"{total_area:.4g}", total_area),
        (f"{r['redchi']:.3g}", float(r["redchi"])),
    ]
    return {"cells": cells, "tag": tag,
            "tooltip": format_composite_fit_output(peak_no, r, tag=tag)}


def format_peak_output(peaks, properties, datax):
    """Build the per-peak result lines shown in the peak-analysis output box.
    One line per peak: its position (``datax`` at the peak index) and FWHM
    (the scipy ``widths`` property)."""
    x = datax.tolist()
    lines = []
    for i in range(len(peaks)):
        lines.append("Peak" + str(i + 1) + "\n\tpeak @ " + str(int(x[peaks[i]]))
                     + ", FWHM=" + str(int(properties['widths'][i])))
    return lines


def format_peak_labels(peaks, properties, datax):
    """One-line label per peak for the checkable peak-selection list.

    Same position/FWHM content as :func:`format_peak_output`, compacted to a
    single line per peak (the list replaced the fixed 12-checkbox grid whose
    labels were bare "Peak N")."""
    x = datax.tolist()
    return ["Peak " + str(i + 1) + " @ " + str(int(x[peaks[i]]))
            + " (FWHM=" + str(int(properties['widths'][i])) + ")"
            for i in range(len(peaks))]
