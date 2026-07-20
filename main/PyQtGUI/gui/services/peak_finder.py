"""Peak finding — the Qt-free core of the peaks cluster.

The peak-analysis feature runs a peak search over the counts of the selected
spectrum, restricted to the currently-visible x range, and reports the found
peaks (position + FWHM). The numeric half — clip the spectrum to the view
window, then locate peaks — is pure numpy/scipy and lives here; the MainWindow
keeps the Qt/matplotlib shell (reading the width/algorithm widgets and axes
limits, drawing the ``v``/vline/hline/text markers, wiring the per-peak
checkboxes).

Four algorithms are offered via the ``PEAK_ALGORITHMS`` dispatch table (the
popup's Algorithm combo shows exactly these names, MainWindow.analyzePeak
dispatches on the selection; the first entry is the combo default):

- ``Mariscotti (2nd difference)`` — smoothed-second-difference search (the
  classic nuclear-spectroscopy method). The second difference cancels smooth
  backgrounds, so it is the best choice on sloping/exponential background.
  The default.
- ``Smoothed (Savitzky-Golay)`` — Savitzky-Golay smoothing + noise-scaled
  prominence (3 sigma of the Poisson level). Best on low statistics;
  ``width`` is read as the *expected FWHM in bins* and peaks down to half of
  it are accepted.
- ``Raw counts (legacy)`` — the historical behavior: pure-Python window clip,
  then ``find_peaks(prominence=1, width=width)`` on the raw counts.
- ``Raw counts (legacy, fast)`` — bit-identical results to the legacy search
  (same find_peaks call), but the clip is a numpy boolean mask (~12x faster
  on large spectra).

All four return the same ``(datax, datay, peaks, properties)`` contract:
``datay`` is always the RAW clipped counts (markers must sit on the real
spectrum) and ``properties`` always carries the five keys the GUI consumes —
``prominences``/``width_heights``/``left_ips``/``right_ips`` (drawSinglePeaks)
and ``widths`` (format_peak_output).

Qt-free by construction: it takes plain arrays and scalars and imports only
numpy/scipy. Extracted from ``GUI.py:analyzePeak`` / ``update_peak_output`` so
the find + output-formatting logic is unit-testable without PyQt5 (mirrors
``geometry_io`` / ``dataframe_export`` / ``display_slot``).
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_prominences, peak_widths, savgol_filter

_FWHM_K = 2.0 * np.sqrt(2.0 * np.log(2.0))   # sigma -> FWHM


def find_peaks_in_range(x_axis, y_data, xmin, xmax, width):
    """Locate peaks in ``y_data`` restricted to the ``[xmin, xmax)`` x window.

    ``x_axis`` is the full bin-centre/edge axis (e.g. from ``createRange``) and
    ``y_data`` the matching counts. Bins whose x falls in the half-open window
    ``[xmin, xmax)`` are kept, then ``scipy.signal.find_peaks`` runs on the
    clipped counts with ``prominence=1`` and the given ``width``.

    Returns ``(datax, datay, peaks, properties)`` where ``datax``/``datay`` are
    the clipped arrays (NumPy), ``peaks`` the indices into them, and
    ``properties`` scipy's property dict. Moved verbatim from
    ``GUI.py:analyzePeak``.
    """
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
    """Savitzky-Golay smoothing + noise-scaled prominence.

    The counts are smoothed with a window ~2x the expected FWHM (``width``),
    and a peak must rise 3 sigma above the Poisson noise of the typical level.
    Peak positions/properties are measured on the smoothed curve (well-behaved
    under counting noise); ``datay`` returned is the raw clipped counts."""
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
    """Mariscotti-style smoothed second difference.

    The second difference of the counts is smoothed (3 moving-average passes,
    window ~ the expected FWHM) and compared to its Poisson standard deviation;
    significant negative dips mark peaks, so smooth backgrounds cancel out.
    Properties for the GUI markers are then measured on a Savitzky-Golay
    smoothed curve at each located peak."""
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

def _gauss_lin(x, A, mu, sigma, m, b):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + m * x + b


_GL_PARAMS = ("A", "mu", "sigma", "m", "b")


def fit_gaussian_linear_range(x_axis, y_data, lo, hi, fixed=None, seeds=None):
    """Fit ``A*exp(-(x-mu)^2/2sigma^2) + m*x + b`` over the explicit window
    ``[lo, hi]`` (may be asymmetric about the peak; clipped to the spectrum;
    reversed bounds are swapped).

    ``fixed`` pins parameters by name (e.g. ``{'mu': 662.0}``): the pinned value
    is substituted into the model and only the remaining parameters vary (pinned
    uncertainties come back 0.0 and the dof drops accordingly). ``seeds``
    overrides the automatic starting values by name. Automatic seeds: background
    from the window's edge bins, mu at the residual maximum, A from peak minus
    background.

    Returns a dict. On success (``ok=True``): params ``A/mu/sigma/m/b`` with
    uncertainties ``dA/dmu/dsigma``, ``fwhm``/``dfwhm``, the NET gaussian area
    in counts ``area``/``darea`` (``A*sigma*sqrt(2pi)/bin_width``; background
    excluded), ``redchi``, the effective window ``win_lo``/``win_hi``, and
    sampled curves ``xx``/``y_fit``/``y_bg`` for drawing (the blue fill goes
    between ``y_bg`` and ``y_fit``). On failure (``ok=False``): an ``error``
    message string."""
    x = np.asarray(x_axis, dtype=float)
    y = np.asarray(y_data, dtype=float)
    lo, hi = (float(lo), float(hi)) if lo <= hi else (float(hi), float(lo))
    mask = (x >= lo) & (x <= hi)
    xs, ys = x[mask], y[mask]
    if xs.size < 6:
        return dict(ok=False, error="fit window holds fewer than 6 bins; "
                                    "widen the window or click inside the spectrum")

    bw = float(np.median(np.diff(xs))) if xs.size > 1 else 1.0
    fixed = {k: float(v) for k, v in (fixed or {}).items()}
    seeds = {k: float(v) for k, v in (seeds or {}).items()}
    unknown = (set(fixed) | set(seeds)) - set(_GL_PARAMS)
    if unknown:
        return dict(ok=False, error=f"unknown parameter(s): {sorted(unknown)}")

    # automatic seeds: background from the window edges, centroid from the
    # background-subtracted local maximum
    n_edge = max(2, xs.size // 10)
    edge_x = np.concatenate([xs[:n_edge], xs[-n_edge:]])
    edge_y = np.concatenate([ys[:n_edge], ys[-n_edge:]])
    try:
        m0, b0 = np.polyfit(edge_x, edge_y, 1)
    except Exception:
        m0, b0 = 0.0, float(np.min(ys))
    resid = ys - (m0 * xs + b0)
    i_pk = int(np.argmax(resid))
    p0 = {"A": max(float(resid[i_pk]), 1e-3), "mu": float(xs[i_pk]),
          "sigma": max((hi - lo) / 12.0, bw), "m": float(m0), "b": float(b0)}
    p0.update(seeds)
    lb = {"A": 0.0, "mu": float(xs[0]), "sigma": bw * 0.25, "m": -np.inf, "b": -np.inf}
    ub = {"A": np.inf, "mu": float(xs[-1]), "sigma": float(xs[-1] - xs[0]),
          "m": np.inf, "b": np.inf}

    free = [n for n in _GL_PARAMS if n not in fixed]
    if not free:
        return dict(ok=False, error="every parameter is fixed; nothing to fit")

    def model(xv, *free_vals):
        vals = dict(fixed)
        vals.update(zip(free, free_vals))
        return _gauss_lin(xv, vals["A"], vals["mu"], vals["sigma"],
                          vals["m"], vals["b"])

    try:
        popt, pcov = curve_fit(
            model, xs, ys, p0=[p0[n] for n in free],
            # Poisson weights for counting data: per-bin variance = the
            # counts, and pcov reports absolute uncertainties
            sigma=np.sqrt(np.clip(ys, 1.0, None)), absolute_sigma=True,
            bounds=([lb[n] for n in free], [ub[n] for n in free]),
            maxfev=5000)
    except Exception as e:
        return dict(ok=False, error=f"fit did not converge: {e}")

    vals = dict(fixed)
    vals.update(zip(free, (float(v) for v in popt)))
    perr_free = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))
    err = {n: 0.0 for n in _GL_PARAMS}
    err.update(zip(free, (float(e) for e in perr_free)))

    A, mu, sigma, m, b = (vals[n] for n in _GL_PARAMS)
    dA, dmu, dsigma = err["A"], err["mu"], err["sigma"]

    fwhm = _FWHM_K * sigma
    dfwhm = _FWHM_K * dsigma
    area = A * sigma * np.sqrt(2.0 * np.pi) / bw
    # propagate A and sigma errors (correlation ignored; quoted as estimate)
    darea = area * float(np.hypot(dA / A if A else 0.0,
                                  dsigma / sigma if sigma else 0.0))

    yhat = _gauss_lin(xs, A, mu, sigma, m, b)
    dof = max(xs.size - len(free), 1)
    redchi = float(np.sum((ys - yhat) ** 2 / np.clip(yhat, 1.0, None)) / dof)

    xx = np.linspace(xs[0], xs[-1], 400)
    return dict(ok=True, A=A, dA=dA, mu=mu, dmu=dmu, sigma=sigma,
                dsigma=dsigma, m=m, b=b, fwhm=fwhm, dfwhm=dfwhm,
                area=area, darea=darea, redchi=redchi,
                win_lo=lo, win_hi=hi,
                xx=xx, y_fit=_gauss_lin(xx, A, mu, sigma, m, b), y_bg=m * xx + b)


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
    background, and measure a crude FWHM at half of (summit - background).
    Returns ``{ok, mu, fwhm, half_window}`` with ``half_window = 3*FWHM``,
    floored at 8 bins. Returns ``{ok: False, error}`` when the click shows no
    significant peak (summit fails a 3-sigma Poisson test against the local
    background)."""
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
    """Click-to-fit with an automatic window (plan A).

    Estimate the window from the data (:func:`estimate_fit_window`), fit, then
    refine once over ``mu_fit +- 4*sigma_fit`` so the final window adapts to
    the *fitted* width. The returned dict also carries the window actually used
    (``win_lo``/``win_hi``).

    ``max_half_window`` (the Config cap, in x units) clamps BOTH passes' half
    windows; a window clamped below the fit minimum returns ``ok=False`` so the
    caller can skip it."""
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
    ``center`` (x units), NOT chosen from the data.

    ``estimate_fit_window`` is deliberately avoided here — its hill-climb snaps
    to the strongest local summit, which is the wrong peak in the exact case
    Fix Peak exists for (a small peak beside a big neighbour). The half-width is
    half the cap when a Config cap is set (so the full window equals
    ``cap_bins`` — the same meaning the cap has in the auto path), else ``50``
    bins. Returns ``(lo, hi)`` in x units."""
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


def format_peak_output(peaks, properties, datax):
    """Build the per-peak result lines shown in the peak-analysis output box.

    One line per peak: its position (``datax`` at the peak index) and FWHM (the
    scipy ``widths`` property). Moved verbatim from ``GUI.py:update_peak_output``.
    """
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
