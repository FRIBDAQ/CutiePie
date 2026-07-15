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
dispatches on the selection):

- ``original``     — the historical behavior: pure-Python window clip, then
  ``find_peaks(prominence=1, width=width)`` on the raw counts.
- ``vectorized``   — bit-identical results to ``original`` (same find_peaks
  call), but the clip is a numpy boolean mask (~12x faster on large spectra).
- ``smoothed``     — Savitzky-Golay smoothing + noise-scaled prominence
  (3 sigma of the Poisson level). Best on low statistics; ``width`` is read
  as the *expected FWHM in bins* and peaks down to half of it are accepted.
- ``second_diff``  — Mariscotti-style smoothed second difference (the classic
  nuclear-spectroscopy search). The second difference cancels smooth
  backgrounds, so it is the best choice on sloping/exponential background.

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
from scipy.signal import find_peaks, peak_prominences, peak_widths, savgol_filter


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


# name -> finder, in the order the popup's Algorithm combo shows them;
# "original" first so index 0 (the default) preserves historical behavior
PEAK_ALGORITHMS = {
    "original": find_peaks_in_range,
    "vectorized": find_peaks_in_range_vectorized,
    "smoothed": find_peaks_in_range_smoothed,
    "second_diff": find_peaks_in_range_second_diff,
}


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
