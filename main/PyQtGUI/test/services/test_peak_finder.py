"""Headless net for the peaks cluster extracted from GUI.py.

Pins the numeric half of peak analysis that was previously buried in
MainWindow.analyzePeak/update_peak_output: clip the spectrum to the visible
x window ([xmin, xmax)), run scipy find_peaks over the clipped counts, and
format the per-peak output lines. Qt-free (numpy/scipy only), so it runs in the
system python3 with no PyQt5.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

import numpy as np
import pytest

from services.peak_finder import (
    PEAK_ALGORITHMS,
    find_peaks_in_range,
    find_peaks_in_range_second_diff,
    find_peaks_in_range_smoothed,
    find_peaks_in_range_vectorized,
    format_peak_labels,
    format_peak_output,
)

# every algorithm must feed GUI.drawSinglePeaks / format_peak_output
REQUIRED_PROPERTY_KEYS = ("prominences", "widths", "width_heights", "left_ips", "right_ips")


def _bump(x, center, height, sigma=3.0):
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _noisy_spectrum(nbins=1024, centers=(200, 500, 800), amp=200.0, bg=5.0, seed=7):
    """Gamma-like test spectrum: Gaussian peaks + flat background, Poisson-sampled."""
    rng = np.random.default_rng(seed)
    x = np.arange(nbins, dtype=float)
    mu = np.full(nbins, bg)
    for c in centers:
        mu += _bump(x, c, amp)
    return x, rng.poisson(mu).astype(float)


# ------------------------------------------------------------- find_peaks_in_range

def test_finds_a_peak_inside_the_window():
    x = np.arange(0, 100, dtype=float)
    y = _bump(x, 50, 20)
    datax, datay, peaks, props = find_peaks_in_range(x, y, 0, 100, width=1)
    assert len(peaks) == 1
    # peak index is into the clipped datax; its x should be ~50
    assert abs(datax[peaks[0]] - 50) <= 1


def test_subrange_filter_excludes_out_of_window_peaks():
    x = np.arange(0, 100, dtype=float)
    y = _bump(x, 20, 20) + _bump(x, 80, 20)
    # window keeps only the peak at 80
    datax, datay, peaks, props = find_peaks_in_range(x, y, 60, 100, width=1)
    assert len(peaks) == 1
    assert abs(datax[peaks[0]] - 80) <= 1
    # datax is clipped to the window
    assert datax.min() >= 60 and datax.max() < 100


def test_half_open_window_upper_bound_excluded():
    x = np.arange(0, 10, dtype=float)
    y = np.ones(10)
    datax, _, _, _ = find_peaks_in_range(x, y, 2, 5, width=1)
    # [2, 5) keeps 2,3,4 — not 5
    assert list(datax) == [2.0, 3.0, 4.0]


def test_flat_data_yields_no_peaks():
    x = np.arange(0, 50, dtype=float)
    y = np.ones(50)
    _, _, peaks, _ = find_peaks_in_range(x, y, 0, 50, width=1)
    assert len(peaks) == 0


def test_prominence_gate_ignores_tiny_bump():
    # find_peaks is called with prominence=1; a sub-1 bump must be rejected
    x = np.arange(0, 50, dtype=float)
    y = np.zeros(50)
    y[25] = 0.5
    _, _, peaks, _ = find_peaks_in_range(x, y, 0, 50, width=1)
    assert len(peaks) == 0


# --------------------------------------------------------------- format_peak_output

def test_format_output_one_line_per_peak():
    x = np.arange(0, 100, dtype=float)
    y = _bump(x, 30, 20) + _bump(x, 70, 20)
    datax, datay, peaks, props = find_peaks_in_range(x, y, 0, 100, width=1)
    lines = format_peak_output(peaks, props, datax)
    assert len(lines) == len(peaks)
    assert lines[0].startswith("Peak1")
    assert "peak @ " in lines[0] and "FWHM=" in lines[0]


def test_format_output_reports_peak_position():
    x = np.arange(0, 100, dtype=float)
    y = _bump(x, 50, 20)
    datax, datay, peaks, props = find_peaks_in_range(x, y, 0, 100, width=1)
    lines = format_peak_output(peaks, props, datax)
    # the reported position should be ~50
    pos = int(lines[0].split("peak @ ")[1].split(",")[0])
    assert abs(pos - 50) <= 1


def test_format_output_empty_when_no_peaks():
    assert format_peak_output([], {'widths': []}, np.array([])) == []


def test_format_labels_one_per_peak_with_position_and_fwhm():
    x = np.arange(0, 100, dtype=float)
    y = _bump(x, 30, 20) + _bump(x, 70, 20)
    datax, datay, peaks, props = find_peaks_in_range(x, y, 0, 100, width=1)
    labels = format_peak_labels(peaks, props, datax)
    assert len(labels) == len(peaks) == 2
    assert labels[0].startswith("Peak 1 @ ")
    assert "(FWHM=" in labels[0]
    # single-line (list labels), unlike format_peak_output's two-line entries
    assert all("\n" not in lab for lab in labels)
    pos = int(labels[0].split("@ ")[1].split(" ")[0])
    assert abs(pos - 30) <= 1


def test_format_labels_empty_when_no_peaks():
    assert format_peak_labels([], {'widths': []}, np.array([])) == []


# ------------------------------------------------------- vectorized (drop-in)

def test_vectorized_is_bit_identical_to_original():
    x, y = _noisy_spectrum()
    for xmin, xmax in ((0, 1024), (150.3, 860.9)):
        dx1, dy1, p1, pr1 = find_peaks_in_range(x, y.tolist(), xmin, xmax, 5)
        dx2, dy2, p2, pr2 = find_peaks_in_range_vectorized(x, y.tolist(), xmin, xmax, 5)
        assert np.array_equal(dx1, dx2)
        assert np.array_equal(dy1, dy2)
        assert np.array_equal(p1, p2)
        assert np.allclose(pr1["widths"], pr2["widths"])
        assert np.allclose(pr1["prominences"], pr2["prominences"])


def test_vectorized_half_open_window_upper_bound_excluded():
    x = np.arange(0, 10, dtype=float)
    y = np.ones(10)
    datax, _, _, _ = find_peaks_in_range_vectorized(x, y, 2, 5, width=1)
    assert list(datax) == [2.0, 3.0, 4.0]


# ---------------------------------------------------------------- smoothed

def test_smoothed_finds_true_peaks_without_noise_junk():
    x, y = _noisy_spectrum(centers=(200, 500, 800))
    datax, datay, peaks, props = find_peaks_in_range_smoothed(x, y, 0, 1024, width=7)
    found = sorted(datax[p] for p in peaks)
    assert len(found) == 3
    for got, want in zip(found, (200, 500, 800)):
        assert abs(got - want) <= 4


def test_smoothed_properties_contract():
    x, y = _noisy_spectrum()
    datax, datay, peaks, props = find_peaks_in_range_smoothed(x, y, 0, 1024, width=7)
    assert len(peaks) > 0
    for key in REQUIRED_PROPERTY_KEYS:
        assert key in props
        assert len(props[key]) == len(peaks)
        assert np.all(np.isfinite(props[key]))
    # datay is the RAW clipped data (markers must sit on the real spectrum)
    assert np.array_equal(datay, y)


def test_smoothed_tiny_window_yields_no_peaks_and_no_crash():
    x = np.arange(0, 8, dtype=float)
    y = np.ones(8)
    datax, datay, peaks, props = find_peaks_in_range_smoothed(x, y, 0, 8, width=3)
    assert len(peaks) == 0
    assert format_peak_output(peaks, props, datax) == []


# -------------------------------------------------------------- second_diff

def test_second_diff_finds_peaks_on_sloping_background():
    # exponential background: raw prominence-based search drowns here,
    # the second difference cancels the smooth slope
    rng = np.random.default_rng(11)
    x = np.arange(2048, dtype=float)
    mu = 200.0 * np.exp(-x / 700.0) + 2.0
    for c in (400, 1000, 1600):
        mu += _bump(x, c, 120.0)
    y = rng.poisson(mu).astype(float)
    datax, datay, peaks, props = find_peaks_in_range_second_diff(x, y, 0, 2048, width=7)
    found = sorted(datax[p] for p in peaks)
    assert len(found) == 3
    for got, want in zip(found, (400, 1000, 1600)):
        assert abs(got - want) <= 4


def test_second_diff_properties_contract():
    x, y = _noisy_spectrum()
    datax, datay, peaks, props = find_peaks_in_range_second_diff(x, y, 0, 1024, width=7)
    assert len(peaks) > 0
    for key in REQUIRED_PROPERTY_KEYS:
        assert key in props
        assert len(props[key]) == len(peaks)
        assert np.all(np.isfinite(props[key]))
    assert np.array_equal(datay, y)
    # output formatting must work end-to-end (int() casts on widths)
    lines = format_peak_output(peaks, props, datax)
    assert len(lines) == len(peaks)


def test_second_diff_flat_data_yields_no_peaks():
    x = np.arange(0, 200, dtype=float)
    y = np.full(200, 5.0)
    _, _, peaks, _ = find_peaks_in_range_second_diff(x, y, 0, 200, width=7)
    assert len(peaks) == 0


def test_second_diff_tiny_window_yields_no_peaks_and_no_crash():
    x = np.arange(0, 4, dtype=float)
    y = np.ones(4)
    _, _, peaks, props = find_peaks_in_range_second_diff(x, y, 0, 4, width=7)
    assert len(peaks) == 0


# ------------------------------------------------------------ dispatch table

def test_algorithm_dispatch_table():
    # dict order = combo order; index 0 is the combo default (Mariscotti,
    # user-chosen 2026-07-15 — the legacy raw-counts search is NOT default)
    assert list(PEAK_ALGORITHMS.keys()) == [
        "Mariscotti (2nd difference)",
        "Smoothed (Savitzky-Golay)",
        "Raw counts (legacy)",
        "Raw counts (legacy, fast)",
    ]
    assert PEAK_ALGORITHMS["Mariscotti (2nd difference)"] is find_peaks_in_range_second_diff
    assert PEAK_ALGORITHMS["Smoothed (Savitzky-Golay)"] is find_peaks_in_range_smoothed
    assert PEAK_ALGORITHMS["Raw counts (legacy)"] is find_peaks_in_range
    assert PEAK_ALGORITHMS["Raw counts (legacy, fast)"] is find_peaks_in_range_vectorized


def test_all_algorithms_share_the_return_contract():
    x, y = _noisy_spectrum()
    for name, fn in PEAK_ALGORITHMS.items():
        datax, datay, peaks, props = fn(x, y.tolist(), 100, 900, 7)
        assert isinstance(datax, np.ndarray) and isinstance(datay, np.ndarray)
        assert len(datax) == len(datay)
        for key in REQUIRED_PROPERTY_KEYS:
            assert key in props, f"{name} missing {key}"
        assert all(0 <= p < len(datax) for p in peaks), name
