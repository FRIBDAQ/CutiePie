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

from services.peak_finder import find_peaks_in_range, format_peak_output


def _bump(x, center, height, sigma=3.0):
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


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
