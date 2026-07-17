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


# ------------------------------- Peak Finder 2: click-to-fit gaussian+linear

from services.peak_finder import fit_gaussian_linear


def _gauss_line(x, A, mu, sigma, m, b):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + m * x + b


def test_fit_recovers_known_gaussian_on_slope():
    rng = np.random.default_rng(7)
    x = np.arange(0.0, 400.0, 1.0)
    y = _gauss_line(x, A=120.0, mu=207.0, sigma=6.0, m=-0.05, b=60.0)
    y = y + rng.normal(0.0, 1.0, x.size)          # mild noise

    r = fit_gaussian_linear(x, y, center=204.0, half_window=40.0)
    assert r["ok"] is True
    assert abs(r["mu"] - 207.0) < 0.5
    assert abs(r["sigma"] - 6.0) < 0.5
    assert abs(r["A"] - 120.0) < 5.0
    assert abs(r["fwhm"] - 6.0 * 2.3548) < 1.0
    # net gaussian counts = A*sigma*sqrt(2pi)/bin_width (bin width 1 here)
    expected_area = 120.0 * 6.0 * np.sqrt(2 * np.pi)
    assert abs(r["area"] - expected_area) / expected_area < 0.05
    # uncertainties present and finite
    for k in ("dmu", "dsigma", "dA", "dfwhm", "darea"):
        assert np.isfinite(r[k])
    # sampled curves for drawing: same x grid, fit above bg at the peak
    assert r["xx"].shape == r["y_fit"].shape == r["y_bg"].shape
    i = int(np.argmin(np.abs(r["xx"] - r["mu"])))
    assert r["y_fit"][i] > r["y_bg"][i]


def test_fit_seeds_from_local_max_not_click():
    # click slightly off-peak still converges to the true centroid
    x = np.arange(0.0, 200.0, 1.0)
    y = _gauss_line(x, A=80.0, mu=100.0, sigma=4.0, m=0.0, b=10.0)
    r = fit_gaussian_linear(x, y, center=93.0, half_window=25.0)
    assert r["ok"] and abs(r["mu"] - 100.0) < 0.5


def test_fit_window_clipped_at_spectrum_edge():
    x = np.arange(0.0, 100.0, 1.0)
    y = _gauss_line(x, A=50.0, mu=8.0, sigma=3.0, m=0.0, b=5.0)
    r = fit_gaussian_linear(x, y, center=8.0, half_window=30.0)   # window spills left
    assert r["ok"] and abs(r["mu"] - 8.0) < 1.0


def test_fit_too_few_points_fails_cleanly():
    x = np.arange(0.0, 100.0, 1.0)
    y = np.ones_like(x)
    r = fit_gaussian_linear(x, y, center=50.0, half_window=1.0)
    assert r["ok"] is False
    assert isinstance(r["error"], str) and r["error"]


def test_fit_flat_data_fails_cleanly_or_zero_area():
    x = np.arange(0.0, 100.0, 1.0)
    y = np.full_like(x, 7.0)
    r = fit_gaussian_linear(x, y, center=50.0, half_window=20.0)
    # flat data: either the fit fails, or it converges to ~zero amplitude
    assert (r["ok"] is False) or (abs(r["A"]) < 1.0)


def test_fit_area_respects_bin_width():
    # same gaussian sampled with 2-unit bins: counts-per-bin area halves? No —
    # area in COUNTS = A*sigma*sqrt(2pi)/bin_width; with bw=2 the summed counts
    # under the peak are half those of bw=1 sampling
    x = np.arange(0.0, 400.0, 2.0)
    y = _gauss_line(x, A=100.0, mu=200.0, sigma=8.0, m=0.0, b=0.0)
    r = fit_gaussian_linear(x, y, center=200.0, half_window=60.0)
    assert r["ok"]
    expected = 100.0 * 8.0 * np.sqrt(2 * np.pi) / 2.0
    assert abs(r["area"] - expected) / expected < 0.05


def test_fit_output_line_format():
    from services.peak_finder import format_gauss_fit_output
    r = dict(ok=True, mu=7449.3, dmu=0.4, A=123.4, dA=5.6, sigma=12.3,
             dsigma=0.5, fwhm=29.0, dfwhm=1.1, area=4530.0, darea=120.0,
             m=-0.05, b=60.0, redchi=1.23)
    text = format_gauss_fit_output(3, r)
    assert "Peak 3" in text
    assert "7449.3" in text
    assert "FWHM" in text and "area" in text


# ---------------------------- Peak Finder 2: automatic fit window (plan A)

from services.peak_finder import estimate_fit_window, fit_gaussian_linear_auto


def test_auto_fit_narrow_and_wide_peaks_same_click_style():
    # no width input: the window must adapt to the peak itself
    rng = np.random.default_rng(11)
    x = np.arange(0.0, 600.0, 1.0)
    for sigma in (2.0, 15.0):
        y = _gauss_line(x, A=150.0, mu=300.0, sigma=sigma, m=-0.02, b=40.0)
        y = y + rng.normal(0.0, 1.5, x.size)
        r = fit_gaussian_linear_auto(x, y, center=300.0)
        assert r["ok"], f"sigma={sigma}: {r.get('error')}"
        assert abs(r["mu"] - 300.0) < 1.0, f"sigma={sigma}"
        assert abs(r["sigma"] - sigma) / sigma < 0.15, f"sigma={sigma}"


def test_auto_fit_off_summit_click_converges():
    x = np.arange(0.0, 400.0, 1.0)
    y = _gauss_line(x, A=100.0, mu=200.0, sigma=8.0, m=0.0, b=20.0)
    # click on the flank, 1.5 sigma off the summit
    r = fit_gaussian_linear_auto(x, y, center=212.0)
    assert r["ok"] and abs(r["mu"] - 200.0) < 1.0


def test_auto_fit_flat_background_fails_cleanly():
    rng = np.random.default_rng(3)
    x = np.arange(0.0, 400.0, 1.0)
    y = np.full_like(x, 50.0) + rng.normal(0.0, np.sqrt(50.0), x.size)
    r = fit_gaussian_linear_auto(x, y, center=200.0)
    assert r["ok"] is False
    assert "no peak" in r["error"].lower()


def test_auto_fit_peak_near_spectrum_edge():
    x = np.arange(0.0, 200.0, 1.0)
    y = _gauss_line(x, A=90.0, mu=12.0, sigma=4.0, m=0.0, b=10.0)
    r = fit_gaussian_linear_auto(x, y, center=12.0)
    assert r["ok"] and abs(r["mu"] - 12.0) < 1.0


def test_auto_fit_reports_window_used():
    x = np.arange(0.0, 400.0, 1.0)
    y = _gauss_line(x, A=100.0, mu=200.0, sigma=6.0, m=0.0, b=15.0)
    r = fit_gaussian_linear_auto(x, y, center=200.0)
    assert r["ok"]
    assert r["win_lo"] < r["mu"] < r["win_hi"]
    # window should be a few sigma wide, not the whole spectrum
    span = r["win_hi"] - r["win_lo"]
    assert 4 * r["sigma"] < span < 20 * r["sigma"]
    # and the formatter mentions it
    from services.peak_finder import format_gauss_fit_output
    assert "window" in format_gauss_fit_output(1, r)


def test_estimate_fit_window_snaps_and_scales():
    x = np.arange(0.0, 400.0, 1.0)
    y = _gauss_line(x, A=100.0, mu=200.0, sigma=10.0, m=0.0, b=5.0)
    est = estimate_fit_window(x, y, center=195.0)     # off-summit click
    assert est["ok"]
    assert abs(est["mu"] - 200.0) < 3.0               # snapped to the summit
    fwhm_true = 10.0 * 2.3548
    assert 0.5 * fwhm_true < est["fwhm"] < 2.0 * fwhm_true
    assert est["half_window"] >= est["fwhm"]          # window spans the peak
