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


def test_auto_fit_wide_peak_with_noise_bump_on_flank():
    # a small bump on the flank of a wide peak must not stall the
    # hill-climb — the estimator should still find the true summit
    x = np.arange(0.0, 800.0, 1.0)
    y = _gauss_line(x, A=100.0, mu=400.0, sigma=25.0, m=0.0, b=10.0)
    # a bump strong enough to be a genuine local max on the flank
    # (20 counts, sigma 2.5, 20 bins off-summit); the naive +-3 climb
    # stalls on it (measured: est mu = 419 pre-fix)
    y += 20.0 * np.exp(-0.5 * ((x - 420.0) / 2.5) ** 2)
    est = estimate_fit_window(x, y, center=430.0)   # click beyond the bump
    assert est["ok"]
    assert abs(est["mu"] - 400.0) < 6.0             # summit, not the bump
    r = fit_gaussian_linear_auto(x, y, center=430.0)
    assert r["ok"] and abs(r["mu"] - 400.0) < 2.0


def test_fit_poisson_calibration():
    # characterization: on true-model Poisson data the fit must recover
    # the params and report red-chi2 ~ 1. NOTE: this does NOT discriminate
    # weighted from unweighted curve_fit (measured pull RMS 0.99 vs 1.06;
    # curve_fit's default cov rescaling self-corrects on average); the
    # sigma=sqrt(y), absolute_sigma=True choice is correctness-in-principle,
    # pinned here only against gross regression.
    rng = np.random.default_rng(5)
    x = np.arange(0.0, 300.0, 1.0)
    y = _gauss_line(x, A=400.0, mu=150.0, sigma=8.0, m=0.0, b=100.0)
    y = rng.poisson(np.clip(y, 0, None)).astype(float)
    r = fit_gaussian_linear(x, y, center=150.0, half_window=60.0)
    assert r["ok"]
    assert abs(r["mu"] - 150.0) < 0.7
    # Poisson-weighted red-chi2 on true-model Poisson data must be ~1
    assert 0.5 < r["redchi"] < 2.0


# ===================== INTERSPEC Phase A1: generalized range fit ================

from services.peak_finder import fit_gaussian_linear_range


def test_A1_range_fit_recovers_peak_asymmetric_window():
    x = np.arange(0.0, 400.0, 1.0)
    y = _gauss_line(x, A=120.0, mu=200.0, sigma=6.0, m=-0.02, b=50.0)
    r = fit_gaussian_linear_range(x, y, lo=170.0, hi=260.0)   # 30 left, 60 right
    assert r["ok"]
    assert abs(r["mu"] - 200.0) < 0.7
    assert r["win_lo"] == 170.0 and r["win_hi"] == 260.0


def test_A1_range_fit_swaps_reversed_bounds():
    x = np.arange(0.0, 400.0, 1.0)
    y = _gauss_line(x, A=100.0, mu=200.0, sigma=6.0, m=0.0, b=40.0)
    r = fit_gaussian_linear_range(x, y, lo=250.0, hi=150.0)   # reversed
    assert r["ok"]
    assert r["win_lo"] == 150.0 and r["win_hi"] == 250.0


def test_A1_fit_gaussian_linear_delegates_to_range():
    # The old center/half_window API must equal the explicit-window one.
    x = np.arange(0.0, 400.0, 1.0)
    y = _gauss_line(x, A=90.0, mu=205.0, sigma=5.0, m=-0.03, b=45.0)
    a = fit_gaussian_linear(x, y, center=205.0, half_window=45.0)
    b = fit_gaussian_linear_range(x, y, lo=160.0, hi=250.0)
    assert a["ok"] and b["ok"]
    for k in ("A", "mu", "sigma", "m", "b"):
        assert abs(a[k] - b[k]) < 1e-9


def test_A1_range_fit_fixed_mu_pins_on_shoulder():
    x = np.arange(0.0, 400.0, 1.0)
    y = (_gauss_line(x, A=500.0, mu=180.0, sigma=12.0, m=0.0, b=20.0)
         + 60.0 * np.exp(-0.5 * ((x - 232.0) / 6.0) ** 2))
    pinned = fit_gaussian_linear_range(x, y, lo=205.0, hi=260.0,
                                       fixed={"mu": 232.0})
    assert pinned["ok"]
    assert pinned["mu"] == 232.0        # pinned exactly
    assert pinned["dmu"] == 0.0         # pinned -> zero uncertainty
    assert pinned["A"] > 0.0


def test_A1_range_fit_fixed_sigma_pins_width():
    x = np.arange(0.0, 400.0, 1.0)
    y = _gauss_line(x, A=100.0, mu=200.0, sigma=8.0, m=0.0, b=30.0)
    r = fit_gaussian_linear_range(x, y, lo=160.0, hi=240.0, fixed={"sigma": 5.0})
    assert r["ok"]
    assert r["sigma"] == 5.0 and r["dsigma"] == 0.0


def test_A1_range_fit_seeds_accepted():
    x = np.arange(0.0, 400.0, 1.0)
    y = _gauss_line(x, A=100.0, mu=200.0, sigma=6.0, m=0.0, b=30.0)
    r = fit_gaussian_linear_range(x, y, lo=160.0, hi=240.0,
                                  seeds={"mu": 201.0, "sigma": 7.0})
    assert r["ok"]
    assert abs(r["mu"] - 200.0) < 1.0


def test_A1_range_fit_unknown_param_name_errors():
    x = np.arange(0.0, 400.0, 1.0)
    y = _gauss_line(x, A=100.0, mu=200.0, sigma=6.0, m=0.0, b=30.0)
    r = fit_gaussian_linear_range(x, y, lo=160.0, hi=240.0, fixed={"bogus": 1.0})
    assert r["ok"] is False and r["error"]


def test_A1_range_fit_too_narrow_fails_cleanly():
    x = np.arange(0.0, 400.0, 1.0)
    y = _gauss_line(x, A=100.0, mu=200.0, sigma=6.0, m=0.0, b=30.0)
    r = fit_gaussian_linear_range(x, y, lo=199.0, hi=201.0)   # ~3 bins
    assert r["ok"] is False and r["error"]


# ===================== INTERSPEC Phase A2: auto-window cap ======================

def test_A2_cap_clamps_window():
    x = np.arange(0.0, 600.0, 1.0)
    y = _gauss_line(x, A=150.0, mu=300.0, sigma=15.0, m=0.0, b=30.0)
    free = fit_gaussian_linear_auto(x, y, center=300.0)
    capped = fit_gaussian_linear_auto(x, y, center=300.0, max_half_window=25.0)
    assert free["ok"] and capped["ok"]
    assert (free["win_hi"] - free["win_lo"]) > 60.0
    assert (capped["win_hi"] - capped["win_lo"]) <= 51.0     # 2*cap + slop
    assert abs(capped["mu"] - 300.0) < 1.5


def test_A2_cap_none_is_uncapped():
    x = np.arange(0.0, 600.0, 1.0)
    y = _gauss_line(x, A=150.0, mu=300.0, sigma=15.0, m=0.0, b=30.0)
    a = fit_gaussian_linear_auto(x, y, center=300.0)
    b = fit_gaussian_linear_auto(x, y, center=300.0, max_half_window=None)
    assert a["ok"] and b["ok"]
    assert (a["win_hi"] - a["win_lo"]) == (b["win_hi"] - b["win_lo"])


def test_A2_cap_too_tight_fails_cleanly():
    x = np.arange(0.0, 200.0, 1.0)
    y = _gauss_line(x, A=100.0, mu=100.0, sigma=5.0, m=0.0, b=10.0)
    r = fit_gaussian_linear_auto(x, y, center=100.0, max_half_window=2.0)
    assert r["ok"] is False and r["error"]


# ===================== INTERSPEC Phase A3: Fix-Peak window + output tag =========

from services.peak_finder import fix_peak_window


def test_A3_fix_window_default_half_width_50_bins():
    lo, hi = fix_peak_window(center=300.0, bin_width=2.0, cap_bins=None)
    assert lo == 300.0 - 50 * 2.0
    assert hi == 300.0 + 50 * 2.0


def test_A3_fix_window_cap_sets_full_window():
    # Cap set: full window = cap bins (half-width = 0.5*cap).
    lo, hi = fix_peak_window(center=300.0, bin_width=2.0, cap_bins=60)
    assert hi - lo == 60 * 2.0
    assert lo == 300.0 - 30 * 2.0 and hi == 300.0 + 30 * 2.0


def test_A3_fix_window_symmetric_about_center():
    lo, hi = fix_peak_window(center=1234.5, bin_width=1.0, cap_bins=None)
    assert abs((hi - 1234.5) - (1234.5 - lo)) < 1e-9


def test_A3_fix_window_recovers_amplitude_with_mu_pinned():
    # Small peak on the flank of a big one: pinning mu recovers it.
    x = np.arange(0.0, 400.0, 1.0)
    y = (_gauss_line(x, A=500.0, mu=180.0, sigma=12.0, m=0.0, b=20.0)
         + 60.0 * np.exp(-0.5 * ((x - 230.0) / 6.0) ** 2))
    lo, hi = fix_peak_window(center=230.0, bin_width=1.0, cap_bins=None)
    r = fit_gaussian_linear_range(x, y, lo, hi, fixed={"mu": 230.0})
    assert r["ok"]
    assert r["mu"] == 230.0 and r["dmu"] == 0.0 and r["A"] > 0.0


def test_A3_format_output_tag_marks_fixed_mu():
    from services.peak_finder import format_gauss_fit_output
    r = dict(ok=True, mu=230.0, dmu=0.0, A=60.0, dA=3.0, sigma=6.0,
             dsigma=0.3, fwhm=14.1, dfwhm=0.7, area=900.0, darea=40.0,
             m=0.0, b=20.0, redchi=1.1)
    tagged = format_gauss_fit_output(4, r, tag="fixed μ")
    assert "Peak 4" in tagged and "fixed μ" in tagged
    # default (no tag) stays byte-identical to today's format
    assert format_gauss_fit_output(4, r) == format_gauss_fit_output(4, r, tag=None)
    assert "(fixed" not in format_gauss_fit_output(4, r)


# ===================== Peak Finder 2: duplicate-fit suppression (auto mode) =====

from services.peak_finder import find_duplicate_mu


def test_dup_none_when_no_existing():
    assert find_duplicate_mu(200.0, [], tol=1.0) is None


def test_dup_matches_within_tolerance():
    # existing peaks at 100, 200, 300; new mu 200.4 within 1-bin tol of index 1
    assert find_duplicate_mu(200.4, [100.0, 200.0, 300.0], tol=1.0) == 1


def test_dup_none_when_outside_tolerance():
    assert find_duplicate_mu(205.0, [100.0, 200.0, 300.0], tol=1.0) is None


def test_dup_returns_first_match():
    assert find_duplicate_mu(200.0, [200.0, 200.5], tol=1.0) == 0


def test_dup_boundary_exactly_at_tol_is_duplicate():
    assert find_duplicate_mu(201.0, [200.0], tol=1.0) == 0


# ===================== Peak Finder 2: results-table row formatter ==============

from services.peak_finder import (
    format_gauss_fit_row, format_gauss_fit_output, PEAK2_TABLE_COLUMNS,
)


def _row_result():
    return dict(ok=True, mu=6623.4, dmu=0.42, A=123.4, dA=5.6, sigma=12.3,
                dsigma=0.51, fwhm=29.02, dfwhm=1.12, area=4530.0, darea=120.0,
                m=-0.05, b=60.1, redchi=1.23, win_lo=6580.0, win_hi=6670.0)


def test_row_columns_are_five_in_order():
    assert PEAK2_TABLE_COLUMNS == ("#", "μ", "FWHM", "area", "χ²ᵣ")


def test_row_cells_display_and_sortkeys():
    row = format_gauss_fit_row(1, _row_result())
    cells = row["cells"]
    assert len(cells) == 5
    # display text carries the ± and value; sort key is the raw numeric
    assert cells[0][0] == "1" and cells[0][1] == 1.0
    assert cells[1][0].startswith("6623.4") and "±" in cells[1][0]
    assert cells[1][1] == 6623.4
    assert cells[2][1] == 29.02        # FWHM sort key
    assert cells[3][1] == 4530.0       # area sort key
    assert cells[4][0] == "1.23" and cells[4][1] == 1.23


def test_row_tag_marks_number_cell_and_tooltip():
    row = format_gauss_fit_row(2, _row_result(), tag="fixed μ")
    assert "*" in row["cells"][0][0]           # # cell marked
    assert "fixed μ" in row["tooltip"]


def test_row_tooltip_reuses_detailed_formatter():
    r = _row_result()
    row = format_gauss_fit_row(3, r)
    assert row["tooltip"] == format_gauss_fit_output(3, r)


def test_row_untagged_has_no_marker():
    row = format_gauss_fit_row(1, _row_result())
    assert row["cells"][0][0] == "1" and row["tag"] is None


# ===================== Peak Finder 2: drag-to-refit grab geometry ==============

from services.peak_finder import nearest_window_edge


def test_grab_none_when_far_from_both_edges():
    assert nearest_window_edge(50.0, lo=0.0, hi=100.0, tol=8.0) is None


def test_grab_lo_edge_within_tol():
    assert nearest_window_edge(3.0, lo=0.0, hi=100.0, tol=8.0) == "lo"


def test_grab_hi_edge_within_tol():
    assert nearest_window_edge(96.0, lo=0.0, hi=100.0, tol=8.0) == "hi"


def test_grab_ties_to_closer_edge():
    # x=40 is closer to lo(0) than hi(100)? no — 40 to 0 is 40, to 100 is 60;
    # but both outside tol → None. Use a case inside tol for both:
    assert nearest_window_edge(4.0, lo=0.0, hi=8.0, tol=8.0) == "lo"   # dlo=4 < dhi=4? tie→lo
    assert nearest_window_edge(5.0, lo=0.0, hi=8.0, tol=8.0) == "hi"   # dlo=5 > dhi=3


def test_grab_exactly_at_tol_grabs():
    assert nearest_window_edge(8.0, lo=0.0, hi=100.0, tol=8.0) == "lo"


# ===================== Peak Finder 2: sigma <-> FWHM (edit popup link) =========

from services.peak_finder import sigma_to_fwhm, fwhm_to_sigma


def test_sigma_to_fwhm_factor():
    # FWHM = 2*sqrt(2*ln2) * sigma ~= 2.35482 * sigma
    assert abs(sigma_to_fwhm(10.0) - 23.5482) < 1e-3


def test_fwhm_to_sigma_inverse():
    assert abs(fwhm_to_sigma(23.5482) - 10.0) < 1e-3


def test_sigma_fwhm_roundtrip():
    for s in (1.0, 6.3, 42.0):
        assert abs(fwhm_to_sigma(sigma_to_fwhm(s)) - s) < 1e-9


# ===================== Peak Finder 2: edit-param validation (K13f) ============

from services.peak_finder import validate_gauss_edit


def test_validate_rejects_negative_sigma():
    assert validate_gauss_edit({"sigma": -2.0}, lo=30.0, hi=70.0) is not None


def test_validate_rejects_zero_sigma():
    assert validate_gauss_edit({"sigma": 0.0}, lo=30.0, hi=70.0) is not None


def test_validate_rejects_mu_outside_window():
    # mu pinned outside the fit window is what makes the fit degenerate
    assert validate_gauss_edit({"mu": -5.0}, lo=30.0, hi=70.0) is not None
    assert validate_gauss_edit({"mu": 999.0}, lo=30.0, hi=70.0) is not None


def test_validate_accepts_negative_mu_inside_negative_window():
    # a spectrum whose x-axis runs negative has legitimately-negative peak centers
    assert validate_gauss_edit({"mu": -40.0}, lo=-100.0, hi=-10.0) is None


def test_validate_accepts_mu_inside_window():
    assert validate_gauss_edit({"mu": 50.0, "sigma": 4.0}, lo=30.0, hi=70.0) is None


def test_validate_accepts_empty():
    assert validate_gauss_edit({}, lo=30.0, hi=70.0) is None


# ===================== INTERSPEC Phase E1: composite shape registry ============
# Signal shapes gaussian + crystal_ball; background poly1 (raw m*x+b) / poly2 /
# poly3 (mapped t in [-1,1]); fit_composite is the one engine, and
# fit_gaussian_linear_range is a pure gaussian x1 + poly1 delegate of it.

from services.peak_finder import (
    SIGNAL_SHAPES, BACKGROUND_SHAPES, fit_composite,
)


def test_E1_signal_registry_has_gaussian_and_crystal_ball():
    assert set(SIGNAL_SHAPES) == {"gaussian", "crystal_ball"}
    assert SIGNAL_SHAPES["gaussian"]["params"] == ("A", "mu", "sigma")
    assert SIGNAL_SHAPES["gaussian"]["shared"] == ()
    assert SIGNAL_SHAPES["crystal_ball"]["params"] == ("A", "mu", "sigma")
    assert SIGNAL_SHAPES["crystal_ball"]["shared"] == ("alpha", "n")


def test_E1_background_registry_has_poly1_2_3():
    assert set(BACKGROUND_SHAPES) == {"poly1", "poly2", "poly3"}
    assert BACKGROUND_SHAPES["poly1"]["params"] == ("m", "b")
    assert len(BACKGROUND_SHAPES["poly2"]["params"]) == 3
    assert len(BACKGROUND_SHAPES["poly3"]["params"]) == 4


def test_E1_gaussian_eval_matches_formula():
    x = np.linspace(0, 100, 51)
    got = SIGNAL_SHAPES["gaussian"]["eval"](x, (10.0, 50.0, 5.0), {}, {})
    want = 10.0 * np.exp(-0.5 * ((x - 50.0) / 5.0) ** 2)
    assert np.allclose(got, want)


def test_E1_poly1_eval_is_raw_linear():
    x = np.array([0.0, 10.0, 20.0])
    got = BACKGROUND_SHAPES["poly1"]["eval"](x, (2.0, 3.0), 0.0, 20.0)
    assert np.allclose(got, 2.0 * x + 3.0)


def test_E1_poly2_eval_uses_mapped_variable():
    # t = -1 at lo, 0 at mid, +1 at hi; poly2 = c0 + c1*t + c2*t^2
    lo, hi = 100.0, 200.0
    x = np.array([lo, 150.0, hi])
    got = BACKGROUND_SHAPES["poly2"]["eval"](x, (5.0, 2.0, 1.0), lo, hi)
    assert np.allclose(got, [5.0 - 2.0 + 1.0, 5.0, 5.0 + 2.0 + 1.0])


def test_E1_composite_single_gaussian_recovers_peak():
    rng = np.random.default_rng(7)
    x = np.arange(0.0, 400.0, 1.0)
    y = _gauss_line(x, A=120.0, mu=207.0, sigma=6.0, m=-0.05, b=60.0)
    y = y + rng.normal(0.0, 1.0, x.size)
    spec = {"signal": "gaussian", "n_components": 1, "background": "poly1"}
    r = fit_composite(x, y, 167.0, 247.0, spec)
    assert r["ok"]
    assert len(r["components"]) == 1
    c = r["components"][0]
    assert abs(c["mu"] - 207.0) < 0.5
    assert abs(c["sigma"] - 6.0) < 0.5
    assert abs(c["A"] - 120.0) < 5.0
    assert abs(c["fwhm"] - 6.0 * 2.3548) < 1.0
    assert r["xx"].shape == r["y_fit"].shape == r["y_bg"].shape
    assert len(r["y_comp"]) == 1


def test_E1_composite_fixed_mu1_pins_exactly():
    x = np.arange(0.0, 400.0, 1.0)
    y = (_gauss_line(x, A=500.0, mu=180.0, sigma=12.0, m=0.0, b=20.0)
         + 60.0 * np.exp(-0.5 * ((x - 232.0) / 6.0) ** 2))
    spec = {"signal": "gaussian", "n_components": 1, "background": "poly1"}
    r = fit_composite(x, y, 205.0, 260.0, spec, fixed={"mu1": 232.0})
    assert r["ok"]
    assert r["components"][0]["mu"] == 232.0
    assert r["components"][0]["dmu"] == 0.0


def test_E1_composite_unknown_param_errors():
    x = np.arange(0.0, 400.0, 1.0)
    y = _gauss_line(x, A=100.0, mu=200.0, sigma=6.0, m=0.0, b=30.0)
    spec = {"signal": "gaussian", "n_components": 1, "background": "poly1"}
    r = fit_composite(x, y, 160.0, 240.0, spec, fixed={"bogus": 1.0})
    assert r["ok"] is False and r["error"]


def test_E1_composite_too_narrow_fails_cleanly():
    x = np.arange(0.0, 400.0, 1.0)
    y = _gauss_line(x, A=100.0, mu=200.0, sigma=6.0, m=0.0, b=30.0)
    spec = {"signal": "gaussian", "n_components": 1, "background": "poly1"}
    r = fit_composite(x, y, 199.0, 201.0, spec)
    assert r["ok"] is False and r["error"]


def _cb(x, A, mu, sigma, alpha, n, side="low"):
    return SIGNAL_SHAPES["crystal_ball"]["eval"](
        x, (A, mu, sigma), {"alpha": alpha, "n": n}, {"tail_side": side})


def test_E1_crystal_ball_core_matches_gaussian():
    # in the core region (the non-tail flank) CB is exactly the gaussian
    x = np.linspace(150.0, 180.0, 31)   # high side of mu=150, always core for low tail
    cb = _cb(x, 100.0, 150.0, 5.0, 1.5, 3.0, side="low")
    g = 100.0 * np.exp(-0.5 * ((x - 150.0) / 5.0) ** 2)
    assert np.allclose(cb, g)


def test_E1_crystal_ball_low_tail_is_heavier_than_gaussian():
    # far on the low side the power-law tail sits well above the gaussian
    xlo = 150.0 - 5.0 * 5.0            # z = -5, past alpha=1.5
    cb = _cb(xlo, 100.0, 150.0, 5.0, 1.5, 3.0, side="low")
    g = 100.0 * np.exp(-0.5 * (5.0 ** 2))
    assert cb > 10.0 * g


def test_E1_crystal_ball_tail_side_flips():
    z5_lo = 150.0 - 5.0 * 5.0
    z5_hi = 150.0 + 5.0 * 5.0
    g = 100.0 * np.exp(-0.5 * (5.0 ** 2))
    # low-side tail: heavy below, gaussian-light above
    assert _cb(z5_lo, 100.0, 150.0, 5.0, 1.5, 3.0, "low") > 10.0 * g
    assert abs(_cb(z5_hi, 100.0, 150.0, 5.0, 1.5, 3.0, "low") - g) < 1e-6
    # high-side tail: mirror image
    assert _cb(z5_hi, 100.0, 150.0, 5.0, 1.5, 3.0, "high") > 10.0 * g
    assert abs(_cb(z5_lo, 100.0, 150.0, 5.0, 1.5, 3.0, "high") - g) < 1e-6


def test_E1_composite_recovers_crystal_ball_peak():
    rng = np.random.default_rng(3)
    x = np.arange(0.0, 400.0, 1.0)
    y = _cb(x, 300.0, 200.0, 6.0, 1.4, 4.0, side="low") + 40.0
    y = rng.poisson(np.clip(y, 0, None)).astype(float)
    spec = {"signal": "crystal_ball", "n_components": 1,
            "background": "poly1", "tail_side": "low"}
    r = fit_composite(x, y, 160.0, 250.0, spec)
    assert r["ok"]
    c = r["components"][0]
    assert abs(c["mu"] - 200.0) < 1.5
    assert abs(c["sigma"] - 6.0) < 2.0
    assert c["area"] > 0.0


def test_E1_composite_resolves_a_doublet():
    rng = np.random.default_rng(11)
    x = np.arange(0.0, 400.0, 1.0)
    y = (_gauss_line(x, A=150.0, mu=185.0, sigma=5.0, m=0.0, b=30.0)
         + 150.0 * np.exp(-0.5 * ((x - 215.0) / 5.0) ** 2))
    y = rng.poisson(np.clip(y, 0, None)).astype(float)
    spec = {"signal": "gaussian", "n_components": 2, "background": "poly1"}
    r = fit_composite(x, y, 160.0, 240.0, spec)
    assert r["ok"] and len(r["components"]) == 2
    mus = sorted(c["mu"] for c in r["components"])
    assert abs(mus[0] - 185.0) < 2.5
    assert abs(mus[1] - 215.0) < 2.5


def test_E1_composite_n_components_out_of_range_errors():
    x = np.arange(0.0, 400.0, 1.0)
    y = _gauss_line(x, A=100.0, mu=200.0, sigma=6.0, m=0.0, b=30.0)
    spec = {"signal": "gaussian", "n_components": 6, "background": "poly1"}
    assert fit_composite(x, y, 160.0, 240.0, spec)["ok"] is False


def test_E1_composite_unknown_shape_errors():
    x = np.arange(0.0, 400.0, 1.0)
    y = _gauss_line(x, A=100.0, mu=200.0, sigma=6.0, m=0.0, b=30.0)
    spec = {"signal": "lorentzian", "n_components": 1, "background": "poly1"}
    assert fit_composite(x, y, 160.0, 240.0, spec)["ok"] is False


# ---- Phase E1 formatters + the gaussian x1 + poly1 delegate pin --------------

from services.peak_finder import (
    format_composite_fit_output, format_composite_fit_row,
)


def _fit_g1poly1():
    x = np.arange(0.0, 400.0, 1.0)
    y = _gauss_line(x, A=120.0, mu=207.0, sigma=6.0, m=-0.05, b=60.0)
    spec = {"signal": "gaussian", "n_components": 1, "background": "poly1"}
    return fit_composite(x, y, 167.0, 247.0, spec)


def test_E1_output_g1poly1_delegates_to_gauss_formatter():
    # the classic case prints the linear "bg = m·x + b" line only the gauss
    # formatter produces — proof the delegation path is taken
    text = format_composite_fit_output(1, _fit_g1poly1())
    assert "Peak 1" in text
    assert "bg =" in text and "·x" in text
    assert "(auto)" in text


def test_E1_output_multicomponent_header_and_component_lines():
    rng = np.random.default_rng(11)
    x = np.arange(0.0, 400.0, 1.0)
    y = (_gauss_line(x, A=150.0, mu=185.0, sigma=5.0, m=0.0, b=30.0)
         + 150.0 * np.exp(-0.5 * ((x - 215.0) / 5.0) ** 2))
    y = rng.poisson(np.clip(y, 0, None)).astype(float)
    spec = {"signal": "gaussian", "n_components": 2, "background": "poly1"}
    r = fit_composite(x, y, 160.0, 240.0, spec)
    text = format_composite_fit_output(4, r)
    assert "Peak 4" in text
    assert "gaussian x2 + linear" in text
    assert text.count("μ =") == 2          # one component line per peak


def test_E1_output_crystal_ball_and_cubic_labels():
    r = _fit_g1poly1()
    r["spec"] = {"signal": "crystal_ball", "n_components": 1, "background": "poly3"}
    text = format_composite_fit_output(2, r)
    assert "crystal ball" in text and "cubic" in text


def test_E1_output_failed_result():
    text = format_composite_fit_output(3, {"ok": False, "error": "boom"})
    assert "Peak 3" in text and "boom" in text


def test_E1_row_g1poly1_delegates_and_has_no_multiplier():
    row = format_composite_fit_row(1, _fit_g1poly1())
    assert len(row["cells"]) == 5
    assert row["cells"][0][0] == "1"       # no ×k marker for a single component
    assert row["tooltip"] == format_composite_fit_output(1, _fit_g1poly1())


from services.peak_finder import fit_composite_auto


def test_E2_auto_matches_gaussian_auto_for_default_spec():
    # the gaussian x1 + poly1 auto path must track the original flat auto
    x = np.arange(0.0, 600.0, 1.0)
    y = _gauss_line(x, A=150.0, mu=300.0, sigma=15.0, m=-0.02, b=30.0)
    spec = {"signal": "gaussian", "n_components": 1, "background": "poly1"}
    comp = fit_composite_auto(x, y, 300.0, spec)
    flat = fit_gaussian_linear_auto(x, y, 300.0)
    assert comp["ok"] and flat["ok"]
    assert abs(comp["components"][0]["mu"] - flat["mu"]) < 1e-6
    assert abs((comp["win_hi"] - comp["win_lo"])
               - (flat["win_hi"] - flat["win_lo"])) < 1e-6


def test_E2_auto_caps_window():
    x = np.arange(0.0, 600.0, 1.0)
    y = _gauss_line(x, A=150.0, mu=300.0, sigma=15.0, m=0.0, b=30.0)
    spec = {"signal": "gaussian", "n_components": 1, "background": "poly1"}
    free = fit_composite_auto(x, y, 300.0, spec)
    capped = fit_composite_auto(x, y, 300.0, spec, max_half_window=25.0)
    assert free["ok"] and capped["ok"]
    assert (capped["win_hi"] - capped["win_lo"]) <= 51.0


def test_E2_auto_fits_crystal_ball_spec():
    rng = np.random.default_rng(3)
    x = np.arange(0.0, 400.0, 1.0)
    y = _cb(x, 300.0, 200.0, 6.0, 1.4, 4.0, side="low") + 40.0
    y = rng.poisson(np.clip(y, 0, None)).astype(float)
    spec = {"signal": "crystal_ball", "n_components": 1,
            "background": "poly1", "tail_side": "low"}
    r = fit_composite_auto(x, y, 200.0, spec)
    assert r["ok"]
    assert abs(r["components"][0]["mu"] - 200.0) < 2.0
    assert r["spec"]["signal"] == "crystal_ball"


def test_E2_cb_on_symmetric_peak_still_recovers_mu():
    # a near-symmetric peak leaves CB's alpha/n unconstrained (the case that
    # used to grind curve_fit to its iteration cap); the loosened tolerances
    # must still converge to the right centroid, not just faster
    rng = np.random.default_rng(1)
    x = np.arange(0.0, 1024.0, 1.0)
    y = 120.0 * np.exp(-0.5 * ((x - 512.0) / 8.0) ** 2) + 50.0
    y = rng.poisson(np.clip(y, 0, None)).astype(float)
    spec = {"signal": "crystal_ball", "n_components": 1,
            "background": "poly1", "tail_side": "low"}
    r = fit_composite_auto(x, y, 512.0, spec)
    assert r["ok"]
    assert abs(r["components"][0]["mu"] - 512.0) < 1.0


def test_E2_auto_flat_background_fails_cleanly():
    x = np.arange(0.0, 200.0, 1.0)
    y = np.full_like(x, 7.0)
    spec = {"signal": "gaussian", "n_components": 1, "background": "poly1"}
    r = fit_composite_auto(x, y, 100.0, spec)
    assert r["ok"] is False and r["error"]


def test_E1_row_single_component_nongaussian_has_no_multiplier():
    # a single Crystal Ball (or gaussian+quadratic) fit is still one peak — no ×k
    r = _fit_g1poly1()
    r["spec"] = {"signal": "crystal_ball", "n_components": 1, "background": "poly1"}
    row = format_composite_fit_row(5, r)
    assert row["cells"][0][0] == "5"


def test_E1_row_multicomponent_marks_count():
    rng = np.random.default_rng(11)
    x = np.arange(0.0, 400.0, 1.0)
    y = (_gauss_line(x, A=150.0, mu=185.0, sigma=5.0, m=0.0, b=30.0)
         + 150.0 * np.exp(-0.5 * ((x - 215.0) / 5.0) ** 2))
    y = rng.poisson(np.clip(y, 0, None)).astype(float)
    spec = {"signal": "gaussian", "n_components": 2, "background": "poly1"}
    r = fit_composite(x, y, 160.0, 240.0, spec)
    row = format_composite_fit_row(7, r)
    assert "×2" in row["cells"][0][0]
    assert len(row["cells"]) == 5
