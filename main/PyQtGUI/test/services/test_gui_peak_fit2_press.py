"""Characterization tests for Peak Finder 2's press dispatch and drag: the unified
press handler and its priority order, the armed-mode fit, the end-handle grab,
the drag guide, the refit on release, and the right-click edit."""

import logging
import os
import sys
import types

import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import gui_stubs


# --------------------------------------------------------------- fake widgets

class FakeLabel:
    def __init__(self):
        self._text = ""

    def text(self):
        return self._text

    def setText(self, t):
        self._text = t


class FakeCombo:
    def __init__(self, value):
        self.value = value

    def currentText(self):
        return self.value


class FakePeakTab:
    def __init__(self):
        self.peak2_status = FakeLabel()
        self.peak2_signal = FakeCombo("Gaussian")
        self.peak2_bg = FakeCombo("Linear")
        self.peak2_cb_tail = FakeCombo("left")


class FakeExtraPopup:
    def __init__(self):
        self.peak = FakePeakTab()


class FakeTabs:
    def __init__(self):
        self.selected = 0
        self.current = 0

    def currentIndex(self):
        return self.current

    def selectedPad(self, index):
        return self.selected

    def plot(self, index):
        return types.SimpleNamespace(canvas=None)


class Event:
    """matplotlib MouseEvent stand-in."""

    def __init__(self, inaxes=None, xdata=1.0, x=100.0, y=0.0, button=1,
                 dblclick=False):
        self.inaxes = inaxes
        self.xdata = xdata
        self.x = x
        self.y = y
        self.button = button
        self.dblclick = dblclick


def fit_result(lo=40.0, hi=60.0, mu=50.0, ok=True, ncomp=1, error=None):
    xx = np.linspace(lo, hi, 21)
    y_bg = np.ones_like(xx)
    y_comp = [np.exp(-0.5 * ((xx - mu) / 2.0) ** 2) * 10.0 for _ in range(ncomp)]
    r = {"ok": ok, "xx": xx, "y_bg": y_bg, "y_fit": y_bg + sum(y_comp),
         "y_comp": y_comp if ncomp > 1 else [],
         "components": [{"mu": mu, "area": 100.0, "sigma": 2.0}
                        for _ in range(ncomp)],
         "spec": {"n_components": ncomp}}
    if error is not None:
        r["error"] = error
    return r


SPECTRUM = {"dim": 1, "binx": 100, "minx": 0.0, "maxx": 100.0,
            "data": np.arange(101, dtype=float)}


@pytest.fixture
def win(monkeypatch):
    gui = gui_stubs.import_gui()
    w = gui.MainWindow.__new__(gui.MainWindow)
    w.logger = logging.getLogger("test.peakfit2.press")
    w.extraPopup = FakeExtraPopup()
    w.wTab = FakeTabs()

    figure = Figure()
    FigureCanvasAgg(figure)
    w.ax = figure.add_subplot(111)
    w.ax.set_xlim(0, 100)
    w.ax.set_ylim(0, 100)
    w.other_ax = figure.add_subplot(212)
    w.currentPlot = types.SimpleNamespace(
        figure=figure, canvas=figure.canvas, isEnlarged=False,
        zoomPress=False, toCreateGate=False, toEditGate=False,
        toCreateSumRegion=False)
    w.gatePopup = types.SimpleNamespace(isVisible=lambda: False)
    w.sumRegionPopup = types.SimpleNamespace(isVisible=lambda: False)

    w.store = dict(SPECTRUM)
    w.getSpectrumStoreInfo = lambda field, index=None, name=None: w.store[field]
    w.nameFromIndex = lambda index: "alpha"
    w.plot_controller = types.SimpleNamespace(
        createRange=lambda bins, vmin, vmax: np.linspace(
            float(vmin), float(vmax), int(bins) + 1))

    w.peak2_fits = []
    w.peak2_count = 0
    w.peak2_drag = None
    w.peak2_armed = False
    w.peak2_fix_armed = False
    w.settings = {}

    # 8d's table methods, recorded rather than run: this file characterizes the
    # press path, and what the table does with a fit is test_gui_peak_fit2_table's
    # subject. They are installed on the controller as well as the window,
    # because 8d moved the real ones onto it and the seam that used to point
    # back here is gone.
    w.rows_added = []
    w.rows_updated = []
    w.edits_opened = []
    w._peak2_add_row = lambda n, r, tag=None: w.rows_added.append((n, r, tag))
    w._peak2_update_row = lambda n, r, tag=None: w.rows_updated.append((n, r, tag))
    w._peak2_open_edit = lambda rec, x: w.edits_opened.append((rec, x))

    class FakeSettings:
        def value(self, key, default=None, type=None):
            return w.settings.get(key, default)

        def setValue(self, key, value):
            w.settings[key] = value

    # the fit engine is Qt-free and covered elsewhere; here it is scripted
    w.fits = {"auto": fit_result(), "fixed": fit_result(), "refit": fit_result()}
    w.calls = []

    def fake_auto(xc, y, cx, spec, max_half_window=None):
        w.calls.append(("auto", cx, max_half_window))
        return w.fits["auto"]

    def fake_composite(xc, y, lo, hi, spec, fixed=None, seeds=None):
        w.calls.append(("composite", lo, hi, fixed))
        return w.fits["fixed"]

    def fake_refit(xc, y, lo, hi, prev):
        w.calls.append(("refit", lo, hi))
        return w.fits["refit"]

    monkeypatch.setattr(gui, "QSettings", FakeSettings, raising=False)

    from controllers import peak_fit2_controller as pfc
    monkeypatch.setattr(pfc, "QSettings", FakeSettings)
    # the fit engine is imported into the controller's namespace now
    monkeypatch.setattr(pfc, "fit_composite_auto", fake_auto)
    monkeypatch.setattr(pfc, "fit_composite", fake_composite)
    monkeypatch.setattr(pfc, "autocomponent_refit", fake_refit)
    w.peak_fit2_controller = pfc.PeakFit2Controller(
        peak_tab=w.extraPopup.peak,
        spectra=types.SimpleNamespace(contains=lambda name: name == "alpha"),
        get_store_info=lambda field, index=None, name=None: w.getSpectrumStoreInfo(
            field, index=index, name=name),
        get_view_info=lambda field, index=None: w.ax,
        plot_controller=w.plot_controller,
        tabs=w.wTab,
        get_current_plot=lambda: w.currentPlot,
        get_gate_popup=lambda: w.gatePopup,
        get_sum_popup=lambda: w.sumRegionPopup,
        name_from_index=lambda index: w.nameFromIndex(index),
        parent_widget=w,
        logger=w.logger,
    )
    # The cluster's state lives on the controller; these proxies let the
    # unchanged test bodies keep reading it on the window.
    for attr in ("peak2_fits", "peak2_count", "peak2_armed", "peak2_fix_armed",
                 "peak2_drag", "peak2_conns"):
        monkeypatch.setattr(
            type(w), attr,
            property(lambda self, a=attr: getattr(self.peak_fit2_controller, a),
                     lambda self, v, a=attr: setattr(self.peak_fit2_controller, a, v)),
            raising=False)
    # 8d dropped every shim that is not a .connect() target, so the fixture
    # binds those names onto the window instance: the test bodies still drive
    # MainWindow exactly as they did before the move, and the ones MainWindow
    # still defines keep routing through its own shim.
    for _name in ("_peak2_add_row", "_peak2_update_row", "_peak2_open_edit"):
        setattr(w.peak_fit2_controller, _name, getattr(w, _name))
    for _name in dir(w.peak_fit2_controller):
        if (_name.startswith(("_peak2_", "peakFit2", "onPeakFit2"))
                and not hasattr(type(w), _name)
                and _name not in w.__dict__):     # keep this fixture's own doubles
            setattr(w, _name, getattr(w.peak_fit2_controller, _name))
    return w


def drawn_fit(win, lo=40.0, hi=60.0, number=1, name="alpha", ax=None):
    """Put a fit on the pad the way a successful press would."""
    r = fit_result(lo=lo, hi=hi)
    target = win.ax if ax is None else ax
    rec = {"number": number, "index": 0, "name": name, "result": r,
           "artists": win.peak_fit2_controller._peak2_draw(target, r)}
    win.peak2_fits.append(rec)
    return rec


def fill_event(win, x=50.0, ax=None):
    """A right-click at pixel coordinates that land INSIDE a fit's blue fill.

    `PolyCollection.contains` hit-tests in pixels, so the data point has to be
    projected — the fill spans from the background up to the fitted curve.
    """
    target = win.ax if ax is None else ax
    px, py = target.transData.transform((x, 5.0))
    return Event(inaxes=target, xdata=x, x=px, y=py, button=3)


def status(win):
    return win.extraPopup.peak.peak2_status.text()


# ------------------------------------------------------- dispatcher priority

def test_a_press_off_any_pad_does_nothing(win):
    """Nothing at all — not even a complaint. Without the guard the press
    reaches the fit path and reports '[error] Fit failed' for a click that
    never touched a spectrum."""
    win.peak2_armed = True
    win.onPeakFit2Press(Event(inaxes=None))
    assert win.peak2_fits == []
    assert status(win) == ""


def test_a_press_with_no_x_coordinate_does_nothing(win):
    win.peak2_armed = True
    win.onPeakFit2Press(Event(inaxes=win.ax, xdata=None))
    assert win.peak2_fits == []
    assert status(win) == ""


@pytest.mark.parametrize("flag", ["zoomPress", "toCreateGate", "toEditGate",
                                  "toCreateSumRegion"])
def test_another_pad_interaction_blocks_the_press(win, flag):
    """K6(i). Zoom, gate create/edit and summing-region create own the click."""
    win.peak2_armed = True
    setattr(win.currentPlot, flag, True)
    win.onPeakFit2Press(Event(inaxes=win.ax))
    assert win.peak2_fits == []


def test_a_left_press_on_an_end_handle_starts_a_drag_not_a_fit(win):
    """Priority: grab beats fit, even while armed."""
    rec = drawn_fit(win)
    win.peak2_armed = True
    x_px = win.ax.transData.transform((rec["result"]["xx"][0], 0.0))[0]
    win.onPeakFit2Press(Event(inaxes=win.ax, xdata=40.0, x=x_px))
    assert win.peak2_drag is not None
    assert len(win.peak2_fits) == 1        # no second fit was added


def test_a_right_click_opens_the_edit_popup_and_never_fits(win):
    rec = drawn_fit(win)
    win.peak2_armed = True
    win.onPeakFit2Press(fill_event(win))
    assert win.edits_opened and win.edits_opened[0][0] is rec
    assert len(win.peak2_fits) == 1


def test_a_double_click_is_ignored(win):
    """Double-click is the enlarge gesture; it must not also fit."""
    win.peak2_armed = True
    win.onPeakFit2Press(Event(inaxes=win.ax, dblclick=True))
    assert win.peak2_fits == []


def test_a_middle_click_is_ignored(win):
    win.peak2_armed = True
    win.onPeakFit2Press(Event(inaxes=win.ax, button=2))
    assert win.peak2_fits == []


def test_a_left_click_while_disarmed_does_not_fit(win):
    win.onPeakFit2Press(Event(inaxes=win.ax, xdata=50.0))
    assert win.peak2_fits == []


def test_a_left_click_while_armed_fits(win):
    win.peak2_armed = True
    win.onPeakFit2Press(Event(inaxes=win.ax, xdata=50.0))
    assert len(win.peak2_fits) == 1


def test_a_left_click_while_fix_armed_fits(win):
    win.peak2_fix_armed = True
    win.onPeakFit2Press(Event(inaxes=win.ax, xdata=50.0))
    assert len(win.peak2_fits) == 1


# ------------------------------------------------------------ armed-mode fit

def test_a_fit_records_the_spectrum_name(win):
    """H11: the record is what a later refit resolves from."""
    win.peak2_armed = True
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    assert win.peak2_fits[0]["name"] == "alpha"


def test_a_fit_numbers_the_peak_and_adds_a_row(win):
    win.peak2_armed = True
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    assert win.peak2_count == 1
    assert win.rows_added[0][0] == 1


def test_a_fit_draws_its_artists_on_the_clicked_pad(win):
    win.peak2_armed = True
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    assert len(win.peak2_fits[0]["artists"]) == 4


def test_a_click_on_a_colorbar_is_skipped(win):
    win.peak2_armed = True
    cb = win.currentPlot.figure.add_subplot(313)
    cb.set_label("colorbar_1")
    win._peak2_fit_at_press(Event(inaxes=cb, xdata=50.0))
    assert win.peak2_fits == []


def test_a_pad_with_no_spectrum_is_reported(win):
    win.peak2_armed = True
    win.nameFromIndex = lambda index: None
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    assert "[skip]" in status(win) and "no spectrum" in status(win)


def test_a_2d_spectrum_is_refused(win):
    win.peak2_armed = True
    win.store["dim"] = 2
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    assert "1D" in status(win)


def test_an_enlarged_pad_uses_the_selected_index(win):
    """While enlarged the figure holds one axes, so the pad index has to come
    from the tab's selection instead of the axes position."""
    win.peak2_armed = True
    win.currentPlot.isEnlarged = True
    win.wTab.selected = 3
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    assert win.peak2_fits[0]["index"] == 3


def test_fix_peak_pins_mu_at_the_click(win):
    """K8. The window is centred on the click; the automatic estimator is never
    consulted, because it would snap onto a bigger neighbour."""
    win.peak2_fix_armed = True
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    kind, lo, hi, fixed = win.calls[0]
    assert kind == "composite"
    assert fixed == {"mu1": 50.0}
    assert lo < 50.0 < hi


def test_fix_peak_tags_the_row(win):
    win.peak2_fix_armed = True
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    assert win.rows_added[0][2] == "fixed μ"


def test_auto_mode_passes_the_cap_as_a_half_window(win):
    """K7: the Config cap is in bins; the fitter wants x units."""
    win.settings["PeakFinder2/max_window_bins"] = "40"
    win.peak2_armed = True
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    kind, cx, max_hw = win.calls[0]
    assert kind == "auto"
    assert max_hw == pytest.approx(0.5 * 40 * (100.0 / 100))


def test_auto_mode_without_a_cap_passes_none(win):
    win.peak2_armed = True
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    assert win.calls[0][2] is None


def test_a_capped_auto_failure_is_silent(win):
    """K7's contract: with a cap set, clicks that cannot be fitted are skipped
    without a message — otherwise scanning a spectrum spams the status line."""
    win.settings["PeakFinder2/max_window_bins"] = "40"
    win.fits["auto"] = fit_result(ok=False, error="no peak")
    win.peak2_armed = True
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    assert status(win) == ""
    assert win.peak2_fits == []


def test_an_uncapped_auto_failure_is_reported(win):
    win.fits["auto"] = fit_result(ok=False, error="no peak")
    win.peak2_armed = True
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    assert "[failed]" in status(win)


def test_a_fix_peak_failure_is_reported_even_with_a_cap(win):
    """Fix Peak is an explicit click at a chosen centre; silence would read as
    the GUI ignoring the user."""
    win.settings["PeakFinder2/max_window_bins"] = "40"
    win.fits["fixed"] = fit_result(ok=False, error="singular")
    win.peak2_fix_armed = True
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    assert "[failed]" in status(win)


def test_a_failed_fit_does_not_consume_a_peak_number(win):
    win.fits["auto"] = fit_result(ok=False, error="no peak")
    win.peak2_armed = True
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    assert win.peak2_count == 0


def test_a_duplicate_click_is_skipped(win):
    """K22-adjacent: an off-peak flank click re-finds a peak already fitted on
    this spectrum."""
    win.peak2_armed = True
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.4))
    assert len(win.peak2_fits) == 1
    assert "already fitted" in status(win)


def test_a_duplicate_click_does_not_consume_a_peak_number(win):
    win.peak2_armed = True
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.4))
    assert win.peak2_count == 1


def test_duplicate_suppression_is_per_spectrum(win):
    """The same μ on a DIFFERENT spectrum is a different peak."""
    win.peak2_armed = True
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    win.nameFromIndex = lambda index: "beta"
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    assert len(win.peak2_fits) == 2


def test_fix_peak_does_not_suppress_duplicates(win):
    """Pinning μ is an explicit instruction to fit exactly there."""
    win.peak2_fix_armed = True
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    assert len(win.peak2_fits) == 2


def test_a_crash_in_the_fit_path_is_reported_not_raised(win):
    """A click must never take the GUI down."""
    win.peak2_armed = True
    win.nameFromIndex = lambda index: (_ for _ in ()).throw(RuntimeError("boom"))
    win._peak2_fit_at_press(Event(inaxes=win.ax, xdata=50.0))
    assert "[error]" in status(win)


# --------------------------------------------------------------- grab + drag

def test_a_press_near_the_low_edge_grabs_it(win):
    rec = drawn_fit(win)
    x_px = win.ax.transData.transform((rec["result"]["xx"][0], 0.0))[0]
    assert win._peak2_try_grab(Event(inaxes=win.ax, xdata=40.0, x=x_px)) is True
    assert win.peak2_drag["edge"] == "lo"


def test_a_press_near_the_high_edge_grabs_it(win):
    rec = drawn_fit(win)
    x_px = win.ax.transData.transform((rec["result"]["xx"][-1], 0.0))[0]
    assert win._peak2_try_grab(Event(inaxes=win.ax, xdata=60.0, x=x_px)) is True
    assert win.peak2_drag["edge"] == "hi"


def test_a_press_in_the_middle_grabs_nothing(win):
    drawn_fit(win)
    mid_px = win.ax.transData.transform((50.0, 0.0))[0]
    assert win._peak2_try_grab(Event(inaxes=win.ax, xdata=50.0, x=mid_px)) is False
    assert win.peak2_drag is None


def test_a_second_grab_is_refused_while_one_is_active(win):
    rec = drawn_fit(win)
    x_px = win.ax.transData.transform((rec["result"]["xx"][0], 0.0))[0]
    win._peak2_try_grab(Event(inaxes=win.ax, xdata=40.0, x=x_px))
    first = win.peak2_drag
    assert win._peak2_try_grab(Event(inaxes=win.ax, xdata=40.0, x=x_px)) is False
    assert win.peak2_drag is first


def test_a_grab_ignores_fits_drawn_on_another_pad(win):
    """The pixel is the one that WOULD grab if this fit were on the pressed
    pad, so only the axes check can refuse it."""
    rec = drawn_fit(win, ax=win.other_ax)
    x_px = win.ax.transData.transform((rec["result"]["xx"][0], 0.0))[0]
    assert win._peak2_try_grab(Event(inaxes=win.ax, xdata=40.0, x=x_px)) is False
    assert win.peak2_drag is None


def test_a_grab_puts_a_guide_line_on_the_pad(win):
    rec = drawn_fit(win)
    before = len(win.ax.lines)
    x_px = win.ax.transData.transform((rec["result"]["xx"][0], 0.0))[0]
    win._peak2_try_grab(Event(inaxes=win.ax, xdata=40.0, x=x_px))
    assert len(win.ax.lines) == before + 1


def test_a_grab_says_which_edge_is_moving(win):
    rec = drawn_fit(win)
    x_px = win.ax.transData.transform((rec["result"]["xx"][0], 0.0))[0]
    win._peak2_try_grab(Event(inaxes=win.ax, xdata=40.0, x=x_px))
    assert "[drag]" in status(win) and "lo" in status(win)


def test_drag_motion_follows_the_cursor(win):
    rec = drawn_fit(win)
    x_px = win.ax.transData.transform((rec["result"]["xx"][0], 0.0))[0]
    win._peak2_try_grab(Event(inaxes=win.ax, xdata=40.0, x=x_px))
    win._peak2_on_drag_motion(Event(inaxes=win.ax, xdata=44.0))
    assert list(win.peak2_drag["guide"].get_xdata()) == [44.0, 44.0]


def test_drag_motion_off_the_pad_is_ignored(win):
    rec = drawn_fit(win)
    x_px = win.ax.transData.transform((rec["result"]["xx"][0], 0.0))[0]
    win._peak2_try_grab(Event(inaxes=win.ax, xdata=40.0, x=x_px))
    before = list(win.peak2_drag["guide"].get_xdata())
    win._peak2_on_drag_motion(Event(inaxes=win.other_ax, xdata=44.0))
    assert list(win.peak2_drag["guide"].get_xdata()) == before


def test_drag_motion_without_a_drag_is_a_no_op(win):
    win._peak2_on_drag_motion(Event(inaxes=win.ax, xdata=44.0))
    assert win.peak2_drag is None


# ------------------------------------------------------------ drag release

def start_drag(win, rec, edge="lo"):
    xx = rec["result"]["xx"]
    x = xx[0] if edge == "lo" else xx[-1]
    x_px = win.ax.transData.transform((x, 0.0))[0]
    win._peak2_try_grab(Event(inaxes=win.ax, xdata=float(x), x=x_px))
    return win.peak2_drag


def test_a_release_clears_the_drag_state(win):
    rec = drawn_fit(win)
    start_drag(win, rec)
    win._peak2_on_drag_release(Event(inaxes=win.ax, xdata=44.0))
    assert win.peak2_drag is None


def test_a_release_takes_the_guide_line_away(win):
    rec = drawn_fit(win)
    start_drag(win, rec)
    before = len(win.ax.lines)
    win._peak2_on_drag_release(Event(inaxes=win.ax, xdata=44.0))
    assert len(win.ax.lines) < before


def test_a_release_off_the_pad_cancels_without_refitting(win):
    rec = drawn_fit(win)
    start_drag(win, rec)
    win.calls.clear()
    win._peak2_on_drag_release(Event(inaxes=None, xdata=None))
    assert "cancelled" in status(win)
    assert win.calls == []


def test_dragging_the_low_edge_moves_only_that_edge(win):
    """K16/K17 depend on the window the release computes."""
    rec = drawn_fit(win, lo=40.0, hi=60.0)
    start_drag(win, rec, edge="lo")
    win.calls.clear()
    win._peak2_on_drag_release(Event(inaxes=win.ax, xdata=44.0))
    kind, lo, hi = win.calls[0]
    assert (lo, hi) == (44.0, 60.0)


def test_dragging_the_high_edge_moves_only_that_edge(win):
    rec = drawn_fit(win, lo=40.0, hi=60.0)
    start_drag(win, rec, edge="hi")
    win.calls.clear()
    win._peak2_on_drag_release(Event(inaxes=win.ax, xdata=66.0))
    kind, lo, hi = win.calls[0]
    assert (lo, hi) == (40.0, 66.0)


def test_a_release_refits_through_the_autocomponent_path(win):
    """K16/K17: the new window may cover extra peaks or have dropped some, so
    the component set is refitted to match rather than carried over."""
    rec = drawn_fit(win)
    start_drag(win, rec)
    win.calls.clear()
    win._peak2_on_drag_release(Event(inaxes=win.ax, xdata=44.0))
    assert win.calls[0][0] == "refit"


def test_a_successful_release_replaces_the_artists(win):
    rec = drawn_fit(win)
    old = rec["artists"]
    start_drag(win, rec)
    win._peak2_on_drag_release(Event(inaxes=win.ax, xdata=44.0))
    assert rec["artists"] != old
    assert all(a not in win.ax.lines for a in old if hasattr(a, "get_xdata"))


def test_a_successful_release_updates_the_row(win):
    rec = drawn_fit(win)
    start_drag(win, rec)
    win._peak2_on_drag_release(Event(inaxes=win.ax, xdata=44.0))
    assert win.rows_updated and win.rows_updated[0][2] == "edited"


def test_a_successful_release_reports_the_new_mu(win):
    rec = drawn_fit(win)
    start_drag(win, rec)
    win._peak2_on_drag_release(Event(inaxes=win.ax, xdata=44.0))
    assert "window edited" in status(win) and "μ" in status(win)


def test_a_multi_component_release_says_how_many(win):
    rec = drawn_fit(win)
    win.fits["refit"] = fit_result(ncomp=3)
    start_drag(win, rec)
    win._peak2_on_drag_release(Event(inaxes=win.ax, xdata=44.0))
    assert "3 components" in status(win)


def test_a_failed_release_keeps_the_previous_fit(win):
    """A drag is explicit, so its failure is always reported — and the fit that
    was there stays exactly as it was."""
    rec = drawn_fit(win)
    old_result = rec["result"]
    old_artists = rec["artists"]
    win.fits["refit"] = fit_result(ok=False, error="fit failed")
    start_drag(win, rec)
    win._peak2_on_drag_release(Event(inaxes=win.ax, xdata=44.0))
    assert "[failed]" in status(win)
    assert rec["result"] is old_result
    assert rec["artists"] is old_artists


def test_a_release_with_no_spectrum_reports_and_keeps_the_fit(win):
    rec = drawn_fit(win, name="gone")
    old_artists = rec["artists"]
    start_drag(win, rec)
    win._peak2_on_drag_release(Event(inaxes=win.ax, xdata=44.0))
    assert "unavailable" in status(win)
    assert rec["artists"] is old_artists


# --------------------------------------------------------------- right-click

def test_a_right_click_inside_the_fill_opens_that_fit(win):
    rec = drawn_fit(win)
    assert win._peak2_try_edit(fill_event(win)) is True
    assert win.edits_opened[0][0] is rec


def test_the_edit_popup_is_told_where_the_click_landed(win):
    """K19's entry point: the popup edits the component nearest the click."""
    drawn_fit(win)
    win._peak2_try_edit(fill_event(win))
    assert win.edits_opened[0][1] == 50.0


def test_a_right_click_outside_every_fill_opens_nothing(win):
    drawn_fit(win, lo=40.0, hi=60.0)
    assert win._peak2_try_edit(fill_event(win, x=5.0)) is False
    assert win.edits_opened == []


def test_a_right_click_ignores_fits_on_another_pad(win):
    drawn_fit(win, ax=win.other_ax)
    assert win._peak2_try_edit(fill_event(win)) is False


def test_a_record_without_a_fill_is_skipped(win):
    """A record whose artists were stripped has no fill to hit-test."""
    rec = drawn_fit(win)
    rec["artists"] = rec["artists"][:2]
    assert win._peak2_try_edit(fill_event(win)) is False


# ------------------------------------------------- the enlarge-gesture guard

def test_a_slow_double_click_does_not_fit_twice(win, monkeypatch):
    """A double-click slow enough that matplotlib does not pair it arrives as
    two independent presses, each with dblclick False. Armed, the first one
    used to drop a fit nobody asked for on the way into enlarged mode."""
    import controllers.peak_fit2_controller as pfc
    clock = [100.0]
    monkeypatch.setattr(pfc.time, "monotonic", lambda: clock[0])
    win.peak2_armed = True
    win.onPeakFit2Press(Event(inaxes=win.ax, xdata=50.0))
    clock[0] += 0.20                      # inside the double-click interval
    win.onPeakFit2Press(Event(inaxes=win.ax, xdata=50.0))
    # asserted on the FITTER, not on the record count: duplicate suppression
    # would keep the count at 1 whether or not the press was debounced
    assert len(win.calls) == 1


def test_two_deliberate_clicks_still_fit_twice(win, monkeypatch):
    """The guard must not cost an ordinary second fit: wait past the interval
    and the click works as it always did."""
    import controllers.peak_fit2_controller as pfc
    clock = [100.0]
    monkeypatch.setattr(pfc.time, "monotonic", lambda: clock[0])
    win.peak2_armed = True
    win.onPeakFit2Press(Event(inaxes=win.ax, xdata=50.0))
    clock[0] += 2.0
    win.nameFromIndex = lambda index: "beta"      # a different spectrum
    win.onPeakFit2Press(Event(inaxes=win.ax, xdata=50.0))
    assert len(win.calls) == 2
    assert len(win.peak2_fits) == 2


def test_the_guard_does_not_block_a_drag_grab(win, monkeypatch):
    """Only the FIT is debounced. Grabbing an end handle is a deliberate press
    and must still work immediately after another one."""
    import controllers.peak_fit2_controller as pfc
    clock = [100.0]
    monkeypatch.setattr(pfc.time, "monotonic", lambda: clock[0])
    rec = drawn_fit(win)
    win.peak2_armed = True
    win.onPeakFit2Press(Event(inaxes=win.ax, xdata=50.0))
    clock[0] += 0.10
    x_px = win.ax.transData.transform((rec["result"]["xx"][0], 0.0))[0]
    win.onPeakFit2Press(Event(inaxes=win.ax, xdata=40.0, x=x_px))
    assert win.peak2_drag is not None


def test_the_guard_does_not_block_a_right_click_edit(win, monkeypatch):
    import controllers.peak_fit2_controller as pfc
    clock = [100.0]
    monkeypatch.setattr(pfc.time, "monotonic", lambda: clock[0])
    drawn_fit(win)
    win.peak2_armed = True
    win.onPeakFit2Press(Event(inaxes=win.ax, xdata=50.0))
    clock[0] += 0.10
    win.onPeakFit2Press(fill_event(win))
    assert win.edits_opened
