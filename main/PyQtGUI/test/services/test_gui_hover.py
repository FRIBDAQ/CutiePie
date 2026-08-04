"""Characterization tests for MainWindow's pointer-readout and mouse-dispatch
path. `histoHover`, `getPointerInfo`, `_blankHoverLabels`, `on_press`,
`on_release` and the asynchronous gate-name pair."""

import logging
import os
import sys
import types

import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib.figure import Figure

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import gui_stubs
from services.spectrum_store import SpectrumStore
from services.log_throttle import LogThrottle


class FakeLabel:
    def __init__(self):
        self._text = ""
        self.set_calls = 0

    def text(self):
        return self._text

    def setText(self, value):
        self._text = value
        self.set_calls += 1


class FakeAction:
    def __init__(self):
        self.emitted = 0
        self.checked = True
        self.triggered = types.SimpleNamespace(emit=self._emit)

    def _emit(self):
        self.emitted += 1

    def setChecked(self, value):
        self.checked = value


class FakeButton:
    def __init__(self):
        self.down = True

    def setDown(self, value):
        self.down = value


class FakePlot:
    """The per-tab plot widget: a REAL matplotlib figure plus fake readout."""

    def __init__(self, npads=2):
        self.figure = Figure()
        for i in range(npads):
            self.figure.add_subplot(npads, 1, i + 1)
        self.h_dict_geo = {}
        self.histoLabel = FakeLabel()
        self.pointerLabel = FakeLabel()
        self.gateLabel = FakeLabel()
        self.zoomPress = False
        self.zoom_action = FakeAction()
        self.customZoomButton = FakeButton()
        self.isEnlarged = False
        self.selected_plot_index = None
        self.toCreateGate = False
        self.toEditGate = False
        self.toCreateSumRegion = False


class FakeTabs:
    def __init__(self):
        self.slots = {0: {}}
        self.zoom = {0: None}
        self.selected = {0: 0}

    def currentIndex(self):
        return 0

    def tabSlots(self, index):
        return self.slots[index]

    def zoomInfo(self, index):
        return self.zoom[index]

    def setZoomInfo(self, index, value):
        self.zoom[index] = value

    def selectedPad(self, index):
        return self.selected[index]


class Recorder:
    """Stands in for a service; records the calls the router makes."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args))
        return record


def fake_event(ax, x=0.0, y=0.0, button=1, dblclick=False, inaxes=True):
    ev = types.SimpleNamespace()
    ev.inaxes = ax if inaxes else None
    ev.xdata, ev.ydata = x, y
    ev.button, ev.dblclick = button, dblclick
    ev.guiEvent = None
    if ax is not None:
        ev.x, ev.y = ax.transData.transform([x, y])
    else:
        ev.x, ev.y = 0.0, 0.0
    return ev


@pytest.fixture
def win(monkeypatch):
    gui = gui_stubs.import_gui()
    w = gui.MainWindow.__new__(gui.MainWindow)
    w.logger = logging.getLogger("test.hover")
    w.spectra = SpectrumStore()
    w.wTab = FakeTabs()
    w.currentPlot = FakePlot()

    # These accessors live on ViewState; bind them back onto the window so
    # the unchanged test bodies keep driving MainWindow.
    from view_state import ViewState
    import types as _types
    w.view_state = ViewState(
        spectra=w.spectra,
        tabs=w.wTab,
        get_current_plot=lambda: w.currentPlot,
        applylistgate=lambda name: None,
        gate_name_fetched=_types.SimpleNamespace(emit=lambda *a: None),
        logger=w.logger,
    )
    for _name in ("getSpectrumStoreInfo", "setSpectrumViewInfo", "getSpectrumViewInfo",
                  "getSpectrumViewDict", "getSpectrumStoreDict", "nameFromIndex",
                  "setGeo", "getGeo", "setEnlargedSpectrum", "getEnlargedSpectrum",
                  "getAppliedGateName"):
        if _name not in w.__dict__:
            setattr(w, _name, getattr(w.view_state, _name))
    # the gate-name cache moved with its only reader; the hover slot that FILLS
    # it stayed here, so the test bodies keep reading it on the window
    for _attr in ("_gate_name_cache", "_gate_name_inflight", "_GATE_NAME_TTL"):
        monkeypatch.setattr(
            type(w), _attr,
            property(lambda self, a=_attr: getattr(self.view_state, a),
                     lambda self, v, a=_attr: setattr(self.view_state, a, v)),
            raising=False)
    w._hoveredSpectrumName = None
    w._hoverLogThrottle = LogThrottle(interval_secs=30.0)
    w._gate_name_cache = {}
    w._gate_name_inflight = set()
    w.mouse_x = None
    w.gate_manager = Recorder()
    w.sum_region_manager = Recorder()
    w.fit_manager = types.SimpleNamespace(_cal=None)
    w.plot_controller = Recorder()
    w.refreshed = []
    # the async refresh moved to ViewState with the cache (stage 9), and
    # getAppliedGateName calls it on that object — so the recorder goes there
    w._refreshGateNameAsync = w.refreshed.append
    w.view_state._refreshGateNameAsync = w.refreshed.append
    return w


def add_pad(win, name, index, dim=1, counts=None, minx=0.0, maxx=100.0, binx=10,
            parameters=("p1",), sp_type="1"):
    """Populate the store and the pad the way a connect + draw does."""
    if counts is None:
        counts = np.arange(binx + 2, dtype=float)
    win.spectra.set(name, dim=dim, binx=binx, minx=minx, maxx=maxx,
                    biny=binx, miny=minx, maxy=maxx, data=counts,
                    parameters=list(parameters), type=sp_type)
    win.setGeo(index, name)
    ax = win.currentPlot.figure.axes[index]
    ax.set_xlim(minx, maxx)
    ax.set_ylim(0, 100)
    win.setSpectrumViewInfo(axis=ax, index=index)
    return ax


def readout(win):
    return (win.currentPlot.histoLabel.text(),
            win.currentPlot.pointerLabel.text(),
            win.currentPlot.gateLabel.text())


BLANK = ("Spectrum: \nX: Y:", "Pointer:\nX: Y: Count: ", "Gate applied: \n")


# ------------------------------------------------------------ _setLabelText

def test_label_is_written_only_when_the_text_actually_changes(win):
    # P3: every setText triggers a Qt relayout and this runs per motion event
    label = FakeLabel()
    win._setLabelText(label, "same")
    win._setLabelText(label, "same")
    win._setLabelText(label, "different")
    assert label.set_calls == 2


# ------------------------------------------------------------- histoHover

def test_hover_outside_any_axes_leaves_the_readout_untouched(win):
    add_pad(win, "h1", 0)
    win.histoHover(fake_event(None, inaxes=False))
    assert readout(win) == ("", "", "")          # nothing written at all


def test_hover_over_a_1d_pad_reports_name_parameter_and_count(win):
    ax = add_pad(win, "h1", 0, counts=np.arange(12, dtype=float))
    win.histoHover(fake_event(ax, x=25.0, y=3.0))
    histo, pointer, _ = readout(win)
    assert histo == "Spectrum: h1\nX: p1"
    assert pointer.startswith("Pointer:\nX: 25.00")
    assert "Count:" in pointer


def test_hover_marks_a_gamma_spectrum_parameter_as_truncated(win):
    ax = add_pad(win, "g", 0, sp_type="g1", parameters=("p1", "p2"))
    win.histoHover(fake_event(ax, x=10.0))
    assert win.currentPlot.histoLabel.text() == "Spectrum: g\nX: p1, ..."


def test_hover_over_a_2d_pad_reports_both_axis_parameters(win):
    counts = np.arange(100, dtype=float).reshape(10, 10)
    ax = add_pad(win, "h2", 0, dim=2, counts=counts, parameters=("px", "py"),
                 sp_type="2")
    win.histoHover(fake_event(ax, x=25.0, y=35.0))
    assert win.currentPlot.histoLabel.text() == "Spectrum: h2\nX: px Y: py"


def test_hover_blanks_when_the_pad_names_a_spectrum_the_store_dropped(win):
    # PIN (BUGS.md L20): the dim==1/dim==2 branches had no else, so the labels
    # kept the PREVIOUS pad's name and counts — a readout describing a spectrum
    # the pointer is not over, with nothing raised and nothing logged.
    ax0 = add_pad(win, "alive", 0)
    ax1 = add_pad(win, "ghost", 1)
    win.histoHover(fake_event(ax0, x=25.0))
    assert "alive" in win.currentPlot.histoLabel.text()

    win.spectra.remove("ghost")                  # deleted server-side
    win.histoHover(fake_event(ax1, x=25.0))
    assert readout(win) == BLANK
    assert win._hoveredSpectrumName is None


def test_hover_over_an_empty_pad_blanks_without_logging(win):
    # PIN (AUDIT.md M32): the expected misses are caught narrowly and stay
    # silent. Only the unexpected ones reach the log.
    add_pad(win, "h1", 0)
    empty_ax = win.currentPlot.figure.axes[1]    # no spectrum, no slot
    win.histoHover(fake_event(empty_ax, x=1.0))
    assert readout(win) == BLANK


def test_hover_logs_an_unexpected_error_once_behind_the_throttle(win, caplog):
    # PIN (AUDIT.md M32): a defect below used to look exactly like the pointer
    # leaving the axes. It is reported now — but at most once per interval, or
    # a broken store record would write a traceback per motion event.
    ax = add_pad(win, "h1", 0)

    def boom(*a, **k):
        raise RuntimeError("store is broken")
    win.getPointerInfo = boom

    with caplog.at_level(logging.WARNING, logger="test.hover"):
        for _ in range(5):
            win.histoHover(fake_event(ax, x=25.0))

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1                    # 5 events, 1 log line
    assert readout(win) == BLANK                 # and blanked every time


def test_hover_reports_the_cached_gate_name(win):
    ax = add_pad(win, "h1", 0)
    win._gate_name_cache["h1"] = ("gateA", __import__("time").monotonic())
    win.histoHover(fake_event(ax, x=25.0))
    assert win.currentPlot.gateLabel.text() == "Gate applied: gateA\n"


def test_hover_never_blocks_on_the_gate_lookup(win):
    # PIN (PERFORMANCE.md P3): a cold cache answers immediately with no name
    # and schedules the fetch behind the readout. The hover path must never
    # wait on HTTP.
    ax = add_pad(win, "h1", 0)
    win.histoHover(fake_event(ax, x=25.0))
    assert win.currentPlot.gateLabel.text() == "Gate applied: \n"
    assert win.refreshed == ["h1"]


# ---------------------------------------------------------- getPointerInfo

def test_pointer_info_reads_bin_math_from_the_store_tier(win):
    # PIN (BUGS.md E7): the store tier is the axis DEFINITION; the per-tab view
    # tier is the current zoom. Bin math that reads the view tier draws the
    # spectrum compressed into the zoomed window.
    counts = np.arange(12, dtype=float)
    ax = add_pad(win, "h1", 0, counts=counts, minx=0.0, maxx=100.0, binx=10)
    win.setSpectrumViewInfo(minx=40.0, maxx=60.0, index=0)   # a zoom
    x, y, count = win.getPointerInfo(fake_event(ax, x=25.0, y=1.0),
                                     "coordinates", 0)
    # bin from the STORE range: (25-0)/10 = 2, +1 for the underflow channel
    assert count == counts[3]


def test_pointer_info_clamps_a_position_left_of_the_axis(win):
    # a sliver left of minx used to make a negative index, which silently
    # wraps to the far end of the array and reports the wrong count
    counts = np.arange(12, dtype=float)
    ax = add_pad(win, "h1", 0, counts=counts, minx=0.0, maxx=100.0, binx=10)
    _, _, count = win.getPointerInfo(fake_event(ax, x=-5.0, y=1.0),
                                     "coordinates", 0)
    assert count == counts[1]                    # clamped to bin 0, +1 shift


def test_pointer_info_clamps_a_position_right_of_the_axis(win):
    counts = np.arange(12, dtype=float)
    ax = add_pad(win, "h1", 0, counts=counts, minx=0.0, maxx=100.0, binx=10)
    _, _, count = win.getPointerInfo(fake_event(ax, x=500.0, y=1.0),
                                     "coordinates", 0)
    assert count == counts[10]                   # clamped to the last bin


def test_pointer_info_returns_blanks_for_a_pad_with_no_data(win):
    win.spectra.set("empty", dim=1, binx=10, minx=0.0, maxx=100.0, biny=0,
                    miny=0.0, maxy=0.0, data=[], parameters=["p1"], type="1")
    win.setGeo(0, "empty")
    ax = win.currentPlot.figure.axes[0]
    win.setSpectrumViewInfo(axis=ax, index=0)
    assert win.getPointerInfo(fake_event(ax, x=25.0), "coordinates", 0) == ['', '', '']


def test_pointer_info_answers_for_the_enlarged_pad_whatever_index_it_is_given(win):
    counts = np.arange(12, dtype=float)
    add_pad(win, "h1", 0, counts=counts)
    ax1 = add_pad(win, "h2", 1, counts=counts * 10)
    win.setEnlargedSpectrum(1, "h2")
    _, _, count = win.getPointerInfo(fake_event(ax1, x=25.0), "coordinates", 0)
    assert count == (counts * 10)[3]             # index 0 ignored while enlarged


# ------------------------------------------------------------ on_press

def test_press_outside_the_axes_cancels_an_armed_zoom(win):
    win.currentPlot.zoomPress = True
    win.cleanPopupExit = lambda *a: None
    win.on_press(fake_event(None, inaxes=False))
    assert win.currentPlot.zoomPress is False
    assert win.currentPlot.zoom_action.emitted == 1
    assert win.currentPlot.zoom_action.checked is False
    assert win.currentPlot.customZoomButton.down is False


def test_press_on_a_colorbar_is_ignored(win):
    add_pad(win, "h1", 0)
    win.cleanPopupExit = lambda *a: None
    bar = win.currentPlot.figure.add_subplot(313)
    bar.set_label("colorbar_0")
    win.on_press(fake_event(bar, x=1.0))
    assert win.currentPlot.selected_plot_index is None


def test_press_selects_the_pad_under_the_pointer(win):
    ax = add_pad(win, "h1", 1)
    win.cleanPopupExit = lambda *a: None
    win.on_singleclick = lambda idx: None
    win.on_press(fake_event(ax, x=25.0))
    assert win.currentPlot.selected_plot_index == 1


def test_press_routes_a_gate_click_to_the_gate_manager(win):
    ax = add_pad(win, "h1", 0)
    win.cleanPopupExit = lambda *a: None
    win.currentPlot.toCreateGate = True
    win.on_press(fake_event(ax, x=25.0, button=1))
    assert [c[0] for c in win.gate_manager.calls] == ["on_singleclick_gate"]

    win.gate_manager.calls.clear()
    win.on_press(fake_event(ax, x=25.0, button=3))
    assert [c[0] for c in win.gate_manager.calls] == ["on_singleclick_gate_right"]


def test_press_routes_a_sum_region_click_with_the_spectrum_name(win):
    ax = add_pad(win, "h1", 0)
    win.cleanPopupExit = lambda *a: None
    win.currentPlot.toCreateSumRegion = True
    win.on_press(fake_event(ax, x=25.0, button=1))
    call = win.sum_region_manager.calls[0]
    assert call[0] == "on_singleclick_sumRegion"
    assert call[1][2] == "h1"                    # name resolved, not the index


def test_press_while_enlarged_uses_the_selected_pad_not_the_axes_index(win):
    ax = add_pad(win, "h1", 1)
    win.cleanPopupExit = lambda *a: None
    win.on_singleclick = lambda idx: None
    win.currentPlot.isEnlarged = True
    win.wTab.selected[0] = 7
    win.on_press(fake_event(ax, x=25.0))
    assert win.currentPlot.selected_plot_index == 7


# ------------------------------------------------------------ on_release

def test_release_ends_an_active_zoom_and_defers_the_limit_update(win):
    win.currentPlot.zoomPress = True
    win.on_release(fake_event(None, inaxes=False))
    assert win.currentPlot.zoomPress is False
    assert win.currentPlot.zoom_action.emitted == 1
    assert win.currentPlot.customZoomButton.down is False


def test_release_without_an_active_zoom_does_nothing(win):
    win.currentPlot.zoomPress = False
    win.on_release(fake_event(None, inaxes=False))
    assert win.currentPlot.zoom_action.emitted == 0


# ------------------------------------------- asynchronous gate-name refresh

@pytest.fixture
def gate_win(win):
    win.refreshed = []
    win.connection_manager = types.SimpleNamespace(
        applylistgate=lambda name: win.gate_reply)
    win.gate_reply = []
    win._gateNameFetched = types.SimpleNamespace(
        emit=lambda name, gate: win._on_gate_name_fetched(name, gate))
    # The async refresh moved to ViewState with the cache (stage 9). Point that
    # object at this fixture's REST double and signal double, then expose the
    # real method on the window — the base fixture stubbed it out.
    win.view_state._applylistgate = lambda name: win.connection_manager.applylistgate(name)
    win.view_state._gate_name_fetched = win._gateNameFetched
    win._refreshGateNameAsync = win.view_state._refreshGateNameAsync
    return win


def test_fetched_gate_name_lands_in_the_cache(gate_win):
    gate_win._on_gate_name_fetched("h1", [{"gate": "gateA"}])
    assert gate_win._gate_name_cache["h1"][0] == "gateA"


@pytest.mark.parametrize("reply", [None, [], [{"gate": "-TRUE-"}],
                                   [{"gate": "-Ungated-"}]])
def test_an_ungated_spectrum_caches_as_no_gate(gate_win, reply):
    gate_win._on_gate_name_fetched("h1", reply)
    assert gate_win._gate_name_cache["h1"][0] is None


def test_a_malformed_reply_caches_as_no_gate_instead_of_raising(gate_win):
    gate_win._on_gate_name_fetched("h1", [{"not_a_gate_key": 1}])
    assert gate_win._gate_name_cache["h1"][0] is None


def test_the_label_is_corrected_only_while_the_pointer_is_still_there(gate_win):
    gate_win._hoveredSpectrumName = "h1"
    gate_win._on_gate_name_fetched("h1", [{"gate": "gateA"}])
    assert gate_win.currentPlot.gateLabel.text() == "Gate applied: gateA\n"

    gate_win.currentPlot.gateLabel.setText("Gate applied: gateA\n")
    gate_win._hoveredSpectrumName = "somewhere-else"
    gate_win._on_gate_name_fetched("h1", [{"gate": "gateB"}])
    assert gate_win.currentPlot.gateLabel.text() == "Gate applied: gateA\n"


def test_a_second_request_for_a_spectrum_already_in_flight_is_dropped(gate_win):
    # one fetch per spectrum, however many motion events arrive meanwhile
    gate_win._gate_name_inflight.add("h1")
    gate_win._refreshGateNameAsync("h1")
    assert gate_win._gate_name_inflight == {"h1"}


def test_a_completed_fetch_clears_the_inflight_marker(gate_win):
    gate_win._gate_name_inflight.add("h1")
    gate_win._on_gate_name_fetched("h1", [{"gate": "gateA"}])
    assert "h1" not in gate_win._gate_name_inflight
