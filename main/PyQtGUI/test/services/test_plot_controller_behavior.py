"""Characterization tests for PlotController.

These pin PlotController's CURRENT observable behavior — rendering data flow,
axis scaling, cutoff masking, the wConf/wTab/cutoff-popup widget effects —
before the step-1 inversion. The ~50 widget touches that go through the
injected `_get_current_plot()` seam are exercised via a fake plot widget
carrying a REAL matplotlib figure (Agg), so line/imshow/axis behavior is real.

Two standing regression pins live here: the E7 two-tier axis rule (bin edges
from the REST store tier, never the per-tab view tier) and the P1 customMinMax
semantics.
"""

import importlib
import logging
import os
import sys
import threading

import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import qt_stubs


_AFFECTED_MODULES = (
    "PyQt5", "PyQt5.QtCore", "PyQt5.QtWidgets",
    "CPyConverter", "httplib2",
    "services.plot_controller",
)


@pytest.fixture(scope="module")
def pc_mod():
    saved = {name: sys.modules.get(name) for name in _AFFECTED_MODULES}
    installed = qt_stubs.install_missing_runtime_stubs()
    if installed:
        sys.modules.pop("services.plot_controller", None)
    module = importlib.import_module("services.plot_controller")
    yield module
    for name, prev in saved.items():
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


class FakeAction:
    def __init__(self):
        self.triggered = qt_stubs.BoundStubSignal()
        self.checked = False

    def setChecked(self, checked):
        self.checked = checked


class FakePlotWidget:
    """The per-tab plot widget: REAL matplotlib figure + fake Qt controls."""

    def __init__(self, nrows=1, ncols=1):
        self.figure = Figure()
        FigureCanvasAgg(self.figure)
        for i in range(nrows * ncols):
            self.figure.add_subplot(nrows, ncols, i + 1)
        self.canvas = self.figure.canvas
        self.histo_autoscale = qt_stubs.FakeCheckBox()
        self.logButton = qt_stubs.FakeButton()
        self.customZoomButton = qt_stubs.FakeButton()
        self.zoom_action = FakeAction()
        self.isEnlarged = False
        self.isLoaded = False
        self.isSelected = False
        self.zoomPress = False
        self.selected_plot_index = None
        self.next_plot_index = -1
        self.rec = None
        self.recDashed = None


class Rig:
    """Composition-root double: wires PlotController exactly as MainWindow does."""

    def __init__(self, module, monkeypatch, nrows=1, ncols=1):
        from services.spectrum_store import SpectrumStore
        self.mod = module
        self.store = SpectrumStore()
        self.cp = FakePlotWidget(nrows, ncols)
        self.geo = {}
        self.info = {}
        self.enlarged = object()          # not-None: skip colorbar creation
        self.auto_index_val = 0
        self.next_index_val = 0
        self.draw_gate_calls = []
        self.clean_popup_calls = []
        self.bind_calls = []
        self.auto_update_calls = []
        self.stop_auto = threading.Event()
        self.msgbox = qt_stubs.fresh_message_box()
        monkeypatch.setattr(module, "QMessageBox", self.msgbox)

        self.pc = module.PlotController(
            spectra=self.store,
            get_current_plot=lambda: self.cp,
            get_geo=lambda: self.geo,
            set_geo=self.geo.__setitem__,
            get_spectrum_info=self.get_info,
            set_spectrum_info=self.set_info,
            get_spectrum_info_dict=lambda: self.info,
            name_from_index=lambda i: self.geo.get(i),
            get_enlarged_spectrum=lambda: self.enlarged,
            auto_index=lambda: self.auto_index_val,
            next_index=lambda: self.next_index_val,
            bind_dynamic_signal=lambda: self.bind_calls.append(1),
            draw_gate=self.draw_gate_calls.append,
            clean_popup_exit=self.clean_popup_calls.append,
            auto_update_start=lambda: self.auto_update_calls.append(1),
            stop_auto_update_thread=self.stop_auto,
            min_y=0.001, max_y=1024, min_z=0.001, max_z=256,
            parent_widget=None,
            logger=logging.getLogger("test.plot_controller"),
        )
        # output signals, recorded from construction on
        self.prepared = qt_stubs.record_signal(self.pc.cutoffPopupPrepared)
        self.close_req = qt_stubs.record_signal(self.pc.cutoffPopupCloseRequested)

    def get_info(self, key, index=None):
        return self.info.get(index, {}).get(key)

    def set_info(self, index=None, **kwargs):
        self.info.setdefault(index, {}).update(kwargs)

    def add_1d(self, name="h1", index=0, binx=10, minx=0.0, maxx=10.0, data=None):
        if data is None:
            data = np.arange(binx + 1, dtype=float)
        self.store.set(name, dim=1, binx=binx, minx=minx, maxx=maxx,
                       data=data, parameters=[], type="1")
        self.geo[index] = name
        ax = self.cp.figure.axes[index]
        self.info.setdefault(index, {}).update(axis=ax, name=name)
        return ax


@pytest.fixture
def rig(pc_mod, monkeypatch):
    return Rig(pc_mod, monkeypatch)


# ---------------------------------------------------------------- pure logic

def test_custom_min_max_semantics(rig):
    # pin: min/max over strictly positive values only
    assert rig.pc.customMinMax(np.array([0, 3, 1, 7])) == (1, 7)
    assert rig.pc.customMinMax(np.zeros(4)) == (None, None)
    masked = np.ma.masked_where(np.array([5, 1, 9]) > 6, np.array([5, 1, 9]))
    assert rig.pc.customMinMax(masked) == (1, 5)


def test_create_range(pc_mod):
    r = pc_mod.PlotController.createRange(4, 0, 8)
    assert list(r) == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_cutoff_masked_data(rig):
    rig.add_1d(data=np.array([0.0, 2.0, 5.0, 9.0]), binx=3)
    rig.set_info(index=0, cutoff=[3.0, 8.0])
    w = rig.pc._cutoff_masked_data(0)
    assert list(w.compressed()) == [5.0]
    # canonical store data untouched
    assert list(rig.store.get("h1", "data")) == [0.0, 2.0, 5.0, 9.0]


def test_get_min_max_in_range_1d(rig):
    rig.add_1d(binx=10, minx=0.0, maxx=10.0, data=np.arange(11, dtype=float))
    # bins for x in (2, 8]: data[3:10].max() * 1.1
    assert rig.pc.getMinMaxInRange(0, xmin=2.0, xmax=8.0) == pytest.approx(9 * 1.1)


def test_get_axis_properties(rig):
    ax = rig.add_1d()
    ax.set_xlim(1, 9)
    ax.set_ylim(2, 8)
    xr, yr = rig.pc.getAxisProperties(0)
    assert xr == [1.0, 9.0] and yr == [2.0, 8.0]


# ------------------------------------------------------------- axis scaling

def test_set_axis_scale_1d_linear_and_log(rig):
    ax = rig.add_1d()
    rig.set_info(index=0, minx=0.0, maxx=10.0, miny=1.0, maxy=100.0, log=False)
    rig.pc.setAxisScale(ax, 0, "x", "y")
    assert ax.get_xlim() == (0.0, 10.0)
    assert ax.get_ylim() == (1.0, 100.0)
    assert ax.get_yscale() == "linear"

    rig.set_info(index=0, log=True)
    rig.pc.setAxisScale(ax, 0, "log")
    assert ax.get_yscale() == "log"
    # miny/maxy written back to the display tier
    assert rig.get_info("miny", index=0) == 1.0
    assert rig.get_info("maxy", index=0) == 100.0


def test_set_axis_scale_log_with_miny_none_does_not_crash(rig):
    # Regression: the log branch did `if ymin <= 0` where ymin could be None
    # when only miny (not maxy) was unset — the both-empty fallback on line 99
    # doesn't fire in the asymmetric case, so `None <= 0` raised TypeError.
    # The asymmetric-None guard coerces ymin to the log-safe default self.minY.
    ax = rig.add_1d()
    rig.cp.histo_autoscale.setChecked(False)
    # maxy is a real value but miny is left UNSET -> get_info("miny") is None
    rig.set_info(index=0, minx=0.0, maxx=10.0, maxy=100.0, log=True)
    rig.pc.setAxisScale(ax, 0, "log")            # must not raise TypeError
    assert ax.get_yscale() == "log"
    lo, hi = ax.get_ylim()
    assert lo == pytest.approx(0.001)            # coerced to self.minY
    assert hi == pytest.approx(100.0)


def test_set_axis_scale_1d_autoscale_uses_visible_range(rig):
    ax = rig.add_1d(binx=10, minx=0.0, maxx=10.0, data=np.arange(11, dtype=float))
    rig.set_info(index=0, minx=2.0, maxx=8.0, miny=0.0, maxy=0.0, log=False)
    rig.cp.histo_autoscale.setChecked(True)
    rig.pc.setAxisScale(ax, 0, "x", "y")
    # ymax recomputed from the data inside the visible x-range
    assert ax.get_ylim()[1] == pytest.approx(9 * 1.1)


# ---------------------------------------------------------------- rendering

def make_line(ax):
    line, = ax.plot([], [], drawstyle='steps')
    return line


def test_plot_plot_1d_sets_bin_edges_from_store(rig):
    ax = rig.add_1d(binx=10, minx=0.0, maxx=10.0)
    line = make_line(ax)
    rig.set_info(index=0, spectrum=line)
    rig.pc.plotPlot(0)
    x, y = line.get_data()
    assert len(x) == 11 and x[0] == 0.0 and x[-1] == 10.0
    assert list(y) == list(np.arange(11, dtype=float))


def test_setup_plot_1d_uses_rest_tier_bin_edges(rig):
    # pin: view tier (per-tab minx/maxx) holds a ZOOM range; bin edges must
    # come from the store tier or the spectrum compresses into the zoom window.
    ax = rig.add_1d(binx=10, minx=0.0, maxx=10.0)
    rig.set_info(index=0, minx=4.0, maxx=6.0)      # zoomed view range
    rig.pc.setupPlot(ax, 0)
    line = rig.get_info("spectrum", index=0)
    x = line.get_xdata()
    assert x[0] == 0.0 and x[-1] == 10.0           # store tier, not 4..6
    assert ax.get_xlim() == (4.0, 6.0)             # view restore untouched


def test_setup_plot_2d_imshow_extent_from_store(rig):
    rig.store.set("m2", dim=2, binx=4, minx=0.0, maxx=4.0,
                  biny=4, miny=0.0, maxy=8.0,
                  data=np.arange(16, dtype=float).reshape(4, 4),
                  parameters=[], type="2")
    rig.geo[0] = "m2"
    ax = rig.cp.figure.axes[0]
    rig.info.setdefault(0, {}).update(axis=ax, name="m2",
                                      minx=0.0, maxx=4.0, binx=4, biny=4)
    rig.pc.setupPlot(ax, 0)
    spectrum = rig.get_info("spectrum", index=0)
    assert list(spectrum.get_extent()) == [0.0, 4.0, 0.0, 8.0]


def test_update_plot_grid_flow(rig):
    ax = rig.add_1d()
    line = make_line(ax)
    rig.set_info(index=0, spectrum=line)
    rig.pc._layout_dirty = False
    rig.pc.updatePlot()
    assert rig.clean_popup_calls == [False]
    assert rig.draw_gate_calls == [0]
    x, _ = line.get_data()
    assert len(x) == 11                             # plotPlot ran


def test_update_plot_no_axis_never_pops_modal(rig):
    # regression: an auto-update tick that finds a geometry slot with
    # no built axis (polling started before Add/geometry, or a config change)
    # must NOT raise a modal dialog — the timer keeps firing, so a modal here
    # stacks a new blocking dialog every tick and freezes the GUI.
    rig.geo[0] = "h1"          # geometry references a spectrum...
    assert rig.info == {}      # ...but no axis has been built for it
    rig.pc.updatePlot()
    assert rig.msgbox.calls == []          # no dialog from the render tick
    assert rig.draw_gate_calls == []       # nothing drawn, cleanly skipped


def test_update_plot_skips_empty_slot_but_draws_valid_one(pc_mod, monkeypatch):
    # A half-configured grid (slot 0 unbuilt, slot 1 valid) must still refresh
    # the valid pad instead of bailing on the whole tick.
    rig = Rig(pc_mod, monkeypatch, nrows=1, ncols=2)
    rig.geo[0] = "missing"
    ax = rig.add_1d(name="h1", index=1)
    line = make_line(ax)
    rig.set_info(index=1, spectrum=line)
    rig.pc._layout_dirty = False
    rig.pc.updatePlot()
    assert rig.msgbox.calls == []
    assert rig.draw_gate_calls == [1]      # valid slot 1 was drawn


def test_zoom_in_out_1d(rig):
    ax = rig.add_1d()
    line = make_line(ax)
    ax.set_ylim(0.0, 100.0)
    rig.set_info(index=0, spectrum=line)
    rig.cp.histo_autoscale.setChecked(True)
    rig.pc.zoomInOut("in")
    assert ax.get_ylim() == (0.0, 50.0)
    assert rig.cp.histo_autoscale.isChecked() is False
    assert rig.draw_gate_calls == [0]
    rig.pc.zoomInOut("out")
    assert ax.get_ylim() == (0.0, 100.0)


def test_log_button_toggles_scale(rig):
    ax = rig.add_1d()
    line = make_line(ax)
    rig.set_info(index=0, spectrum=line, minx=0.0, maxx=10.0,
                 miny=1.0, maxy=100.0, log=False)
    rig.pc.logButtonCallback(0)
    assert rig.get_info("log", index=0) is True
    assert ax.get_yscale() == "log"
    rig.pc.logButtonCallback(0)
    assert rig.get_info("log", index=0) is False
    assert ax.get_yscale() == "linear"


# --------------------------------------------- layout / add-plot arguments

def test_mark_geometry_applied(rig):
    assert rig.pc.geometry_applied is False
    rig.pc.markGeometryApplied()
    assert rig.pc.geometry_applied is True


def test_plot_position_walks_given_layout(rig):
    assert rig.pc.plotPosition(0, [2, 2]) == (0, 0)
    assert rig.pc.plotPosition(1, [2, 2]) == (0, 1)
    assert rig.pc.plotPosition(3, [2, 2]) == (1, 1)


def test_add_plot_requires_geometry(rig):
    rig.pc.geometry_applied = False
    rig.pc.addPlot("h1")
    assert rig.geo == {} and rig.msgbox.calls == []


def test_add_plot_warns_when_no_spectra_connected(rig):
    rig.pc.geometry_applied = True
    rig.pc.addPlot(None)                           # empty histo_list
    assert rig.msgbox.calls[0][0] == "about"
    assert any("Connection" in text for _, _, text in rig.msgbox.calls)


def test_add_plot_places_selected_spectrum(rig):
    rig.pc.geometry_applied = True
    rig.store.set("h1", dim=1, binx=10, minx=0.0, maxx=10.0,
                  data=np.arange(11, dtype=float), parameters=[], type="1")
    ax = rig.cp.figure.axes[0]
    rig.info.setdefault(0, {}).update(axis=ax, name="h1", miny=0.0)
    rig.pc.addPlot("h1", tab_click_bound=True)
    assert rig.geo == {0: "h1"}
    assert rig.cp.histo_autoscale.isChecked() is True
    assert rig.cp.logButton.down is False
    assert rig.get_info("log", index=0) is False
    assert 0 in rig.draw_gate_calls
    assert rig.cp.recDashed is not None
    assert rig.cp.isSelected is False


# ------------------------------------------------------------- cutoff popup

def test_ok_cutoff_applies_ranges_and_requests_close(rig):
    ax = rig.add_1d()
    line = make_line(ax)
    rig.set_info(index=0, spectrum=line)
    rig.cp.selected_plot_index = 0
    rig.pc.okCutoff("8", "2", "1", "50")           # x min/max swapped on purpose
    assert ax.get_xlim() == (2.0, 8.0)             # swapped back
    assert ax.get_ylim() == (1.0, 50.0)
    assert rig.get_info("minx", index=0) == 2.0
    assert rig.get_info("maxx", index=0) == 8.0
    assert rig.draw_gate_calls == [0]
    assert rig.close_req == [()]


def test_ok_cutoff_invalid_input_aborts(rig):
    rig.add_1d()
    rig.cp.selected_plot_index = 0
    rig.set_info(index=0, spectrum=make_line(rig.info[0]["axis"]))
    rig.pc.okCutoff("junk", "2", "1", "50")
    assert rig.close_req == []                     # bails before closing


def test_cutoff_button_publishes_popup_payload(rig):
    ax = rig.add_1d()
    ax.set_xlim(1.0, 9.0)
    ax.set_ylim(2.0, 60.0)
    rig.cp.selected_plot_index = 0
    rig.set_info(index=0, cutoff=[None, None])
    rig.pc.cutoffButtonCallback()
    assert rig.prepared == [({"name": "h1", "dim": 1,
                              "xmin": 1.0, "xmax": 9.0,
                              "ymin": 2.0, "ymax": 60.0,
                              "zmin": None, "zmax": None},)]
    assert rig.cp.histo_autoscale.isChecked() is False


def test_cutoff_button_without_selection_warns(rig):
    rig.cp.selected_plot_index = None
    rig.pc.cutoffButtonCallback()
    assert rig.msgbox.calls[0][0] == "about"
    assert rig.prepared == []


def test_reset_cutoff_clears_and_requests_close(rig):
    rig.add_1d()
    rig.cp.selected_plot_index = 0
    rig.set_info(index=0, cutoff=[3.0, 8.0])
    rig.pc.resetCutoff(doUpdate=False)
    assert rig.get_info("cutoff", index=0) == [None, None]
    assert rig.close_req == [()]


# ------------------------------------------------ change-driven redraw skip

def _spy_draw_idle(rig):
    """Replace the real Agg draw_idle with a call counter."""
    draws = []
    rig.cp.canvas.draw_idle = lambda: draws.append(1)
    return draws


def test_timer_tick_skips_redraw_when_data_unchanged(rig):
    # the auto-update timer tick (_updatePlotOnGui -> updatePlot(force=False))
    # must NOT redraw when no pad's counts changed since the previous tick.
    ax = rig.add_1d()
    line = make_line(ax)
    rig.set_info(index=0, spectrum=line)
    rig.pc._layout_dirty = False
    draws = _spy_draw_idle(rig)

    rig.pc._updatePlotOnGui()                     # first tick: draws + records sig
    assert draws == [1]
    assert rig.draw_gate_calls == [0]

    rig.pc._updatePlotOnGui()                     # data identical: skip entirely
    assert draws == [1]                           # no second draw
    assert rig.draw_gate_calls == [0]             # loop body did not re-run


def test_timer_tick_redraws_when_counts_change(rig):
    # A cumulative-counter increment must move the signature -> redraw.
    ax = rig.add_1d()
    line = make_line(ax)
    rig.set_info(index=0, spectrum=line)
    rig.pc._layout_dirty = False
    draws = _spy_draw_idle(rig)

    rig.pc._updatePlotOnGui()
    assert draws == [1]

    w = rig.store.get("h1", "data")               # live shm view; counts grow
    w[:] = w + 1
    rig.pc._updatePlotOnGui()
    assert draws == [1, 1]                         # changed -> drew again
    assert rig.draw_gate_calls == [0, 0]


def test_forced_updateplot_always_redraws_even_if_unchanged(rig):
    # The hide-gates toggle / gate / sum-region / geometry paths call updatePlot()
    # with force=True (default) and must ALWAYS render, even with static data —
    # otherwise a toggle would not take effect until counts next changed.
    ax = rig.add_1d()
    line = make_line(ax)
    rig.set_info(index=0, spectrum=line)
    rig.pc._layout_dirty = False
    draws = _spy_draw_idle(rig)

    rig.pc._updatePlotOnGui()                      # timer tick records signature
    assert draws == [1]

    rig.pc.updatePlot()                            # force=True, data unchanged
    assert draws == [1, 1]                         # still drew
    rig.pc.updatePlot()
    assert draws == [1, 1, 1]


def test_first_timer_tick_always_draws(rig):
    # No prior signature (None) -> never skip the very first tick.
    ax = rig.add_1d()
    line = make_line(ax)
    rig.set_info(index=0, spectrum=line)
    rig.pc._layout_dirty = False
    draws = _spy_draw_idle(rig)
    assert rig.pc._last_tick_signature is None

    rig.pc._updatePlotOnGui()
    assert draws == [1]
    assert rig.pc._last_tick_signature is not None


def test_add_plot_2d_scans_y_window_from_ylim_not_xlim(pc_mod, monkeypatch):
    # the 2D branch of addPlot read `ymin, ymax = ax.get_xlim()`, so the
    # initial z-autoscale scanned the wrong y-bin window on asymmetric spectra.
    r = Rig(pc_mod, monkeypatch)
    r.store.set("m2", dim=2, binx=10, minx=0.0, maxx=10.0,
                biny=20, miny=0.0, maxy=100.0,
                data=np.ones((20, 10)), parameters=[], type="2")
    # pre-seed the axis (the Rig's set_info doesn't mirror spectrum->axis)
    r.info[0] = {"axis": r.cp.figure.axes[0]}
    calls = []

    def fake_min_max(index, **limits):
        calls.append(limits)
        return (1.0, 2.0)

    monkeypatch.setattr(r.pc, "getMinMaxInRange", fake_min_max)
    r.pc.markGeometryApplied()
    r.pc.addPlot("m2", tab_click_bound=True)

    two_d = [c for c in calls if "ymin" in c][-1]   # the addPlot-branch call
    assert two_d["xmin"] == pytest.approx(0.0)
    assert two_d["xmax"] == pytest.approx(10.0)
    assert two_d["ymin"] == pytest.approx(0.0)
    assert two_d["ymax"] == pytest.approx(100.0)    # pre-fix: 10.0 (the xlim)


def test_ok_cutoff_accepts_decimal_values(rig):
    # cutoff fields were gated by isdigit(), silently dropping "10.5".
    rig.store.set("m2", dim=2, binx=4, minx=0.0, maxx=4.0,
                  biny=4, miny=0.0, maxy=4.0,
                  data=np.ones((4, 4)), parameters=[], type="2")
    rig.geo[0] = "m2"
    ax = rig.cp.figure.axes[0]
    art = ax.imshow(np.ones((4, 4)))
    rig.info[0] = {"axis": ax, "spectrum": art}
    rig.cp.selected_plot_index = 0
    rig.pc.okCutoff("0", "4", "0", "4", "10.5", "200.5")
    assert rig.get_info("cutoff", index=0) == [10.5, 200.5]   # pre-fix: None


def test_reset_all_continues_past_empty_pad(pc_mod, monkeypatch):
    # customHomeButtonCallback returned at the first pad without a
    # spectrum artist, so "Reset all" never reached later pads.
    r = Rig(pc_mod, monkeypatch, nrows=1, ncols=2)
    r.geo[0] = "ghost"                       # pad with no spectrum artist
    ax1 = r.add_1d(name="h1", index=1, binx=4, minx=0.0, maxx=4.0,
                   data=np.arange(5, dtype=float))
    line, = ax1.plot([], [], drawstyle="steps")
    r.set_info(index=1, spectrum=line)
    ax1.set_xlim(1.0, 2.0)                   # zoomed in; reset should restore
    r.pc.customHomeButtonCallback()          # index=None -> all pads
    assert 1 in r.draw_gate_calls            # pre-fix: [] (returned at pad 0)
    assert ax1.get_xlim() == (0.0, 4.0)


def test_set_cmap_norm_log_coerces_zero_zmin(rig):
    # `if zmin and zmin <= 0` let zmin == 0 through -> LogNorm(vmin=0),
    # which matplotlib rejects at draw time.
    ax = rig.cp.figure.axes[0]
    art = ax.imshow(np.ones((4, 4)))
    art.set_clim(0.0, 100.0)
    rig.geo[0] = "m2"
    rig.info[0] = {"axis": ax, "spectrum": art}
    rig.pc.setCmapNorm("log", 0)
    assert art.norm.vmin > 0                 # pre-fix: 0.0
    assert art.norm.vmax == pytest.approx(100.0)
