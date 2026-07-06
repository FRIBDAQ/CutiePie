"""Behavior tests for SumRegionManager (H2 steps 0+2).

After the H2 inversion the service holds no histogram combo and no integrate
popup: widget writes leave as signals (`regionReadoutChanged`,
`sumRegionCreatePrepared`, `sumRegionSelectionChanged`, popup-close requests,
`integrationResultsReady`) and widget reads arrive as method arguments. The ONE
injected widget kept is the sum-region popup, used ONLY for its co-owned
`listRegionLine`/`prevPoint` drawing buffer (shared with gate_manager); those
assertions read that buffer directly. matplotlib runs under Agg so Line2D/axis
behavior is real.
"""

import importlib
import logging
import os
import sys
import threading

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.lines as mlines
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import qt_stubs


_AFFECTED_MODULES = (
    "PyQt5", "PyQt5.QtCore", "PyQt5.QtWidgets",
    "CPyConverter", "httplib2",
    "services.sum_region_manager",
)


@pytest.fixture(scope="module")
def srm_mod():
    saved = {name: sys.modules.get(name) for name in _AFFECTED_MODULES}
    installed = qt_stubs.install_missing_runtime_stubs()
    if installed:
        sys.modules.pop("services.sum_region_manager", None)
    module = importlib.import_module("services.sum_region_manager")
    yield module
    for name, prev in saved.items():
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


class FakeSumPopup:
    """Only the co-owned drawing buffer survives the inversion."""
    def __init__(self):
        self.listRegionLine = []
        self.prevPoint = None


class FakeIntegrateRest:
    def __init__(self):
        self.calls_1d = []
        self.calls_2d = []
    def integrate1D(self, name, lo, hi):
        self.calls_1d.append((name, lo, hi))
        return {"counts": 10, "centroid": 5.0, "fwhm": 2.0}
    def integrate2D(self, name, points):
        self.calls_2d.append((name, points))
        return {"counts": 20, "centroid": [1.0, 2.0], "fwhm": [3.0, 4.0]}


class Rig:
    """Composition-root double: wires SumRegionManager as MainWindow does (H2)."""
    def __init__(self, module, monkeypatch):
        from services.spectrum_store import SpectrumStore
        self.mod = module
        self.store = SpectrumStore()
        self.histo_names = []
        self.popup = FakeSumPopup()
        self.skip_auto = threading.Event()
        self.rest = FakeIntegrateRest()
        self.info = {}                       # index -> {key: value}
        self.add_line_calls = []
        self.remove_prev_calls = []
        self.check_cancel_calls = []
        self.msgbox = qt_stubs.fresh_message_box()
        monkeypatch.setattr(module, "QMessageBox", self.msgbox)

        self.srm = module.SumRegionManager(
            spectra=self.store,
            name_from_index=lambda i: self.info.get(i, {}).get("name"),
            get_spectrum_info=self.get_info,
            get_histo_names=lambda: self.histo_names,
            skip_auto=self.skip_auto,
            add_line=self.add_line,
            remove_prev_line=lambda: self.remove_prev_calls.append(1),
            get_rest=lambda: self.rest,
            check_and_cancel_gate=lambda doClose: self.check_cancel_calls.append(doClose),
            sum_popup=self.popup,
            parent_widget=None,
            logger=logging.getLogger("test.srm"),
        )
        self.started   = qt_stubs.record_signal(self.srm.sumRegionStarted)
        self.ended     = qt_stubs.record_signal(self.srm.sumRegionEnded)
        self.draw      = qt_stubs.record_signal(self.srm.canvasDrawRequested)
        self.tight     = qt_stubs.record_signal(self.srm.figureTightLayoutRequested)
        self.replot    = qt_stubs.record_signal(self.srm.updatePlotRequested)
        self.gatedisc  = qt_stubs.record_signal(self.srm.gateSignalsDisconnectRequested)
        self.readout   = qt_stubs.record_signal(self.srm.regionReadoutChanged)
        self.prepared  = qt_stubs.record_signal(self.srm.sumRegionCreatePrepared)
        self.selection = qt_stubs.record_signal(self.srm.sumRegionSelectionChanged)
        self.popclose  = qt_stubs.record_signal(self.srm.sumRegionPopupCloseRequested)
        self.results   = qt_stubs.record_signal(self.srm.integrationResultsReady)
        self.intclose  = qt_stubs.record_signal(self.srm.integratePopupCloseRequested)

    def get_info(self, key, index=None):
        return self.info.get(index, {}).get(key)

    def add_line(self, x, y, index, label=""):
        ln = mlines.Line2D([float(x), float(x)], [float(y), float(y)], label=label)
        self.add_line_calls.append((float(x), float(y), index, label))
        return ln

    def add_1d(self, name="h1", index=0):
        self.store.set(name, dim=1, binx=10, minx=0.0, maxx=10.0,
                       data=list(range(11)), parameters=[], type="1")
        self.info.setdefault(index, {})["name"] = name


@pytest.fixture
def rig(srm_mod, monkeypatch):
    return Rig(srm_mod, monkeypatch)


def _ax():
    fig = Figure()
    FigureCanvasAgg(fig)
    return fig.add_subplot(1, 1, 1)


class Evt:
    def __init__(self, x, y=0.0): self.xdata = x; self.ydata = y


# --------------------------------------------------------------- dict CRUD

def test_set_and_get_sum_region_round_trip(rig):
    line = mlines.Line2D([1, 1], [0, 1], label="sumReg_-_R1_-_0")
    rig.srm.setSumRegion(0, line, "h1")
    assert rig.srm.getSumRegion(0, "h1") == [line]


def test_set_sum_region_ignores_non_region_label(rig):
    line = mlines.Line2D([1, 1], [0, 1], label="gate_-_G1_-_0")
    rig.srm.setSumRegion(0, line, "h1")
    assert rig.srm.getSumRegion(0, "h1") is None


def test_get_sum_region_none_index_returns_none(rig):
    assert rig.srm.getSumRegion(None, "h1") is None


def test_delete_sum_region_dict_removes_from_axes_and_dict(rig):
    ax = _ax()
    line = mlines.Line2D([1, 1], [0, 1], label="sumReg_-_R1_-_0")
    ax.add_line(line)
    rig.srm.setSumRegion(0, line, "h1")
    rig.srm.deleteSumRegionDict("sumReg_-_R1_-_0", ax.figure.axes)
    assert rig.srm.sumRegionDict["h1"] == []
    assert line not in ax.get_children()


def test_refresh_prunes_regions_not_in_histo_list(rig):
    rig.srm.sumRegionDict = {"h1": ["x"], "gone": ["y"]}
    rig.histo_names = ["h1", "h2"]                       # via the get_histo_names seam
    rig.srm.refreshSpectrumSumRegionDict()
    assert set(rig.srm.sumRegionDict) == {"h1"}


# --------------------------------------------------------------- integration math

def test_set_precision_2d_list(rig):
    out = rig.srm.setPrecisionIntegrationResult(
        {"centroid": [1.23456, 2.0], "fwhm": [3.14159, 4.0], "counts": 7.9})
    assert out["counts"] == 7
    assert out["centroid"][0] == "1.235E+00"
    assert out["fwhm"][1] == "4.000E+00"


def test_set_precision_1d_scalar(rig):
    out = rig.srm.setPrecisionIntegrationResult(
        {"centroid": 9.0, "fwhm": 1.5, "counts": 3.2})
    assert out["counts"] == 3
    assert out["centroid"] == "9.000E+00"


def test_integrate_gate_local_1d_sorts_boundaries_and_tags_region(rig):
    rig.add_1d("h1")
    g0 = mlines.Line2D([3.0, 3.0], [0, 1], label="gate_-_G1_-_0")
    g1 = mlines.Line2D([1.0, 1.0], [0, 1], label="gate_-_G1_-_1")
    out = rig.srm.integrateGateLocal(0, "h1", [g0, g1])
    assert rig.rest.calls_1d == [("h1", 1.0, 3.0)]     # sorted lo..hi
    assert out["h1"][0]["regionName"] == "G1"


def test_integrate_gate_local_no_lines_returns_none(rig):
    assert rig.srm.integrateGateLocal(0, "h1", []) is None


def test_integrate_gate_local_no_rest_returns_none(rig, monkeypatch):
    monkeypatch.setattr(rig.srm, "_get_rest", lambda: None)
    g0 = mlines.Line2D([3.0, 3.0], [0, 1], label="gate_-_G1_-_0")
    assert rig.srm.integrateGateLocal(0, "h1", [g0]) is None


def test_format_rows_blanks_repeated_spectrum_name(rig):
    results = {"h1": [
        {"centroid": 1.0, "fwhm": 2.0, "counts": 5, "regionName": "A"},
        {"centroid": 3.0, "fwhm": 4.0, "counts": 6, "regionName": "B"},
    ]}
    rows = rig.srm._format_integration_rows(results)
    assert rows[0][0] == "h1"                            # first occurrence keeps name
    assert rows[1][0] == ""                              # repeat blanked (old table dedup)
    assert [r[1] for r in rows] == ["A", "B"]


# --------------------------------------------------------------- CRUD + signals

def test_cancel_sum_region_ends_requests_replot_and_close(rig):
    rig.srm._creating_sum_region = True
    rig.srm.cancelSumRegion(doClose=True)
    assert rig.srm._creating_sum_region is False
    assert len(rig.ended) == 1
    assert len(rig.replot) == 1
    assert len(rig.popclose) == 1                        # close is a signal now


def test_cancel_sum_region_no_close_keeps_popup(rig):
    rig.srm.cancelSumRegion(doClose=False)
    assert len(rig.popclose) == 0
    assert len(rig.replot) == 1


def test_create_sum_region_without_index_warns(rig):
    rig.srm.createSumRegion(None, None, _ax())
    assert rig.msgbox.calls == [("about", "Warning!", "Please add/select a spectrum")]
    assert len(rig.started) == 0
    assert len(rig.prepared) == 0


def test_create_sum_region_prepares_names_and_signals(rig):
    rig.add_1d("h1", index=0)
    ax = _ax()
    ax.add_line(mlines.Line2D([1, 1], [0, 1], label="sumReg_-_R1_-_0"))
    rig.srm.createSumRegion(0, "h1", ax)
    assert rig.prepared == [(["R1"],)]                   # names handed to MainWindow
    assert rig.srm._saved_region_names == ["R1"]         # kept as service state
    assert rig.srm._active_sum_index == 0
    assert rig.started == [(0,)]
    assert rig.srm._creating_sum_region is True
    assert rig.skip_auto.is_set()


def test_on_singleclick_1d_appends_line_emits_readout_and_draws(rig):
    rig.add_1d("h1")
    rig.srm.on_singleclick_sumRegion(Evt(4.5), 0, "h1")
    assert len(rig.popup.listRegionLine) == 1            # co-owned buffer still on the popup
    assert len(rig.readout) == 1 and "X= 4.50000" in rig.readout[0][0]
    assert len(rig.draw) == 1
    assert rig.add_line_calls == [(4.5, 0.0, 0, "")]


def test_save_sum_region_1d_builds_two_labeled_lines(rig):
    rig.add_1d("h1")
    ax = _ax()
    spectrum = mlines.Line2D([0, 1], [0, 1]); ax.add_line(spectrum)
    rig.info[0]["spectrum"] = spectrum
    rig.popup.listRegionLine = [mlines.Line2D([2, 2], [0, 1]),
                                mlines.Line2D([6, 6], [0, 1])]
    rig.srm.saveSumRegion(0, "h1", "R1")                 # region name is an argument now
    saved = rig.srm.sumRegionDict["h1"]
    assert [l.get_label() for l in saved] == ["sumReg_-_R1_-_0", "sumReg_-_R1_-_1"]


def test_delete_sum_region_found_resets_selection_and_redraws(rig):
    rig.add_1d("h1")
    ax = _ax()
    for i in (0, 1):
        line = mlines.Line2D([1, 1], [0, 1], label=f"sumReg_-_R1_-_{i}")
        ax.add_line(line); rig.srm.setSumRegion(0, line, "h1")
    rig.srm._saved_region_names = ["R1"]
    rig.srm.deleteSumRegion(0, "h1", ax, "R1")
    assert rig.selection == [("None",)]                  # combo reset via signal
    assert len(rig.tight) == 1 and len(rig.draw) == 1
    assert rig.srm.sumRegionDict["h1"] == []


# --------------------------------------------------------------- integrate flow

def test_integrate_without_axis_warns(rig):
    rig.srm.integrate(0, "h1", None)
    assert rig.msgbox.calls == [("about", "Warning!", "Please add/select one spectrum")]
    assert len(rig.results) == 0


def test_integrate_nothing_to_integrate_emits_empty_rows(rig):
    rig.add_1d("h1")
    ax = _ax()
    rig.srm.integrate(0, "h1", ax)
    assert len(rig.gatedisc) == 1
    assert rig.results == [([],)]                        # empty -> MainWindow shows "Nothing to integrate"


def test_integrate_emits_rows_for_gate_lines(rig):
    rig.add_1d("h1")
    ax = _ax()
    ax.add_line(mlines.Line2D([3.0, 3.0], [0, 1], label="gate_-_G1_-_0"))
    ax.add_line(mlines.Line2D([1.0, 1.0], [0, 1], label="gate_-_G1_-_1"))
    rig.srm.integrate(0, "h1", ax)
    assert len(rig.results) == 1
    rows = rig.results[0][0]
    assert rows[0][0] == "h1" and rows[0][1] == "G1"


def test_ok_integrate_disconnects_and_closes(rig):
    rig.srm.okIntegrate()
    assert len(rig.gatedisc) == 1
    assert len(rig.intclose) == 1
