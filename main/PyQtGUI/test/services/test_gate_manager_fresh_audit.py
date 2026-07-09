"""Fresh-audit regression tests for GateManager (M19, M20)."""

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
    "PyQt5", "PyQt5.QtCore", "PyQt5.QtGui", "PyQt5.QtWidgets",
    "CPyConverter", "httplib2",
    "services.gate_manager",
)


@pytest.fixture(scope="module")
def gm_mod():
    saved = {name: sys.modules.get(name) for name in _AFFECTED_MODULES}
    installed = qt_stubs.install_missing_runtime_stubs()
    if installed:
        sys.modules.pop("services.gate_manager", None)
    module = importlib.import_module("services.gate_manager")
    yield module
    for name, prev in saved.items():
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


class FakeRest:
    def __init__(self, gates=None):
        self._gates = gates or []

    def listGate(self, pattern="*"):
        return self._gates


def make_gm(gm_mod, ax, gates):
    from services.spectrum_store import SpectrumStore
    store = SpectrumStore()
    store.set("h1", dim=1, binx=4, minx=0.0, maxx=4.0,
              data=np.arange(5, dtype=float), parameters=["p1"], type="1")
    rest = FakeRest(gates)
    return gm_mod.GateManager(
        spectra=store,
        name_from_index=lambda i: "h1",
        get_spectrum_info=lambda key, index=None: {"axis": ax}.get(key),
        get_is_enlarged=lambda: True,
        get_geo=lambda: {0: "h1"},
        get_sum_region=lambda i, n: None,
        get_current_canvas=lambda: ax.figure.canvas,
        integrate_popup=None,
        get_integrate_copy=lambda: None,
        get_hide=lambda: False,
        get_annotate=lambda: False,
        get_edit_disable=lambda: False,
        get_readout=lambda: "",
        get_gate_type=lambda: "s",
        get_gate_name=lambda: "g-gone",
        sum_region_popup=None,
        skip_auto=threading.Event(),
        get_rest=lambda: rest,
        gate_popup=None,
        parent_widget=None,
        logger=logging.getLogger("t.gm.fresh"),
    )


def _ax():
    fig = Figure()
    FigureCanvasAgg(fig)
    return fig.add_subplot(111)


def test_gate_name_changed_gate_missing_in_rest_does_not_raise(gm_mod):
    # M19: the selected gate exists as an artist/name but was deleted
    # server-side -> rest.listGate() has no entry -> gate[0] raised IndexError.
    ax = _ax()
    gm = make_gm(gm_mod, ax, gates=[])
    gm._editing_gate = True
    gm._gate_names = ["g-gone"]
    readouts = qt_stubs.record_signal(gm.gateReadoutChanged)
    gm.gateNameListChanged()                 # pre-fix: IndexError
    assert readouts                          # readout cleared instead


class FakePickEvent:
    def __init__(self, artist):
        self.artist = artist
        self.mouseevent = type("M", (), {"button": 1})()


def test_click_on_gate_line_gate_missing_in_rest_does_not_raise(gm_mod):
    ax = _ax()
    line, = ax.plot([1, 1], [0, 1], label="gate_-_g-gone_-_0")
    gm = make_gm(gm_mod, ax, gates=[])
    gm.clickOnGateLine(FakePickEvent(line))  # pre-fix: IndexError
    assert gm.editThisGateLine is line


def test_annotation_offsets_stack_per_gate(gm_mod):
    # M20: getXYAnnotation iterated dict KEYS with a substring test and
    # returned in the first iteration — the third gate never got a deeper
    # offset (always -0.05).
    ax = _ax()
    gm = make_gm(gm_mod, ax, gates=[])
    a = gm.getXYAnnotation("h1", "gA", (5.0, 0.95))
    b = gm.getXYAnnotation("h1", "gB", (5.0, 0.95))
    c = gm.getXYAnnotation("h1", "gC", (5.0, 0.95))
    a2 = gm.getXYAnnotation("h1", "gA", (6.0, 0.95))
    assert a == (5.0, pytest.approx(0.95))
    assert b == (5.0, pytest.approx(0.90))
    assert c == (5.0, pytest.approx(0.85))   # pre-fix: 0.90 (never stacked)
    assert a2 == (6.0, pytest.approx(0.95))  # same gate keeps its offset, fresh x
