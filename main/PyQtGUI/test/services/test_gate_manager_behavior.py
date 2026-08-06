"""Characterization tests for GateManager. GateManager is the largest, most
widget-entangled service (~154 sites, ~18 collaborators) and its drawing runs
in the render-tick / hover hot paths."""

import importlib
import logging
import os
import sys
import threading
import types

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.lines as mlines
import matplotlib.text
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import qt_stubs


_AFFECTED_MODULES = (
    "PyQt5", "PyQt5.QtCore", "PyQt5.QtWidgets", "PyQt5.QtGui",
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


# ----------------------------------------------------------------- widget doubles

class FakeCheckBox:
    def __init__(self, checked=False): self._checked = checked
    def isChecked(self): return self._checked
    def setChecked(self, c): self._checked = c


class FakeCombo:
    def __init__(self, items=None):
        self.items = list(items or [])
        self.current_text = None
        self.editable = None
        self.insert_policy = None
        self.currentTextChanged = qt_stubs.BoundStubSignal()
        self.currentIndexChanged = qt_stubs.BoundStubSignal()
    def addItem(self, t): self.items.append(t)
    def clear(self): self.items = []
    def itemText(self, i): return self.items[i]
    def count(self): return len(self.items)
    def currentText(self): return self.current_text if self.current_text is not None else (self.items[0] if self.items else "")
    def setCurrentText(self, t): self.current_text = t
    def setEditable(self, e): self.editable = e
    def setInsertPolicy(self, p): self.insert_policy = p


class FakeText:
    def __init__(self, text=""): self._text = text
    def toPlainText(self): return self._text
    def clear(self): self._text = ""
    def insertPlainText(self, t): self._text += t


class FakeButton:
    def __init__(self):
        self.clicked = qt_stubs.BoundStubSignal()
        self.checked = None
        self.enabled = None
    def setChecked(self, c): self.checked = c
    def setEnabled(self, e): self.enabled = e


class FakeGatePopup:
    # Residual drawing buffer only (listRegionLine/prevPoint/regionPoint) plus
    # the widgets the service used to touch, kept so tests can seed reads via
    # the seams. After step 1 the service reaches none of these directly
    # except listRegionLine/prevPoint.
    def __init__(self):
        self.gateNameList = FakeCombo()
        self.listGateType = FakeCombo()
        self.regionPoint = FakeText()
        self.listRegionLine = []
        self.prevPoint = FakeText()
        self.gateActionCreate = FakeButton()
        self.gateActionEdit = FakeButton()
        self.preview = FakeButton()
        self.clearinfo_calls = 0
        self.shown = 0
        self.closed = 0
    def clearInfo(self): self.clearinfo_calls += 1
    def show(self): self.shown += 1
    def close(self): self.closed += 1


class FakeShortcut:
    # QShortcut stand-in with a connectable .activated (the qt_stubs QShortcut
    # is attribute-less, so editGate needs this monkeypatched in).
    def __init__(self, *a, **kw):
        self.activated = qt_stubs.BoundStubSignal()
        self.enabled = None
        self.parent_set = "unset"
    def setEnabled(self, e): self.enabled = e
    def setParent(self, p): self.parent_set = p


class FakeCanvas:
    def __init__(self): self.connected = []; self.disconnected = []
    def mpl_connect(self, evt, cb): self.connected.append(evt); return len(self.connected)
    def mpl_disconnect(self, cid): self.disconnected.append(cid)


class FakeGateRest:
    def __init__(self, gates=None):
        self._gates = gates if gates is not None else []
        self.created = []
    def listGate(self): return self._gates
    def createGate(self, name, gtype, params, boundaries):
        self.created.append((name, gtype, params, boundaries))


class Rig:
    def __init__(self, module, monkeypatch):
        from services.spectrum_store import SpectrumStore
        self.mod = module
        self.store = SpectrumStore()
        self.info = {}                    # index -> {key: value}
        self.geo = {}
        self.enlarged = False
        self.sum_region = None
        self.canvas = FakeCanvas()
        self.integ = type("I", (), {"resultsText": type("T", (), {"itemSelectionChanged": qt_stubs.BoundStubSignal()})()})()
        self.hide_cb = FakeCheckBox(False)
        self.annot_cb = FakeCheckBox(False)
        self.edit_disable_cb = FakeCheckBox(False)
        self.sum_popup = type("P", (), {"listRegionLine": [], "prevPoint": None})()
        self.skip_auto = threading.Event()
        self.rest = FakeGateRest()
        self.popup = FakeGatePopup()
        self.msgbox = qt_stubs.fresh_message_box()
        monkeypatch.setattr(module, "QMessageBox", self.msgbox)

        self.view_state = types.SimpleNamespace(
            nameFromIndex=lambda i: self.info.get(i, {}).get("name"),
            getSpectrumViewInfo=self.get_info,
            getGeo=lambda: self.geo,
        )
        self.gm = module.GateManager(
            spectra=self.store,
            view_state=self.view_state,
            get_is_enlarged=lambda: self.enlarged,
            get_sum_region=lambda idx, name: self.sum_region,
            get_current_canvas=lambda: self.canvas,
            integrate_popup=self.integ,
            get_integrate_copy=lambda: None,
            get_hide=self.hide_cb.isChecked,
            get_annotate=self.annot_cb.isChecked,
            get_edit_disable=self.edit_disable_cb.isChecked,
            get_readout=lambda: self.popup.regionPoint.toPlainText(),
            get_gate_type=lambda: self.popup.listGateType.currentText(),
            get_gate_name=lambda: self.popup.gateNameList.currentText(),
            sum_region_popup=self.sum_popup,
            skip_auto=self.skip_auto,
            get_rest=lambda: self.rest,
            gate_popup=self.popup,
            parent_widget=None,
            logger=logging.getLogger("test.gm"),
        )
        self.draw     = qt_stubs.record_signal(self.gm.canvasDrawRequested)
        self.drawidle = qt_stubs.record_signal(self.gm.canvasDrawIdleRequested)
        self.replot   = qt_stubs.record_signal(self.gm.updatePlotRequested)
        self.created  = qt_stubs.record_signal(self.gm.gateCreationStarted)
        self.editing  = qt_stubs.record_signal(self.gm.gateEditingStarted)
        self.ended    = qt_stubs.record_signal(self.gm.gateEnded)
        self.readout  = qt_stubs.record_signal(self.gm.gateReadoutChanged)
        self.readout_editable = qt_stubs.record_signal(self.gm.gateReadoutEditable)
        self.gtype_cleared = qt_stubs.record_signal(self.gm.gateTypeCleared)
        self.gtype_added   = qt_stubs.record_signal(self.gm.gateTypeItemAdded)
        self.names_prepared = qt_stubs.record_signal(self.gm.gateNamesPrepared)
        self.name_selected  = qt_stubs.record_signal(self.gm.gateNameSelected)
        self.clearinfo_req  = qt_stubs.record_signal(self.gm.gateClearInfoRequested)
        self.show_req       = qt_stubs.record_signal(self.gm.gatePopupShowRequested)
        self.close_req      = qt_stubs.record_signal(self.gm.gatePopupCloseRequested)
        self.create_checked = qt_stubs.record_signal(self.gm.gateActionCreateChecked)
        self.edit_checked   = qt_stubs.record_signal(self.gm.gateActionEditChecked)
        self.edit_enabled   = qt_stubs.record_signal(self.gm.gateActionEditEnabled)
        self.completer_cfg  = qt_stubs.record_signal(self.gm.gateNameCompleterConfigured)
        self.editable_req   = qt_stubs.record_signal(self.gm.gateNameListEditable)

    def get_info(self, key, index=None):
        return self.info.get(index, {}).get(key)

    def add_1d(self, name="h1", index=0, type_="1", params=("p1",)):
        self.store.set(name, dim=1, binx=10, minx=0.0, maxx=10.0,
                       data=list(range(11)), parameters=list(params), type=type_)
        self.info.setdefault(index, {})["name"] = name


@pytest.fixture
def rig(gm_mod, monkeypatch):
    return Rig(gm_mod, monkeypatch)


def _ax():
    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 10); ax.set_ylim(0, 100)
    return ax


# --------------------------------------------------------------- pure geometry

def test_dist_is_euclidean(rig):
    assert rig.gm.dist(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == 5.0


def test_dist_point_to_segment_three_branches(rig):
    s0, s1 = np.array([0.0, 0.0]), np.array([10.0, 0.0])
    # before s0 -> distance to s0
    assert rig.gm.dist_point_to_segment(np.array([-3.0, 4.0]), s0, s1) == 5.0
    # after s1 -> distance to s1
    assert rig.gm.dist_point_to_segment(np.array([13.0, 4.0]), s0, s1) == 5.0
    # perpendicular foot
    assert rig.gm.dist_point_to_segment(np.array([5.0, 4.0]), s0, s1) == 4.0


def test_pixel_to_data_distance(rig):
    # 5 px over a 100-wide axis rendered in 200 px -> 2.5 data units
    assert rig.gm.pixel_to_data_distance(5, (0, 100), 200) == 2.5


# --------------------------------------------------------------- color / annotation state

def test_get_gate_color_cycles_and_is_stable(rig):
    c1 = rig.gm.getGateColor("G1")
    c2 = rig.gm.getGateColor("G2")
    assert c1 != c2
    assert rig.gm.getGateColor("G1") == c1        # stable per gate


def test_get_xy_annotation_first_then_stack(rig):
    xy = (5.0, 0.95)
    assert rig.gm.getXYAnnotation("h1", "G1", xy) == xy          # first for spectrum
    stacked = rig.gm.getXYAnnotation("h1", "G2", (5.0, 0.95))
    assert stacked[0] == 5.0 and stacked[1] == pytest.approx(0.90)   # second gate offset down


# --------------------------------------------------------------- name helpers

def test_next_gate_name_is_max_plus_one(rig):
    assert rig.gm._nextGateName(["gate-001", "gate-007", "foo"]) == "gate-008"


def test_next_gate_name_empty_is_001(rig):
    assert rig.gm._nextGateName([]) == "gate-001"


def test_format_gate_point_text_1d_keeps_last_two(rig):
    rig.popup.regionPoint = FakeText("0: X= 1.0\n1: X= 2.0\n2: X= 3.0")
    assert rig.gm.formatGatePopupPointText(1) == [2.0, 3.0]


def test_format_gate_point_text_2d_pairs(rig):
    rig.popup.regionPoint = FakeText("0: X= 1.0 Y= 2.0\n1: X= 3.0 Y= 4.0")
    assert rig.gm.formatGatePopupPointText(2) == [[1.0, 2.0], [3.0, 4.0]]


def test_format_gate_point_text_malformed_returns_none(rig):
    rig.popup.regionPoint = FakeText("no point number here")
    assert rig.gm.formatGatePopupPointText(1) is None


# --------------------------------------------------------------- annotations don't accumulate

def test_set_gate_annotation_is_gid_keyed_and_idempotent(rig):
    rig.add_1d("h1")
    ax = _ax()
    ax.add_line(mlines.Line2D([4.0, 4.0], [0, 100], label="gate_-_G1_-_0"))
    rig.info[0]["axis"] = ax

    def n_annotations():
        return len([a for a in ax.get_children()
                    if isinstance(a, matplotlib.text.Annotation) and a.get_gid() == "G1_low"])

    rig.gm.setGateAnnotation(0, True)
    assert n_annotations() == 1
    rig.gm.setGateAnnotation(0, True)      # redraw must not stack a second one
    assert n_annotations() == 1
    rig.gm.setGateAnnotation(0, False)     # toggle off removes it
    assert n_annotations() == 0


# --------------------------------------------------------------- persistent gate artists

def test_draw_gate_reuses_line_across_redraws(rig):
    rig.add_1d("h1", type_="1", params=("p1",))
    ax = _ax()
    rig.info[0]["axis"] = ax
    rig.rest = FakeGateRest([
        {"name": "G1", "type": "s", "parameters": ["p1"], "low": 2.0, "high": 8.0}])

    rig.gm.drawGate(0)
    lines0 = [l for l in ax.lines if l.get_label() == "gate_-_G1_-_0"]
    assert len(lines0) == 1
    the_line = lines0[0]

    # move the gate; a redraw must UPDATE the same Line2D (set_data), not add one
    rig.rest._gates[0]["low"] = 3.0
    rig.gm._gate_cache_ts = 0.0            # force cache refresh
    rig.gm.drawGate(0)
    lines0b = [l for l in ax.lines if l.get_label() == "gate_-_G1_-_0"]
    assert len(lines0b) == 1
    assert lines0b[0] is the_line                       # same object reused
    assert list(the_line.get_xdata()) == [3.0, 3.0]     # updated in place


def test_draw_gate_hidden_removes_lines(rig):
    rig.add_1d("h1")
    ax = _ax()
    rig.info[0]["axis"] = ax
    rig.rest = FakeGateRest([
        {"name": "G1", "type": "s", "parameters": ["p1"], "low": 2.0, "high": 8.0}])
    rig.gm.drawGate(0)
    assert any(l.get_label() == "gate_-_G1_-_0" for l in ax.lines)
    rig.hide_cb.setChecked(True)
    rig.gm._gate_cache_ts = 0.0
    rig.gm.drawGate(0)
    assert not any(l.get_label().startswith("gate_-_G1") for l in ax.lines)


# --------------------------------------------------------------- off-axes drag guard

def test_followmouse_off_axes_is_noop(rig):
    line = mlines.Line2D([4.0, 4.0], [0, 100])
    rig.gm.editThisGateLine = line
    rig.gm._gate_edit_option = "1d_move_line"

    class Evt:
        xdata = None; ydata = None
    rig.gm.followmouse(Evt())
    assert list(line.get_xdata()) == [4.0, 4.0]     # untouched
    assert len(rig.drawidle) == 0                   # no redraw requested


def test_followmouse_in_axes_moves_1d_line(rig):
    line = mlines.Line2D([4.0, 4.0], [0, 100])
    rig.gm.editThisGateLine = line
    rig.gm._gate_edit_option = "1d_move_line"

    class Evt:
        xdata = 7.0; ydata = 50.0
    rig.gm.followmouse(Evt())
    assert list(line.get_xdata()) == [7.0, 7.0]
    assert len(rig.drawidle) == 1


# --------------------------------------------------------------- lifecycle signals

def test_cancel_gate_ends_disconnects_and_closes(rig):
    rig.gm._creating_gate = True
    rig.gm.cancelGate(doClose=True)
    assert rig.gm._creating_gate is False and rig.gm._editing_gate is False
    assert len(rig.ended) == 1
    assert len(rig.replot) == 1
    assert len(rig.close_req) == 1


def test_cancel_gate_no_close_keeps_popup(rig):
    rig.gm.cancelGate(doClose=False)
    assert len(rig.close_req) == 0
    assert len(rig.replot) == 1


def test_create_gate_without_index_warns(rig):
    rig.gm.createGate(None)
    assert rig.msgbox.calls == [("about", "Warning!", "Please add at least one spectrum")]
    assert len(rig.created) == 0


# --------------------------------------------------------------- REST push

def test_push_gate_to_rest_1d_sorts_boundaries(rig):
    rig.add_1d("h1", type_="1", params=("p1",))
    rig.gm._active_gate_index = 0
    rig.info[0]["dim"] = 1
    rig.popup.listRegionLine = [mlines.Line2D([8.0, 8.0], [0, 1]),
                                mlines.Line2D([2.0, 2.0], [0, 1])]
    rig.gm.pushGateToREST("G1", "s")
    assert len(rig.rest.created) == 1
    name, gtype, params, boundaries = rig.rest.created[0]
    assert name == "G1" and gtype == "s"
    assert boundaries == [2.0, 8.0]              # sorted low..high


# ---------------------------------------------------------------
# leak-proof canvas-callback + shortcut registry
# ---------------------------------------------------------------

def test_mpl_connect_replaces_prior_role_cid(rig):
    # MUTATION-worthy: reconnecting a role must disconnect the previous cid
    # (the followmouse per-move 'release' rebind used to leak it).
    rig.gm._mpl_connect('release', 'button_press_event', lambda e: None)   # cid 1
    rig.gm._mpl_connect('release', 'button_press_event', lambda e: None)   # cid 2
    assert 1 in rig.canvas.disconnected          # prior connection dropped
    assert rig.gm._mpl_cids['release'][1] == 2   # registry now holds the new cid


def test_mpl_disconnect_all_clears_registry(rig):
    rig.gm._mpl_connect('pick', 'pick_event', lambda e: None)              # cid 1
    rig.gm._mpl_connect('follow', 'motion_notify_event', lambda e: None)   # cid 2
    rig.gm._mpl_disconnect_all()
    assert set(rig.canvas.disconnected) >= {1, 2}
    assert rig.gm._mpl_cids == {}


def test_releaseonclick_disconnects_follow_and_release(rig):
    rig.gm._mpl_connect('follow', 'motion_notify_event', lambda e: None)   # cid 1
    rig.gm._mpl_connect('release', 'button_press_event', lambda e: None)   # cid 2

    class Evt:
        pass
    rig.gm.releaseonclick(Evt())
    assert 'follow' not in rig.gm._mpl_cids and 'release' not in rig.gm._mpl_cids
    assert set(rig.canvas.disconnected) >= {1, 2}


def test_edit_gate_replaces_prior_shortcut(rig, gm_mod, monkeypatch):
    monkeypatch.setattr(gm_mod, "QShortcut", FakeShortcut)
    monkeypatch.setattr(gm_mod, "QKeySequence", lambda *a: None)
    rig.add_1d("h1", index=0, type_="1")
    ax = _ax()
    ax.add_line(mlines.Line2D([4.0, 4.0], [0, 100], label="gate_-_G1_-_0"))
    rig.info[0]["axis"] = ax
    rig.gm._active_gate_index = 0

    rig.gm.editGate()
    first = rig.gm._edit_shortcut
    assert first is not None

    rig.gm.editGate()                       # second session must dispose the first
    assert rig.gm._edit_shortcut is not first
    assert first.enabled is False           # disabled...
    assert first.parent_set is None         # ...and released for GC (not lingered)


def test_disconnect_gate_signals_disposes_shortcut_and_cids(rig, gm_mod, monkeypatch):
    monkeypatch.setattr(gm_mod, "QShortcut", FakeShortcut)
    monkeypatch.setattr(gm_mod, "QKeySequence", lambda *a: None)
    rig.gm._install_edit_shortcut()
    rig.gm._mpl_connect('pick', 'pick_event', lambda e: None)
    sc = rig.gm._edit_shortcut
    rig.gm.disconnectGateSignals()
    assert rig.gm._mpl_cids == {}           # all canvas callbacks gone
    assert rig.gm._edit_shortcut is None     # shortcut disposed
    assert sc.parent_set is None


# ---------------------------------------------------------------
# step-1 inversion — signal contracts (widget effects now travel as signals)
# ---------------------------------------------------------------

def test_create_gate_valid_emits_lifecycle(rig):
    # createGate on a 1D spectrum drives the whole popup via signals:
    # action-radio state, gate-type population, name list, and show.
    rig.add_1d("h1", index=0, type_="1", params=("p1",))
    rig.gm.createGate(0)
    assert rig.create_checked == [(True,)]          # gateActionCreate checked
    assert rig.edit_enabled == [(True,)]            # edit not disabled
    assert rig.editable_req == [()]                 # name combo made editable
    assert rig.gtype_added == [("s",)]              # gateTypesDict["1"] == ["s"]
    assert rig.created == [(0,)]                     # gateCreationStarted(index)
    assert rig.names_prepared[-1] == ([], "gate-001")
    assert rig.show_req == [()]
    assert rig.gm._creating_gate is True


def test_create_gate_edit_disabled_emits_edit_off(rig):
    rig.add_1d("h1", index=0, type_="1")
    rig.edit_disable_cb.setChecked(True)
    rig.gm.createGate(0)
    assert rig.edit_checked == [(False,)]            # gateActionEdit unchecked
    assert rig.edit_enabled == [(False,)]            # and disabled


def test_edit_gate_emits_names_completer_and_readout_editable(rig, gm_mod, monkeypatch):
    monkeypatch.setattr(gm_mod, "QShortcut", FakeShortcut)
    monkeypatch.setattr(gm_mod, "QKeySequence", lambda *a: None)
    rig.add_1d("h1", index=0, type_="1")
    ax = _ax()
    ax.add_line(mlines.Line2D([4.0, 4.0], [0, 100], label="gate_-_G1_-_0"))
    rig.info[0]["axis"] = ax
    rig.gm._active_gate_index = 0
    rig.gm.editGate()
    assert rig.create_checked == [(False,)]          # leaving create mode
    assert rig.names_prepared[-1] == (["G1"], "-- select a gate --")
    assert rig.completer_cfg == [()]                 # completer configured
    assert rig.readout_editable == [(True,)]         # region text now editable
    assert rig.gm._editing_gate is True


def test_ok_gate_pushes_arg_name_then_clears_and_closes(rig):
    # okGate uses its gate_name ARGUMENT (not a widget read) and tears down.
    rig.add_1d("h1", index=0, type_="1", params=("p1",))
    rig.gm._active_gate_index = 0
    rig.info[0]["dim"] = 1
    rig.popup.listRegionLine = [mlines.Line2D([2.0, 2.0], [0, 1]),
                                mlines.Line2D([8.0, 8.0], [0, 1])]
    rig.popup.listGateType = FakeCombo(["s"]); rig.popup.listGateType.setCurrentText("s")
    rig.gm.okGate("G1")
    assert rig.rest.created[0][0] == "G1"            # pushed under the ARG name
    assert len(rig.clearinfo_req) == 1
    assert len(rig.close_req) == 1                   # cancelGate(doClose=True)
    assert len(rig.ended) == 1


def test_single_click_1d_emits_readout(rig):
    # MUTATION CHECK: break _set_gate_readout's emit and this fails.
    rig.add_1d("h1", index=0, type_="1")
    ax = _ax()
    rig.enlarged = True
    rig.info[0]["dim"] = 1
    rig.info[0]["spectrum"] = type("S", (), {"axes": ax})()

    class Evt:
        xdata = 4.0; ydata = 0.0
    rig.gm.on_singleclick_gate(Evt(), 0)
    assert rig.readout == [("0: X= 4.000",)]


def test_gate_type_list_changed_gated_by_create_mode(rig):
    # MUTATION CHECK: remove the _creating_gate guard and the no-op case emits.
    rig.gm._creating_gate = False
    rig.gm.gateTypeListChanged()
    assert rig.names_prepared == []                  # guarded: nothing happens

    rig.gm._creating_gate = True
    rig.gm.gateTypeListChanged()
    assert rig.names_prepared[-1] == ([], "gate-001")   # create mode: resets names


def test_gate_name_list_changed_gated_by_edit_mode(rig):
    # Guard blocks the slot outside edit mode even with a matching gate present.
    rig.add_1d("h1", index=0)
    ax = _ax()
    ax.add_line(mlines.Line2D([4.0, 4.0], [0, 100], label="gate_-_G1_-_"))
    rig.info[0]["axis"] = ax
    rig.gm._active_gate_index = 0
    rig.gm._gate_names = ["G1"]
    rig.popup.gateNameList = FakeCombo(["G1"]); rig.popup.gateNameList.setCurrentText("G1")
    rig.rest = FakeGateRest([{"name": "G1", "type": "s"}])
    rig.gm._editing_gate = False
    rig.gm.gateNameListChanged()
    assert rig.name_selected == [] and rig.gtype_added == []   # guard blocked all
