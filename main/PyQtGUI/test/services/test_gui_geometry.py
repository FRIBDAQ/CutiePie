"""Characterization tests for MainWindow's geometry save/load orchestration.
`saveGeo`, `saveGeoAll`, `loadGeo`, `loadGeoAll`, `_applySession`,
`_applyGeometryToCurrentTab`, `_resolveSpectrumName` — 284 lines that decide
what happens to the user's whole workspace when they open a file."""

import logging
import os
import sys
import types

import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib.figure import Figure

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import gui_stubs
from services import geometry_io
from services.spectrum_store import SpectrumStore
from services.display_slot import DisplaySlot


class FakeCombo:
    def __init__(self, value="1"):
        self.value = value
        self.index = 0
        self.enabled = True

    def currentText(self):
        return self.value

    def setCurrentIndex(self, i):
        self.index = i
        self.value = str(i + 1)

    def setEnabled(self, on):
        self.enabled = on


class FakeWConf:
    def __init__(self):
        self.histo_geo_row = FakeCombo("2")
        self.histo_geo_col = FakeCombo("3")
        self.histo_geo_add = FakeCombo()
        self.createGate = FakeCombo()


class FakePlotWidget:
    def __init__(self):
        self.figure = Figure()
        self.h_dict_geo = {}
        self.isEnlarged = False
        self.isLoaded = False
        self.selected_plot_index = 4
        self.next_plot_index = 9
        self.toCreateGate = False
        self.toEditGate = False
        self.toCreateSumRegion = False


class FakeSessions:
    def __init__(self, n=1):
        self.n = n

    def indices(self):
        return list(range(self.n))

    def __len__(self):
        return self.n

    def __contains__(self, i):
        return 0 <= i < self.n


class FakeTabs:
    def __init__(self, ntabs=1):
        self.sessions = FakeSessions(ntabs)
        self.plots = {i: FakePlotWidget() for i in range(ntabs)}
        self.slots = {i: {} for i in range(ntabs)}
        self.layouts = {i: [2, 3] for i in range(ntabs)}
        self.texts = {i: f"Tab {i+1}" for i in range(ntabs)}
        self.zoom = {i: None for i in range(ntabs)}
        self.selected = {i: 3 for i in range(ntabs)}
        self.current = 0
        self.events = []

    def currentIndex(self):
        return self.current

    def setCurrentIndex(self, i):
        self.current = i
        self.events.append(("setCurrentIndex", i))

    def plot(self, i):
        return self.plots[i]

    def tabSlots(self, i):
        return self.slots[i]

    def tabLayout(self, i):
        return self.layouts[i]

    def tabText(self, i):
        return self.texts[i]

    def setTabText(self, i, text):
        self.texts[i] = text

    def setSelectedPad(self, i, value):
        self.selected[i] = value

    def zoomInfo(self, i):
        return self.zoom[i]

    def setZoomInfo(self, i, value):
        self.zoom[i] = value

    def addTab(self, k):
        self.sessions.n += 1
        n = self.sessions.n - 1
        self.plots[n] = FakePlotWidget()
        self.slots[n] = {}
        self.layouts[n] = [2, 3]
        self.texts[n] = f"Tab {n+1}"
        self.zoom[n] = None
        self.selected[n] = None
        self.events.append(("addTab", k))

    def deleteTab(self, k):
        self.sessions.n -= 1
        for d in (self.plots, self.slots, self.layouts, self.texts,
                  self.zoom, self.selected):
            d.pop(k, None)
        self.events.append(("deleteTab", k))


class Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args))
            if name == "getAxisProperties":
                return [0.0, 100.0], [1.0, 50.0]
            return None
        return record


def recording_message_box(answer="No"):
    class Box:
        calls = []
        # ints, not strings: the caller ORs the two button flags together the
        # way real Qt expects
        Yes, No = 0x4000, 0x10000
        reply = None

        @classmethod
        def about(cls, parent, title, text):
            cls.calls.append(("about", title, text))

        @classmethod
        def warning(cls, parent, title, text):
            cls.calls.append(("warning", title, text))

        @classmethod
        def question(cls, parent, title, text, buttons=None, default=None):
            cls.calls.append(("question", title, text))
            return cls.reply

    Box.reply = Box.Yes if answer == "Yes" else Box.No
    return Box


@pytest.fixture
def win(monkeypatch):
    gui = gui_stubs.import_gui()
    w = gui.MainWindow.__new__(gui.MainWindow)
    w.logger = logging.getLogger("test.geometry")
    w.spectra = SpectrumStore()
    w.wTab = FakeTabs()
    w.currentPlot = w.wTab.plot(0)

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
    w.wConf = FakeWConf()
    w.plot_controller = Recorder()
    w.gate_manager = Recorder()
    w.sum_region_manager = Recorder()
    w.connection_manager = Recorder()
    w.gatePopup = types.SimpleNamespace(isVisible=lambda: False)
    w.sumRegionPopup = types.SimpleNamespace(isVisible=lambda: False)
    w.calls = []
    for name in ("setCanvasLayout", "addPlot", "autoUpdateStart", "bindDynamicSignal"):
        setattr(w, name, (lambda n: lambda *a: w.calls.append(n))(name))
    w.box = recording_message_box()
    w.box.calls = []
    monkeypatch.setattr(gui, "QMessageBox", w.box, raising=False)

    # The only thing that changed when the cluster moved out: this fixture now
    # builds the controller that __init__ builds in production, since __init__
    # is deliberately not run. Every test body below is unchanged from before
    # the extraction.
    from controllers import geometry_controller as gc
    monkeypatch.setattr(gc, "QMessageBox", w.box)
    w.tabGeoWidgetAndFlags = lambda k: w.calls.append(("tabGeo", k))
    w.geometry_controller = gc.GeometryController(
        tabs=w.wTab,
        conf=w.wConf,
        spectra=w.spectra,
        get_current_plot=lambda: w.currentPlot,
        set_current_plot=lambda plot: setattr(w, "currentPlot", plot),
        view_state=w.view_state,
        plot_controller=w.plot_controller,
        gate_manager=w.gate_manager,
        sum_region_manager=w.sum_region_manager,
        connection_manager=w.connection_manager,
        gate_popup=w.gatePopup,
        sum_region_popup=w.sumRegionPopup,
        set_canvas_layout=lambda: w.calls.append("setCanvasLayout"),
        add_plot=lambda: w.calls.append("addPlot"),
        auto_update_start=lambda: w.calls.append("autoUpdateStart"),
        bind_dynamic_signal=lambda: w.calls.append("bindDynamicSignal"),
        tab_geo_widget_and_flags=w.tabGeoWidgetAndFlags,
        open_file_dialog=lambda: w.openFileNameDialog(),
        save_file_dialog=lambda: w.saveFileDialog(),
        parent_widget=w,
        logger=w.logger,
    )
    return w


def add_spectrum(win, name, dim=1):
    win.spectra.set(name, dim=dim, binx=10, minx=0.0, maxx=100.0, biny=10,
                    miny=0.0, maxy=100.0, data=[], parameters=["p1"], type="1")


def geo_payload(names, row=2, col=3, x=None, y=None, scale=False):
    return {"row": row, "col": col,
            "geo": {i: {"name": n, "x": x, "y": y, "scale": scale}
                    for i, n in enumerate(names)}}


# --------------------------------------------------------- _resolveSpectrumName

def test_resolve_returns_an_exact_match(win):
    add_spectrum(win, "hSpec")
    assert win._resolveSpectrumName("hSpec") == "hSpec"


def test_resolve_tolerates_case_for_a_unique_match(win):
    # legacy .win files store names upper-cased
    add_spectrum(win, "hSpec")
    assert win._resolveSpectrumName("HSPEC") == "hSpec"


def test_resolve_refuses_an_ambiguous_case_insensitive_match(win):
    add_spectrum(win, "hspec")
    add_spectrum(win, "hSpec")
    assert win._resolveSpectrumName("HSPEC") is None


def test_resolve_returns_none_for_an_unknown_name(win):
    assert win._resolveSpectrumName("nope") is None


# ------------------------------------------------ _applyGeometryToCurrentTab

def test_apply_geometry_sets_the_grid_one_less_than_the_stored_count(win):
    # the combos are 0-indexed while the file stores counts
    add_spectrum(win, "h1")
    win._applyGeometryToCurrentTab(geo_payload(["h1"], row=3, col=4))
    assert win.wConf.histo_geo_row.index == 2
    assert win.wConf.histo_geo_col.index == 3
    assert "setCanvasLayout" in win.calls


def test_apply_geometry_places_the_named_spectra(win):
    add_spectrum(win, "h1")
    add_spectrum(win, "h2")
    win._applyGeometryToCurrentTab(geo_payload(["h1", "h2"]))
    assert win.getGeo()[0] == "h1"
    assert win.getGeo()[1] == "h2"


def test_apply_geometry_skips_empty_pads_without_reporting_them(win):
    add_spectrum(win, "h1")
    assert win._applyGeometryToCurrentTab(geo_payload(["", "h1"])) == []
    assert 0 not in win.getGeo()


def test_apply_geometry_reports_names_the_store_does_not_have(win):
    add_spectrum(win, "h1")
    notFound = win._applyGeometryToCurrentTab(geo_payload(["h1", "ghost"]))
    assert notFound == ["ghost"]
    assert 1 not in win.getGeo()          # the pad is left empty, not guessed at


def test_apply_geometry_restores_the_saved_view_range(win):
    add_spectrum(win, "h1")
    win._applyGeometryToCurrentTab(
        geo_payload(["h1"], x=[10.0, 90.0], y=[2.0, 40.0], scale=True))
    assert win.getSpectrumViewInfo("minx", index=0) == 10.0
    assert win.getSpectrumViewInfo("maxx", index=0) == 90.0
    assert win.getSpectrumViewInfo("miny", index=0) == 2.0
    assert win.getSpectrumViewInfo("log", index=0) is True


def test_apply_geometry_leaves_the_natural_range_when_the_file_omits_it(win):
    # old .win files have no "Expanded" block; the spectrum keeps the store range
    add_spectrum(win, "h1")
    win._applyGeometryToCurrentTab(geo_payload(["h1"], x=None, y=None))
    assert win.getSpectrumViewInfo("minx", index=0) == 0.0     # from the store record
    assert win.getSpectrumViewInfo("maxx", index=0) == 100.0


def test_apply_geometry_clears_the_selection_and_redraws(win):
    add_spectrum(win, "h1")
    win.currentPlot.selected_plot_index = 4
    win._applyGeometryToCurrentTab(geo_payload(["h1"]))
    assert win.currentPlot.selected_plot_index is None
    assert win.currentPlot.next_plot_index == -1
    assert win.wTab.selected[0] is None
    assert win.currentPlot.isLoaded is False          # toggled on, then off
    assert "addPlot" in win.calls
    assert ("updatePlot", ()) in win.plot_controller.calls


def test_apply_geometry_with_a_degenerate_grid_places_nothing(win):
    add_spectrum(win, "h1")
    win._applyGeometryToCurrentTab(geo_payload(["h1"], row=0, col=0))
    assert win.getGeo() == {}
    assert "setCanvasLayout" not in win.calls
    assert "addPlot" in win.calls                     # still redraws


# ---------------------------------------------------------------- saveGeo

def test_save_geo_writes_a_readable_file(win, tmp_path):
    add_spectrum(win, "h1")
    win.setGeo(0, "h1")
    target = tmp_path / "one.win"
    win.saveFileDialog = lambda: str(target)
    win.saveGeo()
    kind, payload = geometry_io.read_geometry_any(str(target), win.logger)
    assert kind == "single"
    assert payload["geo"][0]["name"] == "h1"
    assert ("about", "Saving...", "Window configuration saved!") in win.box.calls


def test_save_geo_writes_nothing_when_the_dialog_is_cancelled(win, tmp_path):
    win.saveFileDialog = lambda: None
    win.saveGeo()
    assert win.box.calls == []
    assert list(tmp_path.iterdir()) == []


def test_save_geo_reports_a_failed_write_and_claims_no_success(win):
    # the old code showed "saved!" before the write, so a failure looked fine
    add_spectrum(win, "h1")
    win.setGeo(0, "h1")
    win.saveFileDialog = lambda: "/nonexistent-directory/one.win"
    win.saveGeo()
    kinds = [c[0] for c in win.box.calls]
    assert "warning" in kinds
    assert "about" not in kinds


def test_save_geo_keeps_going_when_one_pad_cannot_be_read(win, tmp_path):
    add_spectrum(win, "h1")
    add_spectrum(win, "h2")
    win.setGeo(0, "h1")
    win.setGeo(1, "h2")

    def flaky(index):
        if index == 0:
            raise RuntimeError("no axes on this pad")
        return [0.0, 100.0], [1.0, 50.0]
    win.plot_controller.getAxisProperties = flaky

    target = tmp_path / "one.win"
    win.saveFileDialog = lambda: str(target)
    win.saveGeo()
    _, payload = geometry_io.read_geometry_any(str(target), win.logger)
    assert payload["geo"][0]["name"] == ""          # placeholder, not a crash
    assert payload["geo"][1]["name"] == "h2"


# ------------------------------------------------------------- saveGeoAll

def make_slot(**fields):
    slot = DisplaySlot()
    for k, v in fields.items():
        setattr(slot, k, v)
    return slot


def test_save_all_reads_the_view_tier_not_the_live_axes(win, tmp_path):
    # deliberate asymmetry with saveGeo: a background tab's axes are not
    # reliably current, and the view tier is exactly what load consumes
    win.wTab.plots[0].h_dict_geo = {0: "h1"}
    win.wTab.slots[0][0] = make_slot(minx=11.0, maxx=99.0, miny=2.0, maxy=44.0,
                                     log=True)
    win.plot_controller.getAxisProperties = lambda i: (_ for _ in ()).throw(
        AssertionError("saveGeoAll must not touch the axes"))
    target = tmp_path / "all.win"
    win.saveFileDialog = lambda: str(target)
    win.saveGeoAll()
    kind, payload = geometry_io.read_geometry_any(str(target), win.logger)
    assert kind == "session"
    geo = payload["tabs"][0]["geo"]
    assert geo[0]["x"] == [11.0, 99.0] and geo[0]["y"] == [2.0, 44.0]
    assert geo[0]["scale"] is True


def test_save_all_treats_zero_as_a_real_limit(win, tmp_path):
    # 0.0 is falsy; testing truthiness here would drop a legitimate range
    win.wTab.plots[0].h_dict_geo = {0: "h1"}
    win.wTab.slots[0][0] = make_slot(minx=0.0, maxx=0.0, miny=0.0, maxy=0.0)
    target = tmp_path / "all.win"
    win.saveFileDialog = lambda: str(target)
    win.saveGeoAll()
    _, payload = geometry_io.read_geometry_any(str(target), win.logger)
    assert payload["tabs"][0]["geo"][0]["x"] == [0.0, 0.0]


def test_save_all_writes_a_blank_name_for_an_empty_pad(win, tmp_path):
    win.wTab.plots[0].h_dict_geo = {0: "empty"}
    target = tmp_path / "all.win"
    win.saveFileDialog = lambda: str(target)
    win.saveGeoAll()
    _, payload = geometry_io.read_geometry_any(str(target), win.logger)
    assert payload["tabs"][0]["geo"][0]["name"] == ""


def test_save_all_covers_every_tab(win, tmp_path):
    win.wTab.addTab(1)
    win.wTab.plots[0].h_dict_geo = {0: "h1"}
    win.wTab.plots[1].h_dict_geo = {0: "h2"}
    win.wTab.setTabText(1, "second")
    target = tmp_path / "all.win"
    win.saveFileDialog = lambda: str(target)
    win.saveGeoAll()
    _, payload = geometry_io.read_geometry_any(str(target), win.logger)
    assert len(payload["tabs"]) == 2
    assert payload["tabs"][1]["name"] == "second"


# ---------------------------------------------------------------- loadGeo

def test_load_geo_changes_nothing_on_an_unreadable_file(win, tmp_path):
    add_spectrum(win, "h1")
    junk = tmp_path / "junk.win"
    junk.write_bytes(b"\x00\x01 not a geometry file \xff")
    win.openFileNameDialog = lambda: str(junk)
    win.loadGeo()
    assert [c[0] for c in win.box.calls] == ["warning"]
    assert win.getGeo() == {}
    assert "setCanvasLayout" not in win.calls


def test_load_geo_does_nothing_when_the_dialog_is_cancelled(win):
    win.openFileNameDialog = lambda: None
    win.loadGeo()
    assert win.box.calls == []
    assert win.calls == []


def test_load_geo_asks_before_replacing_tabs_with_a_session_file(win, tmp_path):
    add_spectrum(win, "h1")
    session = tmp_path / "sess.win"
    session.write_text(geometry_io.serialize_session(
        [{"name": "T1", "row": 2, "col": 3,
          "geo": {0: {"name": "h1", "x": None, "y": None, "scale": False}}}]))
    win.openFileNameDialog = lambda: str(session)
    win.box.reply = win.box.No
    win.loadGeo()
    assert [c[0] for c in win.box.calls] == ["question"]
    assert win.getGeo() == {}                    # refusing changes nothing


def test_load_geo_applies_the_session_when_the_user_agrees(win, tmp_path):
    add_spectrum(win, "h1")
    session = tmp_path / "sess.win"
    session.write_text(geometry_io.serialize_session(
        [{"name": "T1", "row": 2, "col": 3,
          "geo": {0: {"name": "h1", "x": None, "y": None, "scale": False}}}]))
    win.openFileNameDialog = lambda: str(session)
    win.box.reply = win.box.Yes
    win.loadGeo()
    assert win.getGeo()[0] == "h1"


def test_load_geo_applies_a_single_tab_file_directly(win, tmp_path):
    add_spectrum(win, "h1")
    single = tmp_path / "one.win"
    single.write_text(geometry_io.serialize_geometry(
        "2", "3", {0: {"name": "h1", "x": [0.0, 100.0], "y": [0.0, 50.0],
                       "scale": False}}))
    win.openFileNameDialog = lambda: str(single)
    win.loadGeo()
    assert win.getGeo()[0] == "h1"
    assert [c[0] for c in win.box.calls] == []   # no question for a single tab


# -------------------------------------------------------------- loadGeoAll

def test_load_all_changes_nothing_on_an_unreadable_file(win, tmp_path):
    junk = tmp_path / "junk.win"
    junk.write_bytes(b"\x00\xff not geometry")
    win.openFileNameDialog = lambda: str(junk)
    win.loadGeoAll()
    assert [c[0] for c in win.box.calls] == ["warning"]
    assert win.wTab.events == []                 # no tab was touched


def test_load_all_wraps_a_single_tab_file_as_a_one_tab_session(win, tmp_path):
    add_spectrum(win, "h1")
    single = tmp_path / "one.win"
    single.write_text(geometry_io.serialize_geometry(
        "2", "3", {0: {"name": "h1", "x": None, "y": None, "scale": False}}))
    win.openFileNameDialog = lambda: str(single)
    win.loadGeoAll()
    assert win.getGeo()[0] == "h1"
    assert win.wTab.texts[0] == "Tab 1"


# ------------------------------------------------------------ _applySession

def test_apply_session_quiesces_before_touching_the_workspace(win):
    add_spectrum(win, "h1")
    win.currentPlot.toCreateGate = True
    win.currentPlot.toCreateSumRegion = True
    win._applySession([{"name": "T1", "row": 2, "col": 3,
                        "geo": {0: {"name": "h1", "x": None, "y": None,
                                    "scale": False}}}])
    assert ("cancelGate", ()) in win.gate_manager.calls
    assert ("cancelSumRegion", ()) in win.sum_region_manager.calls
    assert ("_stop_auto_thread", ()) in win.connection_manager.calls
    assert "autoUpdateStart" in win.calls        # and restarted at the end


def test_apply_session_rebuilds_the_tab_set_to_match_the_file(win):
    add_spectrum(win, "h1")
    win.wTab.addTab(1)
    win.wTab.addTab(2)                            # 3 tabs on screen
    tabs = [{"name": f"T{i}", "row": 2, "col": 3,
             "geo": {0: {"name": "h1", "x": None, "y": None, "scale": False}}}
            for i in range(2)]                    # file has 2
    win._applySession(tabs)
    assert len(win.wTab.sessions) == 2
    assert win.wTab.texts[0] == "T0" and win.wTab.texts[1] == "T1"


def test_apply_session_returns_to_the_first_tab(win):
    add_spectrum(win, "h1")
    tabs = [{"name": f"T{i}", "row": 2, "col": 3,
             "geo": {0: {"name": "h1", "x": None, "y": None, "scale": False}}}
            for i in range(3)]
    win._applySession(tabs)
    assert win.wTab.current == 0
    assert "bindDynamicSignal" in win.calls


def test_apply_session_names_the_tabs_that_lost_spectra(win):
    add_spectrum(win, "h1")
    tabs = [{"name": "good", "row": 2, "col": 3,
             "geo": {0: {"name": "h1", "x": None, "y": None, "scale": False}}},
            {"name": "bad", "row": 2, "col": 3,
             "geo": {0: {"name": "ghost", "x": None, "y": None, "scale": False}}}]
    win._applySession(tabs)
    warnings = [c for c in win.box.calls if c[0] == "warning"]
    assert len(warnings) == 1
    assert "bad: ghost" in warnings[0][2]
    assert "good" not in warnings[0][2]


def test_apply_session_falls_back_to_a_positional_tab_name(win):
    add_spectrum(win, "h1")
    win._applySession([{"name": "", "row": 2, "col": 3,
                        "geo": {0: {"name": "h1", "x": None, "y": None,
                                    "scale": False}}}])
    assert win.wTab.texts[0] == "Tab 1"
