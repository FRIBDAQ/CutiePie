"""Characterization tests for Peak Finder 2's arming and canvas connections: the
Start and Fix Peak toggles, the connect/disconnect pair, the sync that decides
which canvases keep the press handler, and the Config dialog."""

import logging
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import gui_stubs


# --------------------------------------------------------------- fake widgets

class FakeCanvas:
    """Records mpl_connect/mpl_disconnect the way a real canvas would."""

    _next_cid = [100]

    def __init__(self, name="canvas"):
        self.name = name
        self.connected = {}          # cid -> event name
        self.disconnected = []
        self.fail_disconnect = False

    def mpl_connect(self, event, handler):
        cid = FakeCanvas._next_cid[0]
        FakeCanvas._next_cid[0] += 1
        self.connected[cid] = (event, handler)
        return cid

    def mpl_disconnect(self, cid):
        if self.fail_disconnect:
            raise RuntimeError("canvas already torn down")
        self.disconnected.append(cid)
        self.connected.pop(cid, None)

    def __repr__(self):
        return f"<{self.name}>"


class FakeButton:
    def __init__(self):
        self.text_ = ""
        self.style = ""
        self.checked = False

    def setText(self, t):
        self.text_ = t

    def text(self):
        return self.text_

    def setStyleSheet(self, s):
        self.style = s

    def setChecked(self, on):
        self.checked = bool(on)

    def isChecked(self):
        return self.checked


class FakeLabel:
    def __init__(self):
        self._text = ""

    def text(self):
        return self._text

    def setText(self, t):
        self._text = t


class FakePeakTab:
    def __init__(self):
        self.peak2_start = FakeButton()
        self.peak2_fix = FakeButton()
        self.peak2_status = FakeLabel()


class FakeExtraPopup:
    def __init__(self):
        self.peak = FakePeakTab()


class FakePlot:
    def __init__(self, canvas):
        self.canvas = canvas


class FakeTabs:
    """wTab stand-in: a canvas per tab index."""

    def __init__(self, canvases):
        self.canvases = canvases
        self.current = 0

    def currentIndex(self):
        return self.current

    def plot(self, index):
        return FakePlot(self.canvases[index])


class FakePopup:
    def __init__(self, visible=False):
        self.visible = visible

    def isVisible(self):
        return self.visible


def artist_on(canvas):
    """A stand-in fit artist whose axes lead back to `canvas`."""
    figure = types.SimpleNamespace(canvas=canvas)
    return types.SimpleNamespace(axes=types.SimpleNamespace(figure=figure))


# ------------------------------------------------------------------- fixtures

@pytest.fixture
def win(monkeypatch):
    gui = gui_stubs.import_gui()
    w = gui.MainWindow.__new__(gui.MainWindow)
    w.logger = logging.getLogger("test.peakfit2.arming")
    w.extraPopup = FakeExtraPopup()

    w.canvas_a = FakeCanvas("tab-a")
    w.canvas_b = FakeCanvas("tab-b")
    w.wTab = FakeTabs([w.canvas_a, w.canvas_b])

    w.currentPlot = types.SimpleNamespace(
        zoomPress=False, toCreateGate=False, toEditGate=False,
        toCreateSumRegion=False)
    w.gatePopup = FakePopup()
    w.sumRegionPopup = FakePopup()

    w.peak2_conns = {}
    w.peak2_armed = False
    w.peak2_fix_armed = False
    w.peak2_fits = []

    w.presses = []
    w.onPeakFit2Press = lambda event: w.presses.append(event)

    settings = {}
    w.settings = settings

    class FakeSettings:
        def value(self, key, default=None, type=None):
            return settings.get(key, default)

        def setValue(self, key, value):
            settings[key] = value

    w.dialog_reply = ("", False)

    class FakeInputDialog:
        @staticmethod
        def getText(parent, title, label, text=""):
            w.dialog_prefill = text
            return w.dialog_reply

    monkeypatch.setattr(gui, "QSettings", FakeSettings, raising=False)
    monkeypatch.setattr(gui, "QInputDialog", FakeInputDialog, raising=False)

    # These methods call 8a's floor — the status line and the Config cap reader
    # — which already lives on PeakFit2Controller, so the window needs it built
    # whether or not this group has moved yet.
    from controllers import peak_fit2_controller as pfc
    monkeypatch.setattr(pfc, "QSettings", FakeSettings)
    monkeypatch.setattr(pfc, "QInputDialog", FakeInputDialog)
    w.view_state = types.SimpleNamespace(
        getSpectrumStoreInfo=None,
        getSpectrumViewInfo=None,
        nameFromIndex=lambda index: "alpha",
    )
    w.peak_fit2_controller = pfc.PeakFit2Controller(
        peak_tab=w.extraPopup.peak,
        spectra=None,
        view_state=w.view_state,
        plot_controller=None,
        tabs=w.wTab,
        get_current_plot=lambda: w.currentPlot,
        get_gate_popup=lambda: w.gatePopup,
        get_sum_popup=lambda: w.sumRegionPopup,
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
    for _name in dir(w.peak_fit2_controller):
        if (_name.startswith(("_peak2_", "peakFit2", "onPeakFit2"))
                and not hasattr(type(w), _name)
                and _name not in w.__dict__):     # keep this fixture's own doubles
            setattr(w, _name, getattr(w.peak_fit2_controller, _name))
    return w


# --------------------------------------------------------- connect/disconnect

def test_connect_binds_the_press_handler_once(win):
    win._peak2_connect(win.canvas_a)
    assert len(win.canvas_a.connected) == 1
    assert win.canvas_a in win.peak2_conns


def test_connect_binds_the_button_press_event(win):
    win._peak2_connect(win.canvas_a)
    (event, _), = win.canvas_a.connected.values()
    assert event == "button_press_event"


def test_connecting_twice_does_not_stack_handlers(win):
    """A second bind on the same canvas would fit twice per click."""
    win._peak2_connect(win.canvas_a)
    win._peak2_connect(win.canvas_a)
    assert len(win.canvas_a.connected) == 1


def test_connect_ignores_a_missing_canvas(win):
    win._peak2_connect(None)
    assert win.peak2_conns == {}


def test_disconnect_removes_the_binding_and_the_record(win):
    win._peak2_connect(win.canvas_a)
    cid = win.peak2_conns[win.canvas_a]
    win._peak2_disconnect(win.canvas_a)
    assert win.canvas_a.disconnected == [cid]
    assert win.canvas_a not in win.peak2_conns


def test_disconnecting_an_unconnected_canvas_is_a_no_op(win):
    win._peak2_disconnect(win.canvas_b)
    assert win.canvas_b.disconnected == []


def test_a_canvas_that_refuses_to_disconnect_is_still_forgotten(win):
    """The canvas may already be torn down. Keeping the entry would make the
    next sync try again forever."""
    win._peak2_connect(win.canvas_a)
    win.canvas_a.fail_disconnect = True
    win._peak2_disconnect(win.canvas_a)
    assert win.canvas_a not in win.peak2_conns


# ------------------------------------------------------------- fit canvases

def test_fit_canvases_finds_the_canvas_behind_a_fit(win):
    win.peak2_fits = [{"artists": (artist_on(win.canvas_a),)}]
    assert win._peak2_fit_canvases() == {win.canvas_a}


def test_fit_canvases_covers_every_tab_that_holds_a_fit(win):
    win.peak2_fits = [{"artists": (artist_on(win.canvas_a),)},
                      {"artists": (artist_on(win.canvas_b),)}]
    assert win._peak2_fit_canvases() == {win.canvas_a, win.canvas_b}


def test_fit_canvases_skips_a_record_with_no_artists(win):
    win.peak2_fits = [{"artists": ()}, {}]
    assert win._peak2_fit_canvases() == set()


def test_fit_canvases_survives_an_artist_whose_pad_is_gone(win):
    dead = types.SimpleNamespace(axes=None)
    win.peak2_fits = [{"artists": (dead,)},
                      {"artists": (artist_on(win.canvas_a),)}]
    assert win._peak2_fit_canvases() == {win.canvas_a}


# ------------------------------------------------------- sync (K6(j) + K6(g))

def test_sync_keeps_the_handler_where_fits_live(win):
    """K6(j): after Stop, drag-to-refit and the edit popup must keep working
    on whatever tab still shows a fit."""
    win._peak2_connect(win.canvas_a)
    win.peak2_fits = [{"artists": (artist_on(win.canvas_a),)}]
    win._peak2_sync_connections()
    assert win.canvas_a in win.peak2_conns


def test_sync_drops_the_handler_from_a_canvas_with_nothing_on_it(win):
    win._peak2_connect(win.canvas_a)
    win._peak2_sync_connections()
    assert win.peak2_conns == {}


def test_sync_keeps_the_current_tab_while_armed(win):
    win._peak2_connect(win.canvas_a)
    win.peak2_armed = True
    win._peak2_sync_connections()
    assert win.canvas_a in win.peak2_conns


def test_sync_keeps_the_current_tab_while_fix_armed(win):
    win._peak2_connect(win.canvas_a)
    win.peak2_fix_armed = True
    win._peak2_sync_connections()
    assert win.canvas_a in win.peak2_conns


def test_sync_arming_protects_only_the_current_tab(win):
    """K6(g)'s other half: arming covers the tab that is current now, so a
    handler left on another tab with no fits still goes."""
    win._peak2_connect(win.canvas_a)
    win._peak2_connect(win.canvas_b)
    win.peak2_armed = True
    win.wTab.current = 0
    win._peak2_sync_connections()
    assert win.canvas_a in win.peak2_conns
    assert win.canvas_b not in win.peak2_conns


def test_sync_keeps_both_a_fit_canvas_and_the_armed_one(win):
    win._peak2_connect(win.canvas_a)
    win._peak2_connect(win.canvas_b)
    win.peak2_fits = [{"artists": (artist_on(win.canvas_b),)}]
    win.peak2_armed = True
    win.wTab.current = 0
    win._peak2_sync_connections()
    assert set(win.peak2_conns) == {win.canvas_a, win.canvas_b}


# ------------------------------------------------------------- Start toggle

def test_start_arms_and_connects_the_current_tab(win):
    win.peakFit2Toggle(True)
    assert win.peak2_armed is True
    assert win.canvas_a in win.peak2_conns


def test_start_arms_the_tab_that_is_current_at_that_moment(win):
    """K6(g). Documented limitation: switching tabs while armed does not carry
    the arming over."""
    win.wTab.current = 1
    win.peakFit2Toggle(True)
    assert win.canvas_b in win.peak2_conns
    assert win.canvas_a not in win.peak2_conns


def test_start_relabels_the_button_to_stop(win):
    win.peakFit2Toggle(True)
    assert win.extraPopup.peak.peak2_start.text() == "Stop"


def test_start_says_it_is_armed(win):
    win.peakFit2Toggle(True)
    assert "[armed]" in win.extraPopup.peak.peak2_status.text()


def test_start_turns_fix_peak_off(win):
    """Two arming modes, never both."""
    win.extraPopup.peak.peak2_fix.setChecked(True)
    win.peakFit2Toggle(True)
    assert win.extraPopup.peak.peak2_fix.isChecked() is False


def test_stop_disarms_and_restores_the_button(win):
    win.peakFit2Toggle(True)
    win.peakFit2Toggle(False)
    assert win.peak2_armed is False
    assert win.extraPopup.peak.peak2_start.text() == "Start"


def test_stop_drops_the_handler_when_no_fits_remain(win):
    win.peakFit2Toggle(True)
    win.peakFit2Toggle(False)
    assert win.peak2_conns == {}


def test_stop_keeps_the_handler_where_a_fit_stands(win):
    """K6(j) again, through the button rather than the sync directly."""
    win.peakFit2Toggle(True)
    win.peak2_fits = [{"artists": (artist_on(win.canvas_a),)}]
    win.peakFit2Toggle(False)
    assert win.canvas_a in win.peak2_conns


# ---------------------------------------------------------- Fix Peak toggle

def test_fix_arms_and_connects_the_current_tab(win):
    win.peakFit2FixToggle(True)
    assert win.peak2_fix_armed is True
    assert win.canvas_a in win.peak2_conns


def test_fix_says_which_mode_is_armed(win):
    win.peakFit2FixToggle(True)
    assert "fix" in win.extraPopup.peak.peak2_status.text().lower()


def test_fix_turns_start_off(win):
    win.extraPopup.peak.peak2_start.setChecked(True)
    win.peakFit2FixToggle(True)
    assert win.extraPopup.peak.peak2_start.isChecked() is False


def test_unfixing_disarms_and_syncs(win):
    win.peakFit2FixToggle(True)
    win.peakFit2FixToggle(False)
    assert win.peak2_fix_armed is False
    assert win.peak2_conns == {}


# --------------------------------------------------- other-mode guard (K6(i))

@pytest.mark.parametrize("flag", ["zoomPress", "toCreateGate", "toEditGate",
                                  "toCreateSumRegion"])
def test_other_mode_active_for_each_pad_interaction(win, flag):
    """K6(i): rubber-band zoom, gate create, gate edit and summing-region
    create each own the click while they are running."""
    setattr(win.currentPlot, flag, True)
    assert win._peak2_other_mode_active() is True


def test_other_mode_active_while_the_gate_popup_is_open(win):
    win.gatePopup.visible = True
    assert win._peak2_other_mode_active() is True


def test_other_mode_active_while_the_sum_region_popup_is_open(win):
    win.sumRegionPopup.visible = True
    assert win._peak2_other_mode_active() is True


def test_other_mode_inactive_when_nothing_else_is_running(win):
    assert win._peak2_other_mode_active() is False


def test_other_mode_says_no_when_a_popup_cannot_be_asked(win):
    """A torn-down popup must read as 'not blocking', not as an exception into
    the press handler."""
    win.gatePopup = None
    assert win._peak2_other_mode_active() is False


# ------------------------------------------------------------------- Config

def test_config_cancel_changes_nothing(win):
    win.settings["PeakFinder2/max_window_bins"] = "60"
    win.dialog_reply = ("999", False)
    win.peakFit2Config()
    assert win.settings["PeakFinder2/max_window_bins"] == "60"


def test_config_stores_a_positive_cap(win):
    win.dialog_reply = ("60", True)
    win.peakFit2Config()
    assert win.settings["PeakFinder2/max_window_bins"] == "60"
    assert "60" in win.extraPopup.peak.peak2_status.text()


def test_config_empty_clears_the_cap(win):
    win.settings["PeakFinder2/max_window_bins"] = "60"
    win.dialog_reply = ("", True)
    win.peakFit2Config()
    assert win.settings["PeakFinder2/max_window_bins"] == ""
    assert "no cap" in win.extraPopup.peak.peak2_status.text()


def test_config_rejects_a_non_positive_cap_and_keeps_the_old_one(win):
    win.settings["PeakFinder2/max_window_bins"] = "60"
    win.dialog_reply = ("0", True)
    win.peakFit2Config()
    assert win.settings["PeakFinder2/max_window_bins"] == "60"
    assert "unchanged" in win.extraPopup.peak.peak2_status.text()


def test_config_rejects_junk_and_keeps_the_old_one(win):
    win.settings["PeakFinder2/max_window_bins"] = "60"
    win.dialog_reply = ("wide", True)
    win.peakFit2Config()
    assert win.settings["PeakFinder2/max_window_bins"] == "60"
    assert "unchanged" in win.extraPopup.peak.peak2_status.text()


def test_config_offers_the_current_cap_as_the_prefill(win):
    win.settings["PeakFinder2/max_window_bins"] = "42"
    win.dialog_reply = ("42", True)
    win.peakFit2Config()
    assert win.dialog_prefill == "42"
