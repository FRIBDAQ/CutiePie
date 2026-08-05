"""Characterization tests for MainWindow's image-overlay and Jupyter clusters.
Two small independent clusters that share no state, characterized together
because pairs them for the same extraction."""

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


class FakeSlider:
    def __init__(self, v=5):
        self.v = v

    def value(self):
        return self.v


class FakeLabel:
    def __init__(self):
        self._text = ""

    def text(self):
        return self._text

    def setText(self, t):
        self._text = t


class FakeButton:
    def __init__(self):
        self.enabled = True
        self.style = ""

    def setEnabled(self, on):
        self.enabled = on

    def setStyleSheet(self, s):
        self.style = s


class FakeImaging:
    def __init__(self):
        self.loadLISE_name = FakeLabel()
        self.alpha_slider = FakeSlider(5)
        self.zoomX_slider = FakeSlider(3)
        self.zoomY_slider = FakeSlider(4)
        self.alpha_label = FakeLabel()
        self.zoomX_label = FakeLabel()
        self.zoomY_label = FakeLabel()
        self.joystick = types.SimpleNamespace(direction="up", distance=2)


class FakePeakTab:
    def __init__(self):
        self.jup_df_filename = FakeLabel()
        self.jup_df_filename.setText("dump.csv")
        self.jup_start = FakeButton()
        self.jup_stop = FakeButton()


class FakeExtraPopup:
    def __init__(self):
        self.imaging = FakeImaging()
        self.peak = FakePeakTab()


class FakeCanvas:
    def __init__(self):
        self.draws = 0

    def draw(self):
        self.draws += 1

    def draw_idle(self):
        self.draws += 1


class FakePlot:
    def __init__(self):
        self.figure = Figure()
        self.figure.add_subplot(111)
        self.canvas = FakeCanvas()
        self.selected_plot_index = 0


class FakeCombo:
    def __init__(self, v):
        self.v = v

    def currentText(self):
        return self.v


def recording_message_box():
    class Box:
        calls = []
        Ok = 0x400

        @classmethod
        def warning(cls, parent, title, text):
            cls.calls.append(("warning", title, text))

        @classmethod
        def information(cls, parent, title, text, buttons=None):
            cls.calls.append(("information", title, text))

        @classmethod
        def about(cls, parent, title, text):
            cls.calls.append(("about", title, text))
    return Box


# The overlay state moved onto the controller with the methods. These proxies
# let the unchanged test bodies keep reading and writing it on the window,
# exactly as they did before the extraction.
_OVERLAY_STATE = ("LISEpic", "imgplot", "overlay_ax", "onFigure",
                  "xstart", "ystart", "alpha", "zoomX", "zoomY")


def _proxy(attr):
    return property(lambda self: getattr(self.overlay_controller, attr),
                    lambda self, v: setattr(self.overlay_controller, attr, v))


@pytest.fixture
def win(monkeypatch):
    gui = gui_stubs.import_gui()
    w = gui.MainWindow.__new__(gui.MainWindow)
    w.logger = logging.getLogger("test.overlay")
    w.currentPlot = FakePlot()
    w.extraPopup = FakeExtraPopup()
    w.wConf = types.SimpleNamespace(histo_geo_row=FakeCombo("2"),
                                    histo_geo_col=FakeCombo("2"))
    w.plotPosition = lambda index: (0, 0)
    w.box = recording_message_box()
    w.box.calls = []
    monkeypatch.setattr(gui, "QMessageBox", w.box, raising=False)
    w._gui = gui

    from controllers import overlay_controller as oc
    monkeypatch.setattr(oc, "QMessageBox", w.box)
    w._oc = oc
    w.overlay_controller = oc.OverlayController(
        imaging=w.extraPopup.imaging,
        get_current_plot=lambda: w.currentPlot,
        get_selected_index=lambda: w.currentPlot.selected_plot_index,
        get_grid=lambda: (int(w.wConf.histo_geo_row.currentText()),
                          int(w.wConf.histo_geo_col.currentText())),
        plot_position=lambda index: w.plotPosition(index),
        open_image_dialog=lambda: w.openFigureDialog(),
        parent_widget=w,
        logger=w.logger,
    )
    for attr in _OVERLAY_STATE:
        monkeypatch.setattr(type(w), attr, _proxy(attr), raising=False)
    for name in ("indexToStartPosition", "drawFigure",
                 "_removeOverlayArtist"):
        setattr(w, name, getattr(w.overlay_controller, name))

    # overlay state, exactly as the controller's __init__ leaves it
    w.LISEpic = None
    w.imgplot = None
    w.overlay_ax = None
    w.onFigure = False
    w.xstart = w.ystart = 0.0
    w.alpha = w.zoomX = w.zoomY = 1.0
    return w


def load_image(win, shape=(4, 4)):
    win.LISEpic = np.arange(shape[0] * shape[1]).reshape(shape).astype(float)


# ================================================================== overlay

def test_every_overlay_slot_survives_being_pressed_before_a_load(win):
    # PIN: six slots were wired straight to buttons with no
    # guard, each opening on a bare self.imgplot.remove().
    for slot in (win.deleteFigure, win.fineUpMove, win.fineDownMove,
                 win.fineLeftMove, win.fineRightMove, win.moveFigure,
                 win.transFigure, win.zoomFigureX, win.zoomFigureY):
        slot()                                    # must not raise
    assert win.onFigure is False


def test_add_without_an_image_says_so(win):
    # PIN: the dead `except NameError: raise` never caught this
    win.addFigure()
    assert win.box.calls == [("warning", "Overlay", "Load an image first.")]
    assert win.onFigure is False


def test_add_without_a_selected_pad_says_so(win):
    load_image(win)
    win.currentPlot.selected_plot_index = None
    win.addFigure()
    assert win.box.calls[0][2] == "Please select one histogram."
    assert win.onFigure is False


def test_add_draws_one_overlay_and_sets_the_flag(win):
    load_image(win)
    before = len(win.currentPlot.figure.axes)
    win.addFigure()
    assert win.onFigure is True
    assert len(win.currentPlot.figure.axes) == before + 1
    assert win.imgplot is not None and win.overlay_ax is not None


def test_a_second_add_does_not_stack_another_overlay(win):
    load_image(win)
    win.addFigure()
    n = len(win.currentPlot.figure.axes)
    win.addFigure()
    assert len(win.currentPlot.figure.axes) == n


def test_delete_removes_both_the_image_and_its_axes(win):
    # PIN: only the image used to be removed, so every redraw
    # appended an empty axes to the list the pad lookups index
    load_image(win)
    before = len(win.currentPlot.figure.axes)
    win.addFigure()
    win.deleteFigure()
    assert len(win.currentPlot.figure.axes) == before
    assert win.imgplot is None and win.overlay_ax is None
    assert win.onFigure is False


@pytest.mark.parametrize("slider", ["transFigure", "zoomFigureX", "zoomFigureY"])
def test_a_slider_drag_never_accumulates_axes(win, slider):
    load_image(win)
    win.addFigure()
    n = len(win.currentPlot.figure.axes)
    for _ in range(10):                           # valueChanged fires per tick
        getattr(win, slider)()
    assert len(win.currentPlot.figure.axes) == n


@pytest.mark.parametrize("slider", ["transFigure", "zoomFigureX", "zoomFigureY"])
def test_a_slider_leaves_the_overlay_flag_alone(win, slider):
    # PIN: routing these through deleteFigure cleared onFigure while
    # the image was still on screen, so the next Add stacked a second overlay
    load_image(win)
    win.addFigure()
    getattr(win, slider)()
    assert win.onFigure is True


def test_the_sliders_report_their_level_in_the_label(win):
    win.extraPopup.imaging.alpha_slider.v = 7
    win.transFigure()
    assert win.extraPopup.imaging.alpha_label.text() == "Transparency Level (70 %)"


def test_a_nudge_moves_the_overlay_without_changing_the_axes_count(win):
    load_image(win)
    win.addFigure()
    n = len(win.currentPlot.figure.axes)
    x0, y0 = win.xstart, win.ystart
    win.fineUpMove()
    assert (win.xstart, win.ystart) != (x0, y0)
    assert len(win.currentPlot.figure.axes) == n


def test_a_joystick_move_with_no_overlay_is_a_no_op(win):
    load_image(win)
    x0, y0 = win.xstart, win.ystart
    win.moveFigure()                              # nothing drawn yet
    assert (win.xstart, win.ystart) == (x0, y0)


def test_draw_clears_the_flag_when_the_image_went_away(win):
    # reachable when a later Load fails under a live overlay: cv2.imread
    # answers None rather than raising, so the flag must not still claim one
    load_image(win)
    win.addFigure()
    win.LISEpic = None
    win._removeOverlayArtist()
    win.drawFigure()
    assert win.onFigure is False


def test_remove_tolerates_an_axes_detached_behind_our_back(win):
    # a geometry change or an enlarge runs InitializeCanvas, which delaxes
    # everything; the membership test is what keeps this from raising
    load_image(win)
    win.addFigure()
    win.currentPlot.figure.delaxes(win.overlay_ax)
    win.deleteFigure()                            # must not raise
    assert win.imgplot is None and win.overlay_ax is None


def test_load_figure_does_nothing_when_the_dialog_is_cancelled(win):
    win.openFigureDialog = lambda: None
    win.loadFigure()
    assert win.LISEpic is None
    assert win.extraPopup.imaging.loadLISE_name.text() == ""


def test_load_figure_records_the_path_even_when_the_image_is_unreadable(win):
    # cv2.imread answers None for a non-image; the name box still shows what
    # the user picked, and nothing raises
    win.openFigureDialog = lambda: "/no/such/image.png"
    win.loadFigure()
    assert win.extraPopup.imaging.loadLISE_name.text() == "/no/such/image.png"
    assert win.LISEpic is None


def test_start_position_comes_from_the_grid(win):
    win.wConf.histo_geo_row = FakeCombo("2")
    win.wConf.histo_geo_col = FakeCombo("2")
    win.plotPosition = lambda index: (1, 0)
    win.indexToStartPosition(0)
    assert 0.0 <= win.xstart <= 1.0 and 0.0 <= win.ystart <= 1.0


# ================================================================== jupyter

@pytest.fixture
def jwin(win, monkeypatch, tmp_path):
    win.view_state = types.SimpleNamespace(getSpectrumStoreDict=lambda: {})
    win.connection_manager = types.SimpleNamespace(
        getSpectrumStatistics=lambda: {})
    win.events = []

    class FakeView:
        def __init__(self, *a):
            self.closed = False
            self.loaded = None
            self.loggerdock = types.SimpleNamespace(log=lambda m: None)

        def setWindowTitle(self, t):
            pass

        def loadmain(self, addr):
            self.loaded = addr

        def close(self):
            self.closed = True
    from controllers import jupyter_controller as jc
    monkeypatch.setattr(jc, "WebWindow", FakeView)
    for name in ("log", "stopnotebook", "setup_logging", "set_logger"):
        monkeypatch.setattr(jc, name,
                            (lambda n: lambda *a, **k: win.events.append(n))(name))
    monkeypatch.setattr(jc, "export_spectrum_csv",
                        lambda *a, **k: win.events.append("export"))
    monkeypatch.setattr(jc, "testnotebook", lambda name: True)
    monkeypatch.setattr(jc, "startnotebook",
                        lambda execname, directory=None: "http://localhost:8888")
    monkeypatch.setattr(jc, "QMessageBox", win.box)
    monkeypatch.setattr(jc.QDir, "currentPath", staticmethod(lambda: str(tmp_path)))
    monkeypatch.setattr(jc.QDir, "homePath", staticmethod(lambda: str(tmp_path)))
    win._jc = jc
    # the Jupyter module globals the test bodies patch now live here
    win._gui = jc
    win.jupyter_controller = jc.JupyterController(
        peak_tab=win.extraPopup.peak,
        view_state=win.view_state,
        get_statistics=win.connection_manager.getSpectrumStatistics,
        qt_logger_factory=lambda view: types.SimpleNamespace(
            newlog=types.SimpleNamespace(connect=lambda f: None,
                                         emit=lambda m: None)),
        parent_widget=win,
        logger=win.logger,
    )
    monkeypatch.setattr(type(win), "jupyterView", property(
        lambda self: self.jupyter_controller.jupyterView,
        lambda self, v: setattr(self.jupyter_controller, "jupyterView", v)),
        raising=False)
    return win


def test_export_failure_is_logged_and_never_raised(jwin, monkeypatch, caplog):
    # PIN: a failure here must not crash the GUI or block
    # jupyterStart — the notebook can still open — but must not be silent
    monkeypatch.setattr(jwin._gui, "export_spectrum_csv",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("ragged")))
    with caplog.at_level(logging.ERROR, logger="test.overlay"):
        jwin.createDf()
    assert any("createDf" in r.message for r in caplog.records)


def test_start_flips_the_buttons_when_the_notebook_comes_up(jwin):
    jwin.jupyterStart()
    assert jwin.extraPopup.peak.jup_start.enabled is False
    assert jwin.extraPopup.peak.jup_stop.enabled is True
    assert jwin.jupyterView.loaded == "http://localhost:8888"


def test_start_exports_the_dataframe_first(jwin):
    jwin.jupyterStart()
    assert jwin.events[0] == "export"


def test_a_cancelled_locate_aborts_the_start_and_keeps_the_gui(jwin, monkeypatch):
    # PIN: the old tuple-truthiness check made Cancel unreachable,
    # and the cancel path called sys.exit(0) — killing the GUI
    gui = jwin._gui
    monkeypatch.setattr(gui, "testnotebook", lambda name: False)
    monkeypatch.setattr(gui.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    jwin.jupyterStart()                           # must return, not exit
    assert jwin.jupyterView is None
    assert jwin.extraPopup.peak.jup_start.enabled is True


def test_a_failed_server_start_reports_and_leaves_no_window(jwin, monkeypatch):
    monkeypatch.setattr(jwin._gui, "startnotebook",
                        lambda execname, directory=None: (_ for _ in ()).throw(
                            RuntimeError("port in use")))
    jwin.jupyterStart()
    assert jwin.jupyterView is None
    assert [c[0] for c in jwin.box.calls] == ["warning"]
    assert "port in use" in jwin.box.calls[0][2]


def test_stop_closes_the_view_and_restores_the_buttons(jwin):
    jwin.jupyterStart()
    view = jwin.jupyterView
    jwin.jupyterStop()
    assert view.closed is True
    assert jwin.jupyterView is None
    assert jwin.extraPopup.peak.jup_start.enabled is True
    assert jwin.extraPopup.peak.jup_stop.enabled is False
    assert "stopnotebook" in jwin.events


def test_stop_without_a_running_notebook_is_safe(jwin):
    jwin.jupyterStop()                            # never started
    assert jwin.jupyterView is None
    assert "stopnotebook" in jwin.events
