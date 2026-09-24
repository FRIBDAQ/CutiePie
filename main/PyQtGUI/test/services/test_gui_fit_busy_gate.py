"""MainWindow refuses the slots that rebuild or destroy axes while a fit is
running. The fit pumps the Qt event loop so Abort stays clickable, which is
what lets these slots run mid-fit in the first place."""

import logging
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import gui_stubs


class Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args))
            return None
        return record


class Strict:
    """Any attribute access is a test failure: the gated slot must return
    before it touches the window."""

    def __getattr__(self, name):
        raise AssertionError(f"slot touched .{name} while a fit was running")


@pytest.fixture
def win():
    w = gui_stubs.bare_window("test.fit_busy_gate")
    w.fit_manager = types.SimpleNamespace(is_busy=lambda: True)
    for attr in ("wTab", "wConf", "currentPlot", "plot_controller",
                 "geometry_controller", "gatePopup", "sumRegionPopup",
                 "connection_manager", "connectConfig"):
        setattr(w, attr, Strict())
    return w


@pytest.mark.parametrize("slot,args", [
    ("addPlot", ()),
    ("closeTab", (1,)),
    ("setCanvasLayout", ()),
    ("loadGeo", ()),
    ("loadGeoAll", ()),
    ("okConnect", ()),      # a mirror re-transfer rebuilds every canvas
])
def test_rebuild_slots_return_without_touching_the_window(win, slot, args):
    assert getattr(win, slot)(*args) is None


def test_gates_open_again_when_the_fit_is_over(win):
    win.fit_manager.is_busy = lambda: False
    win.geometry_controller = Recorder()
    win.loadGeo()
    win.loadGeoAll()
    assert [c[0] for c in win.geometry_controller.calls] == ["loadGeo", "loadGeoAll"]

    win.plot_controller = Recorder()
    win.wConf = types.SimpleNamespace(
        histo_list=types.SimpleNamespace(count=lambda: 1, currentText=lambda: "h1"))
    win.wTab = types.SimpleNamespace(currentIndex=lambda: 0, isClickBound=lambda i: True)
    win.addPlot()
    assert win.plot_controller.calls == [("addPlot", ("h1", True))]
