"""The enlarge/un-enlarge round trip must not destroy Peak Finder 2's fits:
leaving enlarged mode re-adds the pad's spectrum, and that re-add clears the pad
and every fit artist on it. The fit records survive the trip, so these pin the
redraw that puts the fits back.
"""

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


def fit_result(lo=40.0, hi=60.0, mu=50.0):
    xx = np.linspace(lo, hi, 21)
    y_bg = np.ones_like(xx)
    y_fit = y_bg + np.exp(-0.5 * ((xx - mu) / 2.0) ** 2) * 10.0
    return {"xx": xx, "y_fit": y_fit, "y_bg": y_bg, "y_comp": [],
            "components": [{"mu": mu, "area": 100.0, "sigma": 2.0}]}


@pytest.fixture
def win(monkeypatch):
    gui = gui_stubs.import_gui()
    w = gui.MainWindow.__new__(gui.MainWindow)
    w.logger = logging.getLogger("test.enlarge")

    w.figure = Figure()
    FigureCanvasAgg(w.figure)
    w.pads = [w.figure.add_subplot(2, 2, i + 1) for i in range(4)]
    w.axes_by_index = {i: ax for i, ax in enumerate(w.pads)}
    w.names = {i: f"spec{i}" for i in range(4)}
    w.currentPlot = types.SimpleNamespace(figure=w.figure, canvas=w.figure.canvas,
                                          isEnlarged=False, _saved_axes=None)

    from controllers import peak_fit2_controller as pfc
    w.view_state = types.SimpleNamespace(
        getSpectrumStoreInfo=None,
        getSpectrumViewInfo=lambda field, index=None: w.axes_by_index.get(index),
        nameFromIndex=lambda index: w.names.get(index),
    )
    w.peak_fit2_controller = pfc.PeakFit2Controller(
        peak_tab=types.SimpleNamespace(),
        spectra=None,
        view_state=w.view_state,
        plot_controller=None,
        logger=w.logger,
    )
    monkeypatch.setattr(
        type(w), "peak2_fits",
        property(lambda self: self.peak_fit2_controller.peak2_fits,
                 lambda self, v: setattr(self.peak_fit2_controller, "peak2_fits", v)),
        raising=False)
    w.peakFit2RedrawAll = w.peak_fit2_controller.peakFit2RedrawAll
    return w


def add_fit(win, index, number=1):
    r = fit_result()
    rec = {"number": number, "index": index, "name": f"spec{index}", "result": r,
           "artists": win.peak_fit2_controller._peak2_draw(win.pads[index], r)}
    win.peak2_fits.append(rec)
    return rec


def drawn_on(win, index):
    """Fit artists ON SCREEN for that pad.

    A detached or hidden axes keeps its children, so counting the artists alone
    would report fits the user cannot see — the axes has to be in the figure
    and visible for them to count.
    """
    ax = win.pads[index]
    if ax not in win.figure.axes or not ax.get_visible():
        return 0
    return sum(1 for a in list(ax.lines) + list(ax.collections)
               if getattr(a, "get_gid", lambda: None)() == "peakfit2")


def enter_enlarged(win, idx):
    """What on_dblclick does entering enlarged mode."""
    win.currentPlot._saved_axes = win.figure.axes.copy()
    for ax in win.currentPlot._saved_axes:
        ax.set_visible(False)
    for ax in list(win.figure.axes):
        win.figure.delaxes(ax)
    enlarged = win.figure.add_subplot(111)
    win.currentPlot.isEnlarged = True
    win.axes_by_index = {i: enlarged for i in range(4)}
    return enlarged


def leave_enlarged(win, idx):
    """What on_dblclick does leaving it, including the re-add that clears."""
    for ax in list(win.figure.axes):
        win.figure.delaxes(ax)
    for ax in win.currentPlot._saved_axes:
        win.figure.add_axes(ax)
        ax.set_visible(True)
    win.currentPlot._saved_axes = None
    win.currentPlot.isEnlarged = False
    win.axes_by_index = {i: ax for i, ax in enumerate(win.pads)}
    win.pads[idx].clear()          # plot_controller.add(index) -> a.clear()


def test_a_fit_is_on_its_pad_to_begin_with(win):
    add_fit(win, 0)
    assert drawn_on(win, 0) == 4


def test_entering_enlarged_takes_the_fits_off_screen(win):
    """Expected under the restore-only fix: they go, and come back on the way
    out. Nothing redraws them onto the enlarged axes."""
    add_fit(win, 0)
    enter_enlarged(win, 0)
    assert drawn_on(win, 0) == 0


def test_the_round_trip_destroys_the_artists_without_a_redraw(win):
    """The defect itself. The re-add clears the restored pad, so the fits are
    gone for good — nothing else in the tick path puts them back."""
    add_fit(win, 0)
    enter_enlarged(win, 0)
    leave_enlarged(win, 0)
    assert drawn_on(win, 0) == 0


def test_the_records_survive_the_round_trip(win):
    """Which is why a redraw can restore them: the curves are stored."""
    add_fit(win, 0)
    enter_enlarged(win, 0)
    leave_enlarged(win, 0)
    assert len(win.peak2_fits) == 1
    assert win.peak2_fits[0]["result"]["xx"] is not None


def test_a_redraw_after_the_round_trip_puts_the_fits_back(win):
    """The fix's contract."""
    add_fit(win, 0)
    enter_enlarged(win, 0)
    leave_enlarged(win, 0)
    win.peakFit2RedrawAll()
    assert drawn_on(win, 0) == 4


def test_the_redraw_does_not_double_the_artists(win):
    """Called twice — as a stray extra call would — it must still leave one
    set, or every un-enlarge would stack another copy."""
    add_fit(win, 0)
    enter_enlarged(win, 0)
    leave_enlarged(win, 0)
    win.peakFit2RedrawAll()
    win.peakFit2RedrawAll()
    assert drawn_on(win, 0) == 4


def test_a_fit_on_another_pad_is_untouched_by_the_round_trip(win):
    """Confirmed live 2026-08-04: the re-add runs only for the enlarged index,
    so the other pads keep their fits throughout."""
    add_fit(win, 0, number=1)
    add_fit(win, 2, number=2)
    enter_enlarged(win, 0)
    leave_enlarged(win, 0)
    assert drawn_on(win, 2) == 4


def test_the_redraw_restores_every_pad_not_just_the_enlarged_one(win):
    add_fit(win, 0, number=1)
    add_fit(win, 2, number=2)
    enter_enlarged(win, 0)
    leave_enlarged(win, 0)
    win.peakFit2RedrawAll()
    assert drawn_on(win, 0) == 4 and drawn_on(win, 2) == 4


# --------------------------------------------------------------- the wiring

def _on_dblclick_ast():
    import ast
    gui = gui_stubs.import_gui()
    with open(gui.__file__.replace(".pyc", ".py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "MainWindow")
    return ast, next(m for m in cls.body
                     if isinstance(m, ast.FunctionDef) and m.name == "on_dblclick")


def test_the_un_enlarge_branch_calls_the_redraw():
    """The tests above prove the redraw restores the fits; this proves it is
    actually wired into the path that destroys them. Nothing headless drives
    on_dblclick, so the wiring is checked by reading it."""
    ast, fn = _on_dblclick_ast()
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "peakFit2RedrawAll"]
    assert len(calls) == 1, "expected exactly one redraw call in on_dblclick"

    branch = next(n for n in ast.walk(fn) if isinstance(n, ast.If)
                  and "isEnlarged" in ast.dump(n.test))
    in_enter = any(c in ast.walk(ast.Module(body=branch.body, type_ignores=[]))
                   for c in calls)
    in_leave = any(c in ast.walk(ast.Module(body=branch.orelse, type_ignores=[]))
                   for c in calls)
    assert in_leave and not in_enter, (
        "the redraw belongs in the un-enlarge branch only: on enter it would "
        "strip the artists off the saved axes and make this call mandatory "
        "rather than sufficient")


def test_the_redraw_runs_after_the_pad_is_re_added():
    """Order matters: the re-add clears the pad, so a redraw before it would be
    wiped by the very thing it is compensating for."""
    ast, fn = _on_dblclick_ast()
    redraw = next(n for n in ast.walk(fn) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "peakFit2RedrawAll")
    adds = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == "add"
            and n.lineno > 0]
    assert adds and redraw.lineno > max(adds)


# ------------------------------------------------- fits are not resurrected

def test_a_fit_whose_spectrum_was_removed_is_not_redrawn(win):
    """Removing a spectrum clears its pad but leaves the fit record behind, so
    a later redraw would put the fit back on a pad that no longer holds it."""
    add_fit(win, 0)
    win.pads[0].clear()            # what the removal adapter does
    win.names[0] = None            # the pad holds nothing now
    win.peakFit2RedrawAll()
    assert drawn_on(win, 0) == 0


def test_a_fit_is_not_redrawn_onto_a_different_spectrum(win):
    """A geometry change can leave a different spectrum at the same pad index.
    Redrawing by index alone would draw the old fit over the new data."""
    add_fit(win, 0)
    win.pads[0].clear()
    win.names[0] = "somethingelse"
    win.peakFit2RedrawAll()
    assert drawn_on(win, 0) == 0


def test_a_record_the_redraw_refuses_forgets_its_artists(win):
    """So nothing else is left holding handles to artists that are gone."""
    rec = add_fit(win, 0)
    win.pads[0].clear()
    win.names[0] = None
    win.peakFit2RedrawAll()
    assert rec["artists"] == ()


def test_the_guard_does_not_block_an_ordinary_redraw(win):
    """The pad still holds the same spectrum, so the round trip restores it."""
    add_fit(win, 0)
    enter_enlarged(win, 0)
    leave_enlarged(win, 0)
    win.peakFit2RedrawAll()
    assert drawn_on(win, 0) == 4
