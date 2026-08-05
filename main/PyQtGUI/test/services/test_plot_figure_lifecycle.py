"""A tab's figure belongs to its canvas, not to pyplot. `Plot.__init__` builds
a real Qt toolbar and cannot run here, so the two lines that decide figure
ownership are checked against the source and the teardown they make necessary is
checked by running it."""

import ast
import os
import sys

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import gui_stubs

PLOTGUI = os.path.join(os.path.dirname(__file__), "..", "..", "gui", "PlotGUI.py")


def _func(name, cls=None):
    """The AST of a function, optionally inside a class."""
    with open(PLOTGUI, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    scope = tree
    if cls is not None:
        scope = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in ast.walk(scope)
                if isinstance(n, ast.FunctionDef) and n.name == name)


def _calls(node):
    """Every call in a subtree as a dotted string: "Figure", "plt.figure"."""
    out = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        parts = []
        f = sub.func
        while isinstance(f, ast.Attribute):
            parts.append(f.attr)
            f = f.value
        if isinstance(f, ast.Name):
            parts.append(f.id)
            out.append(".".join(reversed(parts)))
    return out


# ------------------------------------------------------------- creation

def test_plot_builds_its_figure_outside_pyplot():
    calls = _calls(_func("__init__", cls="Plot"))
    assert "Figure" in calls, "Plot.__init__ must build a Figure() of its own"
    assert not [c for c in calls if c.startswith("plt.")], \
        "Plot.__init__ still reaches pyplot: %s" % calls


def test_plot_module_imports_figure():
    with open(PLOTGUI, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    imported = {a.name for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.module == "matplotlib.figure"
                for a in n.names}
    assert "Figure" in imported


# -------------------------------------------------------------- teardown

def test_delete_tab_tears_the_figure_down_itself():
    """plt.close() cannot free a figure pyplot never registered, so deleteTab
    has to do it — the call being gone is half the check, destroyFigure being
    called in its place is the other half."""
    calls = _calls(_func("deleteTab"))
    assert "plt.close" not in calls, "plt.close on an unmanaged figure does nothing"
    assert "destroyFigure" in calls


def test_destroy_figure_empties_the_figure():
    gui_stubs.import_gui()
    import PlotGUI

    figure = Figure()
    figure.add_subplot(111).plot([0, 1], [0, 1])
    assert figure.axes

    PlotGUI.destroyFigure(_widget(figure))
    assert figure.axes == []


def test_destroy_figure_leaves_pyplot_alone():
    """The teardown must not reach for pyplot either: swapping plt.close() for
    a call that still goes through the registry would move the bug rather than
    fix it. Checked at the source, and by running it against a figure the
    registry has never heard of."""
    assert not [c for c in _calls(_func("destroyFigure")) if c.startswith("plt.")]

    gui_stubs.import_gui()
    import PlotGUI

    before = set(plt.get_fignums())
    PlotGUI.destroyFigure(_widget(Figure()))
    assert set(plt.get_fignums()) == before


def test_destroy_figure_survives_a_widget_with_no_figure():
    # deleteTab must not raise on a session whose widget never drew
    gui_stubs.import_gui()
    import PlotGUI

    PlotGUI.destroyFigure(_widget(None))
    PlotGUI.destroyFigure(object())


class _widget:
    """Stands in for the Plot widget deleteTab hands over."""

    def __init__(self, figure):
        if figure is not None:
            self.figure = figure
