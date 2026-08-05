"""The Fit-CSV plot lives in a window CutiePie owns.

`CsvPlotWindow.__init__` cannot be RUN here — it lays out real Qt widgets, and
the stub widgets carry no interaction methods by design — so what it builds is
checked against the source, the way `Plot.__init__` is. What the fit manager
DOES with it is checked by running it against a fake window."""

import ast
import importlib
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib.figure import Figure

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import qt_stubs

CSVGUI = os.path.join(os.path.dirname(__file__), "..", "..", "gui", "CsvPlotGUI.py")


def _calls_in(name):
    with open(CSVGUI, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == name)
    out = []
    for sub in ast.walk(fn):
        if not isinstance(sub, ast.Call):
            continue
        parts, f = [], sub.func
        while isinstance(f, ast.Attribute):
            parts.append(f.attr)
            f = f.value
        if isinstance(f, ast.Name):
            parts.append(f.id)
            out.append(".".join(reversed(parts)))
    return out


# --------------------------------------------------------- the window itself

def test_window_builds_its_own_figure_and_canvas():
    calls = _calls_in("__init__")
    assert "Figure" in calls, "the window must build a Figure() of its own"
    assert "FigureCanvas" in calls, "and hand it to a canvas it owns"
    assert "self.figure.add_subplot" in calls


def test_window_never_reaches_pyplot():
    """Checked as calls, not as text: the docstring explains what pyplot would
    have done, and a substring search would trip over its own explanation."""
    with open(CSVGUI, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    imported = [n for n in ast.walk(tree)
                if (isinstance(n, ast.Import)
                    and any(a.name.endswith("pyplot") for a in n.names))
                or (isinstance(n, ast.ImportFrom)
                    and (n.module or "").endswith("pyplot"))]
    assert not imported
    assert not [c for c in _calls_in("__init__") if c.startswith(("plt.", "pyplot."))]


# ------------------------------------------------ what fit_manager does with it

class FakeCsvWindow:
    """Stands in for the Qt window: same three attributes fit_manager uses."""

    instances = []

    def __init__(self, parent=None):
        self.parent = parent
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.shown = 0
        FakeCsvWindow.instances.append(self)

    def showWindow(self):
        self.shown += 1


@pytest.fixture
def fm(monkeypatch, tmp_path):
    qt_stubs.install_missing_runtime_stubs()
    for name in ("services.fit_manager", "alpha_filter_dialog"):
        sys.modules.pop(name, None)
    module = importlib.import_module("services.fit_manager")
    FakeCsvWindow.instances = []
    # the real factory imports a Qt window; this is the seam that keeps the
    # service importable, and the test replaces it rather than the class
    monkeypatch.setattr(module, "_make_csv_window",
                        lambda parent: FakeCsvWindow(parent))

    csv = tmp_path / "data.csv"
    csv.write_text("\n".join("%d,%d" % (i, i * i) for i in range(5)))
    second = tmp_path / "other.csv"
    second.write_text("\n".join("%d,%d" % (i, 2 * i) for i in range(5)))
    monkeypatch.setattr(module, "QFileDialog",
                        type("D", (), {"getOpenFileName": staticmethod(
                            lambda *a, **k: (str(csv), ""))}))

    manager = module.FitManager(fit_factory=None, spectra=None,
                                parent_widget=None,
                                logger=logging.getLogger("test.csv"))
    return module, manager, str(csv), str(second)


def test_plot_csv_opens_one_window_and_draws_into_it(fm):
    _module, manager, csv, _second = fm
    manager.on_plot_csv_clicked()

    assert len(FakeCsvWindow.instances) == 1
    win = FakeCsvWindow.instances[0]
    assert win.shown == 1
    assert manager._csv_ax is win.ax
    line, = win.ax.get_lines()
    assert np.allclose(line.get_xdata(), [0, 1, 2, 3, 4])
    assert np.allclose(line.get_ydata(), [0, 1, 4, 9, 16])
    assert win.ax.get_title() == os.path.basename(csv)


def test_a_second_csv_reuses_the_same_window(fm):
    module, manager, _csv, second = fm
    manager.on_plot_csv_clicked()
    monkey = type("D", (), {"getOpenFileName": staticmethod(
        lambda *a, **k: (second, ""))})
    module.QFileDialog = monkey
    manager.on_plot_csv_clicked()

    # one window, raised twice, holding only the newest curve
    assert len(FakeCsvWindow.instances) == 1
    win = FakeCsvWindow.instances[0]
    assert win.shown == 2
    line, = win.ax.get_lines()
    assert np.allclose(line.get_ydata(), [0, 2, 4, 6, 8])
