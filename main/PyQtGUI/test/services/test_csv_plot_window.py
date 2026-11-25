"""The Fit-CSV plot lives in a window CutiePie owns. `CsvPlotWindow.__init__`
lays out real Qt widgets and cannot run here, so what it builds is checked
against the source and what the fit manager does with it is checked against a
fake window."""

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


def test_window_is_a_window_of_its_own():
    """It is built with the main window as parent. Without the window flag Qt
    makes such a widget a child drawn inside its parent, and Plot CSV showed
    nothing but a sliver in the main window's corner."""
    with open(CSVGUI, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    flags = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
             and isinstance(c.func, ast.Attribute) and c.func.attr == "setWindowFlag"]
    assert len(flags) == 1
    flag, on = flags[0].args
    assert (flag.value.id, flag.attr) == ("Qt", "Window") and on.value is True


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


# ------------------------------------------------------------ InterSpec CSV

INTERSPEC_HEAD = """Spectrum Name:
Description:
Detector Used:
Comments:
Acquisition Mode:,PHA Pre-Amp
High Voltage:,0
Conversion Gain:,1024
Fine Gain:,1.00
Calibration Coefficients:,-751.0,33.0
Calibration Channels:,179,189
Calibration Energies:,5156.0,5486.0
Elapsed Live Time:,7200.0
Start Time:,August 18, 2025, 04:59:18
ROIs:
ROI:,Low,High,Gross,Net,FWHM,Centroid


Channel Data:
Channel,Energy,Counts
"""


def interspec_file(tmp_path, counts=(0, 1, 13, 63, 2), name="spec.csv", head=INTERSPEC_HEAD):
    rows = "".join("%d,%.1f,%d\n" % (ch, -751.0 + 33.0 * ch, n)
                   for ch, n in enumerate(counts))
    path = tmp_path / name
    path.write_text(head + rows)
    return str(path)


def test_interspec_csv_gives_counts_against_energy(fm, tmp_path):
    module = fm[0]
    x, y = module.read_interspec_csv(interspec_file(tmp_path))
    assert np.allclose(x, [-751.0, -718.0, -685.0, -652.0, -619.0])
    assert np.allclose(y, [0, 1, 13, 63, 2])


def test_interspec_header_numbers_are_not_taken_for_data(fm, tmp_path):
    """The header has rows of numbers of its own (calibration channels and
    energies, an ROI column row); only the rows under the column names count."""
    module = fm[0]
    x, y = module.read_interspec_csv(interspec_file(tmp_path, counts=(7, 8)))
    assert len(x) == 2 and np.allclose(y, [7, 8])


def test_interspec_csv_without_energy_uses_the_channel(fm, tmp_path):
    module = fm[0]
    path = tmp_path / "ch.csv"
    path.write_text("Comments:\nChannel,Counts\n0,4\n1,5\n2,6\n")
    x, y = module.read_interspec_csv(str(path))
    assert np.allclose(x, [0, 1, 2]) and np.allclose(y, [4, 5, 6])


def test_interspec_csv_with_energy_and_data_columns(fm, tmp_path):
    module = fm[0]
    path = tmp_path / "ed.csv"
    path.write_text("Energy, Data\r\n0.5, 3\r\n1.5, 9\r\n")
    x, y = module.read_interspec_csv(str(path))
    assert np.allclose(x, [0.5, 1.5]) and np.allclose(y, [3, 9])


def test_interspec_table_ends_at_the_first_row_that_is_not_data(fm, tmp_path):
    module = fm[0]
    path = interspec_file(tmp_path, counts=(1, 2, 3))
    with open(path, "a") as f:
        f.write("\nSomething else:\n9,9,9\n")
    x, y = module.read_interspec_csv(path)
    assert np.allclose(y, [1, 2, 3])


def test_a_file_without_a_spectrum_table_is_refused(fm, tmp_path):
    module, _manager, plain_xy, _second = fm
    with pytest.raises(ValueError):
        module.read_interspec_csv(plain_xy)             # plain x,y rows
    empty = tmp_path / "empty.csv"
    empty.write_text(INTERSPEC_HEAD)
    with pytest.raises(ValueError):
        module.read_interspec_csv(str(empty))           # column names, no rows


def _point_dialog_at(module, path):
    module.QFileDialog = type("D", (), {"getOpenFileName": staticmethod(
        lambda *a, **k: (path, ""))})


def test_plot_csv_draws_an_interspec_file(fm, tmp_path):
    module, manager, _csv, _second = fm
    path = interspec_file(tmp_path)
    _point_dialog_at(module, path)
    manager.on_plot_csv_clicked("InterSpec")

    win = FakeCsvWindow.instances[0]
    line, = win.ax.get_lines()
    assert np.allclose(line.get_xdata(), [-751.0, -718.0, -685.0, -652.0, -619.0])
    assert np.allclose(line.get_ydata(), [0, 1, 13, 63, 2])
    assert win.ax.get_title() == "spec.csv" and win.shown == 1
    # Fit CSV works from these
    assert np.allclose(manager._csv_x, line.get_xdata())
    assert np.allclose(manager._csv_y, line.get_ydata())
    assert manager._csv_path == path


def test_plot_csv_keeps_reading_plain_rows_as_spectcl(fm):
    _module, manager, _csv, _second = fm
    manager.on_plot_csv_clicked("SpecTcl")
    line, = FakeCsvWindow.instances[0].ax.get_lines()
    assert np.allclose(line.get_ydata(), [0, 1, 4, 9, 16])


def test_a_wrong_file_under_interspec_warns_and_keeps_the_loaded_data(fm, monkeypatch):
    module, manager, csv, _second = fm
    manager.on_plot_csv_clicked()                       # plain file, loaded
    before = (manager._csv_path, manager._csv_x.copy())

    warned = []
    monkeypatch.setattr(module, "QMessageBox", type("M", (), {
        "warning": staticmethod(lambda *a: warned.append(a[1:]))}))
    manager.on_plot_csv_clicked("InterSpec")            # same plain file: no table

    assert len(warned) == 1 and warned[0][0] == "Cannot read InterSpec CSV"
    assert manager._csv_path == before[0]
    assert np.allclose(manager._csv_x, before[1])
    assert FakeCsvWindow.instances[0].shown == 1        # not redrawn or raised


# ------------------------------------------------------------ Clear fit CSV

def _draw_csv_fit(ax):
    """What a fit leaves on the CSV plot: a labelled total, a component curve,
    a peak label and a legend, all under one gid."""
    total, = ax.plot([1, 2, 3], [1, 4, 1], label="fit total")
    comp, = ax.plot([1, 2, 3], [0.5, 2, 0.5], ls="--", label="224Th")
    text = ax.text(2, 4.2, "224Th 7170")
    ax.legend([total, comp], ["fit total", "224Th"], loc="best", frameon=False)
    total.set_label("fit-_-0")
    for a in (total, comp, text):
        a.set_gid("fit-0")
    return total, comp, text


def test_clear_fit_csv_removes_the_fits_and_keeps_the_data(fm):
    _module, manager, _csv, _second = fm
    manager.on_plot_csv_clicked()
    ax = manager._csv_ax
    data, = ax.get_lines()
    total, comp, text = _draw_csv_fit(ax)

    assert manager.on_clear_csv_fit_clicked() == 3
    assert list(ax.get_lines()) == [data]
    assert text not in ax.texts and ax.get_legend() is None
    # the loaded data is still there for the next Fit CSV
    assert np.allclose(manager._csv_y, [0, 1, 4, 9, 16])


def test_clear_fit_csv_takes_several_fits_at_once(fm):
    _module, manager, _csv, _second = fm
    manager.on_plot_csv_clicked()
    ax = manager._csv_ax
    _draw_csv_fit(ax)
    second, = ax.plot([5, 6], [1, 2], label="fit-_-1")
    second.set_gid("fit-1")

    assert manager.on_clear_csv_fit_clicked() == 4
    assert len(ax.get_lines()) == 1


def test_clear_fit_csv_leaves_the_fit_labels_field_alone(fm):
    """That field lists the fits of the selected pad in the main window."""
    _module, manager, _csv, _second = fm
    labels = qt_stubs.record_signal(manager.fitLabelsTextChanged)
    manager.on_plot_csv_clicked()
    _draw_csv_fit(manager._csv_ax)

    manager.on_clear_csv_fit_clicked()
    assert labels == []


def test_clear_fit_csv_before_any_csv_was_plotted_does_nothing(fm):
    _module, manager, _csv, _second = fm
    assert manager.on_clear_csv_fit_clicked() == 0


def test_clear_fit_csv_waits_for_a_running_fit(fm):
    _module, manager, _csv, _second = fm
    manager.on_plot_csv_clicked()
    ax = manager._csv_ax
    total, _comp, _text = _draw_csv_fit(ax)
    manager._fit_busy = True

    assert manager.on_clear_csv_fit_clicked() == 0
    assert total in ax.get_lines()


# ------------------------------------------- a file that cannot be read

def _catch_warnings(module, monkeypatch):
    warned = []
    monkeypatch.setattr(module, "QMessageBox", type("M", (), {
        "warning": staticmethod(lambda *a: warned.append(a[1:]))}))
    return warned


def test_an_interspec_file_under_spectcl_warns_instead_of_raising(fm, tmp_path, monkeypatch):
    """Its header rows have other column counts than its data rows, which
    numpy refuses with a ValueError. Raised out of the button's slot, that
    ended the program."""
    module, manager, csv, _second = fm
    manager.on_plot_csv_clicked("SpecTcl")              # plain file, loaded
    before = (manager._csv_path, manager._csv_x.copy())

    warned = _catch_warnings(module, monkeypatch)
    path = interspec_file(tmp_path, counts=range(40))
    _point_dialog_at(module, path)
    manager.on_plot_csv_clicked("SpecTcl")              # must not raise

    assert len(warned) == 1
    title, text = warned[0]
    assert title == "Cannot read SpecTcl CSV"
    assert text.startswith("spec.csv\n\n")
    # what was loaded before is still there for Fit CSV
    assert manager._csv_path == before[0]
    assert np.allclose(manager._csv_x, before[1])
    assert FakeCsvWindow.instances[0].shown == 1


def test_the_error_popup_shows_only_the_first_lines(fm, tmp_path, monkeypatch):
    """numpy names every row it could not use; for a 1024-channel file that
    is a message of a thousand lines."""
    module, manager, _csv, _second = fm
    warned = _catch_warnings(module, monkeypatch)
    _point_dialog_at(module, interspec_file(tmp_path, counts=range(1024)))
    manager.on_plot_csv_clicked("SpecTcl")
    text = warned[0][1]
    assert len(text.splitlines()) <= 9                  # name, blank, 6 lines, the rest counted
    assert "more lines)" in text.splitlines()[-1]


def test_a_missing_file_warns_instead_of_raising(fm, tmp_path, monkeypatch):
    module, manager, _csv, _second = fm
    warned = _catch_warnings(module, monkeypatch)
    _point_dialog_at(module, str(tmp_path / "gone.csv"))
    for fmt in ("SpecTcl", "InterSpec"):
        manager.on_plot_csv_clicked(fmt)
    assert [w[0] for w in warned] == ["Cannot read SpecTcl CSV", "Cannot read InterSpec CSV"]
    assert FakeCsvWindow.instances == []                # no window for nothing


def test_a_single_number_file_warns_instead_of_raising(fm, tmp_path, monkeypatch):
    module, manager, _csv, _second = fm
    warned = _catch_warnings(module, monkeypatch)
    one = tmp_path / "one.csv"
    one.write_text("5\n")
    _point_dialog_at(module, str(one))
    manager.on_plot_csv_clicked("SpecTcl")
    assert [w[0] for w in warned] == ["Cannot read SpecTcl CSV"]
