"""The tab-rename dialog, the per-pad cutoff dialog and the Qt log sink are Qt
shells with no MainWindow reference. The stub harness cannot construct a QDialog
subclass, so these check the shape only: the classes are where they belong, kept
every method, and took on no dependency on the window.
"""

import ast
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import gui_stubs

import pathlib

GUI = pathlib.Path(__file__).resolve().parents[2] / "gui"

# what each class carried before the move; a dropped method is a silent
# regression that only shows when a user opens the dialog
EXPECTED = {
    "QtLogger": (["__init__"], ["newlog"]),
    "TabPopup": (["__init__"], []),
    "cutoffPopup": (["__init__", "setVisibleFields", "layout1d", "layout2d"], []),
}


def _classes(path):
    tree = ast.parse(pathlib.Path(path).read_text())
    return {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}


def test_the_dialogs_module_exists():
    assert (GUI / "dialogs.py").exists()


def test_it_imports_headless():
    """No Qt beyond what the stub harness provides, and no import of GUI."""
    gui_stubs.install_gui_import_stubs()
    import importlib
    mod = importlib.import_module("dialogs")
    assert mod is not None


def test_every_class_moved():
    moved = _classes(GUI / "dialogs.py")
    assert set(EXPECTED) <= set(moved), sorted(set(EXPECTED) - set(moved))


def test_no_class_was_left_behind_in_gui():
    """A copy in both places would drift; MainWindow must import, not define."""
    left = set(EXPECTED) & set(_classes(GUI / "GUI.py"))
    assert not left, "still defined in GUI.py: %s" % sorted(left)


def test_gui_imports_them_from_the_new_module():
    src = (GUI / "GUI.py").read_text()
    assert "from dialogs import" in src


def test_every_method_survived_the_move():
    moved = _classes(GUI / "dialogs.py")
    for name, (methods, _) in EXPECTED.items():
        got = [m.name for m in moved[name].body if isinstance(m, ast.FunctionDef)]
        assert got == methods, "%s: %s" % (name, got)


def test_the_class_level_signal_survived():
    """QtLogger's `newlog` is what the notebook view subscribes to."""
    moved = _classes(GUI / "dialogs.py")
    attrs = [t.id for n in moved["QtLogger"].body if isinstance(n, ast.Assign)
             for t in n.targets if isinstance(t, ast.Name)]
    assert "newlog" in attrs


def test_nothing_moved_reaches_back_into_the_window():
    """The reason this move is safe: these are plain widgets. A reference to
    MainWindow state would make dialogs.py import GUI and close the loop."""
    # the CLASS bodies, not the file: the module docstring says "no MainWindow
    # reference", and a raw text scan matches its own documentation
    moved = _classes(GUI / "dialogs.py")
    body = "\n".join(ast.unparse(c) for c in moved.values())
    for probe in ("MainWindow", "wTab", "currentPlot", "self.spectra"):
        assert probe not in body, probe
    assert "import GUI" not in (GUI / "dialogs.py").read_text()


def test_the_cutoff_dialog_keeps_its_two_layouts():
    """1-D hides the Z fields, 2-D shows them; losing either method leaves the
    dialog showing the wrong axes for the pad."""
    moved = _classes(GUI / "dialogs.py")
    body = ast.unparse(moved["cutoffPopup"])
    assert "def layout1d" in body and "def layout2d" in body
    assert "setVisibleFields" in body
