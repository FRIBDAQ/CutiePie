"""Guards for the seam between FitManager and the dialogs it used to build
itself: the service takes an injected builder, builds one lazily when none
is given, and MainWindow injects the real one."""

import ast
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import qt_stubs

GUI = os.path.join(os.path.dirname(__file__), "..", "..", "gui")

_AFFECTED = ("PyQt5", "PyQt5.QtCore", "PyQt5.QtWidgets", "CPyConverter",
             "httplib2", "alpha_filter_dialog", "services.fit_manager",
             "fit_dialogs")


@pytest.fixture(scope="module")
def mods():
    saved = {n: sys.modules.get(n) for n in _AFFECTED}
    installed = qt_stubs.install_missing_runtime_stubs()
    if installed:
        for n in ("services.fit_manager", "fit_dialogs", "alpha_filter_dialog"):
            sys.modules.pop(n, None)
    fm = importlib.import_module("services.fit_manager")
    fd = importlib.import_module("fit_dialogs")
    yield fm, fd
    for n, prev in saved.items():
        if prev is None:
            sys.modules.pop(n, None)
        else:
            sys.modules[n] = prev


def test_fit_manager_uses_the_injected_dialogs(mods):
    fm, _ = mods
    marker = object()
    m = fm.FitManager(fit_factory=None, spectra=None, dialogs=marker)
    assert m.dialogs is marker


def test_fit_manager_builds_fit_dialogs_lazily_when_none_is_injected(mods):
    fm, fd = mods
    parent = object()
    m = fm.FitManager(fit_factory=None, spectra=None, parent_widget=parent)
    assert m._dialogs is None                    # nothing built at construction
    built = m.dialogs
    assert isinstance(built, fd.FitDialogs)
    assert built._parent is parent
    assert m.dialogs is built                    # built once, then reused


def test_main_window_injects_fit_dialogs_at_the_composition_root():
    with open(os.path.join(GUI, "GUI.py")) as fh:
        tree = ast.parse(fh.read())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "FitManager"]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert isinstance(kw["dialogs"], ast.Call)
    assert kw["dialogs"].func.id == "FitDialogs"
