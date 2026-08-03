"""Headless import harness for ``gui/GUI.py``.

``qt_stubs`` gets the *service* layer importable without PyQt5. This module
goes the rest of the way and makes ``import GUI`` work, so ``MainWindow``
methods can be characterized the same way service methods already are.

Four things stand between qt_stubs and ``import GUI``:

* ``cv2`` is not installed in the system python3 (the figure-overlay import),
* the stub PyQt5 exposes only what the services touch, and ``GUI.py`` plus its
  siblings reach for 29 more names,
* ``matplotlib.backends.backend_qt5agg`` needs a real ``sip``, so it is shimmed
  onto the Agg canvas the existing tests already draw with,
* ``WebWindow`` imports ``PyQt5.QtWebEngineWidgets``.

None of it is installed when the real package is importable, so on a machine
with PyQt5 the same tests run against the real thing.

**The name lists below are deliberately explicit.** A module that answers every
attribute would let a test pass against code referencing a widget that does not
exist, which is the blind spot ``py_compile`` already has. Adding a name here
when production code grows one is the review point; keep it that way.

Usage::

    import gui_stubs
    win = gui_stubs.bare_window()      # MainWindow with no __init__ run
    win.spectra = SpectrumStore()      # inject only what the method touches

``bare_window`` deliberately skips ``__init__``: it builds 9 popups, 5 services
and ~118 signal connections, none of which a seam-level test wants.
"""

import importlib
import logging
import sys
import types

import qt_stubs


class _StubWidget:
    """Accepts any construction, does nothing. Base for the extra Qt names."""

    def __init__(self, *args, **kwargs):
        pass


# The Qt names GUI.py and its siblings import beyond what qt_stubs already
# provides. Measured by importing GUI against a recording stub, 2026-08-03.
_EXTRA_QTCORE = ("QDir", "QLineF", "QPointF", "QRectF", "QUrl")

_EXTRA_QTGUI = ("QCloseEvent", "QCursor", "QMouseEvent", "QPainter")

# QPalette is reached for its role enum, not constructed: connectCopy asks a
# button for palette().color(QPalette.Text).name(). The sentinel just has to be
# something a fake palette can be keyed on.
_QT_ENUMS = {"PyQt5.QtGui": {"QPalette": {"Text": "Text", "WindowText": "WindowText"}}}

_EXTRA_QTWIDGETS = (
    "QAbstractItemView", "QDockWidget", "QFormLayout", "QGridLayout",
    "QGroupBox", "QHeaderView", "QLineEdit", "QListWidget", "QListWidgetItem",
    "QMainWindow", "QPlainTextEdit", "QRadioButton", "QSizePolicy", "QSlider",
    "QTabBar", "QTabWidget", "QTableWidget", "QToolButton", "QWidget",
)

_MPL_QT_BACKENDS = ("matplotlib.backends.backend_qt5agg",
                    "matplotlib.backends.backend_qt5")


def _widen_stub_pyqt5():
    """Add the extra names to the stub PyQt5, and only to the stub."""
    installed = []
    for mod_name, names in (("PyQt5.QtCore", _EXTRA_QTCORE),
                            ("PyQt5.QtGui", _EXTRA_QTGUI),
                            ("PyQt5.QtWidgets", _EXTRA_QTWIDGETS)):
        module = sys.modules[mod_name]
        for name in names:
            if not hasattr(module, name):
                setattr(module, name, type(name, (_StubWidget,), {}))
                installed.append(f"{mod_name}.{name}")

    for mod_name, classes in _QT_ENUMS.items():
        module = sys.modules[mod_name]
        for cls_name, members in classes.items():
            if not hasattr(module, cls_name):
                setattr(module, cls_name, type(cls_name, (_StubWidget,), dict(members)))
                installed.append(f"{mod_name}.{cls_name}")

    if "PyQt5.QtWebEngineWidgets" not in sys.modules:
        web = types.ModuleType("PyQt5.QtWebEngineWidgets")
        web.QWebEngineView = type("QWebEngineView", (_StubWidget,), {})
        sys.modules["PyQt5.QtWebEngineWidgets"] = web
        sys.modules["PyQt5"].QtWebEngineWidgets = web
        installed.append("PyQt5.QtWebEngineWidgets")
    return installed


def _stub_cv2():
    mod = types.ModuleType("cv2")
    mod.imread = lambda *args, **kwargs: None      # what a non-image file yields
    sys.modules["cv2"] = mod
    return ["cv2"]


def _stub_mpl_qt_backends():
    """Point the Qt canvas at Agg: the tests render off-screen either way."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    installed = []
    for name in _MPL_QT_BACKENDS:
        mod = types.ModuleType(name)
        mod.FigureCanvasQTAgg = FigureCanvasAgg
        mod.FigureCanvas = FigureCanvasAgg
        mod.NavigationToolbar2QT = type("NavigationToolbar2QT", (_StubWidget,), {})
        sys.modules[name] = mod
        installed.append(name)
    return installed


def install_gui_import_stubs():
    """Make ``import GUI`` possible headless; return what was installed.

    Idempotent, and import-time side-effect free: call it from a fixture.
    """
    installed = list(qt_stubs.install_missing_runtime_stubs())

    # Only widen a PyQt5 we built ourselves. With the real package present its
    # own names are already there and hasattr() short-circuits every setattr.
    installed += _widen_stub_pyqt5()

    for name, builder in (("cv2", _stub_cv2),):
        if name not in sys.modules:
            try:
                importlib.import_module(name)
            except ImportError:
                installed += builder()

    try:
        importlib.import_module("matplotlib.backends.backend_qt5agg")
    except Exception:      # ImportError, or the sip AttributeError PyQt5 raises
        installed += _stub_mpl_qt_backends()
    return installed


def import_gui():
    """Install what is missing, then return the imported ``GUI`` module."""
    install_gui_import_stubs()
    return importlib.import_module("GUI")


def bare_window(logger_name="test.mainwindow"):
    """A ``MainWindow`` with ``__init__`` deliberately not run.

    Only ``logger`` is set, because every method logs. Give the instance
    whatever else the method under test reads; anything it reaches for and did
    not get raises AttributeError, which is the signal you wanted.
    """
    gui = import_gui()
    win = gui.MainWindow.__new__(gui.MainWindow)
    win.logger = logging.getLogger(logger_name)
    return win
