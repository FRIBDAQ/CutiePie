"""Headless test doubles for the service layer. Two things live here: 1."""

import importlib
import sys
import types


# ---------------------------------------------------------------------------
# Stub PyQt5 (only what services.connection_manager / thread_workers import)
# ---------------------------------------------------------------------------

class BoundStubSignal:
    """Instance-level signal: connect/emit/disconnect, synchronous delivery."""

    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def disconnect(self, slot=None):
        if slot is None:
            self._slots.clear()
        else:
            self._slots.remove(slot)

    def emit(self, *args):
        for slot in list(self._slots):
            # A slot may itself be a signal (signal-to-signal chaining).
            # Real pyqtBoundSignal objects are not callable — use .emit.
            target = slot if callable(slot) else getattr(slot, "emit", None)
            if target is None:
                raise TypeError(f"unsupported slot: {slot!r}")
            target(*args)

    __call__ = emit


class StubSignal:
    """pyqtSignal stand-in: class attribute that binds per instance."""

    def __init__(self, *signature):
        self._signature = signature
        self._name = None

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        key = "_stub_signal_" + (self._name or str(id(self)))
        bound = obj.__dict__.get(key)
        if bound is None:
            bound = BoundStubSignal()
            obj.__dict__[key] = bound
        return bound


def stub_pyqt_slot(*args, **kwargs):
    def decorate(fn):
        return fn
    return decorate


class StubQObject:
    def __init__(self, parent=None):
        self._parent = parent

    def moveToThread(self, thread):
        pass

    def deleteLater(self):
        pass


class StubQThread(StubQObject):
    started = StubSignal()

    def start(self):
        pass

    def quit(self):
        pass

    def wait(self):
        pass


class StubQTimer:
    @staticmethod
    def singleShot(msec, callback):
        pass


class _StubQtNamespace:
    MatchContains = object()
    NonModal = object()


class StubQComboBox:
    NoInsert = 0


class StubQCompleter:
    PopupCompletion = 0


class _StubWidget:
    """Import-surface stand-in for widget classes tests never instantiate
    for real (dialogs, labels, layouts). Accepts any ctor args; interaction
    methods are absent on purpose — behavioral doubles are monkeypatched at
    the service-module attribute per test instead."""

    def __init__(self, *args, **kwargs):
        pass


class StubQMessageBox(_StubWidget):
    YesRole = 0
    NoRole = 1
    RejectRole = 2

    @staticmethod
    def about(*args, **kwargs):
        pass

    @staticmethod
    def warning(*args, **kwargs):
        pass

    @staticmethod
    def information(*args, **kwargs):
        pass


class StubQFileDialog(_StubWidget):
    @staticmethod
    def getOpenFileName(*args, **kwargs):
        return ("", "")


class StubQDialog(_StubWidget):
    Accepted = 1
    Rejected = 0


class StubQMenu(_StubWidget):
    def addAction(self, *args, **kwargs):
        class _Action:
            triggered = BoundStubSignal()
        return _Action()

    def exec_(self, *args, **kwargs):
        pass


class StubQSettings:
    def __init__(self, *args, **kwargs):
        pass

    def value(self, key, default=None, type=None):
        return default

    def setValue(self, key, value):
        pass


class StubQEventLoop:
    def exec_(self):
        pass

    def quit(self):
        pass


def _build_stub_pyqt5():
    qtcore = types.ModuleType("PyQt5.QtCore")
    qtcore.QObject = StubQObject
    qtcore.QThread = StubQThread
    qtcore.QTimer = StubQTimer
    qtcore.pyqtSignal = StubSignal
    qtcore.pyqtSlot = stub_pyqt_slot
    qtcore.Qt = _StubQtNamespace
    qtcore.QSettings = StubQSettings
    qtcore.QEventLoop = StubQEventLoop

    qtwidgets = types.ModuleType("PyQt5.QtWidgets")
    qtwidgets.QComboBox = StubQComboBox
    qtwidgets.QCompleter = StubQCompleter
    qtwidgets.QMessageBox = StubQMessageBox
    qtwidgets.QFileDialog = StubQFileDialog
    qtwidgets.QDialog = StubQDialog
    qtwidgets.QApplication = _StubWidget
    qtwidgets.QLabel = _StubWidget
    qtwidgets.QPushButton = _StubWidget
    qtwidgets.QCheckBox = _StubWidget
    qtwidgets.QHBoxLayout = _StubWidget
    qtwidgets.QVBoxLayout = _StubWidget
    qtwidgets.QInputDialog = _StubWidget
    qtwidgets.QTextEdit = _StubWidget
    qtwidgets.QTableWidgetItem = _StubWidget
    qtwidgets.QShortcut = _StubWidget
    qtwidgets.QMenu = StubQMenu

    qtgui = types.ModuleType("PyQt5.QtGui")
    qtgui.QKeySequence = _StubWidget

    pyqt5 = types.ModuleType("PyQt5")
    pyqt5.QtCore = qtcore
    pyqt5.QtWidgets = qtwidgets
    pyqt5.QtGui = qtgui
    return {"PyQt5": pyqt5, "PyQt5.QtCore": qtcore,
            "PyQt5.QtWidgets": qtwidgets, "PyQt5.QtGui": qtgui}


def _build_stub_cpyconverter():
    mod = types.ModuleType("CPyConverter")

    class CPyConverter:
        def Update(self, *args):
            raise RuntimeError("stub CPyConverter: no shared memory in tests")

    mod.CPyConverter = CPyConverter
    return {"CPyConverter": mod}


def _build_stub_httplib2():
    mod = types.ModuleType("httplib2")

    class Http:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            raise RuntimeError("stub httplib2: no network in tests")

    mod.Http = Http
    return {"httplib2": mod}


_BUILDERS = {
    "PyQt5": _build_stub_pyqt5,
    "CPyConverter": _build_stub_cpyconverter,
    "httplib2": _build_stub_httplib2,
}


def install_missing_runtime_stubs():
    """Register stubs for the unimportable runtime deps; return installed names."""
    installed = []
    for name, builder in _BUILDERS.items():
        try:
            importlib.import_module(name)
        except ImportError:
            for mod_name, module in builder().items():
                sys.modules[mod_name] = module
                installed.append(mod_name)
    return installed


# ---------------------------------------------------------------------------
# Fake collaborators for ConnectionManager
# ---------------------------------------------------------------------------

class FakeLineEdit:
    def __init__(self, value=""):
        self.value = value

    def text(self):
        return self.value

    def setText(self, value):
        self.value = value


class FakeButton:
    def __init__(self):
        self.label = None
        self.style = None
        self.enabled = True
        self.down = False

    def setText(self, text):
        self.label = text

    def setStyleSheet(self, style):
        self.style = style

    def setEnabled(self, enabled):
        self.enabled = enabled

    def setDown(self, down):
        self.down = down


class FakeCheckBox:
    def __init__(self, checked=False):
        self.checked = checked

    def isChecked(self):
        return self.checked

    def setChecked(self, checked):
        self.checked = checked


class FakeCompleter:
    def __init__(self):
        self.completion_mode = None
        self.filter_mode = None

    def setCompletionMode(self, mode):
        self.completion_mode = mode

    def setFilterMode(self, mode):
        self.filter_mode = mode


class FakeComboBox:
    def __init__(self):
        self.items = []
        self.signals_blocked = None
        self.editable = None
        self.insert_policy = None
        self.index = 0
        self.current_text = None
        self._completer = FakeCompleter()

    def blockSignals(self, blocked):
        self.signals_blocked = blocked

    def clear(self):
        self.items = []

    def addItems(self, items):
        self.items.extend(items)

    def setEditable(self, editable):
        self.editable = editable

    def setInsertPolicy(self, policy):
        self.insert_policy = policy

    def completer(self):
        return self._completer

    def currentIndex(self):
        return self.index

    def currentText(self):
        if self.current_text is not None:
            return self.current_text
        return self.items[self.index] if self.items else ""

    def count(self):
        return len(self.items)


class FakeRest:
    """Configurable PyREST stand-in."""

    def __init__(self, check=True, shmem_size=None, spectra=None, binds=None,
                 gates=None, gates_error=False):
        self._check = check
        self._shmem_size = shmem_size
        self._spectra = spectra if spectra is not None else []
        self._binds = binds if binds is not None else []
        self._gates = gates if gates is not None else []
        self._gates_error = gates_error
        self.reconfigured = None

    def reconfigure(self, server, rest):
        self.reconfigured = (server, rest)

    def checkSpecTclREST(self):
        return self._check

    def shmemSize(self):
        return self._shmem_size

    def listSpectrum(self, filter=None):
        return self._spectra

    def listsbind(self, pattern):
        return self._binds

    def applylistgate(self, spectrum_name):
        if self._gates_error:
            raise RuntimeError("REST down")
        return self._gates


class FakeThread:
    """QThread stand-in for monkeypatching connection_manager.QThread.

    Never starts an OS thread, so worker code never runs unless a test emits
    the worker's signals explicitly — deterministic under stub AND real Qt."""

    def __init__(self, parent=None):
        self.started = BoundStubSignal()
        self.running = False
        self.quit_count = 0

    def start(self):
        self.running = True

    def quit(self):
        self.quit_count += 1
        self.running = False

    def wait(self):
        pass


class FakeWorker:
    """Stand-in for RestWorker / ConnectWorker / AutoUpdateWorker: records
    ctor args and exposes every signal any of the three declares."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.thread = None
        self.connected = BoundStubSignal()
        self.disconnected = BoundStubSignal()
        self.tracesReady = BoundStubSignal()
        self.spectrumAdded = BoundStubSignal()
        self.succeeded = BoundStubSignal()
        self.failed = BoundStubSignal()
        self.updateTriggered = BoundStubSignal()

    def moveToThread(self, thread):
        self.thread = thread

    def run(self):
        pass


class RecordingTimer:
    """QTimer stand-in: records singleShot callbacks instead of scheduling."""

    def __init__(self):
        self.scheduled = []

    def singleShot(self, msec, callback):
        self.scheduled.append((msec, callback))


class FakeTextEdit:
    """QTextEdit double: records ordered append/insert ops; monkeypatch it
    over the service module's QTextEdit attribute so no real widget (which
    would need a QApplication) is ever created."""

    def __init__(self, *args, **kwargs):
        self.ops = []
        self.read_only = False
        self.window_title = None
        self.size = None
        self.shown = 0

    def append(self, text):
        self.ops.append(("append", text))

    def insertPlainText(self, text):
        self.ops.append(("insert", text))

    def toPlainText(self):
        return "\n".join(text for _, text in self.ops)

    def setReadOnly(self, flag):
        self.read_only = flag

    def setWindowTitle(self, title):
        self.window_title = title

    def resize(self, w, h):
        self.size = (w, h)

    def show(self):
        self.shown += 1


class FakeQSettings:
    """QSettings double backed by a class-level dict so values persist across
    instances (as real QSettings does). Reset `FakeQSettings.store` per test."""

    store = {}

    def __init__(self, *args, **kwargs):
        pass

    def value(self, key, default=None, type=None):
        return self.store.get(key, default)

    def setValue(self, key, value):
        self.store[key] = value


def fresh_message_box():
    """Return a fresh QMessageBox recorder class (per-test isolation): static
    about/warning/information calls land in `cls.calls` as (kind, title, text)."""

    class RecordingMessageBox:
        calls = []
        YesRole = 0
        NoRole = 1
        RejectRole = 2

        @classmethod
        def about(cls, parent, title, text):
            cls.calls.append(("about", title, text))

        @classmethod
        def warning(cls, parent, title, text):
            cls.calls.append(("warning", title, text))

        @classmethod
        def information(cls, parent, title, text):
            cls.calls.append(("information", title, text))

    return RecordingMessageBox


def record_signal(signal):
    """Connect a recorder to a (stub or real) signal; returns the list that
    receives one args-tuple per emission."""
    received = []
    signal.connect(lambda *args: received.append(args))
    return received
