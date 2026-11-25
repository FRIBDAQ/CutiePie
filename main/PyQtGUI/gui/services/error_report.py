"""An error nobody caught is reported, not fatal.

PyQt ends the program (qFatal, abort) when a Python exception leaves a button
handler, a timer, an event handler or a worker thread, unless sys.excepthook
has been replaced. ErrorHook is that replacement: it turns the exception into
a text report the user can copy into a bug report and hands it on to be shown.
Everything here is Qt-free; the window is in ErrorReportGUI.py."""

import platform
import sys
import threading
import time
import traceback

SEPARATOR = "-" * 72


def _library_versions():
    """(name, version) of what a report reader needs to reproduce the error.
    A library that is missing or will not say is left out."""
    found = [("Python", platform.python_version())]
    try:
        from PyQt5.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
        found.append(("PyQt / Qt", f"{PYQT_VERSION_STR} / {QT_VERSION_STR}"))
    except Exception:
        pass
    for name in ("matplotlib", "numpy"):
        try:
            found.append((name, str(sys.modules[name].__version__)))
        except Exception:
            pass
    return found


def format_traceback(exc_type, exc, tb):
    return "".join(traceback.format_exception(exc_type, exc, tb)).rstrip("\n")


def build_report(exc_type, exc, tb, version="", thread_name=None, when=None):
    """The text of one report: what ran, where, and the traceback."""
    if thread_name is None:
        thread_name = threading.current_thread().name
    if when is None:
        when = time.localtime()
    head = [("Time", time.strftime("%Y-%m-%d %H:%M:%S", when)),
            ("Thread", thread_name)]
    head += _library_versions()
    head.append(("Platform", platform.platform()))
    width = max(len(k) for k, _ in head) + 1
    lines = ["CutiePie" + (" " + version if version else "") + " - error report"]
    lines += [f"{k + ':':<{width}} {v}" for k, v in head]
    return "\n".join(lines) + "\n\n" + format_traceback(exc_type, exc, tb)


def report_callback_error(exc):
    """Exception handler for a matplotlib callback registry: hand the error
    to sys.excepthook, as an error in a Qt handler is."""
    sys.excepthook(type(exc), exc, exc.__traceback__)


def report_canvas_errors(canvas):
    """Have errors in this canvas's mouse, key and other event handlers
    reported. matplotlib catches them itself and only prints the traceback,
    so the click or key did nothing and nobody was told."""
    canvas.callbacks.exception_handler = report_callback_error


class ReportLog:
    """The reports the window shows. The same error arriving again is counted
    on its report instead of being listed again, and only the newest reports
    are kept, so an error raised by a timer cannot fill the window or the
    memory."""

    def __init__(self, max_reports=20, storm_seconds=2.0, clock=time.monotonic):
        self._max = max_reports
        self._storm = storm_seconds
        self._clock = clock
        self._reports = []          # [text, key, count]
        self._last_time = None

    def add(self, text, key):
        """Record a report; `key` identifies the error (its traceback).
        Returns True when the window should be brought up for it, False for
        a repeat that follows the same error within `storm_seconds`: a timer
        failing many times a second must not reopen a window the user just
        closed."""
        now = self._clock()
        repeat = bool(self._reports) and self._reports[-1][1] == key
        in_storm = (repeat and self._last_time is not None
                    and now - self._last_time < self._storm)
        self._last_time = now
        if repeat:
            self._reports[-1][0] = text         # newest time stamp
            self._reports[-1][2] += 1
        else:
            self._reports.append([text, key, 1])
            del self._reports[:-self._max]
        return not in_storm

    def count(self):
        """Number of different reports held."""
        return len(self._reports)

    def text(self):
        """Everything to show and copy, newest report first."""
        parts = []
        for text, _key, count in reversed(self._reports):
            if count > 1:
                text = f"(this error occurred {count} times; the time below is the latest)\n{text}"
            parts.append(text)
        return ("\n\n" + SEPARATOR + "\n\n").join(parts)


class ErrorHook:
    """Callable for sys.excepthook. `deliver(text, key)` receives the report;
    it is called on whatever thread the error happened on."""

    def __init__(self, deliver, version="", on_interrupt=None, fallback=None):
        self._deliver = deliver
        self._version = version
        self._on_interrupt = on_interrupt
        self._fallback = fallback or sys.__excepthook__

    def __call__(self, exc_type, exc, tb):
        # the terminal keeps getting the traceback, as before
        self._fallback(exc_type, exc, tb)
        if issubclass(exc_type, KeyboardInterrupt):
            # Ctrl+C in the terminal still ends the program
            if self._on_interrupt is not None:
                self._on_interrupt()
            return
        try:
            self._deliver(build_report(exc_type, exc, tb, self._version),
                          format_traceback(exc_type, exc, tb))
        except Exception:
            # reporting must never be the thing that fails
            self._fallback(*sys.exc_info())
