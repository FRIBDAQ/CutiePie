"""An error nobody caught becomes a report instead of ending the program.
PyQt aborts when an exception leaves a slot unless sys.excepthook has been
replaced; these tests cover the replacement and the text it produces. The
window itself needs a running Qt and is not built here."""

import ast
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

from services import error_report as er

GUI_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "gui")


def raised(make=lambda: ValueError("bad value")):
    try:
        def inner():
            raise make()
        inner()
    except BaseException:
        return sys.exc_info()


# ------------------------------------------------------------ the report

def field(lines, name):
    line, = [l for l in lines if l.startswith(name + ":")]
    return line.split(":", 1)[1].strip()


def test_report_names_the_version_the_error_and_where_it_happened():
    text = er.build_report(*raised(), version="v1.6-011", thread_name="MainThread",
                           when=time.strptime("2026-11-25 10:11:12", "%Y-%m-%d %H:%M:%S"))
    lines = text.splitlines()
    assert lines[0] == "CutiePie v1.6-011 - error report"
    assert field(lines, "Time") == "2026-11-25 10:11:12"
    assert field(lines, "Thread") == "MainThread"
    assert any(l.startswith("Python:") for l in lines)
    assert any(l.startswith("Platform:") for l in lines)
    assert "Traceback (most recent call last):" in lines
    assert lines[-1] == "ValueError: bad value"
    assert any("in inner" in l for l in lines)          # the frames are there


def test_report_without_a_version_still_has_a_title():
    assert er.build_report(*raised()).splitlines()[0] == "CutiePie - error report"


def test_report_takes_the_thread_it_is_built_on():
    out = []
    t = threading.Thread(target=lambda: out.append(er.build_report(*raised())),
                         name="rest-poller")
    t.start(); t.join()
    assert field(out[0].splitlines(), "Thread") == "rest-poller"


def test_report_keeps_the_cause_of_a_chained_error():
    def make():
        try:
            {}["missing"]
        except KeyError as e:
            err = RuntimeError("could not draw")
            err.__cause__ = e
            return err
    text = er.build_report(*raised(make))
    assert "KeyError: 'missing'" in text and text.endswith("RuntimeError: could not draw")


# ------------------------------------------------------------ the hook

class Recorder:
    def __init__(self):
        self.delivered, self.printed, self.interrupts = [], [], 0

    def hook(self, **kw):
        return er.ErrorHook(lambda text, key: self.delivered.append((text, key)),
                            fallback=lambda *a: self.printed.append(a[0]),
                            on_interrupt=self._interrupt, **kw)

    def _interrupt(self):
        self.interrupts += 1


def test_hook_delivers_a_report_and_still_prints_the_traceback():
    r = Recorder()
    r.hook(version="v9")(*raised())
    assert r.printed == [ValueError]                     # the terminal, as before
    (text, key), = r.delivered
    assert text.startswith("CutiePie v9 - error report")
    assert key.startswith("Traceback") and key.endswith("ValueError: bad value")
    assert text.endswith(key)


def test_hook_key_is_the_same_for_the_same_error_at_another_time():
    r = Recorder()
    hook = r.hook()
    info = raised()
    hook(*info); hook(*info)
    assert r.delivered[0][1] == r.delivered[1][1]


def test_ctrl_c_is_not_reported_and_ends_the_program():
    r = Recorder()
    r.hook()(*raised(KeyboardInterrupt))
    assert r.delivered == [] and r.interrupts == 1
    assert r.printed == [KeyboardInterrupt]


def test_hook_does_not_raise_when_showing_the_report_fails():
    printed = []

    def broken(text, key):
        raise RuntimeError("no window")

    er.ErrorHook(broken, fallback=lambda *a: printed.append(a[0]))(*raised())
    assert printed == [ValueError, RuntimeError]         # both reach the terminal


# ------------------------------------------------------------ the list

class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_a_new_error_brings_the_window_up():
    log = er.ReportLog(clock=Clock())
    assert log.add("report A", "A") is True
    assert log.count() == 1 and log.text() == "report A"


def test_a_different_error_is_listed_above_the_older_one():
    clock = Clock()
    log = er.ReportLog(clock=clock)
    log.add("report A", "A")
    assert log.add("report B", "B") is True              # even right after A
    assert log.count() == 2
    assert log.text().index("report B") < log.text().index("report A")
    assert er.SEPARATOR in log.text()


def test_the_same_error_again_is_counted_not_listed_twice():
    clock = Clock()
    log = er.ReportLog(clock=clock)
    log.add("report A at 10:00", "A")
    clock.t += 60
    log.add("report A at 10:01", "A")
    assert log.count() == 1
    assert "occurred 2 times" in log.text()
    assert "report A at 10:01" in log.text() and "10:00" not in log.text()


def test_a_storm_of_one_error_does_not_reopen_the_window():
    """A timer failing many times a second: the window comes up once; closing
    it must not be undone by the next tick."""
    clock = Clock()
    log = er.ReportLog(storm_seconds=2.0, clock=clock)
    assert log.add("r", "A") is True
    for _ in range(50):
        clock.t += 0.08
        assert log.add("r", "A") is False
    assert "occurred 51 times" in log.text()


def test_the_same_error_after_a_pause_brings_the_window_up_again():
    """Closing the report and trying the same action again must show it again."""
    clock = Clock()
    log = er.ReportLog(storm_seconds=2.0, clock=clock)
    log.add("r", "A")
    clock.t += 5
    assert log.add("r", "A") is True


def test_only_the_newest_reports_are_kept():
    log = er.ReportLog(max_reports=3, clock=Clock())
    for i in range(10):
        log.add(f"report {i}", f"key {i}")
    assert log.count() == 3
    assert [l for l in log.text().splitlines() if l.startswith("report")] == \
        ["report 9", "report 8", "report 7"]


# ------------------------------------------- mouse and key handlers on plots

def agg_canvas():
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    return FigureCanvasAgg(Figure())


def test_an_error_in_a_plot_handler_goes_to_the_report(monkeypatch):
    from matplotlib.backend_bases import KeyEvent, MouseEvent
    seen = []
    monkeypatch.setattr(sys, "excepthook", lambda t, e, tb: seen.append((t, str(e), tb)))
    canvas = agg_canvas()
    er.report_canvas_errors(canvas)

    def on_press(event):
        raise IndexError("no such pad")

    def on_key(event):
        raise KeyError("z")

    canvas.mpl_connect("button_press_event", on_press)
    canvas.mpl_connect("key_press_event", on_key)
    canvas.callbacks.process("button_press_event",
                             MouseEvent("button_press_event", canvas, 10, 10, button=1))
    canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, "z"))

    assert [(t, m) for t, m, _ in seen] == [(IndexError, "no such pad"), (KeyError, "'z'")]
    assert all(tb is not None for _, _, tb in seen)      # with the frames to report


def test_the_other_handlers_of_the_event_still_run(monkeypatch):
    from matplotlib.backend_bases import MouseEvent
    monkeypatch.setattr(sys, "excepthook", lambda *a: None)
    canvas = agg_canvas()
    er.report_canvas_errors(canvas)
    ran = []

    def broken(event):
        raise ValueError("first handler")

    canvas.mpl_connect("button_press_event", broken)
    canvas.mpl_connect("button_press_event", lambda event: ran.append(event.button))
    canvas.callbacks.process("button_press_event",
                             MouseEvent("button_press_event", canvas, 1, 1, button=3))
    assert ran == [3]


@pytest.mark.parametrize("module, cls", [("PlotGUI.py", "Plot")])
def test_every_plot_canvas_reports_its_handler_errors(module, cls):
    """The spectrum tabs are the only canvas there is."""
    with open(os.path.join(GUI_DIR, module), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    init = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    made = [n.lineno for n in ast.walk(init) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "FigureCanvas"]
    wired = [n for n in ast.walk(init) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "report_canvas_errors"]
    assert len(made) == 1 and len(wired) == 1
    assert wired[0].lineno > made[0]
    arg = wired[0].args[0]
    assert isinstance(arg, ast.Attribute) and arg.attr == "canvas"


def test_no_other_module_builds_a_canvas():
    """A canvas made anywhere else would go back to printing its errors."""
    makers = set()
    for root, _dirs, files in os.walk(GUI_DIR):
        for name in files:
            if not name.endswith(".py"):
                continue
            with open(os.path.join(root, name), encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
            for n in ast.walk(tree):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                        and n.func.id.startswith("FigureCanvas"):
                    makers.add(name)
    assert makers == {"PlotGUI.py"}


# ------------------------------------------------------------ the wiring

def test_the_report_module_does_not_need_qt_to_import():
    with open(os.path.join(GUI_DIR, "services", "error_report.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [a.name for n in top if isinstance(n, ast.Import) for a in n.names] + \
            [n.module for n in top if isinstance(n, ast.ImportFrom)]
    assert not [n for n in names if n and n.startswith("PyQt5")]


def test_the_main_window_installs_the_reporter_before_anything_else():
    """In the window, not in Main.py: Main.py is a skeleton users copy, and an
    old copy must get the reporter too. First, so that a failure while the
    window is being built is reported as well."""
    with open(os.path.join(GUI_DIR, "GUI.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MainWindow")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    calls = [n.value.func for n in init.body
             if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    names = [f.id if isinstance(f, ast.Name) else f.attr for f in calls]
    assert names[0] == "__init__"                        # super().__init__()
    assert names[1] == "install_error_reporter"


def test_the_window_shows_errors_from_worker_threads_on_the_gui_thread():
    """The hook runs on the thread the error happened on, where no widget may
    be touched; the report has to cross over as a signal."""
    with open(os.path.join(GUI_DIR, "ErrorReportGUI.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ErrorReporter")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    hook = next(n for n in ast.walk(init) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "ErrorHook")
    deliver = hook.args[0]
    assert isinstance(deliver, ast.Attribute) and deliver.attr == "emit"
    assert deliver.value.attr == "_arrived"
