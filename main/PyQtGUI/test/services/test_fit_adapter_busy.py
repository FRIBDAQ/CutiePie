"""The fit adapter turns FitManager's busy signal into what the user sees in
the Extra window: Fit off, Abort on, and the turning ring on the Fit button
for as long as the fit runs."""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

from adapters.fit_adapter import FitAdapter


class Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for slot in self.slots:
            slot(*args)


class Button:
    def __init__(self):
        self.enabled = True
        self.clicked = Signal()

    def setEnabled(self, on):
        self.enabled = bool(on)


class Spinner:
    def __init__(self):
        self.spinning = False
        self.calls = []

    def start(self):
        self.spinning = True
        self.calls.append("start")

    def stop(self):
        self.spinning = False
        self.calls.append("stop")


def make():
    fm = types.SimpleNamespace(
        fitBusyChanged=Signal(), abortEnabledChanged=Signal(),
        fitResultsAppended=Signal(), fitLabelsTextChanged=Signal(),
        on_plot_csv_clicked=lambda: None, on_abort_clicked=lambda: None)
    pc = types.SimpleNamespace(cutoffPopupPrepared=Signal(),
                               cutoffPopupCloseRequested=Signal())
    extra = types.SimpleNamespace(fit_button=Button(), abort_button=Button(),
                                  plot_csv_button=Button(),
                                  clear_fit_csv_button=Button(),
                                  fit_spinner=Spinner())
    cutoff = types.SimpleNamespace(close=lambda: None)
    FitAdapter(fm, pc, extra, cutoff, logger=None)
    return fm, extra


def test_the_ring_turns_while_a_fit_runs():
    fm, extra = make()
    assert extra.fit_spinner.spinning is False
    fm.fitBusyChanged.emit(True)
    assert extra.fit_spinner.spinning is True
    assert extra.fit_button.enabled is False and extra.abort_button.enabled is True
    fm.fitBusyChanged.emit(False)
    assert extra.fit_spinner.spinning is False
    assert extra.fit_button.enabled is True and extra.abort_button.enabled is False
    assert extra.fit_spinner.calls == ["start", "stop"]


def test_the_ring_stops_however_the_fit_ends():
    """FitManager clears busy in a finally, so an aborted or failed fit sends
    the same signal as a finished one; the ring must not be left turning."""
    fm, extra = make()
    for _ in range(3):
        fm.fitBusyChanged.emit(True)
        fm.fitBusyChanged.emit(False)
    assert extra.fit_spinner.spinning is False
    assert extra.fit_spinner.calls == ["start", "stop"] * 3
