"""Plot CSV hands FitManager the program chosen with the radio buttons above
it. FitManager reads no widgets, so the adapter turns the buttons into the
plain name."""

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
        self.clicked = Signal()

    def setEnabled(self, on):
        pass


class Radio:
    def __init__(self, checked=False):
        self.checked = checked

    def isChecked(self):
        return self.checked


def make():
    asked = []
    cleared = []
    fm = types.SimpleNamespace(
        fitBusyChanged=Signal(), abortEnabledChanged=Signal(),
        fitResultsAppended=Signal(), fitLabelsTextChanged=Signal(),
        on_plot_csv_clicked=lambda *a: asked.append(a),
        on_clear_csv_fit_clicked=lambda *a: cleared.append(a),
        on_abort_clicked=lambda: None)
    pc = types.SimpleNamespace(cutoffPopupPrepared=Signal(),
                               cutoffPopupCloseRequested=Signal())
    extra = types.SimpleNamespace(fit_button=Button(), abort_button=Button(),
                                  plot_csv_button=Button(),
                                  clear_fit_csv_button=Button(),
                                  csv_interspec_radio=Radio())
    FitAdapter(fm, pc, extra, types.SimpleNamespace(close=lambda: None), logger=None)
    extra.cleared = cleared
    return extra, asked


def test_plot_csv_passes_the_chosen_program():
    extra, asked = make()
    extra.plot_csv_button.clicked.emit()
    extra.csv_interspec_radio.checked = True
    extra.plot_csv_button.clicked.emit()
    assert asked == [("SpecTcl",), ("InterSpec",)]


def test_clear_fit_csv_asks_the_fit_manager():
    extra, _asked = make()
    extra.clear_fit_csv_button.clicked.emit()
    assert extra.cleared == [()]
