"""Import CSV into a spectrum, the dialogs: file, then spectrum name (the
file name by default), then the import; and every way out of it ends in a
popup or quietly, never in an exception leaving the button's slot."""

import importlib
import logging
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import qt_stubs
from test_csv_import import FakeSpecTcl, interspec_file, plain_file


class Dialogs:
    """What the user does: the file picked, the names typed, the answers to
    questions; and what the program showed."""

    def __init__(self, path="", names=(), answers=()):
        self.path = path
        self.names = list(names)        # None = Cancel in the name dialog
        self.answers = list(answers)    # True = Yes
        self.file_titles = []
        self.name_defaults = []
        self.warnings = []
        self.questions = []
        self.cursor = 0

    # QFileDialog
    def getOpenFileName(self, parent, title, *a):
        self.file_titles.append(title)
        return self.path, ""

    # QInputDialog
    def getText(self, parent, title, label, mode, default):
        self.name_defaults.append(default)
        name = self.names.pop(0)
        return ("", False) if name is None else (name, True)

    # QMessageBox
    Yes, No = 1, 0

    def warning(self, parent, title, text):
        self.warnings.append((title, text))

    def question(self, parent, title, text):
        self.questions.append(text)
        return self.Yes if self.answers.pop(0) else self.No

    # QApplication
    def setOverrideCursor(self, cursor):
        self.cursor += 1

    def restoreOverrideCursor(self):
        self.cursor -= 1


@pytest.fixture
def make(monkeypatch):
    qt_stubs.install_missing_runtime_stubs()
    module = importlib.import_module("controllers.csv_import_controller")

    def build(dialogs, tcl="new"):
        rest = FakeSpecTcl() if tcl == "new" else tcl
        for name in ("QFileDialog", "QInputDialog", "QMessageBox", "QApplication"):
            monkeypatch.setattr(module, name, dialogs)
        monkeypatch.setattr(module, "QLineEdit", types.SimpleNamespace(Normal=0))
        monkeypatch.setattr(module, "Qt", types.SimpleNamespace(WaitCursor=0))
        controller = module.CsvImportController(
            get_rest=lambda: rest, logger=logging.getLogger("test.csvimport"))
        return controller, rest
    return build


def test_file_then_name_then_the_spectrum_is_in_spectcl(make, tmp_path):
    path = interspec_file(tmp_path, name="run12.csv")
    dialogs = Dialogs(path, names=["run12"])
    controller, tcl = make(dialogs)
    controller.import_csv("InterSpec")

    assert dialogs.file_titles == ["Select InterSpec CSV to import"]
    assert dialogs.name_defaults == ["run12"]           # the file name, no extension
    assert tcl.spectra["run12"]["counts"] == [0, 1, 13, 63, 2]
    assert tcl.bound == ["run12"]
    assert dialogs.warnings == [] and dialogs.questions == []
    assert dialogs.cursor == 0                          # busy cursor put back


def test_the_typed_name_is_the_one_used(make, tmp_path):
    dialogs = Dialogs(interspec_file(tmp_path), names=["  Th228.source  "])
    controller, tcl = make(dialogs)
    controller.import_csv("InterSpec")
    assert list(tcl.spectra) == ["Th228.source"]


def test_plain_rows_are_imported_as_spectcl(make, tmp_path):
    dialogs = Dialogs(plain_file(tmp_path, "0,5\n1,6\n2,7\n", "xy.csv"), names=["xy"])
    controller, tcl = make(dialogs)
    controller.import_csv("SpecTcl")
    assert tcl.spectra["xy"]["counts"] == [5, 6, 7]
    assert (tcl.spectra["xy"]["low"], tcl.spectra["xy"]["high"]) == (-0.5, 2.5)


def test_cancelling_the_file_dialog_does_nothing(make):
    dialogs = Dialogs("")
    controller, tcl = make(dialogs)
    controller.import_csv("InterSpec")
    assert tcl.scripts == [] and dialogs.name_defaults == [] and dialogs.warnings == []


def test_cancelling_the_name_dialog_does_nothing(make, tmp_path):
    dialogs = Dialogs(interspec_file(tmp_path), names=[None])
    controller, tcl = make(dialogs)
    controller.import_csv("InterSpec")
    assert tcl.spectra == {} and dialogs.warnings == []


def test_a_file_that_cannot_be_read_is_reported_before_the_name_is_asked(make, tmp_path):
    path = plain_file(tmp_path, "0,0\n1,1\n", "plain.csv")
    dialogs = Dialogs(path)
    controller, tcl = make(dialogs)
    controller.import_csv("InterSpec")                  # no InterSpec table in it
    assert len(dialogs.warnings) == 1
    title, text = dialogs.warnings[0]
    assert title == "Cannot read InterSpec CSV" and text.startswith("plain.csv\n\n")
    assert dialogs.name_defaults == [] and tcl.scripts == []


def test_a_long_reader_error_is_cut_to_its_first_lines(make, tmp_path):
    dialogs = Dialogs(interspec_file(tmp_path, counts=range(40)))
    controller, tcl = make(dialogs)
    controller.import_csv("SpecTcl")                    # numpy lists every bad row
    text = dialogs.warnings[0][1]
    assert "more lines)" in text and len(text.splitlines()) <= 9


def test_a_name_that_cannot_be_used_is_asked_again(make, tmp_path):
    dialogs = Dialogs(interspec_file(tmp_path), names=["two words", "a[1]", "good"])
    controller, tcl = make(dialogs)
    controller.import_csv("InterSpec")
    assert len(dialogs.warnings) == 2
    assert dialogs.name_defaults == ["spec", "two words", "a[1]"]   # what was typed stays
    assert list(tcl.spectra) == ["good"]


def test_an_existing_spectrum_is_replaced_only_after_a_yes(make, tmp_path):
    tcl = FakeSpecTcl()
    tcl.parameters["old"] = (0, 1, 1)
    tcl.script("spectrum {spec} 1 {old} {{0 4 4}}")
    old = tcl.spectra["spec"]

    # No: the name is asked again, and Cancel there leaves everything alone
    dialogs = Dialogs(interspec_file(tmp_path), names=["spec", None], answers=[False])
    controller, _ = make(dialogs, tcl)
    controller.import_csv("InterSpec")
    assert "already has a spectrum named spec" in dialogs.questions[0]
    assert tcl.spectra["spec"] is old

    # Yes: replaced
    dialogs = Dialogs(interspec_file(tmp_path), names=["spec"], answers=[True])
    controller, _ = make(dialogs, tcl)
    controller.import_csv("InterSpec")
    assert tcl.spectra["spec"]["counts"] == [0, 1, 13, 63, 2]


def test_uneven_x_values_are_imported_only_after_a_yes(make, tmp_path):
    rows = "".join("%g,%d\n" % (2.0 * i + 0.01 * i * i, i) for i in range(200))
    path = plain_file(tmp_path, rows, "curved.csv")

    dialogs = Dialogs(path, answers=[False])
    controller, tcl = make(dialogs)
    controller.import_csv("SpecTcl")
    assert "not evenly spaced" in dialogs.questions[0]
    assert dialogs.name_defaults == [] and tcl.scripts == []

    dialogs = Dialogs(path, names=["curved"], answers=[True])
    controller, tcl = make(dialogs)
    controller.import_csv("SpecTcl")
    assert tcl.spectra["curved"]["bins"] == 200


def test_a_refusal_by_spectcl_is_shown_with_its_reason(make, tmp_path):
    dialogs = Dialogs(interspec_file(tmp_path), names=["spec"])
    tcl = FakeSpecTcl()
    tcl.fail = "sbind"
    controller, _ = make(dialogs, tcl)
    controller.import_csv("InterSpec")
    title, text = dialogs.warnings[0]
    assert title == "Cannot import the CSV" and "it broke" in text
    assert tcl.spectra == {} and dialogs.cursor == 0


def test_spectcl_not_answering_is_shown_too(make, tmp_path):
    dialogs = Dialogs(interspec_file(tmp_path), names=["spec"])
    tcl = FakeSpecTcl()
    tcl.dead = True
    controller, _ = make(dialogs, tcl)
    controller.import_csv("InterSpec")
    assert dialogs.warnings[0][0] == "Cannot import the CSV"
    assert "No usable answer" in dialogs.warnings[0][1]


def test_without_a_connection_the_user_is_told_before_picking_a_file(make):
    dialogs = Dialogs("/some/file.csv")
    controller, _ = make(dialogs, tcl=None)
    controller.import_csv("InterSpec")
    assert dialogs.warnings == [("Import CSV into a spectrum", "Connect to SpecTcl first.")]
    assert dialogs.file_titles == []
