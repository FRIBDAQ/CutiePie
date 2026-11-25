"""The window that shows an error nobody caught, with the text to copy into a
bug report. See services/error_report.py for why and for the report itself."""

import sys

from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QVBoxLayout,
)

from services.error_report import ErrorHook, ReportLog


class ErrorReportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CutiePie - error report")
        self.intro = QLabel()
        self.intro.setWordWrap(True)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.text.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.copy_button = QPushButton("Copy to clipboard")
        self.copy_button.clicked.connect(self.copyReport)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)

        buttons = QHBoxLayout()
        buttons.addWidget(self.copy_button)
        buttons.addStretch(1)
        buttons.addWidget(self.close_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.intro)
        layout.addWidget(self.text)
        layout.addLayout(buttons)
        self.resize(860, 520)

    def setReport(self, text, count):
        what = "an error" if count == 1 else f"{count} different errors"
        self.intro.setText(
            f"CutiePie ran into {what}. The action that caused it did not "
            "finish, but the program is still running. To report it, copy the "
            "text below and send it along with what you were doing.")
        self.text.setPlainText(text)
        self.copy_button.setText("Copy to clipboard")

    def copyReport(self):
        QApplication.clipboard().setText(self.text.toPlainText())
        self.copy_button.setText("Copied")


class ErrorReporter(QObject):
    """Owns the hook and the window. An error on a worker thread is carried
    to the GUI thread by the signal before any widget is touched."""

    _arrived = pyqtSignal(str, str)

    def __init__(self, version=""):
        super().__init__()
        self._log = ReportLog()
        self._dialog = None
        self._arrived.connect(self._show)
        self.hook = ErrorHook(self._arrived.emit, version,
                              on_interrupt=lambda: QApplication.exit(130))

    def install(self):
        sys.excepthook = self.hook
        return self

    def _show(self, text, key):
        bring_up = self._log.add(text, key)
        if self._dialog is None:
            self._dialog = ErrorReportDialog()
        self._dialog.setReport(self._log.text(), self._log.count())
        if not bring_up:
            return
        if QThread.currentThread().loopLevel() == 0:
            # no event loop yet (an error while starting up): the program is
            # about to end, so hold it until the report has been read
            self._dialog.exec_()
            return
        self._dialog.show()
        self._dialog.raise_()
        self._dialog.activateWindow()


_reporter = None


def install_error_reporter(version=""):
    """Install the reporter once for the program; later calls return it."""
    global _reporter
    if _reporter is None:
        _reporter = ErrorReporter(version).install()
    return _reporter
