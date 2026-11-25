"""Import CSV into a spectrum (Extra tab): ask for the file and the spectrum
name, then have SpecTcl create and fill the spectrum. The work itself is in
services/csv_import.py; this is the dialogs around it."""

import logging
import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QFileDialog, QInputDialog,
                             QLineEdit, QMessageBox)

from services import csv_import

TITLE = "Import CSV into a spectrum"


class CsvImportController:

    def __init__(self, get_rest, parent_widget=None, logger=None):
        self._get_rest      = get_rest        # () -> PyREST, None before a connection
        self._parent_widget = parent_widget
        self.logger         = logger or logging.getLogger(__name__)

    def import_csv(self, csv_format="InterSpec"):
        """`csv_format` is the program the file comes from, as chosen with the
        radio buttons: "InterSpec" or "SpecTcl" (plain x,y rows)."""
        self.logger.info('import_csv - %s', csv_format)
        rest = self._get_rest()
        if rest is None:
            QMessageBox.warning(self._parent_widget, TITLE,
                                "Connect to SpecTcl first.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self._parent_widget, "Select %s CSV to import" % csv_format, "",
            "CSV files (*.csv *.txt);;All files (*)")
        if not path:
            return

        try:
            x, y = csv_import.read_csv(path, csv_format)
            _low, _high, _bins, even = csv_import.axis_for(x)
        except Exception as e:
            self.logger.warning('import_csv - cannot read %s as %s CSV',
                                path, csv_format, exc_info=True)
            # numpy lists every bad row; the first few say what is wrong
            lines = (str(e) or type(e).__name__).splitlines()
            shown = lines[:6]
            if len(lines) > len(shown):
                shown.append("... (%d more lines)" % (len(lines) - len(shown)))
            QMessageBox.warning(self._parent_widget,
                                "Cannot read %s CSV" % csv_format,
                                os.path.basename(path) + "\n\n" + "\n".join(shown))
            return

        if not even:
            answer = QMessageBox.question(
                self._parent_widget, TITLE,
                "The x values in %s are not evenly spaced, and a SpecTcl "
                "axis is. The spectrum would run evenly from the first x "
                "value to the last, so positions in between are shifted.\n\n"
                "Import it anyway?" % os.path.basename(path))
            if answer != QMessageBox.Yes:
                return

        try:
            name, replace = self._ask_name(rest, csv_import.default_spectrum_name(path))
            if name is None:
                return
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                low, high, bins, parameter = csv_import.import_spectrum(
                    rest, name, x, y, replace=replace)
            finally:
                QApplication.restoreOverrideCursor()
        except csv_import.CsvImportError as e:
            self.logger.warning('import_csv - %s', e)
            QMessageBox.warning(self._parent_widget, "Cannot import the CSV", str(e))
            return
        self.logger.info('import_csv - %s -> spectrum %s on parameter %s, %d bins from %g to %g',
                         path, name, parameter, bins, low, high)

    def _ask_name(self, rest, default):
        """Ask for the spectrum name until it is usable: (name, replace), or
        (None, False) when the user cancels. `replace` is True when a
        spectrum of that name exists and the user agreed to replace it."""
        name = default
        while True:
            name, ok = QInputDialog.getText(
                self._parent_widget, TITLE, "Spectrum name:",
                QLineEdit.Normal, name)
            if not ok:
                return None, False
            name = name.strip()
            problem = csv_import.spectrum_name_problem(name)
            if problem:
                QMessageBox.warning(self._parent_widget, TITLE, problem)
                continue
            if not csv_import.spectrum_exists(rest, name):
                return name, False
            answer = QMessageBox.question(
                self._parent_widget, TITLE,
                "SpecTcl already has a spectrum named %s.\n\n"
                "Replace it?" % name)
            if answer == QMessageBox.Yes:
                return name, True
