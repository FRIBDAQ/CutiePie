"""Jupyter notebook: dump the spectra, start the server, show it, stop it.

Lifted out of MainWindow (ARCH.md §7, D4). The dataframe reshape already lives
in `services/dataframe_export.py`; what is here is the process and window
lifecycle around it.

Two things are load-bearing and easy to undo by accident. The
locate-executable loop is a `while True` whose ONLY exit on the failure side is
the cancel check — remove it and the GUI spins forever rather than merely
misbehaving (the original code was worse still: it called `sys.exit(0)` and
took the GUI with it). And the notebook window is anchored on this controller,
because a parentless local `QMainWindow` is finalized by the cyclic GC as soon
as the method returns and the window silently disappears mid-session.
"""

import logging
import os
import time

from PyQt5.QtCore import QDir, QSettings
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from logger import log, setup_logging, set_logger
from notebook_process import testnotebook, startnotebook, stopnotebook
from services.dataframe_export import export_spectrum_csv
from WebWindow import WebWindow

SETTING_BASEDIR = "workdir"
SETTING_EXECUTABLE = "exec"
DEBUG = False


class JupyterController:

    def __init__(self, peak_tab, get_store_dict, get_statistics,
                 qt_logger_factory, parent_widget=None, logger=None):
        self._peak            = peak_tab          # extraPopup.peak
        self._get_store_dict  = get_store_dict
        self._get_statistics  = get_statistics
        self._qt_logger       = qt_logger_factory  # (view) -> QtLogger
        self._parent_widget   = parent_widget
        self.logger           = logger or logging.getLogger(__name__)
        self.jupyterView      = None

    # Create dataframe for Jupyter and web
    def createDf(self):
        self.logger.info('createDf')
        try:
            export_spectrum_csv(self._get_store_dict(),
                                self._peak.jup_df_filename.text(),
                                statistics_fetcher=self._get_statistics().get)
        except Exception:
            # Export is best-effort: a failure here must not crash the GUI or
            # block jupyterStart (the notebook can still open). But it must not
            # be silent either — otherwise the notebook loads stale/missing data
            # with no clue why. Log the traceback instead of swallowing it.
            self.logger.exception('createDf - spectrum export failed')

    def jupyterStop(self):
        self.logger.info('jupyterStop')
        # stop the notebook process
        log("Sending interrupt signal to jupyter-notebook")
        self._peak.jup_start.setEnabled(True)
        self._peak.jup_stop.setEnabled(False)
        self._peak.jup_start.setStyleSheet("background-color:#3CB371;")
        self._peak.jup_stop.setStyleSheet("")
        if self.jupyterView is not None:
            self.jupyterView.close()
            self.jupyterView = None
        stopnotebook()

    def jupyterStart(self):
        self.logger.info('jupyterStart')
        # dump df to gzip
        self.createDf()
        #starting jupyter server
        s = QSettings()
        execname = s.value(SETTING_EXECUTABLE, "jupyter-notebook")
        if not testnotebook(execname):
            while True:
                QMessageBox.information(None, "Error", "It appears that Jupyter Notebook isn't where it usually is. " +
                                        "Ensure you've installed Jupyter correctly and then press Ok to " +
                                        "find the executable 'jupyter-notebook'", QMessageBox.Ok)
                if testnotebook(execname):
                    break
                path, _ = QFileDialog.getOpenFileName(None, "Find jupyter-notebook executable", QDir.homePath())
                if not path:
                    # user cancelled: abort starting Jupyter, keep the GUI alive
                    # (the old tuple-truthiness check made Cancel unreachable,
                    # and the cancel path called sys.exit(0) — killing the GUI).
                    # This is also the loop's only exit on the failure side.
                    self.logger.warning('jupyterStart - jupyter-notebook not located; start aborted by user')
                    return
                execname = path
                if testnotebook(execname):
                    log("Jupyter found at %s" % execname)
                    #save setting
                    s.setValue(SETTING_EXECUTABLE, execname)
                    break

        # setup logging
        # try to write to a log file, or redirect to stdout if debugging
        logname = "JupyterQtPy-"+time.strftime("%Y%m%d-%H%M%S")+".log"
        logfile = os.path.join(str(QDir.currentPath()), ".JupyterQtPy", logname)
        if not os.path.isdir(os.path.dirname(logfile)):
            os.mkdir(os.path.dirname(logfile))
            try:
                if DEBUG:
                    raise IOError()  # force logging to console
                setup_logging(logfile)
            except IOError:
                # no writable directory, log to console
                setup_logging(None)

        # workdir
        directory = s.value(SETTING_BASEDIR, QDir.currentPath())

        # setting window — anchored on the controller: a parentless local
        # QMainWindow is finalized by the cyclic GC after this method returns
        # (the WebWindow<->CustomWebView reference cycle is its only holder)
        # and the window silently disappears mid-session
        self.jupyterView = view = WebWindow(None, None)
        view.setWindowTitle("Jupyter CutiePie: %s" % directory)
        # logging on docked console
        qtlogger = self._qt_logger(view)
        qtlogger.newlog.connect(view.loggerdock.log)
        set_logger(lambda message: qtlogger.newlog.emit(message))

        log("Setting home directory --> "+str(directory))

        # start the notebook process
        try:
            webaddr = startnotebook(execname, directory=directory)
        except (RuntimeError, ValueError) as e:
            self.logger.error('jupyterStart - notebook failed to start: %s', e)
            view.close()
            self.jupyterView = None
            setup_logging(logfile)
            QMessageBox.warning(self._parent_widget, "Jupyter",
                                "The Jupyter notebook server failed to start:\n%s" % e)
            return
        view.loadmain(webaddr)

        # resume regular logging
        setup_logging(logfile)

        self._peak.jup_start.setEnabled(False)
        self._peak.jup_stop.setEnabled(True)
        self._peak.jup_start.setStyleSheet("")
        self._peak.jup_stop.setStyleSheet("background-color:#DC143C;")
