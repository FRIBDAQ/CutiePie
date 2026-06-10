import logging

import CPyConverter as cpy

from PyQt5.QtCore import QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QComboBox, QCompleter
from PyQt5 import QtCore

from PyREST import PyREST
from services.spectrum_store import SpectrumStore
from services.thread_workers import RestWorker, AutoUpdateWorker, ConnectWorker


class ConnectionManager(QtCore.QObject):
    """Owns REST connection lifecycle, trace polling, and auto-update thread."""

    connectionEstablished = pyqtSignal()
    spectrumRemoved       = pyqtSignal(str)
    spectrumListChanged   = pyqtSignal()
    updatePlotRequested   = pyqtSignal()

    def __init__(self, wConf, connect_config, spectra,
                 update_intervals, update_intervals_user,
                 stop_rest, stop_auto, skip_auto, logger=None):
        super().__init__()
        self._wConf              = wConf
        self._connect_config     = connect_config
        self._spectra            = spectra
        self._update_intervals      = update_intervals
        self._update_intervals_user = update_intervals_user
        self.stopRestThread         = stop_rest
        self.stopAutoUpdateThread   = stop_auto
        self.skipAutoUpdateThread   = skip_auto
        self.logger = logger or logging.getLogger(__name__)
        self._rest           = None
        self._rest_thread    = None
        self._rest_worker    = None
        self._auto_thread    = None
        self._auto_worker    = None
        self._connect_thread = None
        self._connect_worker = None
        self._pending_adds: list   = []
        self._flush_scheduled: bool = False

    # ------------------------------------------------------------------
    # Connection popup callbacks
    # ------------------------------------------------------------------

    def connectPopup(self):
        self.logger.info('callback connectPopup')
        self._connect_config.show()

    def closeConnect(self):
        self.logger.info('closeConnect callback')
        self._connect_config.close()

    def okConnect(self):
        self.logger.info('okConnect')
        self.connectShMem()
        self.closeConnect()

    # ------------------------------------------------------------------
    # REST + shared memory connection
    # ------------------------------------------------------------------

    def connectShMem(self):
        self.logger.info('connectShMem')
        try:
            hostname = str(self._connect_config.server.text())
            port     = str(self._connect_config.rest.text())
            user     = str(self._connect_config.user.text())
            mirror   = str(self._connect_config.mirror.text())
            self.logger.debug('connectShMem - host: %s -- user: %s -- RESTPort: %s -- MirrorPort: %s',
                               hostname, user, port, mirror)

            if self._rest is not None:
                try:
                    self._rest.reconfigure(hostname, port)
                except Exception:
                    self._rest = PyREST(self.logger, hostname, port)
            else:
                self._rest = PyREST(self.logger, hostname, port)

            self._wConf.connectButton.setStyleSheet("background-color:rgb(252, 48, 3);")
            self._wConf.connectButton.setText("Disconnected")

            if not self._rest.checkSpecTclREST():
                self.logger.debug('connectShMem - invalid URL for SpecTclREST')
                return

            self.logger.debug("connectShMem - could make REST request of server")
            self._stop_rest_thread()
            self.stopRestThread.clear()
            self._rest_worker = RestWorker(self._rest, 6, self.stopRestThread)
            self._rest_thread = QThread(self)
            self._rest_worker.moveToThread(self._rest_thread)
            self._rest_thread.started.connect(self._rest_worker.run)
            self._rest_worker.connected.connect(self._on_rest_connected)
            self._rest_worker.disconnected.connect(self._on_rest_disconnected)
            self._rest_worker.tracesReady.connect(self.updateFromTraces)
            self._rest_worker.spectrumAdded.connect(self._on_spectrum_added)
            self._rest_thread.start()

            # Mirror transfer is slow (full shmem copy over TCP) — run it off the GUI thread.
            self._wConf.connectButton.setStyleSheet("background-color:rgb(255, 200, 0);")
            self._wConf.connectButton.setText("Connecting to mirror…")
            self._wConf.connectButton.setEnabled(False)

            if self._connect_thread is not None:
                self._connect_thread.quit()
                self._connect_thread.wait()

            self._connect_worker = ConnectWorker(hostname, port, mirror, user)
            self._connect_thread = QThread(self)
            self._connect_worker.moveToThread(self._connect_thread)
            self._connect_thread.started.connect(self._connect_worker.run)
            self._connect_worker.succeeded.connect(self._on_connect_succeeded)
            self._connect_worker.failed.connect(self._on_connect_failed)
            self._connect_thread.start()

        except Exception:
            self.logger.exception('connectShMem - Exception')
            self._wConf.connectButton.setEnabled(True)
            raise

    @pyqtSlot(object)
    def _on_connect_succeeded(self, s):
        self.logger.debug('connectShMem - mirror transfer done, fetching spectrum list from REST')
        try:
            otherInfo = self.getSpectrumInfoFromReST()
            self.logger.debug('connectShMem - populating spectra from shmem + REST')
            for i, name in enumerate(s[1]):
                if name not in otherInfo:
                    continue
                if s[2][i] == 2:
                    data = s[9][i][1:-1, 1:-1]
                    if "s" in otherInfo[name]["type"]:
                        minx = s[4][i]
                        maxx = s[5][i] + 1
                else:
                    data = s[9][i][0:-1]
                    data[0] = 0

                if "s" in otherInfo[name]["type"]:
                    self._spectra.set(
                        name, dim=s[2][i], binx=s[3][i]-2, minx=minx, maxx=maxx,
                        biny=s[6][i]-2, miny=s[7][i], maxy=s[8][i],
                        data=data, parameters=otherInfo[name]["parameters"],
                        type=otherInfo[name]["type"],
                    )
                else:
                    self._spectra.set(
                        name, dim=s[2][i], binx=s[3][i]-2, minx=s[4][i], maxx=s[5][i],
                        biny=s[6][i]-2, miny=s[7][i], maxy=s[8][i],
                        data=data, parameters=otherInfo[name]["parameters"],
                        type=otherInfo[name]["type"],
                    )

            self.updateSpectrumList(True)
        finally:
            self._wConf.connectButton.setEnabled(True)
            self._connect_thread.quit()

    @pyqtSlot(str)
    def _on_connect_failed(self, msg):
        self.logger.error('connectShMem - mirror transfer failed: %s', msg)
        self._wConf.connectButton.setStyleSheet("background-color:rgb(252, 48, 3);")
        self._wConf.connectButton.setText("Disconnected")
        self._wConf.connectButton.setEnabled(True)
        self._connect_thread.quit()

    # ------------------------------------------------------------------
    # Trace updates from REST worker
    # ------------------------------------------------------------------

    def updateFromTraces(self, tracesDetails):
        self.logger.info('updateFromTraces - tracesDetails: %s', tracesDetails)
        for entry in (tracesDetails.get("binding") or []):
            action, name, _ = entry.split(" ")
            if action == "remove" and self._spectra.contains(name):
                self._spectra.remove(name)
                self.spectrumRemoved.emit(name)
                self.updateSpectrumList()
                self.spectrumListChanged.emit()

    @pyqtSlot(str, dict)
    def _on_spectrum_added(self, name, spec_info):
        """Queue a newly bound spectrum; flush all pending adds in one CPyConverter call."""
        if self._spectra.contains(name):
            return
        self._pending_adds.append((name, spec_info))
        if not self._flush_scheduled:
            self._flush_scheduled = True
            QTimer.singleShot(0, self._flush_spectrum_adds)

    def _flush_spectrum_adds(self):
        """Process all queued adds with a single CPyConverter.Update() call."""
        self._flush_scheduled = False
        pending = self._pending_adds[:]
        self._pending_adds.clear()
        if not pending:
            return

        hostname = self._connect_config.server.text()
        port     = self._connect_config.rest.text()
        user     = self._connect_config.user.text()
        mirror   = self._connect_config.mirror.text()
        try:
            s = cpy.CPyConverter().Update(
                bytes(hostname, encoding='utf-8'),
                bytes(port,     encoding='utf-8'),
                bytes(mirror,   encoding='utf-8'),
                bytes(user,     encoding='utf-8'),
            )
        except Exception:
            self.logger.debug('_flush_spectrum_adds - CPyConverter failed', exc_info=True)
            return

        for name, spec_info in pending:
            if self._spectra.contains(name):
                continue
            self._process_spectrum_add(name, spec_info, s)
        self.updateSpectrumList()

    def _process_spectrum_add(self, name, spec_info, s):
        """Extract and store one spectrum from an already-fetched CPyConverter result."""
        self.logger.debug('_process_spectrum_add - name: %s', name)
        data      = []
        binx      = spec_info["axes"][0]["bins"]
        minx      = spec_info["axes"][0]["low"]
        maxx      = spec_info["axes"][0]["high"]
        spec_type = spec_info["type"]

        if "1" in spec_type or "b" in spec_type or "g1" in spec_type:
            dim  = 1
            biny = miny = maxy = None
            try:
                nameIndex = s[1].index(name)
                data = s[9][nameIndex][0:-1]
                data[0] = 0
            except Exception:
                self.logger.debug('_process_spectrum_add - name not in shmem for %s', name, exc_info=True)
        elif "s" in spec_type:
            dim  = 2
            biny = binx
            miny = minx
            maxy = maxx
            binx = 0
            minx = 9e+6
            maxx = 0
            for par in spec_info["parameters"]:
                ipar = self._get_last_digit_param(par)
                if ipar is not None:
                    if ipar < minx:
                        minx = ipar
                    if ipar > maxx:
                        maxx = ipar
            maxx += 1
            binx = maxx - minx
            try:
                nameIndex = s[1].index(name)
                data = s[9][nameIndex][1:-1, 1:-1]
            except Exception:
                self.logger.debug('_process_spectrum_add - name not in shmem for %s', name, exc_info=True)
        else:
            dim  = 2
            biny = spec_info["axes"][1]["bins"]
            miny = spec_info["axes"][1]["low"]
            maxy = spec_info["axes"][1]["high"]
            try:
                nameIndex = s[1].index(name)
                data = s[9][nameIndex][1:-1, 1:-1]
            except Exception:
                self.logger.debug('_process_spectrum_add - name not in shmem for %s', name, exc_info=True)

        self._spectra.set(name, dim=dim, binx=binx, minx=minx, maxx=maxx,
                          biny=biny, miny=miny, maxy=maxy,
                          parameters=spec_info["parameters"],
                          type=spec_type, data=data)

    # ------------------------------------------------------------------
    # Spectrum list helpers
    # ------------------------------------------------------------------

    def getSpectrumInfoFromReST(self):
        self.logger.info('getSpectrumInfoFromReST')
        outDict  = {}
        inpDict  = self._rest.listSpectrum()
        bindList = self._rest.listsbind("*")
        if not isinstance(inpDict, list):
            self.logger.warning('getSpectrumInfoFromReST - listSpectrum returned non-list: %s', inpDict)
            return outDict
        if not isinstance(bindList, list):
            self.logger.warning('getSpectrumInfoFromReST - listsbind returned non-list: %s', bindList)
            return outDict
        bindings = {}
        for d in bindList:
            bindings[d["name"]] = d["binding"]
        for el in inpDict:
            if el["name"] in bindings:
                outDict[el["name"]] = {
                    "parameters": el["parameters"],
                    "type":       el["type"],
                    "binding":    bindings[el["name"]],
                }
        self.logger.info('getSpectrumInfoFromReST - return: %s', outDict)
        return outDict

    def updateSpectrumList(self, init=False):
        self.logger.debug('updateSpectrumList')
        self._wConf.histo_list.blockSignals(True)
        self._wConf.histo_list.clear()
        self._wConf.histo_list.addItems(self._spectra.all_names())
        self._wConf.histo_list.blockSignals(False)
        if init:
            self._wConf.histo_list.setEditable(True)
            self._wConf.histo_list.setInsertPolicy(QComboBox.NoInsert)
            self._wConf.histo_list.completer().setCompletionMode(QCompleter.PopupCompletion)
            self._wConf.histo_list.completer().setFilterMode(QtCore.Qt.MatchContains)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_last_digit_param(parameter_name):
        parts = parameter_name.split(".")
        if not any(part.isdigit() for part in parts):
            return None
        try:
            return int(parts[-1])
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def _stop_rest_thread(self):
        self.logger.info('_stop_rest_thread')
        self.stopRestThread.set()
        if self._rest_thread is not None:
            self._rest_thread.quit()
            self._rest_thread.wait()
            self._rest_thread = None
            self._rest_worker = None

    def _stop_auto_thread(self):
        self.logger.info('_stop_auto_thread')
        self.stopAutoUpdateThread.set()
        if self._auto_thread is not None:
            self._auto_thread.quit()
            self._auto_thread.wait()
            self._auto_thread = None
            self._auto_worker = None

    @pyqtSlot()
    def _on_rest_connected(self):
        self.logger.info('_on_rest_connected')
        self._wConf.connectButton.setStyleSheet("background-color:#bcee68;")
        self._wConf.connectButton.setText("Connected")
        self.connectionEstablished.emit()

    @pyqtSlot()
    def _on_rest_disconnected(self):
        self.logger.info('_on_rest_disconnected')
        self._wConf.connectButton.setStyleSheet("background-color:rgb(252, 48, 3);")
        self._wConf.connectButton.setText("Disconnected")
        if self._rest_thread is not None:
            self._rest_thread.quit()
            self._rest_thread.wait()
        self._rest_thread = None
        self._rest_worker = None

    # ------------------------------------------------------------------
    # Auto-update
    # ------------------------------------------------------------------

    def autoUpdateResume(self):
        self.logger.info('autoUpdateResume')
        self.skipAutoUpdateThread.clear()

    def autoUpdateStart(self):
        self.logger.info('autoUpdateStart')
        val_auto = self._wConf.autoUpdate2.currentIndex()
        updateInterval     = self._update_intervals[val_auto]
        updateIntervalUser = self._update_intervals_user[val_auto]
        try:
            self._stop_auto_thread()
            self.stopAutoUpdateThread.clear()
            self.skipAutoUpdateThread.clear()
            self._auto_worker = AutoUpdateWorker(
                updateInterval, self.stopAutoUpdateThread, self.skipAutoUpdateThread)
            self._auto_thread = QThread(self)
            self._auto_worker.moveToThread(self._auto_thread)
            self._auto_thread.started.connect(self._auto_worker.run)
            self._auto_worker.updateTriggered.connect(self.updatePlotRequested)
            self._auto_thread.start()
        except ValueError:
            self.logger.debug('autoUpdateStart - ValueError exception', exc_info=True)
