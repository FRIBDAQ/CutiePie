import logging

import CPyConverter as cpy
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

logger = logging.getLogger(__name__)


def _get_last_digit_param(parameter_name):
    parts = parameter_name.split(".")
    if not any(part.isdigit() for part in parts):
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


class RestWorker(QObject):
    """Polls SpecTcl REST traces on a QThread. Emits signals; never touches GUI directly."""

    connected     = pyqtSignal()
    disconnected  = pyqtSignal()
    tracesReady   = pyqtSignal(dict)       # "remove" binding events only
    spectrumAdded = pyqtSignal(str, dict)  # name, spectrum data kwargs

    def __init__(self, rest, retention, stop_event, hostname, port, mirror, user):
        super().__init__()
        self._rest      = rest
        self._retention = retention
        self._stop      = stop_event
        self._hostname  = hostname
        self._port      = port
        self._mirror    = mirror
        self._user      = user

    @pyqtSlot()
    def run(self):
        token = self._rest.startTraces(self._retention)
        if not token:
            self.disconnected.emit()
            return
        self._stop.clear()
        self.connected.emit()
        while not self._stop.is_set():
            if self._stop.wait(self._retention / 2):
                break
            traces = self._rest.pollTraces(token)
            if not traces:
                break
            remove_bindings = []
            for entry in (traces.get("binding") or []):
                action, name, _ = entry.split(" ")
                if action == "add":
                    self._process_add(name)
                elif action == "remove":
                    remove_bindings.append(entry)
            if remove_bindings:
                self.tracesReady.emit({"binding": remove_bindings})
        self.disconnected.emit()

    def _process_add(self, name):
        info = self._rest.listSpectrum(name)
        if not info:
            logger.warning('RestWorker._process_add - listSpectrum returned empty for %s', name)
            return
        try:
            s = cpy.CPyConverter().Update(
                bytes(self._hostname, encoding='utf-8'),
                bytes(self._port,     encoding='utf-8'),
                bytes(self._mirror,   encoding='utf-8'),
                bytes(self._user,     encoding='utf-8'),
            )
        except Exception:
            logger.debug('RestWorker._process_add - CPyConverter failed for %s', name, exc_info=True)
            return

        data = []
        binx = info[0]["axes"][0]["bins"]
        minx = info[0]["axes"][0]["low"]
        maxx = info[0]["axes"][0]["high"]
        spec_type = info[0]["type"]

        if "1" in spec_type or "b" in spec_type or "g1" in spec_type:
            dim  = 1
            biny = miny = maxy = None
            try:
                nameIndex = s[1].index(name)
                data = s[9][nameIndex][0:-1]
                data[0] = 0
            except Exception:
                logger.debug('RestWorker._process_add - name not in shmem for %s', name, exc_info=True)

        elif "s" in spec_type:
            dim  = 2
            biny = binx
            miny = minx
            maxy = maxx
            binx = 0
            minx = 9e+6
            maxx = 0
            for par in info[0]["parameters"]:
                ipar = _get_last_digit_param(par)
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
                logger.debug('RestWorker._process_add - name not in shmem for %s', name, exc_info=True)

        else:
            dim  = 2
            biny = info[0]["axes"][1]["bins"]
            miny = info[0]["axes"][1]["low"]
            maxy = info[0]["axes"][1]["high"]
            try:
                nameIndex = s[1].index(name)
                data = s[9][nameIndex][1:-1, 1:-1]
            except Exception:
                logger.debug('RestWorker._process_add - name not in shmem for %s', name, exc_info=True)

        self.spectrumAdded.emit(name, dict(
            dim=dim, binx=binx, minx=minx, maxx=maxx,
            biny=biny, miny=miny, maxy=maxy,
            parameters=info[0]["parameters"],
            type=spec_type, data=data,
        ))


class AutoUpdateWorker(QObject):
    """Fires a signal on the GUI thread at a configurable interval."""

    updateTriggered = pyqtSignal()

    def __init__(self, interval, stop_event, skip_event):
        super().__init__()
        self._interval = interval
        self._stop     = stop_event
        self._skip     = skip_event

    @pyqtSlot()
    def run(self):
        while not self._stop.is_set():
            if self._skip.is_set():
                self._stop.wait(0.05)
                continue
            if self._stop.wait(self._interval):
                break
            try:
                self.updateTriggered.emit()
            except ValueError:
                logger.debug('AutoUpdateWorker.run - ValueError', exc_info=True)
