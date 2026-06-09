import logging

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

logger = logging.getLogger(__name__)


class RestWorker(QObject):
    """Polls SpecTcl REST traces on a QThread. Emits signals; never touches GUI directly."""

    connected     = pyqtSignal()
    disconnected  = pyqtSignal()
    tracesReady   = pyqtSignal(dict)       # "remove" binding events only
    spectrumAdded = pyqtSignal(str, dict)  # name, raw REST spectrum info dict

    def __init__(self, rest, retention, stop_event):
        super().__init__()
        self._rest      = rest
        self._retention = retention
        self._stop      = stop_event

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
        """Fetch REST metadata for a newly bound spectrum (background thread).
        CPyConverter is intentionally NOT called here — it must run on the GUI thread."""
        info = self._rest.listSpectrum(name)
        if not info:
            logger.warning('RestWorker._process_add - listSpectrum returned empty for %s', name)
            return
        self.spectrumAdded.emit(name, info[0])


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
