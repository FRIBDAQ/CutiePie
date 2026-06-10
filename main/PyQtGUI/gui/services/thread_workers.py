import logging

import CPyConverter as cpy

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

            # Collect removes and fetch add metadata — all on the background thread.
            # Removes MUST be emitted before adds so the GUI thread processes them in
            # the original SpecTcl order. A remove-then-add of the same name (rebind)
            # would otherwise leave the spectrum absent if adds were queued first.
            remove_bindings = []
            add_infos       = []
            for entry in (traces.get("binding") or []):
                action, name, _ = entry.split(" ")
                if action == "remove":
                    remove_bindings.append(entry)
                elif action == "add":
                    info = self._rest.listSpectrum(name)
                    if info:
                        add_infos.append((name, info[0]))
                    else:
                        logger.warning('RestWorker - listSpectrum returned empty for %s', name)

            if remove_bindings:
                self.tracesReady.emit({"binding": remove_bindings})
            for name, spec_info in add_infos:
                self.spectrumAdded.emit(name, spec_info)

        self.disconnected.emit()


class ConnectWorker(QObject):
    """Runs only the blocking CPyConverter mirror transfer off the GUI thread."""

    succeeded = pyqtSignal(object)  # shmem tuple
    failed    = pyqtSignal(str)

    def __init__(self, hostname, port, mirror, user):
        super().__init__()
        self._hostname = hostname
        self._port     = port
        self._mirror   = mirror
        self._user     = user

    @pyqtSlot()
    def run(self):
        try:
            s = cpy.CPyConverter().Update(
                bytes(self._hostname, encoding='utf-8'),
                bytes(self._port,     encoding='utf-8'),
                bytes(self._mirror,   encoding='utf-8'),
                bytes(self._user,     encoding='utf-8'),
            )
            self.succeeded.emit(s)
        except Exception as exc:
            self.failed.emit(str(exc))


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
                self._stop.wait(self._interval)
                continue
            if self._stop.wait(self._interval):
                break
            try:
                self.updateTriggered.emit()
            except ValueError:
                logger.debug('AutoUpdateWorker.run - ValueError', exc_info=True)
