import logging

import CPyConverter as cpy

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

logger = logging.getLogger(__name__)


def parse_binding_entry(entry):
    """Parse a binding trace entry into (action, name, binding_id).

    Entries are Tcl lists rendered as strings: ``action name bindingId``.
    Tcl braces elements that contain spaces (``add {my spec} 5``), so a plain
    ``split(" ")`` 3-tuple unpack mis-parses or raises on such names. The
    action is always the first token and the binding id the last; the name is
    everything in between, with one surrounding brace pair stripped.

    Raises ValueError for entries with fewer than three tokens.
    """
    parts = entry.split(" ")
    if len(parts) < 3:
        raise ValueError(f"malformed binding trace entry: {entry!r}")
    name = " ".join(parts[1:-1])
    if name.startswith("{") and name.endswith("}"):
        name = name[1:-1]
    return parts[0], name, parts[-1]


_GLOB_METACHARACTERS = "*?[]\\"


def lookup_spectrum_info(rest, name):
    """Return the REST info dict for exactly `name`, or None.

    ``listSpectrum(name)`` sends the name as the REST ``filter`` field, which
    SpecTcl matches as a Tcl glob pattern — a name containing ``*``/``?``/``[``
    can match a *different* spectrum, and blindly taking ``info[0]`` would
    record the wrong axes/type under this name (BUGS.md B1). Never trust the
    pattern match: select by exact name, and when the pattern lookup yields no
    exact hit for a metacharacter-bearing name, fall back to listing all
    spectra and matching literally. Plain names keep today's single-request
    behavior exactly.
    """
    candidates = rest.listSpectrum(name)
    exact = [d for d in candidates
             if isinstance(d, dict) and d.get("name") == name]
    if not exact and any(c in name for c in _GLOB_METACHARACTERS):
        exact = [d for d in rest.listSpectrum()
                 if isinstance(d, dict) and d.get("name") == name]
    return exact[0] if exact else None


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
        try:
            token = self._rest.startTraces(self._retention)
        except Exception:
            logger.warning('RestWorker - startTraces failed', exc_info=True)
            token = None
        if not token:
            self.disconnected.emit()
            return
        self._stop.clear()
        self.connected.emit()
        # try/finally: an unexpected error in the polling loop must still emit
        # disconnected(), otherwise the UI shows "Connected" on a dead thread.
        try:
            while not self._stop.is_set():
                if self._stop.wait(self._retention / 2):
                    break
                traces = self._rest.pollTraces(token)
                if not traces:
                    break
                if not isinstance(traces, dict):
                    logger.warning('RestWorker - pollTraces returned non-dict: %s', type(traces).__name__)
                    continue

                # Collect removes and fetch add metadata — all on the background thread.
                # Removes MUST be emitted before adds so the GUI thread processes them in
                # the original SpecTcl order. A remove-then-add of the same name (rebind)
                # would otherwise leave the spectrum absent if adds were queued first.
                remove_bindings = []
                add_infos       = []
                for entry in (traces.get("binding") or []):
                    try:
                        action, name, _ = parse_binding_entry(entry)
                    except ValueError:
                        logger.warning('RestWorker - skipping malformed binding entry: %r', entry)
                        continue
                    if action == "remove":
                        remove_bindings.append(entry)
                    elif action == "add":
                        spec_info = lookup_spectrum_info(self._rest, name)
                        if spec_info is not None:
                            add_infos.append((name, spec_info))
                        else:
                            logger.warning('RestWorker - no REST spectrum info for %r', name)

                if remove_bindings:
                    self.tracesReady.emit({"binding": remove_bindings})
                for name, spec_info in add_infos:
                    self.spectrumAdded.emit(name, spec_info)
        except Exception:
            logger.exception('RestWorker - polling loop crashed; treating as disconnect')
        finally:
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
