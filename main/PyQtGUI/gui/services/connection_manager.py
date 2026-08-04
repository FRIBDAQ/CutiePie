import logging

import CPyConverter as cpy

from PyQt5.QtCore import QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5 import QtCore

from PyREST import PyREST
from services.thread_workers import RestWorker, AutoUpdateWorker, ConnectWorker, parse_binding_entry


class ConnectionManager(QtCore.QObject):
    """Owns REST connection lifecycle, trace polling, and auto-update thread."""

    connectionEstablished = pyqtSignal()
    spectrumRemoved       = pyqtSignal(str)
    spectrumListChanged   = pyqtSignal()
    updatePlotRequested   = pyqtSignal()
    # a re-connect finished a fresh mirror transfer — the GUI must drop every
    # matplotlib artist / cached axis still bound to arrays from the previous
    # transfer before the store is repopulated with new views.
    shmViewsInvalidated   = pyqtSignal()
    # connect attempt refused (shm mapping cannot change within a process);
    # payload is the user-facing message.
    connectionRefused     = pyqtSignal(str)
    # the mirror transfer raised (e.g. getSpecTclMemory returned nullptr —
    # wrong port / dead mirror service). The C++ side now raises instead of
    # segfaulting (CPyConverter::Update null-check); MainWindow shows the message.
    connectFailed         = pyqtSignal(str)
    # a bound spectrum declares more data than the mapped mirror can hold, so
    # its view was discarded instead of touched; payload is the user-facing
    # message. Emitted once per spectrum name per session.
    spectrumDiscarded     = pyqtSignal(str)
    # connect-button rendering inverted into signals — MainWindow owns the
    # widget (adapters _render_connect_state / _on_connect_attempt_busy).
    connectionStateChanged = pyqtSignal(str)   # "connected" | "connecting" | "disconnected"
    connectAttemptBusy     = pyqtSignal(bool)  # True while the mirror transfer runs
    spectrumListUpdated    = pyqtSignal(list, bool)  # (sorted names, init) — replaces histo_list surgery

    def __init__(self, spectra,
                 update_intervals, update_intervals_user,
                 stop_rest, stop_auto, skip_auto, logger=None):
        super().__init__()
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
        # guard state: CPyConverter attaches the shm mirror once per process and
        # can never remap it, so the endpoint and segment size are fixed at the
        # first successful mirror transfer. _pending_* hold the values of the
        # in-flight connect attempt; they are committed on success.
        self._mapped_endpoint    = None   # (hostname, mirror, user)
        self._mapped_shmem_size  = None   # bytes, REST-reported at mapping time
        self._pending_endpoint   = None
        self._pending_shmem_size = None
        # connect parameters of the last attempt that passed the guards;
        # flush-time CPyConverter calls reuse these instead of re-reading the
        # popup fields (which now live in MainWindow). Order matches
        # CPyConverter.Update: (hostname, port, mirror, user).
        self._last_connect_params = None
        # spectra already reported as too big for the mirror; a binding trace
        # re-offers them on every poll, so the dialog fires once per name.
        self._oversized_reported = set()

    # ------------------------------------------------------------------
    # REST + shared memory connection
    # ------------------------------------------------------------------

    def _shm_view_fits(self, arr, name):
        """True when a zero-copy view's own extent fits inside the mapped
        mirror. Nothing between the mirror header and here checks the declared
        shape: CPyConverter wraps whatever address and bin counts it is given,
        and creating the array touches no memory."""
        size = self._mapped_shmem_size
        if size is None:
            # same fail-open as the connect-time size guard: without a
            # REST-reported size there is nothing to compare against
            return True
        try:
            nbytes = int(arr.nbytes)
        except Exception:
            self.logger.debug('_shm_view_fits - no nbytes for %s', name, exc_info=True)
            return True
        if nbytes <= size:
            return True

        self.logger.error(
            '_shm_view_fits - discarding %s: it declares %d bytes but the mapped '
            'shared memory is %d bytes', name, nbytes, size)
        if name not in self._oversized_reported:
            self._oversized_reported.add(name)
            self.spectrumDiscarded.emit(
                f'The spectrum "{name}" declares {nbytes} bytes of data, but '
                f"SpecTcl's display shared memory is only {size} bytes.\n\n"
                "It cannot be read and has been skipped. Reduce the number of "
                "bins, or restart SpecTcl with a larger display memory.")
        return False

    def connectShMem(self, hostname, port, user, mirror):
        """Connect to SpecTcl REST + the shm mirror. The four parameters are
        supplied by the MainWindow adapter from the connection popup."""
        self.logger.info('connectShMem')
        try:
            self.logger.debug('connectShMem - host: %s -- user: %s -- RESTPort: %s -- MirrorPort: %s',
                               hostname, user, port, mirror)

            # the C++ layer maps the mirror only when no mapping exists, so a
            # connect to a different host/mirror/user would silently keep serving
            # views of the OLD SpecTcl's mirror. Refuse it and leave the current
            # session untouched (note: self._rest is not reconfigured either).
            endpoint = (hostname, mirror, user)
            if self._mapped_endpoint is not None and endpoint != self._mapped_endpoint:
                self.logger.error(
                    'connectShMem - refused: shm mirror already mapped for %s, requested %s',
                    self._mapped_endpoint, endpoint)
                self.connectionRefused.emit(
                    "The shared-memory mirror is already mapped for\n"
                    f"host: {self._mapped_endpoint[0]}  mirror: {self._mapped_endpoint[1]}  "
                    f"user: {self._mapped_endpoint[2]}\n\n"
                    "It cannot be redirected to a different SpecTcl in a running session.\n"
                    "Please restart CutiePie to switch servers.")
                return

            if self._rest is not None:
                try:
                    self._rest.reconfigure(hostname, port)
                except Exception:
                    self._rest = PyREST(self.logger, hostname, port)
            else:
                self._rest = PyREST(self.logger, hostname, port)

            self.connectionStateChanged.emit("disconnected")

            if not self._rest.checkSpecTclREST():
                self.logger.debug('connectShMem - invalid URL for SpecTclREST')
                return

            # same endpoint, but SpecTcl may have restarted with a resized
            # display memory — the process-lifetime mapping would then be the
            # wrong size and stale views could read past the recreated segment.
            # Deterministic mismatch -> refuse; size unavailable -> fail open.
            shmem_size = self._rest.shmemSize()
            if (self._mapped_shmem_size is not None and shmem_size is not None
                    and shmem_size != self._mapped_shmem_size):
                self.logger.error(
                    'connectShMem - refused: shmem size changed %s -> %s (SpecTcl restarted with a resized display memory?)',
                    self._mapped_shmem_size, shmem_size)
                self.connectionRefused.emit(
                    "SpecTcl's display shared memory changed size since this session "
                    f"first connected ({self._mapped_shmem_size} -> {shmem_size} bytes).\n\n"
                    "The mirror mapping cannot be resized in a running session.\n"
                    "Please restart CutiePie to reconnect.")
                return
            if shmem_size is None:
                self.logger.warning('connectShMem - shmem size unavailable from REST; size-change guard inactive for this connect')
            self._pending_endpoint   = endpoint
            self._pending_shmem_size = shmem_size
            self._last_connect_params = (hostname, port, mirror, user)

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
            self.connectionStateChanged.emit("connecting")
            self.connectAttemptBusy.emit(True)

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
            self.connectAttemptBusy.emit(False)
            raise

    @pyqtSlot(object)
    def _on_connect_succeeded(self, s):
        self.logger.debug('connectShMem - mirror transfer done, fetching spectrum list from REST')
        if self._mapped_endpoint is None:
            # first successful mirror transfer: the mapping identity is now fixed
            # for the lifetime of the process (guards compare against these)
            self._mapped_endpoint   = self._pending_endpoint
            self._mapped_shmem_size = self._pending_shmem_size
        else:
            # re-connect over the existing mapping: artists still hold views from
            # the previous transfer — have the GUI drop them before the store is
            # repointed below. Synchronous: handler runs before we continue.
            self.shmViewsInvalidated.emit()
        try:
            otherInfo = self.getSpectrumInfoFromREST()
            self.logger.debug('connectShMem - populating spectra from shmem + REST')
            for i, name in enumerate(s[1]):
                if name not in otherInfo:
                    continue
                if not self._shm_view_fits(s[9][i], name):
                    continue
                if s[2][i] == 2:
                    data = s[9][i][1:-1, 1:-1]
                else:
                    data = s[9][i][0:-1]
                    data[0] = 0

                if "s" in otherInfo[name]["type"]:
                    # summary bounds computed per spectrum — previously assigned only
                    # under dim==2, so a dim-1 "s"-type would reuse a stale value from
                    # an earlier iteration (or hit UnboundLocalError on the first)
                    self._spectra.set(
                        name, dim=s[2][i], binx=s[3][i]-2, minx=s[4][i], maxx=s[5][i] + 1,
                        biny=s[6][i]-2, miny=s[7][i], maxy=s[8][i],
                        data=data, parameters=otherInfo[name]["parameters"],
                        type=otherInfo[name]["type"],
                        allow_data_replacement=True,   # fresh views from a new mirror
                    )
                else:
                    self._spectra.set(
                        name, dim=s[2][i], binx=s[3][i]-2, minx=s[4][i], maxx=s[5][i],
                        biny=s[6][i]-2, miny=s[7][i], maxy=s[8][i],
                        data=data, parameters=otherInfo[name]["parameters"],
                        type=otherInfo[name]["type"],
                        allow_data_replacement=True,   # fresh views from a new mirror
                    )

            self.updateSpectrumList(True)
        finally:
            self.connectAttemptBusy.emit(False)
            self._connect_thread.quit()

    @pyqtSlot(str)
    def _on_connect_failed(self, msg):
        self.logger.error('connectShMem - mirror transfer failed: %s', msg)
        self.connectFailed.emit(msg)                  # surface the reason to the user
        self.connectionStateChanged.emit("disconnected")
        self.connectAttemptBusy.emit(False)
        self._connect_thread.quit()

    # ------------------------------------------------------------------
    # Trace updates from REST worker
    # ------------------------------------------------------------------

    def updateFromTraces(self, tracesDetails):
        self.logger.info('updateFromTraces - tracesDetails: %s', tracesDetails)
        for entry in (tracesDetails.get("binding") or []):
            try:
                action, name, _ = parse_binding_entry(entry)
            except ValueError:
                self.logger.warning('updateFromTraces - skipping malformed binding entry: %r', entry)
                continue
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

        if self._last_connect_params is None:
            self.logger.debug('_flush_spectrum_adds - no accepted connect parameters yet')
            return
        hostname, port, mirror, user = self._last_connect_params
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
            # Guard per spectrum, not per batch: the pending queue was cleared
            # above, so an exception escaping here strands every spectrum still
            # to be processed with nothing left to retry — they stay missing
            # from the histogram list until a full reconnect. Skip the offender
            # and keep the batch.
            try:
                self._process_spectrum_add(name, spec_info, s)
            except Exception:
                self.logger.exception(
                    '_flush_spectrum_adds - skipping %s; the rest of the batch continues',
                    name)
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
                if not self._shm_view_fits(s[9][nameIndex], name):
                    return
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
                if not self._shm_view_fits(s[9][nameIndex], name):
                    return
                data = s[9][nameIndex][1:-1, 1:-1]
            except Exception:
                self.logger.debug('_process_spectrum_add - name not in shmem for %s', name, exc_info=True)
        else:
            # Everything that is not recognised as 1-D lands here and is read as
            # 2-D, which needs a second axis. The type test above is
            # case-sensitive, so a one-axis spectrum whose type differs only in
            # case (SpecTcl's strip-chart "S") arrives with a single axis; say
            # so and skip it rather than failing on the subscript.
            axes = spec_info.get("axes") or []
            if len(axes) < 2:
                self.logger.warning(
                    '_process_spectrum_add - %s has type %r and %d axis/axes; '
                    'cannot read it as 2-D, skipping', name, spec_type, len(axes))
                return
            dim  = 2
            biny = spec_info["axes"][1]["bins"]
            miny = spec_info["axes"][1]["low"]
            maxy = spec_info["axes"][1]["high"]
            try:
                nameIndex = s[1].index(name)
                if not self._shm_view_fits(s[9][nameIndex], name):
                    return
                data = s[9][nameIndex][1:-1, 1:-1]
            except Exception:
                self.logger.debug('_process_spectrum_add - name not in shmem for %s', name, exc_info=True)

        self._spectra.set(name, dim=dim, binx=binx, minx=minx, maxx=maxx,
                          biny=biny, miny=miny, maxy=maxy,
                          parameters=spec_info["parameters"],
                          type=spec_type, data=data,
                          allow_data_replacement=True)   # rebind: view from post-Update mirror

    # ------------------------------------------------------------------
    # Spectrum list helpers
    # ------------------------------------------------------------------

    def getSpectrumInfoFromREST(self):
        self.logger.info('getSpectrumInfoFromREST')
        outDict  = {}
        inpDict  = self._rest.listSpectrum()
        bindList = self._rest.listsbind("*")
        if not isinstance(inpDict, list):
            self.logger.warning('getSpectrumInfoFromREST - listSpectrum returned non-list: %s', inpDict)
            return outDict
        if not isinstance(bindList, list):
            self.logger.warning('getSpectrumInfoFromREST - listsbind returned non-list: %s', bindList)
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
        # log the count only — the full registry dict is large
        self.logger.info('getSpectrumInfoFromREST - %d bound spectra', len(outDict))
        return outDict

    def applylistgate(self, spectrum_name):
        """Return the gate-application list for a spectrum via the owned REST client.

        Routes through ConnectionManager so callers don't touch the REST client
        directly: returns [] when REST is unavailable or the call fails, instead of
        raising or exposing a half-open connection."""
        if self._rest is None:
            self.logger.debug('applylistgate - no REST client')
            return []
        try:
            return self._rest.applylistgate(spectrum_name)
        except Exception:
            self.logger.debug('applylistgate - REST call failed', exc_info=True)
            return []

    def getSpectrumStatistics(self, pattern="*"):
        """Return {name: {xunderflow, xoverflow, yunderflow, yoverflow}} via
        the owned REST client. One /spectcl/specstats call covers every
        spectrum matching the pattern (much lighter than per-spectrum
        /spectcl/spectrum/contents, which ships the full channel data)."""
        if self._rest is None:
            self.logger.debug('getSpectrumStatistics - no REST client')
            return {}
        try:
            entries = self._rest.getSpectrumStats(pattern)
            out = {}
            if isinstance(entries, list):
                for entry in entries:
                    if not isinstance(entry, dict) or "name" not in entry:
                        continue
                    stats = {}
                    for key, cols in (("underflows", ("xunderflow", "yunderflow")),
                                      ("overflows", ("xoverflow", "yoverflow"))):
                        vals = entry.get(key)
                        if isinstance(vals, (list, tuple)):
                            stats.update(zip(cols, vals))
                    out[entry["name"]] = stats
            return out
        except Exception:
            self.logger.debug('getSpectrumStatistics - REST call failed',
                              exc_info=True)
            return {}

    def updateSpectrumList(self, init=False):
        self.logger.debug('updateSpectrumList')
        self.spectrumListUpdated.emit(self._spectra.all_names(), init)

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
        # Detach the worker we're discarding BEFORE we stop it. Its run() emits
        # disconnected() (and possibly connected/traces/adds) from the
        # finally-block as it exits; that queued signal would otherwise be
        # delivered AFTER a reconnect has already installed a FRESH worker/
        # thread, and _on_rest_disconnected would then quit()+wait() the fresh
        # thread on the GUI thread — which hangs the GUI, because the fresh
        # thread's run loop never stops (its stop event was just cleared).
        # Severing the connections first makes the superseded worker's late
        # signals no-ops.
        if self._rest_worker is not None:
            for sig in ("disconnected", "connected", "tracesReady", "spectrumAdded"):
                try:
                    getattr(self._rest_worker, sig).disconnect()
                except (TypeError, RuntimeError):
                    pass
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
        self.connectionStateChanged.emit("connected")
        self.connectionEstablished.emit()

    @pyqtSlot()
    def _on_rest_disconnected(self):
        self.logger.info('_on_rest_disconnected')
        self.connectionStateChanged.emit("disconnected")
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

    def autoUpdateStart(self, interval_index):
        """Start the auto-update worker. `interval_index` selects from the
        configured interval tables (the combo read lives in MainWindow)."""
        self.logger.info('autoUpdateStart')
        updateInterval     = self._update_intervals[interval_index]
        updateIntervalUser = self._update_intervals_user[interval_index]
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
