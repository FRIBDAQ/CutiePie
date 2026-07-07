"""Characterization tests for ConnectionManager (H2 step 0).

These pin ConnectionManager's CURRENT observable behavior — signal emissions,
widget effects, store mutations, P7 guard decisions — before the H2 step-1
inversion (widget writes -> signals, widget reads -> method arguments).

They run headless via qt_stubs.install_missing_runtime_stubs(); on a machine
with real PyQt5/CPyConverter/httplib2 the real modules are used instead.
Threading is never real: QThread and the worker classes are monkeypatched
with fakes, and handlers are driven by emitting the fakes' signals directly.
"""

import importlib
import logging
import os
import sys
import threading
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import qt_stubs
from qt_stubs import record_signal


# Modules whose sys.modules entries this file may replace or import under
# stubs; snapshot/restore so later test modules (and importorskip gates on a
# GUI machine) see the environment they expect.
_AFFECTED_MODULES = (
    "PyQt5", "PyQt5.QtCore", "PyQt5.QtWidgets",
    "CPyConverter", "httplib2",
    "PyREST", "services.thread_workers", "services.connection_manager",
)


@pytest.fixture(scope="module")
def conn_mod():
    saved = {name: sys.modules.get(name) for name in _AFFECTED_MODULES}
    installed = qt_stubs.install_missing_runtime_stubs()
    if installed:
        # force the service modules to (re)import against the stubs
        for name in ("services.connection_manager", "services.thread_workers", "PyREST"):
            sys.modules.pop(name, None)
    module = importlib.import_module("services.connection_manager")
    yield module
    for name, prev in saved.items():
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


# connectShMem arguments in signature order (hostname, port, user, mirror),
# and the (hostname, mirror, user) P7 mapping-identity tuple they produce.
CONNECT_ARGS = ("spechost", "8080", "physicist", "8081")
ENDPOINT = ("spechost", "8081", "physicist")


class Env:
    def __init__(self, module):
        from services.spectrum_store import SpectrumStore
        self.mod = module
        self.store = SpectrumStore()
        self.stop_rest = threading.Event()
        self.stop_auto = threading.Event()
        self.skip_auto = threading.Event()
        self.cm = module.ConnectionManager(
            self.store,
            [5, 10, 30], [300, 600, 1800],
            self.stop_rest, self.stop_auto, self.skip_auto,
            logger=logging.getLogger("test.connection_manager"),
        )
        # H2 output signals, recorded from construction on
        self.states = record_signal(self.cm.connectionStateChanged)
        self.busy = record_signal(self.cm.connectAttemptBusy)
        self.lists = record_signal(self.cm.spectrumListUpdated)


@pytest.fixture
def env(conn_mod):
    return Env(conn_mod)


def make_shm(*entries):
    """Build the CPyConverter.Update() result shape the code indexes:
    s[1]=names, s[2]=dims, s[3]=shm binx, s[4]=minx, s[5]=maxx,
    s[6]=shm biny, s[7]=miny, s[8]=maxy, s[9]=data arrays.
    Each entry: (name, dim, binx, minx, maxx, biny, miny, maxy, data)."""
    names, dims, binx, minx, maxx, biny, miny, maxy, data = (
        list(column) for column in zip(*entries))
    return [None, names, dims, binx, minx, maxx, biny, miny, maxy, data]


def spec_info_1d(bins=8, low=0.0, high=10.0, params=("p1",), type_="1"):
    return {"axes": [{"bins": bins, "low": low, "high": high}],
            "parameters": list(params), "type": type_}


def patched_connect(env, monkeypatch, rest):
    """Run connectShMem with fake thread/worker classes installed.

    Signal wiring happens exactly as in production, but nothing runs until a
    test emits a worker signal explicitly."""
    monkeypatch.setattr(env.mod, "QThread", qt_stubs.FakeThread)
    monkeypatch.setattr(env.mod, "RestWorker", qt_stubs.FakeWorker)
    monkeypatch.setattr(env.mod, "ConnectWorker", qt_stubs.FakeWorker)
    env.cm._rest = rest
    env.cm.connectShMem(*CONNECT_ARGS)


# ---------------------------------------------------------------- pure helper

def test_get_last_digit_param(conn_mod):
    f = conn_mod.ConnectionManager._get_last_digit_param
    assert f("event.raw.7") == 7
    assert f("event.raw") is None      # no digit in any part
    assert f("7.raw") is None          # digit present but last part not numeric


# ----------------------------------------------------------------- P7 guards

def test_connect_refused_on_endpoint_switch(env):
    env.cm._mapped_endpoint = ("otherhost", "9999", "someoneelse")
    env.cm._mapped_shmem_size = 4096
    refused = record_signal(env.cm.connectionRefused)
    env.cm.connectShMem(*CONNECT_ARGS)
    assert len(refused) == 1
    message = refused[0][0]
    assert "otherhost" in message and "restart" in message.lower()
    # refused before any client work or state emission
    assert env.cm._rest is None
    assert env.states == []
    assert env.cm._pending_endpoint is None


def test_connect_refused_on_shmem_resize(env):
    env.cm._mapped_endpoint = ENDPOINT
    env.cm._mapped_shmem_size = 1024
    env.cm._rest = qt_stubs.FakeRest(check=True, shmem_size=2048)
    refused = record_signal(env.cm.connectionRefused)
    env.cm.connectShMem(*CONNECT_ARGS)
    assert len(refused) == 1
    assert "1024 -> 2048" in refused[0][0]
    assert env.cm._pending_endpoint is None
    # the attempt had already reported "disconnected" before the guard fired
    assert env.states == [("disconnected",)]


def test_connect_aborts_when_rest_unreachable(env):
    env.cm._rest = qt_stubs.FakeRest(check=False)
    refused = record_signal(env.cm.connectionRefused)
    env.cm.connectShMem(*CONNECT_ARGS)
    assert refused == []
    assert env.cm._pending_endpoint is None
    assert env.states == [("disconnected",)]


# ------------------------------------------------- connect + mirror transfer

def test_connect_starts_workers_and_marks_pending(env, monkeypatch):
    patched_connect(env, monkeypatch, qt_stubs.FakeRest(check=True, shmem_size=4096))
    assert env.cm._pending_endpoint == ENDPOINT
    assert env.cm._pending_shmem_size == 4096
    assert env.cm._mapped_endpoint is None            # not committed yet
    assert env.states == [("disconnected",), ("connecting",)]
    assert env.busy == [(True,)]
    assert env.cm._rest_thread.running and env.cm._connect_thread.running
    # ConnectWorker receives the popup fields
    assert env.cm._connect_worker.args == ("spechost", "8080", "8081", "physicist")


def test_shmem_size_unavailable_fails_open(env, monkeypatch):
    refused = record_signal(env.cm.connectionRefused)
    env.cm._mapped_endpoint = ENDPOINT
    env.cm._mapped_shmem_size = 1024
    patched_connect(env, monkeypatch, qt_stubs.FakeRest(check=True, shmem_size=None))
    assert refused == []                              # size guard fails open
    assert env.cm._pending_endpoint == ENDPOINT
    assert env.cm._pending_shmem_size is None


def test_first_mirror_transfer_commits_mapping_and_populates_store(env, monkeypatch):
    rest = qt_stubs.FakeRest(
        check=True, shmem_size=4096,
        spectra=[
            {"name": "h1", "parameters": ["p1"], "type": "1"},
            {"name": "h2", "parameters": ["px", "py"], "type": "2"},
            {"name": "sum1", "parameters": ["a", "b"], "type": "s"},
            {"name": "unbound", "parameters": ["q"], "type": "1"},
        ],
        binds=[{"name": "h1", "binding": 1}, {"name": "h2", "binding": 2},
               {"name": "sum1", "binding": 3}],
    )
    patched_connect(env, monkeypatch, rest)
    invalidated = record_signal(env.cm.shmViewsInvalidated)
    s = make_shm(
        ("h1", 1, 8, 0.0, 10.0, 0, 0.0, 0.0, np.arange(1, 7)),
        ("h2", 2, 6, 0.0, 4.0, 6, 0.0, 4.0, np.arange(16).reshape(4, 4)),
        ("sum1", 2, 6, 0.0, 4.0, 6, 0.0, 4.0, np.arange(16).reshape(4, 4)),
        ("unbound", 1, 7, 0.0, 5.0, 0, 0.0, 0.0, np.arange(5)),
    )
    env.cm._connect_worker.succeeded.emit(s)

    assert env.cm._mapped_endpoint == ENDPOINT
    assert env.cm._mapped_shmem_size == 4096
    assert invalidated == []                          # first connect: no purge
    # 1-d: slice [0:-1], underflow zeroed; store binx = shm binx - 2
    h1 = env.store.get_record("h1")
    assert h1["data"].shape == (5,) and h1["data"][0] == 0
    assert h1["binx"] == 6 and h1["minx"] == 0.0 and h1["maxx"] == 10.0
    # 2-d: under/overflow frame stripped
    assert env.store.get("h2", "data").shape == (2, 2)
    assert env.store.get("h2", "maxx") == 4.0
    # summary type: maxx gets +1
    assert env.store.get("sum1", "maxx") == 5.0
    # spectra without a REST binding are skipped
    assert not env.store.contains("unbound")
    # spectrum list published (sorted) with init=True
    assert env.lists[-1] == (["h1", "h2", "sum1"], True)
    assert env.busy[-1] == (False,)                   # button re-enabled
    assert env.cm._connect_thread.quit_count == 1


def test_reconnect_emits_invalidation_before_repopulating(env, monkeypatch):
    endpoint = ENDPOINT
    env.cm._mapped_endpoint = endpoint
    env.cm._mapped_shmem_size = 4096
    rest = qt_stubs.FakeRest(
        check=True, shmem_size=4096,
        spectra=[{"name": "h1", "parameters": ["p1"], "type": "1"}],
        binds=[{"name": "h1", "binding": 1}],
    )
    patched_connect(env, monkeypatch, rest)
    snapshots = []
    env.cm.shmViewsInvalidated.connect(
        lambda: snapshots.append(env.store.all_names()))
    s = make_shm(("h1", 1, 8, 0.0, 10.0, 0, 0.0, 0.0, np.arange(1, 7)))
    env.cm._connect_worker.succeeded.emit(s)
    # emitted exactly once, BEFORE the store was repopulated
    assert snapshots == [[]]
    assert env.store.contains("h1")
    # the original mapping identity is retained, not overwritten
    assert env.cm._mapped_endpoint == endpoint
    assert env.cm._mapped_shmem_size == 4096


def test_failed_mirror_transfer_restores_button(env, monkeypatch):
    patched_connect(env, monkeypatch, qt_stubs.FakeRest(check=True, shmem_size=4096))
    env.cm._connect_worker.failed.emit("boom")
    assert env.states[-1] == ("disconnected",)
    assert env.busy[-1] == (False,)
    assert env.cm._connect_thread.quit_count == 1
    assert env.cm._mapped_endpoint is None            # nothing committed


# --------------------------------------------------------- REST worker wiring

def test_reconnect_ignores_stale_disconnect_from_old_worker(env, monkeypatch):
    # SMOKE-A4 regression: reconnecting (same endpoint) tears down the old
    # RestWorker, whose run() emits disconnected() from its finally-block as it
    # exits. That signal is delivered AFTER the reconnect has installed a fresh
    # worker/thread; if _on_rest_disconnected acts on it, it quit()+wait()s the
    # BRAND-NEW thread on the GUI thread — and that thread's run loop never
    # stops (its stop event was just cleared), so the GUI hangs. The superseded
    # worker's signals must be detached at teardown so its late disconnect is a
    # no-op.
    env.cm._mapped_endpoint = ENDPOINT
    env.cm._mapped_shmem_size = 4096
    patched_connect(env, monkeypatch, qt_stubs.FakeRest(check=True, shmem_size=4096))
    old_worker = env.cm._rest_worker
    old_worker.connected.emit()                       # be in the connected state

    env.cm.connectShMem(*CONNECT_ARGS)                 # reconnect, same values
    new_worker = env.cm._rest_worker
    new_thread = env.cm._rest_thread
    assert new_worker is not old_worker

    old_worker.disconnected.emit()                     # its finally-block fires late

    # the fresh thread/worker must survive the stale disconnect
    assert env.cm._rest_worker is new_worker
    assert env.cm._rest_thread is new_thread
    assert new_thread.quit_count == 0


def test_rest_worker_connected_then_disconnected_updates_button(env, monkeypatch):
    patched_connect(env, monkeypatch, qt_stubs.FakeRest(check=True, shmem_size=4096))
    established = record_signal(env.cm.connectionEstablished)
    worker = env.cm._rest_worker
    thread = env.cm._rest_thread

    worker.connected.emit()
    assert env.states[-1] == ("connected",)
    assert len(established) == 1

    worker.disconnected.emit()
    assert env.states[-1] == ("disconnected",)
    assert env.cm._rest_thread is None and env.cm._rest_worker is None
    assert thread.quit_count == 1


# --------------------------------------------------------------- trace removes

def test_trace_remove_updates_store_and_signals(env):
    env.store.set("alpha", dim=1, data=[])
    env.store.set("my spec", dim=1, data=[])
    removed = record_signal(env.cm.spectrumRemoved)
    changed = record_signal(env.cm.spectrumListChanged)
    env.cm.updateFromTraces(
        {"binding": ["remove alpha 3", "remove {my spec} 5"]})
    assert removed == [("alpha",), ("my spec",)]      # braced Tcl names (E3)
    assert len(changed) == 2
    assert env.store.all_names() == []
    assert env.lists[-1] == ([], False)


def test_trace_remove_ignores_unknown_malformed_and_adds(env):
    env.store.set("keep", dim=1)
    removed = record_signal(env.cm.spectrumRemoved)
    env.cm.updateFromTraces(
        {"binding": ["remove ghost 1", "garbage", "add keep 2"]})
    env.cm.updateFromTraces({})
    env.cm.updateFromTraces({"binding": None})
    assert removed == []
    assert env.store.contains("keep")


# ------------------------------------------------------- spectrum add batching

def test_spectrum_adds_batch_into_one_flush(env, monkeypatch):
    timer = qt_stubs.RecordingTimer()
    monkeypatch.setattr(env.mod, "QTimer", timer)
    env.store.set("already", dim=1)
    env.cm._on_spectrum_added("already", spec_info_1d())   # in store: ignored
    env.cm._on_spectrum_added("n1", spec_info_1d())
    env.cm._on_spectrum_added("n2", spec_info_1d())
    assert [name for name, _ in env.cm._pending_adds] == ["n1", "n2"]
    assert len(timer.scheduled) == 1                       # one flush per batch
    assert env.cm._flush_scheduled is True

    update_calls = []
    s = make_shm(("n1", 1, 8, 0.0, 10.0, 0, 0.0, 0.0, np.arange(1, 7)),
                 ("n2", 1, 8, 0.0, 10.0, 0, 0.0, 0.0, np.arange(2, 8)))

    class FakeConverter:
        def Update(self, *args):
            update_calls.append(args)
            return s

    monkeypatch.setattr(env.mod, "cpy",
                        types.SimpleNamespace(CPyConverter=FakeConverter))
    env.cm._last_connect_params = ("spechost", "8080", "8081", "physicist")
    msec, flush = timer.scheduled[0]
    flush()
    assert len(update_calls) == 1                          # single shm fetch
    # flush reuses the last ACCEPTED connect parameters (H2), not live popup text
    assert update_calls[0] == (b"spechost", b"8080", b"8081", b"physicist")
    assert env.store.contains("n1") and env.store.contains("n2")
    assert env.store.get("n1", "data").shape == (5,)
    assert env.store.get("n1", "data")[0] == 0
    # note the asymmetry with the connect path: binx here comes straight from
    # the REST axes dict, NOT shm binx - 2
    assert env.store.get("n1", "binx") == 8
    assert env.cm._pending_adds == [] and env.cm._flush_scheduled is False
    assert env.lists[-1] == (["already", "n1", "n2"], False)


def test_flush_drops_batch_when_converter_fails(env, monkeypatch):
    timer = qt_stubs.RecordingTimer()
    monkeypatch.setattr(env.mod, "QTimer", timer)
    env.cm._on_spectrum_added("n1", spec_info_1d())

    class Boom:
        def Update(self, *args):
            raise RuntimeError("no shm")

    monkeypatch.setattr(env.mod, "cpy", types.SimpleNamespace(CPyConverter=Boom))
    env.cm._last_connect_params = ("spechost", "8080", "8081", "physicist")
    timer.scheduled[0][1]()
    assert not env.store.contains("n1")
    assert env.cm._pending_adds == []                 # batch dropped, no retry


def test_flush_without_connect_params_drops_batch(env, monkeypatch):
    timer = qt_stubs.RecordingTimer()
    monkeypatch.setattr(env.mod, "QTimer", timer)
    env.cm._on_spectrum_added("n1", spec_info_1d())
    timer.scheduled[0][1]()                           # no connect ever accepted
    assert not env.store.contains("n1")
    assert env.cm._pending_adds == []


def test_process_add_summary_bounds_from_parameter_indices(env):
    info = {"axes": [{"bins": 16, "low": 0.0, "high": 16.0}],
            "parameters": ["det.q.2", "det.q.5"], "type": "s"}
    s = make_shm(("summ", 2, 0, 0, 0, 0, 0, 0, np.arange(36).reshape(6, 6)))
    env.cm._process_spectrum_add("summ", info, s)
    rec = env.store.get_record("summ")
    assert rec["dim"] == 2
    # x axis derived from the parameter indices: [2, 5] -> min 2, max 5+1
    assert rec["minx"] == 2 and rec["maxx"] == 6 and rec["binx"] == 4
    # y axis inherits the REST x-axis definition
    assert rec["biny"] == 16 and rec["miny"] == 0.0 and rec["maxy"] == 16.0
    assert rec["data"].shape == (4, 4)


def test_process_add_missing_from_shmem_stores_empty_data(env):
    s = make_shm(("someother", 1, 8, 0.0, 10.0, 0, 0.0, 0.0, np.arange(6)))
    env.cm._process_spectrum_add("ghost", spec_info_1d(), s)
    assert env.store.contains("ghost")
    assert env.store.get("ghost", "data") == []


# ------------------------------------------------------------------ REST info

def test_spectrum_info_joins_only_bound(env):
    env.cm._rest = qt_stubs.FakeRest(
        spectra=[{"name": "a", "parameters": ["p"], "type": "1"},
                 {"name": "b", "parameters": ["q"], "type": "1"}],
        binds=[{"name": "a", "binding": 7}])
    out = env.cm.getSpectrumInfoFromREST()
    assert list(out) == ["a"]
    assert out["a"] == {"parameters": ["p"], "type": "1", "binding": 7}


def test_spectrum_info_tolerates_non_list_replies(env):
    env.cm._rest = qt_stubs.FakeRest(spectra={"status": "error"}, binds=[])
    assert env.cm.getSpectrumInfoFromREST() == {}
    env.cm._rest = qt_stubs.FakeRest(spectra=[], binds="nope")
    assert env.cm.getSpectrumInfoFromREST() == {}


def test_applylistgate_shields_callers(env):
    assert env.cm.applylistgate("h1") == []           # no client yet
    env.cm._rest = qt_stubs.FakeRest(gates_error=True)
    assert env.cm.applylistgate("h1") == []           # REST failure swallowed
    env.cm._rest = qt_stubs.FakeRest(gates=[{"gate": "g1"}])
    assert env.cm.applylistgate("h1") == [{"gate": "g1"}]


# ---------------------------------------------------------------- auto-update

def test_auto_update_start_uses_selected_interval(env, monkeypatch):
    monkeypatch.setattr(env.mod, "QThread", qt_stubs.FakeThread)
    monkeypatch.setattr(env.mod, "AutoUpdateWorker", qt_stubs.FakeWorker)
    env.stop_auto.set()
    env.skip_auto.set()
    env.cm.autoUpdateStart(1)
    worker = env.cm._auto_worker
    assert worker.args == (10, env.stop_auto, env.skip_auto)
    assert not env.stop_auto.is_set() and not env.skip_auto.is_set()
    assert env.cm._auto_thread.running
    # worker ticks propagate to updatePlotRequested
    ticks = record_signal(env.cm.updatePlotRequested)
    worker.updateTriggered.emit()
    assert len(ticks) == 1


def test_auto_update_resume_clears_skip(env):
    env.skip_auto.set()
    env.cm.autoUpdateResume()
    assert not env.skip_auto.is_set()
