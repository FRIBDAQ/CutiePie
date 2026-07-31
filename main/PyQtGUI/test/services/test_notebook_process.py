import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

import pytest

import notebook_process


class FakePopen:
    """Stand-in for the jupyter subprocess: its own stderr.readline() feed.

    - `lines`      : bytes lines returned in order (b'' = EOF).
    - `loop_line`  : if set, returned forever once `lines` is exhausted
                     (an alive server that keeps talking but never prints
                     an address).
    - `alive`      : poll() returns None while True, else `returncode`.
    """
    def __init__(self, lines=None, loop_line=None, returncode=None, alive=True):
        self._lines = list(lines or [])
        self._loop_line = loop_line
        self.returncode = returncode
        self._alive = alive
        self.stderr = self

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        if self._loop_line is not None:
            return self._loop_line
        return b''

    def poll(self):
        return None if self._alive else self.returncode


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    # startnotebook refuses to start if a process is already recorded; keep
    # each test isolated and never spawn a real subprocess.
    notebook_process._process = None
    notebook_process._monitor = None
    notebook_process._webaddr = None
    monkeypatch.setattr(notebook_process, "log", lambda *a, **k: None)
    yield
    notebook_process._process = None


def _patch_popen(monkeypatch, fake):
    monkeypatch.setattr(notebook_process.subprocess, "Popen",
                        lambda *a, **k: fake)


# --------------------------------------------------------------- happy path

def test_returns_published_address(monkeypatch):
    # dead after startup so the monitor thread exits immediately
    fake = FakePopen(lines=[b"[I] loading\n",
                            b"[I] http://127.0.0.1:8888/?token=abc\n"],
                     returncode=0, alive=False)
    _patch_popen(monkeypatch, fake)
    addr = notebook_process.startnotebook("jupyter-notebook", directory="/tmp")
    assert addr == "http://127.0.0.1:8888/?token=abc"


# -------------------------------------------------------- died before publishing

def test_raises_when_server_exits_before_publishing(monkeypatch):
    fake = FakePopen(lines=[], returncode=1, alive=False)   # immediate EOF, exited
    _patch_popen(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="before publishing"):
        notebook_process.startnotebook("jupyter-notebook", directory="/tmp")


# --------------------------------------------------------------- deadline

def test_deadline_fires_when_alive_but_never_publishes(monkeypatch):
    # alive server that keeps printing non-http lines forever (gap 1)
    monkeypatch.setattr(notebook_process, "STARTUP_DEADLINE_SECS", 0.3)
    fake = FakePopen(loop_line=b"[I] still starting up\n", alive=True)
    _patch_popen(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="did not publish a server address within"):
        notebook_process.startnotebook("jupyter-notebook", directory="/tmp")


def test_deadline_fires_on_stderr_eof_while_alive(monkeypatch):
    # stderr EOF but process still alive → old code spun at 10 Hz forever (gap 2)
    monkeypatch.setattr(notebook_process, "STARTUP_DEADLINE_SECS", 0.3)
    fake = FakePopen(lines=[], loop_line=None, alive=True)   # readline()==b'', poll()==None
    _patch_popen(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="did not publish a server address within"):
        notebook_process.startnotebook("jupyter-notebook", directory="/tmp")
