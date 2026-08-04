"""Trace-polling resilience for RestWorker, and the pollTraces contract it rests on.

The polling loop used to end on any falsy poll result, and PyREST reported a
failed request as an empty dict. One request that timed out therefore ended
trace tracking for the whole session: the connect button went red and no
further spectrum add or remove was seen until the user reconnected by hand.

These tests pin the two halves of the fix. PyREST now says None for "the poll
did not get through" and keeps the empty dict for "nothing fired", and
RestWorker retries a failed poll instead of quitting on the first one.

They run headless via qt_stubs.install_missing_runtime_stubs(); on a machine
with the real PyQt5 and httplib2 the real modules are used instead. No OS
thread and no socket is involved: run() is called directly and the stop event
is set by the fake REST client when the test has seen enough polls.
"""

import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import qt_stubs


# Modules whose sys.modules entries the stub install may replace; snapshot and
# restore so later test modules (and importorskip gates on a GUI machine) see
# the environment they expect.
_AFFECTED_MODULES = (
    "PyQt5", "PyQt5.QtCore", "PyQt5.QtWidgets",
    "CPyConverter", "httplib2", "PyREST", "services.thread_workers",
)


@pytest.fixture(scope="module")
def workers_mod():
    saved = {name: sys.modules.get(name) for name in _AFFECTED_MODULES}
    installed = qt_stubs.install_missing_runtime_stubs()
    if installed:
        sys.modules.pop("services.thread_workers", None)
        sys.modules.pop("PyREST", None)
    mod = importlib.import_module("services.thread_workers")
    yield mod
    for name, prev in saved.items():
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


@pytest.fixture(scope="module")
def pyrest_cls(workers_mod):
    return importlib.import_module("PyREST").PyREST


EMPTY_DETAIL = {"binding": [], "parameter": [], "spectrum": [], "gate": []}


class StopAfter:
    """threading.Event stand-in that never blocks.

    The worker waits on this between polls, so a real Event would make every
    test sleep retention/2. wait() returns False (not stopped) until the test
    or the fake REST client sets it.
    """

    def __init__(self):
        self._set = False

    def is_set(self):
        return self._set

    def set(self):
        self._set = True

    def clear(self):
        self._set = False

    def wait(self, timeout=None):
        return self._set


class ScriptedRest:
    """Replays a list of pollTraces results, one per poll.

    A result of None is a failed poll, mirroring what PyREST returns when the
    request did not get through. The script running out stops the worker the
    way a user pressing disconnect would, so a test that reaches the end of
    its script proves the loop was still alive there.
    """

    def __init__(self, script, stop):
        self._script = list(script)
        self._stop = stop
        self.polls = 0

    def startTraces(self, retention):
        return 42

    def pollTraces(self, token):
        assert token == 42
        self.polls += 1
        result = self._script.pop(0)
        if not self._script:
            self._stop.set()
        return result


def drive(workers_mod, script, retention=6):
    """Run RestWorker.run against `script` and return (worker, rest, events)."""
    stop = StopAfter()
    rest = ScriptedRest(script, stop)
    worker = workers_mod.RestWorker(rest, retention, stop)
    events = []
    worker.connected.connect(lambda: events.append("connected"))
    worker.disconnected.connect(lambda: events.append("disconnected"))
    worker.tracesReady.connect(lambda d: events.append(("traces", d)))
    worker.spectrumAdded.connect(lambda n, i: events.append(("added", n)))
    worker.run()
    return worker, rest, events


# --------------------------------------------------------- the retry itself

def test_one_failed_poll_does_not_end_the_loop(workers_mod):
    """The regression: a single failed poll used to disconnect the session."""
    _, rest, events = drive(workers_mod, [dict(EMPTY_DETAIL), None,
                                          dict(EMPTY_DETAIL), dict(EMPTY_DETAIL)])
    assert rest.polls == 4, "the loop stopped early instead of retrying"
    assert events == ["connected", "disconnected"], events


def test_two_failed_polls_in_a_row_still_do_not_end_the_loop(workers_mod):
    _, rest, events = drive(workers_mod, [None, None, dict(EMPTY_DETAIL),
                                          dict(EMPTY_DETAIL)])
    assert rest.polls == 4


def test_three_failed_polls_in_a_row_end_the_loop(workers_mod):
    """The server really is gone: give up, and say so exactly once."""
    _, rest, events = drive(workers_mod, [None, None, None,
                                          dict(EMPTY_DETAIL), dict(EMPTY_DETAIL)])
    assert rest.polls == 3, "kept polling past the failure budget"
    assert events == ["connected", "disconnected"]


def test_the_failure_budget_matches_the_declared_limit(workers_mod):
    assert workers_mod.RestWorker._MAX_POLL_FAILURES == 3


def test_a_good_poll_resets_the_failure_count(workers_mod):
    """Scattered failures must not accumulate across a healthy session.

    Two failures, a success, then two more failures is five bad-ish polls but
    never three in a row, so the connection stands.
    """
    script = [None, None, dict(EMPTY_DETAIL), None, None,
              dict(EMPTY_DETAIL), dict(EMPTY_DETAIL)]
    _, rest, events = drive(workers_mod, script)
    assert rest.polls == len(script)
    assert events == ["connected", "disconnected"]


def test_an_empty_detail_is_not_a_disconnect(workers_mod):
    """`{}` means the server answered and nothing had fired."""
    _, rest, _ = drive(workers_mod, [{}, {}, {}, dict(EMPTY_DETAIL)])
    assert rest.polls == 4


def test_disconnected_is_emitted_exactly_once_when_the_budget_runs_out(workers_mod):
    _, _, events = drive(workers_mod, [None, None, None])
    assert events.count("disconnected") == 1


# ------------------------------------------- the retry does not eat the work

def test_traces_still_flow_after_a_retry(workers_mod):
    """A recovered poll is processed normally: removes first, then adds."""
    rest_info = {"type": "1", "axes": [{"bins": 10, "low": 0, "high": 10}],
                 "parameters": ["p"], "name": "spec"}
    good = {"binding": ["remove old 3", "add spec 4"]}
    stop = StopAfter()
    rest = ScriptedRest([None, good, dict(EMPTY_DETAIL)], stop)
    rest.listSpectrum = lambda pattern=None: [dict(rest_info)]
    worker = workers_mod.RestWorker(rest, 6, stop)
    events = []
    worker.tracesReady.connect(lambda d: events.append(("traces", d["binding"])))
    worker.spectrumAdded.connect(lambda n, i: events.append(("added", n)))
    worker.run()
    assert events == [("traces", ["remove old 3"]), ("added", "spec")]


def test_a_non_dict_poll_result_is_skipped_not_fatal(workers_mod):
    """Pre-existing guard: keep it working through the new failure branch."""
    _, rest, events = drive(workers_mod, ["not a dict", dict(EMPTY_DETAIL),
                                          dict(EMPTY_DETAIL)])
    assert rest.polls == 3
    assert events == ["connected", "disconnected"]


# ------------------------------------------------------ the PyREST contract

class FakeLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def debug(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass

    def info(self, *a, **k):
        pass


@pytest.fixture
def rest(pyrest_cls):
    """A PyREST whose transport is scripted per test."""
    r = pyrest_cls.__new__(pyrest_cls)
    r.server, r.rest = "spechost", "8080"
    r.logger = FakeLogger()
    return r


def test_poll_traces_says_none_when_the_request_fails(rest):
    """sendRequest returns None for any transport error. That must not look
    like an empty result, or the caller cannot tell the two apart."""
    rest.sendRequest = lambda url: None
    assert rest.pollTraces(7) is None


def test_poll_traces_returns_the_detail_object(rest):
    rest.sendRequest = lambda url: b'{"status":"OK","detail":{"binding":["add s 1"]}}'
    assert rest.pollTraces(7) == {"binding": ["add s 1"]}


def test_poll_traces_returns_an_empty_detail_as_an_empty_dict(rest):
    rest.sendRequest = lambda url: b'{"status":"OK","detail":{}}'
    assert rest.pollTraces(7) == {}


def test_poll_traces_says_none_on_an_unparseable_body(rest):
    """A truncated reply used to raise out of the polling loop, which the
    worker's blanket except turned into a disconnect. It is a failed poll."""
    rest.sendRequest = lambda url: b'{"status":"OK","detail":{"bind'
    assert rest.pollTraces(7) is None
    assert rest.logger.warnings


def test_poll_traces_says_none_when_detail_is_missing(rest):
    rest.sendRequest = lambda url: b'{"status":"bad parameter"}'
    assert rest.pollTraces(7) is None


def test_poll_traces_sends_the_token(rest):
    sent = []
    rest.sendRequest = lambda url: sent.append(url)
    rest.pollTraces(99)
    assert "token=99" in sent[0]
