"""URL-construction net for PyREST — runs headless via the httplib2 stub.

`test_pyrest.py` needs the real httplib2 and skips in this environment, so the
query strings PyREST puts on the wire were never checked here. These tests stub
the transport and assert the URL only, which is where the M22 double-encode
lived: a value pre-encoded at the call site is encoded AGAIN by `_build_url`'s
urlencode, so `%2B` reaches SpecTcl as `%252B` and decodes to the literal text
`%2B` rather than `+`.
"""

import importlib
import os
import sys
import urllib.parse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import pytest

import qt_stubs

# Snapshot/restore the modules the stub install touches, so a later test module
# — and the importorskip gates on a machine that HAS the real httplib2 — see the
# environment they expect. Installing stubs at import time leaks them.
_AFFECTED_MODULES = ("httplib2", "PyREST")


@pytest.fixture(scope="module")
def pyrest_cls():
    saved = {name: sys.modules.get(name) for name in _AFFECTED_MODULES}
    installed = qt_stubs.install_missing_runtime_stubs()
    if installed:
        sys.modules.pop("PyREST", None)
    cls = importlib.import_module("PyREST").PyREST
    yield cls
    for name, prev in saved.items():
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


@pytest.fixture
def rest(pyrest_cls):
    """A PyREST that records the URL instead of sending it."""
    r = pyrest_cls.__new__(pyrest_cls)
    r.server, r.rest = "spechost", "8080"
    r.sent = []
    r.sendRequest = lambda url: r.sent.append(url)
    return r


def query_of(url):
    """The decoded query mapping SpecTcl would parse out of `url`."""
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))


# ---------------------------------------------------------------- M22: "+" gate

def test_plus_gate_type_reaches_the_server_as_a_plus(rest):
    """A `+` compound gate must arrive as `+`. Pre-encoding it at the call site
    made the wire value `%252B`, which decodes to the string `%2B`."""
    rest.createGate("m2gate", "+", ["g1", "g2"], [])
    url = rest.sent[-1]
    assert "type=%252B" not in url, "double-encoded: the server sees the text %2B"
    assert query_of(url)["type"] == "+"


def test_plus_gate_still_carries_its_component_gates(rest):
    rest.createGate("m2gate", "+", ["g1", "g2"], [])
    pairs = urllib.parse.parse_qsl(urllib.parse.urlsplit(rest.sent[-1]).query)
    assert [v for k, v in pairs if k == "gate"] == ["g1", "g2"]


# ------------------------------------------------------- M22: vector-or slice

def test_vector_or_slice_type_reaches_the_server_as_vs_plus(rest):
    rest.createVectorOrSlice("vs", "vec", 0.0, 10.0)
    assert query_of(rest.sent[-1])["type"] == "vs+"


def test_vector_and_slice_is_unchanged(rest):
    rest.createVectorSlice("vs", "vs*", "vec", 0.0, 10.0)
    assert query_of(rest.sent[-1])["type"] == "vs*"


def test_legacy_pre_encoded_type_is_still_accepted(rest):
    """`createVectorSlice` is public and its documented type was `vs%2B`. A
    caller passing the old spelling must keep working and must produce the same
    corrected wire value, not a double-encoded one."""
    rest.createVectorSlice("vs", "vs%2B", "vec", 0.0, 10.0)
    assert query_of(rest.sent[-1])["type"] == "vs+"


def test_invalid_slice_type_still_rejected(rest):
    with pytest.raises(Exception):
        rest.createVectorSlice("vs", "nonsense", "vec", 0.0, 10.0)


# ------------------------------------------------ M22 residual: integrate2D

def test_integrate2d_coordinates_are_encoded(rest):
    """The coords were appended raw while every neighbouring append used `_q`."""
    rest.integrate2D("spec name", [(1.5, 2.5), (3.0, 4.0)])
    pairs = urllib.parse.parse_qsl(urllib.parse.urlsplit(rest.sent[-1]).query)
    assert [v for k, v in pairs if k == "xcoord"] == ["1.5", "3.0"]
    assert [v for k, v in pairs if k == "ycoord"] == ["2.5", "4.0"]
    assert query_of(rest.sent[-1])["spectrum"] == "spec name"
