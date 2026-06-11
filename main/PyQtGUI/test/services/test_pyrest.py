import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

import logging
import threading

import pytest

httplib2 = pytest.importorskip("httplib2", reason="httplib2 not available in this environment")

from PyREST import PyREST


def _make_client():
    return PyREST(logging.getLogger("test"), "localhost", "8000")


def test_http_same_thread_reuses_instance():
    client = _make_client()
    assert client._http is client._http


def test_http_distinct_per_thread():
    client = _make_client()
    main_http = client._http
    seen = {}

    def grab():
        seen['worker'] = client._http

    t = threading.Thread(target=grab)
    t.start()
    t.join()

    assert seen['worker'] is not None
    assert seen['worker'] is not main_http


def test_http_has_configured_timeout():
    client = _make_client()
    assert client._http.timeout == PyREST._HTTP_TIMEOUT


def test_build_url_basic_and_params():
    client = _make_client()
    assert client._build_url("spectcl/spectrum/list") == \
        "http://localhost:8000/spectcl/spectrum/list"
    url = client._build_url("spectcl/parameter/list", filter="*")
    assert url == "http://localhost:8000/spectcl/parameter/list?filter=%2A"


def test_build_url_skips_none_params():
    client = _make_client()
    url = client._build_url("spectcl/parameter/edit", name="p", bins=None)
    assert url == "http://localhost:8000/spectcl/parameter/edit?name=p"
