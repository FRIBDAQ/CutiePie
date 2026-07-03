import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

import threading
import pytest

PyQt5 = pytest.importorskip("PyQt5", reason="PyQt5 not available in this environment")
from PyQt5.QtWidgets import QApplication
from unittest.mock import MagicMock


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


def test_rest_worker_instantiates(qapp):
    from services.thread_workers import RestWorker
    stop = threading.Event()
    stop.set()
    mock_rest = MagicMock()
    worker = RestWorker(mock_rest, 6, stop)
    assert hasattr(worker, 'connected')
    assert hasattr(worker, 'disconnected')
    assert hasattr(worker, 'tracesReady')


def test_auto_update_worker_instantiates(qapp):
    from services.thread_workers import AutoUpdateWorker
    stop = threading.Event()
    stop.set()
    skip = threading.Event()
    worker = AutoUpdateWorker(1, stop, skip)
    assert hasattr(worker, 'updateTriggered')


def test_parse_binding_entry_simple():
    from services.thread_workers import parse_binding_entry
    assert parse_binding_entry("add raw00 5") == ("add", "raw00", "5")
    assert parse_binding_entry("remove raw00 5") == ("remove", "raw00", "5")


def test_parse_binding_entry_braced_name_with_spaces():
    from services.thread_workers import parse_binding_entry
    assert parse_binding_entry("add {my spec} 12") == ("add", "my spec", "12")


def test_parse_binding_entry_unbraced_name_with_spaces():
    from services.thread_workers import parse_binding_entry
    assert parse_binding_entry("add my spec 12") == ("add", "my spec", "12")


def test_parse_binding_entry_malformed_raises():
    from services.thread_workers import parse_binding_entry
    with pytest.raises(ValueError):
        parse_binding_entry("add")
    with pytest.raises(ValueError):
        parse_binding_entry("add raw00")


# ---------------------------------------------------------------------------
# B1: listSpectrum's filter is a glob pattern — lookup must be exact-name.
# ---------------------------------------------------------------------------

def test_lookup_spectrum_info_plain_name_single_request():
    from services.thread_workers import lookup_spectrum_info
    rest = MagicMock()
    rest.listSpectrum.return_value = [{"name": "raw00", "type": "1"}]
    assert lookup_spectrum_info(rest, "raw00") == {"name": "raw00", "type": "1"}
    rest.listSpectrum.assert_called_once_with("raw00")


def test_lookup_spectrum_info_glob_name_never_returns_wrong_spectrum():
    from services.thread_workers import lookup_spectrum_info
    rest = MagicMock()
    def fake_list(pattern="*"):
        if pattern == "run[12]":
            # SpecTcl expands the glob: matches run1/run2, NOT the literal name
            return [{"name": "run1"}, {"name": "run2"}]
        return [{"name": "run1"}, {"name": "run2"}, {"name": "run[12]"}]
    rest.listSpectrum.side_effect = fake_list
    assert lookup_spectrum_info(rest, "run[12]") == {"name": "run[12]"}


def test_lookup_spectrum_info_glob_name_exact_hit_in_expansion_no_fallback():
    from services.thread_workers import lookup_spectrum_info
    rest = MagicMock()
    # pattern "a*" matches both "abc" and the literal "a*" — exact filter wins
    rest.listSpectrum.return_value = [{"name": "abc"}, {"name": "a*"}]
    assert lookup_spectrum_info(rest, "a*") == {"name": "a*"}
    rest.listSpectrum.assert_called_once()


def test_lookup_spectrum_info_missing_plain_name_no_fallback():
    from services.thread_workers import lookup_spectrum_info
    rest = MagicMock()
    rest.listSpectrum.return_value = []
    assert lookup_spectrum_info(rest, "gone") is None
    rest.listSpectrum.assert_called_once()   # plain names: no second request


def test_lookup_spectrum_info_missing_glob_name_after_fallback():
    from services.thread_workers import lookup_spectrum_info
    rest = MagicMock()
    rest.listSpectrum.return_value = []
    assert lookup_spectrum_info(rest, "gone[1]") is None
    assert rest.listSpectrum.call_count == 2   # pattern, then full list
