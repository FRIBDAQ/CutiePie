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
