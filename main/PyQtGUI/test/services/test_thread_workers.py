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
