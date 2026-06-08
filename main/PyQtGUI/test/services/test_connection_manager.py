import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

import pytest

PyQt5 = pytest.importorskip("PyQt5", reason="PyQt5 not available in this environment")

def test_connection_manager_importable():
    from services.connection_manager import ConnectionManager
    assert ConnectionManager is not None

def test_connection_manager_constructor_params():
    from services.connection_manager import ConnectionManager
    import inspect
    sig = inspect.signature(ConnectionManager.__init__)
    params = list(sig.parameters)
    for p in ('window', 'wConf', 'connect_config', 'stop_rest', 'stop_auto', 'skip_auto', 'logger'):
        assert p in params, f"Missing: {p}"

def test_connection_manager_has_connect():
    from services.connection_manager import ConnectionManager
    assert callable(ConnectionManager.connectShMem)

def test_connection_manager_has_auto_update():
    from services.connection_manager import ConnectionManager
    assert callable(ConnectionManager.autoUpdateStart)
