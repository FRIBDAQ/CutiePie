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
    for p in ('spectra',
              'update_intervals', 'update_intervals_user',
              'stop_rest', 'stop_auto', 'skip_auto'):
        assert p in params, f"Missing: {p}"
    # H2: the service must not receive widgets (main-window bundle or popup)
    assert 'wConf' not in params
    assert 'connect_config' not in params

def test_connection_manager_has_connect():
    from services.connection_manager import ConnectionManager
    assert callable(ConnectionManager.connectShMem)

def test_connection_manager_has_auto_update():
    from services.connection_manager import ConnectionManager
    assert callable(ConnectionManager.autoUpdateStart)
