import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

import pytest

PyQt5 = pytest.importorskip("PyQt5", reason="PyQt5 not available in this environment")

def test_gate_manager_importable():
    from services.gate_manager import GateManager
    assert GateManager is not None

def test_gate_manager_constructor_params():
    from services.gate_manager import GateManager
    import inspect
    sig = inspect.signature(GateManager.__init__)
    params = list(sig.parameters)
    assert 'window' in params
    assert 'spectra' in params
    assert 'gate_popup' in params
    assert 'logger' in params

def test_gate_manager_has_create_gate():
    from services.gate_manager import GateManager
    assert callable(GateManager.createGate)

def test_gate_manager_has_push_gate():
    from services.gate_manager import GateManager
    assert callable(GateManager.pushGateToREST)
