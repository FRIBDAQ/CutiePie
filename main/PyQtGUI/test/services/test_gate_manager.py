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
    # kept collaborators / documented residuals
    assert 'spectra' in params
    assert 'gate_popup' in params          # residual: listRegionLine/prevPoint buffer
    assert 'sum_region_popup' in params    # residual: shared draw buffer
    assert 'parent_widget' in params
    assert 'logger' in params
    # step-1 seams (widget reads pulled through callables)
    for seam in ('get_hide', 'get_annotate', 'get_edit_disable',
                 'get_readout', 'get_gate_type', 'get_gate_name'):
        assert seam in params
    # widget refs inverted away must be ABSENT
    for gone in ('window', 'gate_hide_cb', 'gate_annotation_cb',
                 'gate_edit_disable_cb'):
        assert gone not in params

def test_gate_manager_has_create_gate():
    from services.gate_manager import GateManager
    assert callable(GateManager.createGate)

def test_gate_manager_has_push_gate():
    from services.gate_manager import GateManager
    assert callable(GateManager.pushGateToREST)
