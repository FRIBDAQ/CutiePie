import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

import pytest

PyQt5 = pytest.importorskip("PyQt5", reason="PyQt5 not available in this environment")

def test_fit_manager_importable():
    from services.fit_manager import FitManager
    assert FitManager is not None

def test_fit_manager_constructor_params():
    # (this asserted a 'window' param that never existed — the real
    # name was parent_widget — so this test failed on any PyQt5 machine)
    from services.fit_manager import FitManager
    import inspect
    sig = inspect.signature(FitManager.__init__)
    params = list(sig.parameters)
    assert 'fit_factory' in params
    assert 'spectra' in params
    assert 'parent_widget' in params
    assert 'logger' in params
    # the service must not receive the extra popup widget
    assert 'extra_popup' not in params

def test_fit_manager_has_fit_method():
    from services.fit_manager import FitManager
    assert callable(FitManager.fit)

def test_fit_manager_has_delete_method():
    from services.fit_manager import FitManager
    assert callable(FitManager.deleteFit)
