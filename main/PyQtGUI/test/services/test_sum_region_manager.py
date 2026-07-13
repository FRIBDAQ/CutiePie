import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

import pytest

PyQt5 = pytest.importorskip("PyQt5", reason="PyQt5 not available in this environment")

def test_sum_region_manager_importable():
    from services.sum_region_manager import SumRegionManager
    assert SumRegionManager is not None

def test_sum_region_manager_constructor_params():
    from services.sum_region_manager import SumRegionManager
    import inspect
    sig = inspect.signature(SumRegionManager.__init__)
    params = list(sig.parameters)
    # histogram combo and integrate popup are gone; the sum popup stays only
    # for the co-owned listRegionLine/prevPoint buffer, reads arrive via seams.
    for p in ('get_histo_names', 'sum_popup', 'parent_widget', 'logger'):
        assert p in params, f"Missing param: {p}"
    for p in ('histo_list', 'integrate_popup', 'window'):
        assert p not in params, f"Widget param should be inverted away: {p}"

def test_sum_region_manager_has_integrate():
    from services.sum_region_manager import SumRegionManager
    assert callable(SumRegionManager.integrate)

def test_sum_region_manager_has_create():
    from services.sum_region_manager import SumRegionManager
    assert callable(SumRegionManager.createSumRegion)
