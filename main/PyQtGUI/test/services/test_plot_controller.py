import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

import pytest

PyQt5 = pytest.importorskip("PyQt5", reason="PyQt5 not available in this environment")


def test_plot_controller_importable():
    from services.plot_controller import PlotController
    assert PlotController is not None


def test_plot_controller_constructor_params():
    from services.plot_controller import PlotController
    import inspect
    sig = inspect.signature(PlotController.__init__)
    params = list(sig.parameters)
    for p in ('wTab', 'wConf', 'spectra', 'parent_widget', 'logger'):
        assert p in params, f"Missing: {p}"


def test_plot_controller_has_update_plot():
    from services.plot_controller import PlotController
    assert callable(PlotController.updatePlot)


def test_plot_controller_has_zoom():
    from services.plot_controller import PlotController
    assert callable(PlotController.zoomInOut)


def test_centred_norm_importable():
    from services.plot_controller import centeredNorm
    assert centeredNorm is not None


# ---- customMinMax (vectorized) ---------------------------------------------
# Called unbound with a mock self so no PlotController construction is needed.

def _custom_min_max(data):
    import numpy as np
    from unittest.mock import MagicMock
    from services.plot_controller import PlotController
    mock_self = MagicMock()
    mock_self.minZ = 0.001
    mock_self.maxZ = 256
    return PlotController.customMinMax(mock_self, np.asanyarray(data))


def test_custom_min_max_small_array():
    assert _custom_min_max([[0, 5], [3, 0]]) == (3, 5)


def test_custom_min_max_large_array():
    import numpy as np
    data = np.zeros((300, 400))
    data[10, 20] = 7
    data[250, 350] = 2
    assert _custom_min_max(data) == (2, 7)


def test_custom_min_max_all_zero_returns_none():
    import numpy as np
    assert _custom_min_max(np.zeros((50, 50))) == (None, None)


def test_custom_min_max_fully_masked_returns_none():
    import numpy as np
    data = np.ma.masked_greater(np.ones((10, 10)), 0)  # everything masked
    assert _custom_min_max(data) == (None, None)


def test_custom_min_max_masked_bins_excluded():
    import numpy as np
    raw = np.array([[1, 100], [5, 0]])
    data = np.ma.masked_greater(raw, 50)  # cutoff masks the 100
    assert _custom_min_max(data) == (1, 5)


def test_custom_min_max_positives_masked_zeros_remain_returns_none():
    import numpy as np
    raw = np.array([[0, 100], [100, 0]])
    data = np.ma.masked_greater(raw, 50)  # positives all masked, zeros remain
    # data.any() is False → (None, None), identical to the pre-P1 behavior
    assert _custom_min_max(data) == (None, None)


def test_custom_min_max_nonzero_but_no_positive_falls_back_to_z_limits():
    import numpy as np
    # only reachable with non-positive nonzero values (impossible for count
    # data; documents the unified large-array-path fallback semantics)
    assert _custom_min_max(np.array([[-1, 0], [0, 0]])) == (0.001, 256)
