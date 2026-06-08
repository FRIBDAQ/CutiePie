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
    for p in ('window', 'wTab', 'wConf', 'logger'):
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
