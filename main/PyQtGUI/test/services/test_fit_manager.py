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


def test_calibration_prompt_is_deferred_out_of_the_press_handler(monkeypatch):
    """A ctrl+click during energy calibration must not open the modal energy
    prompt synchronously from inside the canvas press handler. The prompt
    runs on the next event-loop pass, a second click while one is pending is
    ignored, and the point lands once the prompt returns."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import numpy as np
    from matplotlib.figure import Figure
    from PyQt5.QtWidgets import QApplication, QInputDialog
    from PyQt5.QtCore import QTimer
    from services.fit_manager import FitManager

    QApplication.instance() or QApplication([])
    mgr = FitManager(fit_factory=None, spectra=None, parent_widget=None)
    ax = Figure().add_subplot(111)
    x = np.arange(0.0, 1000.0)
    y = np.zeros_like(x); y[500] = 10.0

    prompts = []
    monkeypatch.setattr(QInputDialog, "getDouble",
                        staticmethod(lambda *a, **k: (prompts.append(a) or (5486.0, True))))

    seen = {}

    def drive():
        cal = mgr._cal
        if cal is None:
            QTimer.singleShot(0, drive)
            return
        cal.add_point(500.0)
        cal.add_point(500.0)          # second click while the first is pending
        seen["prompted_synchronously"] = bool(prompts)
        seen["pending"] = cal.pending

        def check():
            seen["mu"], seen["E"] = list(cal.MU), list(cal.E)
            seen["prompt_count"] = len(prompts)
            seen["pending_after"] = cal.pending
            cal.dlg.reject()
        QTimer.singleShot(50, check)

    QTimer.singleShot(0, drive)
    # failsafe so a broken dialog can never hang the suite
    QTimer.singleShot(5000, lambda: mgr._cal is not None and mgr._cal.dlg.reject())
    result = mgr._prompt_energy_calibration(ax, x, y, min_pts=2, max_pts=4, snap_halfwin=150)

    assert result is None                          # dialog was cancelled
    assert seen["prompted_synchronously"] is False # nothing modal inside add_point
    assert seen["pending"] is True
    assert seen["prompt_count"] == 1               # the duplicate click was dropped
    assert seen["mu"] == [500.0] and seen["E"] == [5486.0]
    assert seen["pending_after"] is False
    assert mgr._cal is None                        # dialog teardown still clears state
