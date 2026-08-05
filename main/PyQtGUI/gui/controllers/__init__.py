"""Qt-owning controllers lifted out of MainWindow. Unlike ``gui/services/``, a
controller here MAY hold widgets: they are the Qt shells left behind when the
Qt-free halves moved into services, and MainWindow still owns their wiring."""

__all__ = [
    "copy_properties_controller",
    "geometry_controller",
    "jupyter_controller",
    "overlay_controller",
    "peak_fit2_controller",
    "peak_scan_controller",
]
