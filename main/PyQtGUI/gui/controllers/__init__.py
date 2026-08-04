"""Qt-owning controllers lifted out of MainWindow. Unlike ``gui/services/``, a
controller here MAY hold widgets: they are the Qt shells left behind when the
Qt-free halves moved into services, and MainWindow still owns their wiring."""
