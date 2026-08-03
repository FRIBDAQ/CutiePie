"""Qt-owning controllers lifted out of MainWindow (ARCH.md §7, D1-D9).

Unlike ``gui/services/``, a controller here MAY hold widgets: these are the Qt
shells that were left behind when the Qt-free halves moved into services. What
they must not do is own the composition root — MainWindow builds them and
supplies the seams they read the rest of the window through.
"""
