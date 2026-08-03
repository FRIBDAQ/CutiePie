"""Image overlay: load a picture, drop it on a pad, nudge and scale it.

Lifted out of MainWindow (ARCH.md §7, D3). The placement math already lives in
`services/figure_overlay.py`; what is here is the Qt shell it left behind — the
file dialog, the sliders, and the artist lifecycle.

**The artist lifecycle is the whole difficulty.** `drawFigure` adds an axes on
every call, so anything that redraws must take the previous one away first or
the figure grows an empty axes per slider tick — and `figure.axes` is the list
pad lookups index, so a leaked axes answers a click with an index past the end
of the grid. `_removeOverlayArtist` is the single teardown every path goes
through, and `onFigure` says only whether an overlay is currently up: the
sliders must leave it alone, or the next Add stacks a second overlay on the
first.
"""

import logging
import os

import cv2

from PyQt5.QtWidgets import QMessageBox

from services.figure_overlay import (compute_overlay_position,
                                     apply_joystick_move, apply_fine_move)


class OverlayController:

    def __init__(self, imaging, get_current_plot, get_selected_index,
                 get_grid, plot_position, open_image_dialog,
                 parent_widget=None, logger=None):
        self._imaging            = imaging          # extraPopup.imaging
        self._get_current_plot   = get_current_plot
        self._get_selected_index = get_selected_index
        self._get_grid           = get_grid         # () -> (rows, cols)
        self._plot_position      = plot_position    # (index) -> (row, col)
        # the picker stays in MainWindow next to the other file dialogs
        self._open_image_dialog  = open_image_dialog
        self._parent_widget      = parent_widget
        self.logger              = logger or logging.getLogger(__name__)

        self.LISEpic = None          # the loaded image, or None
        self.imgplot = None          # the drawn artist, or None
        self.overlay_ax = None       # the axes it was drawn on, or None
        self.onFigure = False        # is an overlay currently up
        self.xstart = self.ystart = 0.0
        self.alpha = self.zoomX = self.zoomY = 1.0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def loadFigure(self):
        self.logger.info('loadFigure')
        fileName = self._open_image_dialog()
        if not fileName:
            return
        self._imaging.loadLISE_name.setText(fileName)
        try:
            if os.path.isfile(fileName):
                self.LISEpic = cv2.imread(fileName, 0)
        except Exception:
            # Best-effort image load — a bad path / unreadable file must not
            # crash the GUI, but must not be silent either.
            self.logger.exception('loadFigure - image load failed')

    # ------------------------------------------------------------------
    # Artist lifecycle
    # ------------------------------------------------------------------

    def _removeOverlayArtist(self):
        """Detach the overlay image and its axes if one is drawn; True when
        there was something to remove.

        The axes goes too. drawFigure adds a fresh one on every call and only
        the image used to be removed, so a slider drag (valueChanged fires
        continuously) appended one empty axes per tick to figure.axes — the
        same list the pad lookups index (`list(figure.axes).index(inaxes)`), so
        a click on a leaked axes answers with an index past the end of the grid.
        A geometry change or an enlarge detaches the axes behind our back
        (InitializeCanvas delaxes everything), hence the membership test. Both
        artists are removed from whatever figure they were drawn on rather than
        from currentPlot's, which is a different figure once the user has
        switched tabs."""
        if self.imgplot is None:
            return False
        self.imgplot.remove()
        self.imgplot = None
        if self.overlay_ax is not None:
            fig = self.overlay_ax.figure
            if fig is not None and self.overlay_ax in fig.axes:
                fig.delaxes(self.overlay_ax)
            self.overlay_ax = None
        return True

    def drawFigure(self):
        self.logger.info('drawFigure')
        if self.LISEpic is None:
            # Reachable from a redraw when a later Load failed under a live
            # overlay (cv2.imread answers None instead of raising): the image
            # is gone from the canvas, so the flag must not still claim one.
            self.logger.warning('drawFigure - no image loaded')
            self.onFigure = False
            return
        self.alpha = self._imaging.alpha_slider.value()/10
        self.zoomX = self._imaging.zoomX_slider.value()/10
        self.zoomY = self._imaging.zoomY_slider.value()/10

        cp = self._get_current_plot()
        self.overlay_ax = ax = cp.figure.add_axes(
            [self.xstart, self.ystart, self.zoomX, self.zoomY], frameon=True)
        ax.axis('off')
        self.imgplot = ax.imshow(self.LISEpic,
                                 aspect='auto',
                                 alpha=self.alpha)

        cp.canvas.draw()

    def deleteFigure(self):
        self.logger.info('deleteFigure')
        if not self._removeOverlayArtist():
            return
        self.onFigure = False
        self._get_current_plot().canvas.draw()

    def _redrawOverlay(self):
        """Slider handlers: redraw the overlay in place, or do nothing when
        none is up. Deliberately leaves onFigure alone — routing these through
        deleteFigure cleared the flag while the image stayed on screen, and the
        next Add then drew a second overlay over the first."""
        if self._removeOverlayArtist():
            self.drawFigure()

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def indexToStartPosition(self, index):
        self.logger.info('indexToStartPosition')
        row, col = self._get_grid()
        i, j = self._plot_position(index)
        self.xstart, self.ystart = compute_overlay_position(row, col, i, j)

    def _fineMove(self, direction):
        if not self._removeOverlayArtist():
            return
        self.xstart, self.ystart = apply_fine_move(self.xstart, self.ystart, direction)
        self.drawFigure()

    def fineUpMove(self):    self._fineMove("up")

    def fineDownMove(self):  self._fineMove("down")

    def fineLeftMove(self):  self._fineMove("left")

    def fineRightMove(self): self._fineMove("right")

    def moveFigure(self):
        self.logger.info('moveFigure')
        try:
            if not self._removeOverlayArtist():
                return
            self.xstart, self.ystart = apply_joystick_move(
                self.xstart, self.ystart,
                self._imaging.joystick.direction,
                self._imaging.joystick.distance)
            self.drawFigure()
        except Exception:
            # Best-effort overlay nudge — a bad joystick read must not crash the
            # GUI, but must not be silent either. (The missing-overlay case is
            # handled above now, not by this clause.)
            self.logger.exception('moveFigure - overlay move failed')

    # ------------------------------------------------------------------
    # Sliders and Add
    # ------------------------------------------------------------------

    def transFigure(self):
        self.logger.info('transFigure')
        self._imaging.alpha_label.setText("Transparency Level ({} %)".format(self._imaging.alpha_slider.value()*10))
        self._redrawOverlay()

    def zoomFigureX(self):
        self.logger.info('zoomFigureX')
        self._imaging.zoomX_label.setText("Zoom X Level ({} %)".format(self._imaging.zoomX_slider.value()*10))
        self._redrawOverlay()

    def zoomFigureY(self):
        self.logger.info('zoomFigureY')
        self._imaging.zoomY_label.setText("Zoom Y Level ({} %)".format(self._imaging.zoomY_slider.value()*10))
        self._redrawOverlay()

    def addFigure(self):
        self.logger.info('addFigure')
        if self.LISEpic is None:
            QMessageBox.warning(self._parent_widget, "Overlay", "Load an image first.")
            return
        # plotPosition returns None for an unselected pad, so the unpack in
        # indexToStartPosition raises TypeError — this is the case the old
        # `except NameError: raise` was reaching for and never caught.
        index = self._get_selected_index()
        if index is None:
            QMessageBox.warning(self._parent_widget, "Overlay", "Please select one histogram.")
            return
        if self.onFigure:
            return
        self.indexToStartPosition(index)
        self.drawFigure()
        self.onFigure = True
