import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from copy import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PyQt5 import QtCore
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMenu, QMessageBox, QFileDialog


class centeredNorm(colors.Normalize):
    def __init__(self, data, vcenter=0, halfrange=None, clip=False):
        if halfrange is None:
            halfrange = np.max(np.abs(data - vcenter))
        super().__init__(vmin=vcenter - halfrange, vmax=vcenter + halfrange, clip=clip)


class PlotController(QObject):
    """Owns all plot rendering, axis control, zoom, and canvas management."""

    # cutoff-popup rendering inverted into signals — MainWindow owns the
    # popup (adapters _show_cutoff_popup / cutoffp.close). Payload keys:
    # name, dim, xmin, xmax, ymin, ymax, zmin, zmax (z entries None for 1D).
    cutoffPopupPrepared      = pyqtSignal(dict)
    cutoffPopupCloseRequested = pyqtSignal()

    def __init__(self, spectra,
                 get_current_plot, get_geo, set_geo,
                 get_spectrum_info, set_spectrum_info, get_spectrum_info_dict,
                 name_from_index, get_enlarged_spectrum,
                 auto_index, next_index, bind_dynamic_signal,
                 draw_gate, clean_popup_exit, auto_update_start,
                 stop_auto_update_thread,
                 min_y, max_y, min_z, max_z,
                 parent_widget=None, logger=None):
        super().__init__()
        self._spectra                 = spectra
        self._get_current_plot        = get_current_plot        # () -> wPlot widget
        self._get_geo                 = get_geo                 # () -> {index: name}
        self._set_geo                 = set_geo                 # (index, name) -> None
        self._get_spectrum_info       = get_spectrum_info       # (key, index=) -> value
        self._set_spectrum_info       = set_spectrum_info       # (key=val, index=) -> None
        self._get_spectrum_info_dict  = get_spectrum_info_dict  # () -> dict
        self._name_from_index         = name_from_index         # (index) -> str
        self._get_enlarged_spectrum   = get_enlarged_spectrum   # () -> spectrum|None
        self._auto_index              = auto_index              # () -> int
        self._next_index              = next_index              # () -> int
        self._bind_dynamic_signal     = bind_dynamic_signal     # () -> None
        self._draw_gate               = draw_gate               # (index) -> None
        self._clean_popup_exit        = clean_popup_exit        # (doClose) -> None
        self._auto_update_start       = auto_update_start       # () -> None
        self._stop_auto_update_thread = stop_auto_update_thread # threading.Event
        self.minY                     = min_y
        self.maxY                     = max_y
        self.minZ                     = min_z
        self.maxZ                     = max_z
        self._parent_widget           = parent_widget
        self.logger                   = logger or logging.getLogger(__name__)

        self.palette          = None
        self.old_cmap         = None
        self.geometry_applied = False
        self._layout_dirty    = False
        # (change-driven redraw): fingerprint of the last frame the
        # auto-update tick rendered. When an unforced (timer) tick produces an
        # identical fingerprint, the redundant full-figure redraw is skipped.
        self._last_tick_signature = None
        # per-pad fingerprints of the last frame actually drawn, so an unforced
        # tick can re-render only the pads whose counts moved
        self._last_pad_signatures = None

    # ------------------------------------------------------------------
    # Canvas / layout
    # ------------------------------------------------------------------

    def markGeometryApplied(self):
        """Called by MainWindow.setCanvasLayout after the canvas grid is
        (re)initialized (the geometry combos and tab widget live there)."""
        self.logger.info('markGeometryApplied')
        self.geometry_applied = True

    # ------------------------------------------------------------------
    # Axis scaling
    # ------------------------------------------------------------------

    # sets x, y scales for 1d and x, y, z scales for 2d depending on the scale
    # identifier and on axisIsLog; does all the scaling operations
    def setAxisScale(self, ax, index, *scale):
        self.logger.debug('setAxisScale - index: %s', index)

        cp            = self._get_current_plot()
        axisIsLog     = self._get_spectrum_info("log", index=index)
        axisIsAutoScale = cp.histo_autoscale.isChecked()
        name          = self._name_from_index(index)
        dim           = self._spectra.get(name, "dim")

        # A pad the store no longer knows has dim None, not 2. Changing the
        # geometry blanks the pad-to-name map while the per-tab slots keep the
        # artist that was drawn there, and a spectrum deleted server-side
        # leaves the same gap.
        if dim not in (1, 2):
            self.logger.warning('setAxisScale - index %s names no spectrum in the store (%s); skipped',
                                index, name)
            return

        if dim == 1:
            xmin = self._get_spectrum_info("minx", index=index)
            xmax = self._get_spectrum_info("maxx", index=index)
            if "x" in scale and xmin is not None and xmax is not None:
                ax.set_xlim(xmin, xmax)
            if "y" in scale or "log" in scale:
                ymin = self._get_spectrum_info("miny", index=index)
                ymax = self._get_spectrum_info("maxy", index=index)
                if (not ymin or ymin is None or ymin == 0) and (not ymax or ymax is None or ymax == 0):
                    ymin = self.minY
                    ymax = self.maxY
                # Asymmetric guard: the fallback above only fires when BOTH are
                # empty; when just one is None (e.g. miny None, maxy a real value)
                # the log branch's `ymin <= 0` would raise TypeError. Coerce each
                # independently to the log-safe defaults.
                if ymin is None:
                    ymin = self.minY
                if ymax is None:
                    ymax = self.maxY
                if axisIsAutoScale:
                    xmin, xmax = ax.get_xlim()
                    ymax = self.getMinMaxInRange(index, xmin=xmin, xmax=xmax)
                if axisIsLog:
                    if ymin <= 0:
                        ymin = 0.001
                    if ymax <= 0:
                        self.logger.warning('setAxisScale - all value <0 for : %s - cannot log scale',
                                            self._name_from_index(index))
                    else:
                        ax.set_ylim(ymin, ymax)
                        ax.set_yscale("log")
                else:
                    ax.set_ylim(ymin, ymax)
                    ax.set_yscale("linear")
                self._set_spectrum_info(miny=ymin, index=index)
                self._set_spectrum_info(maxy=ymax, index=index)
        else:
            xmin = self._get_spectrum_info("minx", index=index)
            xmax = self._get_spectrum_info("maxx", index=index)
            ymin = self._get_spectrum_info("miny", index=index)
            ymax = self._get_spectrum_info("maxy", index=index)

            if "x" in scale and xmin is not None and xmax is not None:
                ax.set_xlim(xmin, xmax)
            if "y" in scale and ymin is not None and ymax is not None:
                ax.set_ylim(ymin, ymax)
            if "z" in scale or "log" in scale:
                zmin     = self._get_spectrum_info("minz", index=index)
                zmax     = self._get_spectrum_info("maxz", index=index)
                spectrum = self._get_spectrum_info("spectrum", index=index)
                if spectrum is None:
                    return
                if (not zmin or zmin is None or zmin == 0) and (not zmax or zmax is None or zmax == 0):
                    zmin = self.minZ
                    zmax = self.maxZ
                if axisIsAutoScale:
                    xmin, xmax = ax.get_xlim()
                    ymin, ymax = ax.get_ylim()
                    zmin, zmax = self.getMinMaxInRange(index, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
                    self._set_spectrum_info(maxz=zmax, index=index)
                    self._set_spectrum_info(minz=zmin, index=index)
                spectrum.set_clim(vmin=zmin, vmax=zmax)
                if axisIsLog:
                    self.setCmapNorm("log", index)
                else:
                    self.setCmapNorm("linear", index)
                self._set_spectrum_info(spectrum=spectrum, index=index)

    # where the color bar is defined
    def setCmapNorm(self, scale, index):
        self.logger.info('setCmapNorm')
        validScales = ["linear", "log", "linearCentered"]
        if scale not in validScales or index is None:
            self.logger.debug('setCmapNorm - scale not in validScales or index is None')
            return
        spectrum = self._get_spectrum_info("spectrum", index=index)
        zmin, zmax = spectrum.get_clim()

        if scale == validScales[0]:
            if zmin > zmax:
                self.logger.warning('setCmapNorm - zmin > zmax')
                spectrum.set_norm(colors.Normalize(vmin=self.minZ, vmax=self.maxZ))
            else:
                spectrum.set_norm(colors.Normalize(vmin=zmin, vmax=zmax))
        elif scale == validScales[1]:
            if zmin is None or zmin <= 0:
                zmin = 0.001
                self.logger.warning('setCmapNorm - LogNorm with zmin<=0 coerced to 0.001, may want to use CenteredNorm')
            if zmax is not None and zmin > zmax:
                self.logger.warning('setCmapNorm - zmin > zmax')
                spectrum.set_norm(colors.LogNorm(vmin=self.minZ, vmax=self.maxZ))
            else:
                spectrum.set_norm(colors.LogNorm(vmin=zmin, vmax=zmax))
        elif scale == validScales[2]:
            palette = copy(plt.cm.jet)
            palette.set_bad(color='white')
            data = self._cutoff_masked_data(index)
            spectrum.set_cmap(palette)
            spectrum.set_norm(centeredNorm(data, 50000))
        if self._get_enlarged_spectrum() is None:
            ax    = spectrum.axes
            label = "colorbar_" + str(index)
            cp    = self._get_current_plot()
            cax   = next((a for a in cp.figure.axes if a.get_label() == label), None)
            if cax is None:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cax.set_label(label)
                cp.figure.colorbar(spectrum, cax=cax, orientation='vertical')
            # else: norm already updated on spectrum; the linked colorbar redraws automatically

    # callback for histo_autoscale, calls setAxisScale
    def autoScaleAxisBox(self, forIndex):
        self.logger.info('autoScaleAxisBox - forIndex: %s', forIndex)
        try:
            cp = self._get_current_plot()
            ax = None
            if cp.isEnlarged:
                ax  = self._get_spectrum_info("axis", index=0)
                dim = self._spectra.get(self._name_from_index(0), "dim")
                if ax is None:
                    self.logger.debug('autoScaleAxisBox - isEnlarged TRUE - ax is None ')
                    return
                if dim == 1:
                    self.setAxisScale(ax, 0, "y")
                elif dim == 2:
                    self.setAxisScale(ax, 0, "z")
                self._draw_gate(0)
            elif forIndex is not None:
                ax  = self._get_spectrum_info("axis", index=forIndex)
                dim = self._spectra.get(self._name_from_index(forIndex), "dim")
                if ax is None:
                    self.logger.debug('autoScaleAxisBox - forIndex - ax is None ')
                    return
                if dim == 1:
                    self.setAxisScale(ax, forIndex, "y")
                elif dim == 2:
                    self.setAxisScale(ax, forIndex, "z")
            else:
                for index, name in self._get_geo().items():
                    if name:
                        ax  = self._get_spectrum_info("axis", index=index)
                        dim = self._spectra.get(self._name_from_index(index), "dim")
                        if ax is None:
                            self.logger.debug('autoScaleAxisBox - isEnlarged FALSE - ax is None ')
                            return
                        if dim == 1:
                            self.setAxisScale(ax, index, "y")
                        elif dim == 2:
                            self.setAxisScale(ax, index, "z")
                        self._draw_gate(index)
            cp.canvas.draw()
        except Exception:
            self.logger.debug('autoScaleAxisBox - exception', exc_info=True)

    # ------------------------------------------------------------------
    # Range / min-max helpers
    # ------------------------------------------------------------------

    # data max within a user defined range. For 2D give two ranges (x,y), for 1D range x.
    def getMinMaxInRange(self, index, **limits):
        self.logger.debug('getMinMaxInRange - limits: %s', limits)
        result = None
        if not limits:
            self.logger.warning('getMinMaxInRange - limits identifier not valid - expect xmin=val, xmax=val etc. for y with 2D')
            return
        if "xmin" in limits and "xmax" in limits:
            xmin = limits["xmin"]
            xmax = limits["xmax"]
        if "ymin" in limits and "ymax" in limits:
            ymin = limits["ymin"]
            ymax = limits["ymax"]

        name  = self._name_from_index(index)
        dim   = self._spectra.get(name, "dim")
        minx  = self._spectra.get(name, "minx")
        maxx  = self._spectra.get(name, "maxx")
        binx  = self._spectra.get(name, "binx")
        data  = self._cutoff_masked_data(index)
        stepx = (float(maxx) - float(minx)) / float(binx)
        binminx = int((xmin - minx) / stepx)
        binmaxx = int((xmax - minx) / stepx)
        if dim == 1:
            try:
                result = data[binminx + 1:binmaxx + 2].max() * 1.1
            except Exception:
                self.logger.debug('getMinMaxInRange - dim == 1 - exception occured', exc_info=True)
                return self.maxY
        elif dim == 2:
            try:
                miny  = self._spectra.get(name, "miny")
                maxy  = self._spectra.get(name, "maxy")
                biny  = self._spectra.get(name, "biny")
                stepy = (float(maxy) - float(miny)) / float(biny)
                binminy = int((ymin - miny) / stepy)
                binmaxy = int((ymax - miny) / stepy)
                minimum, maximum = self.customMinMax(data[binminy:binmaxy + 1, binminx:binmaxx + 1])
                result = minimum, maximum
            except Exception:
                self.logger.debug('getMinMaxInRange - dim == 2 - exception occured', exc_info=True)
                return self.minZ, self.maxZ
        return result

    # have seen a malloc error if the data array is too large, so split it into
    # sub-arrays with sub-(min, max) and take the global (min, max) of those
    def customMinMax(self, data):
        """Return (min, max) of the positive counts in `data`, one vectorized
        pass. Replaces a tiled Python-loop scan that visited every element
        anyway."""
        self.logger.debug('customMinMax')
        if not data.any():
            return None, None
        values = data.compressed() if isinstance(data, np.ma.MaskedArray) else data
        positive = values[values > 0]
        if positive.size == 0:
            return self.minZ, self.maxZ
        return positive.min(), positive.max()

    # axis limits in the form [[xmin, xmax], [ymin, ymax]]
    def getAxisProperties(self, index):
        self.logger.info('getAxisProperties')
        try:
            ax = self._get_spectrum_info("axis", index=index)
            if ax is None:
                return None
            else:
                return list(ax.get_xlim()), list(ax.get_ylim())
        except Exception:
            self.logger.debug('getAxisProperties - exception occured', exc_info=True)

    # ------------------------------------------------------------------
    # Zoom / toolbar callbacks
    # ------------------------------------------------------------------

    # callback for plusButton/minusButton
    def zoomInOut(self, arg):
        self.logger.info('zoomInOut - arg: %s', arg)
        cp = self._get_current_plot()
        cp.histo_autoscale.setChecked(False)
        index = self._auto_index()
        spectrum = self._get_spectrum_info("spectrum", index=index)
        if spectrum is None:
            return
        ax  = spectrum.axes
        dim = self._spectra.get(self._name_from_index(index), "dim")
        if dim == 1:
            ymin, ymax = ax.get_ylim()
            if arg == "in":
                ymax = ymax * 0.5
            elif arg == "out":
                ymax = ymax * 2
            ax.set_ylim(ymin, ymax)
            self._set_spectrum_info(miny=ymin, index=index)
            self._set_spectrum_info(maxy=ymax, index=index)
            self._set_spectrum_info(spectrum=spectrum, index=index)
        elif dim == 2:
            zmin, zmax = spectrum.get_clim()
            if arg == "in":
                zmax = zmax * 0.5
            elif arg == "out":
                zmax = zmax * 2
            spectrum.set_clim(zmin, zmax)
            self._set_spectrum_info(minz=zmin, index=index)
            self._set_spectrum_info(maxz=zmax, index=index)
            self._set_spectrum_info(spectrum=spectrum, index=index)
        self._draw_gate(index)
        cp.canvas.draw()

    def zoomCallback(self, event):
        self.logger.info('zoomCallback')
        self._get_current_plot().zoomPress = True

    # used by customZoom button, triggers the toolbar zoom action
    def customZoomButtonCallback(self):
        self.logger.info('customZoomButtonCallback')
        cp = self._get_current_plot()
        cp.histo_autoscale.setChecked(False)
        if cp.zoomPress:
            cp.zoom_action.triggered.emit()
            cp.zoom_action.setChecked(False)
            cp.customZoomButton.setDown(False)
            cp.zoomPress = False
        else:
            cp.zoom_action.triggered.emit()
            cp.zoom_action.setChecked(True)
            cp.customZoomButton.setDown(True)

    # used by customHome button; resets the axis limits to the ReST definitions,
    # for the plot at index or for all plots when index is not given
    def customHomeButtonCallback(self, index=None):
        self.logger.info('customHomeButtonCallback - index: %s', index)
        index_list = [idx for idx, name in self._get_geo().items() if index is None]
        if index is not None:
            index_list = [index]
        for idx in index_list:
            ax       = None
            spectrum = self._get_spectrum_info("spectrum", index=idx)
            if spectrum is None:
                continue
            ax   = spectrum.axes
            name = self._name_from_index(idx)
            dim  = self._spectra.get(name, "dim")
            xmin = self._spectra.get(name, "minx")
            xmax = self._spectra.get(name, "maxx")
            ymin = self._spectra.get(name, "miny")
            ymax = self._spectra.get(name, "maxy")

            ax.set_xlim(xmin, xmax)
            if dim == 1:
                # counts start at zero; the store holds no y range for a 1-D
                # spectrum, so Home pins the bottom here rather than reading it
                ymin = 0
                ymax = self.getMinMaxInRange(idx, xmin=xmin, xmax=xmax)
                ax.set_ylim(ymin, ymax)
                if self._get_spectrum_info("log", index=idx):
                    ax.set_yscale("linear")
            if dim == 2:
                ax.set_ylim(ymin, ymax)
                zmin, zmax = self.getMinMaxInRange(idx, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
                spectrum.set_clim(vmin=zmin, vmax=zmax)
                self.setCmapNorm("linear", idx)
                self._set_spectrum_info(maxz=zmax, index=idx)
                self._set_spectrum_info(minz=zmin, index=idx)
            self._draw_gate(idx)

            self._set_spectrum_info(log=None, index=idx)
            self._set_spectrum_info(minx=xmin, index=idx)
            self._set_spectrum_info(maxx=xmax, index=idx)
            self._set_spectrum_info(miny=ymin, index=idx)
            self._set_spectrum_info(maxy=ymax, index=idx)
            self._set_spectrum_info(spectrum=spectrum, index=idx)
        self._get_current_plot().canvas.draw()

    # used by logButton; sets the log scale for the plot at index, or for all
    # plots on logAll/unlogAll. Calls setAxisScale
    def logButtonCallback(self, *arg):
        self.logger.info('logButtonCallback - arg: %s', arg)
        index       = None
        logAllPlot  = False
        unlogAllPlot = False
        if "logAll" in arg:
            logAllPlot = True
        elif "unlogAll" in arg:
            unlogAllPlot = True
        else:
            index = arg[0]

        cp         = self._get_current_plot()
        index_list = [idx for idx, name in self._get_geo().items() if logAllPlot or unlogAllPlot]

        if index is not None:
            index_list = [index]
        for idx in index_list:
            ax       = None
            spectrum = self._get_spectrum_info("spectrum", index=idx)
            if spectrum is None:
                continue
            ax = spectrum.axes
            if logAllPlot:
                self._set_spectrum_info(index=idx, log=True)
            elif unlogAllPlot:
                self._set_spectrum_info(index=idx, log=False)
            elif self._get_spectrum_info("log", index=idx) and not logAllPlot and not unlogAllPlot:
                self._set_spectrum_info(index=idx, log=False)
            elif not self._get_spectrum_info("log", index=idx) and not logAllPlot and not unlogAllPlot:
                self._set_spectrum_info(index=idx, log=True)

            self.setAxisScale(ax, idx, "log")
        cp.canvas.draw()

    # callback on right click on customZoomButton
    def zoom_handle_right_click(self):
        self.logger.info('zoom_handle_right_click')
        menu  = QMenu()
        item1 = menu.addAction("Set Zoom Range Manually")
        cp    = self._get_current_plot()
        index = cp.selected_plot_index
        if index is None:
            return QMessageBox.about(self._parent_widget, "Warning!", "Please add/select a spectrum")
        else:
            item1.triggered.connect(self.cutoffButtonCallback)
            menuPosX = cp.mapToGlobal(QtCore.QPoint(0, 0)).x() + cp.customZoomButton.geometry().topLeft().x()
            menuPosY = cp.mapToGlobal(QtCore.QPoint(0, 0)).y() + cp.customZoomButton.geometry().topLeft().y()
            menuPos  = QtCore.QPoint(menuPosX, menuPosY)
            menu.exec_(menuPos)

    # callback on right click on customHomeButton
    def handle_right_click(self):
        self.logger.info('handle_right_click')
        menu  = QMenu()
        item1 = menu.addAction("Reset all")
        item1.triggered.connect(lambda: self.customHomeButtonCallback())
        plotgui  = self._get_current_plot()
        menuPosX = plotgui.mapToGlobal(QtCore.QPoint(0, 0)).x() + plotgui.customHomeButton.geometry().topLeft().x()
        menuPosY = plotgui.mapToGlobal(QtCore.QPoint(0, 0)).y() + plotgui.customHomeButton.geometry().topLeft().y()
        menuPos  = QtCore.QPoint(menuPosX, menuPosY)
        menu.exec_(menuPos)

    # callback on right click on logButton
    def log_handle_right_click(self):
        self.logger.info('log_handle_right_click')
        menu  = QMenu()
        item1 = menu.addAction("Log all")
        item2 = menu.addAction("unLog all")
        item1.triggered.connect(lambda: self.logButtonCallback("logAll"))
        item2.triggered.connect(lambda: self.logButtonCallback("unlogAll"))
        plotgui  = self._get_current_plot()
        menuPosX = plotgui.mapToGlobal(QtCore.QPoint(0, 0)).x() + plotgui.logButton.geometry().topLeft().x()
        menuPosY = plotgui.mapToGlobal(QtCore.QPoint(0, 0)).y() + plotgui.logButton.geometry().topLeft().y()
        menuPos  = QtCore.QPoint(menuPosX, menuPosY)
        menu.exec_(menuPos)

    # ------------------------------------------------------------------
    # Cutoff popup
    # ------------------------------------------------------------------

    def okCutoff(self, xmin_text="", xmax_text="", ymin_text="", ymax_text="",
                 zmin_text="", zmax_text=""):
        """Apply the cutoff/zoom popup values, supplied as text by the
        MainWindow adapter."""
        self.logger.info('okCutoff')
        cp    = self._get_current_plot()
        index = cp.selected_plot_index
        if index is None:
            self.logger.debug('okCutoff - index is None')
            return
        spectrum = self._get_spectrum_info("spectrum", index=index)
        if spectrum is None:
            return
        ax = self._get_spectrum_info("axis", index=index)
        if ax is None:
            self.logger.debug('okCutoff - ax is None')
            return

        name       = self._name_from_index(index)
        dim        = self._spectra.get(name, "dim")
        rangeXmin  = xmin_text
        rangeXmax  = xmax_text
        rangeYmin  = ymin_text
        rangeYmax  = ymax_text

        cutoffVal  = [None, None]
        cutoffMin  = zmin_text
        cutoffMax  = zmax_text

        try:
            rangeXmin = float(rangeXmin)
            rangeXmax = float(rangeXmax)
            rangeYmin = float(rangeYmin)
            rangeYmax = float(rangeYmax)
        except ValueError:
            self.logger.warning("okCutoff Range - Invalid input format for zoom range(s). Please enter valid numbers.")
            return

        if rangeXmin is not None and rangeXmax is not None and rangeXmax < rangeXmin:
            buff = rangeXmin; rangeXmin = rangeXmax; rangeXmax = buff
            self.logger.warning('okCutoff Range - new range X values swapped because min > max')
        if rangeYmin is not None and rangeYmax is not None and rangeYmax < rangeYmin:
            buff = rangeYmin; rangeYmin = rangeYmax; rangeYmax = buff
            self.logger.warning('okCutoff Range - new range Y values swapped because min > max')
        if dim == 2:
            if cutoffMin != "":
                try:
                    cutoffVal[0] = float(cutoffMin)
                    self._set_spectrum_info(cutoff=cutoffVal, index=index)
                except (TypeError, ValueError):
                    self.logger.warning('okCutoff - invalid Z-min cutoff %r ignored', cutoffMin)
            if cutoffMax != "":
                try:
                    cutoffVal[1] = float(cutoffMax)
                    self._set_spectrum_info(cutoff=cutoffVal, index=index)
                except (TypeError, ValueError):
                    self.logger.warning('okCutoff - invalid Z-max cutoff %r ignored', cutoffMax)
            if cutoffVal[0] is not None and cutoffVal[1] is not None and cutoffVal[1] < cutoffVal[0]:
                cutoffVal = [cutoffVal[1], cutoffVal[0]]
                self._set_spectrum_info(cutoff=cutoffVal, index=index)
        try:
            spectrum = self._get_spectrum_info("spectrum", index=index)
            ax.set_xlim(float(rangeXmin), float(rangeXmax))
            ax.set_ylim(float(rangeYmin), float(rangeYmax))
            self._set_spectrum_info(minx=rangeXmin, index=index)
            self._set_spectrum_info(maxx=rangeXmax, index=index)
            self._set_spectrum_info(miny=rangeYmin, index=index)
            self._set_spectrum_info(maxy=rangeYmax, index=index)
            if dim == 2:
                spectrum.set_clim(cutoffVal[0], cutoffVal[1])
                self._set_spectrum_info(minz=cutoffVal[0], index=index)
                self._set_spectrum_info(maxz=cutoffVal[1], index=index)
                self._set_spectrum_info(spectrum=spectrum, index=index)
            self._set_spectrum_info(spectrum=spectrum, index=index)
            self._draw_gate(index)
            cp.canvas.draw()
        except Exception:
            self.logger.debug('okCutoff - exception', exc_info=True)

        self.cutoffPopupCloseRequested.emit()

    def resetCutoff(self, doUpdate):
        self.logger.info('resetCutoff - doUpdate: %s', doUpdate)
        index = self._get_current_plot().selected_plot_index
        if index is None:
            return
        cutoffVal = [None, None]
        self._set_spectrum_info(cutoff=cutoffVal, index=index)
        if doUpdate:
            self.updatePlot()
        self.cutoffPopupCloseRequested.emit()

    # called by cutoffButton, sets the information in the cutoff window
    def cutoffButtonCallback(self, *arg):
        self.logger.info('cutoffButtonCallback')
        cp    = self._get_current_plot()
        cp.histo_autoscale.setChecked(False)
        index = cp.selected_plot_index
        if index is None:
            return QMessageBox.about(self._parent_widget, "Warning!", "Please Add/Select a Spectrum")
        name = self._name_from_index(index)
        if self._get_spectrum_info("cutoff", index=index) is not None and len(self._get_spectrum_info("cutoff", index=index)) > 0:
            ax  = self._get_spectrum_info("axis", index=index)
            dim = self._spectra.get(name, "dim")
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            zmin = zmax = None
            if dim == 2:
                spectrum = self._get_spectrum_info("spectrum", index=index)
                zmin, zmax = spectrum.get_clim()
            # MainWindow renders the popup from this payload
            self.cutoffPopupPrepared.emit({
                "name": name, "dim": dim,
                "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
                "zmin": zmin, "zmax": zmax,
            })
        else:
            QMessageBox.about(self._parent_widget, "Warning!", "Please Add/Select a Spectrum")
            self.logger.warning('cutoffButtonCallback - you broke something really bad - spectrum dict: %s',
                                self._get_spectrum_info("cutoff", index=index))

    # used in zoomCallback to save the new axis limits
    def updatePlotLimits(self):
        self.logger.debug('updatePlotLimits')
        cp    = self._get_current_plot()
        index = cp.selected_plot_index
        ax    = self._get_spectrum_info("axis", index=index)
        if ax is None:
            self.logger.debug('updatePlotLimits - ax is None')
            return

        try:
            x_range, y_range = self.getAxisProperties(index)
            self._set_spectrum_info(minx=x_range[0], index=index)
            self._set_spectrum_info(maxx=x_range[1], index=index)
            self._set_spectrum_info(miny=y_range[0], index=index)
            self._set_spectrum_info(maxy=y_range[1], index=index)

            spectrum = self._get_spectrum_info("spectrum", index=index)
            ax.set_xlim(x_range[0], x_range[1])
            ax.set_ylim(y_range[0], y_range[1])

            self._set_spectrum_info(spectrum=spectrum, index=index)
            self._draw_gate(index)
        except Exception:
            self.logger.debug('updatePlotLimits - exception', exc_info=True)

    # ------------------------------------------------------------------
    # Colorbar / canvas helpers
    # ------------------------------------------------------------------

    # remove colorbar
    def removeCb(self, axis):
        im = axis.images
        if im is not None and len(im) > 0:
            try:
                cb = im[-1].colorbar
                cb.remove()
            except Exception:
                self.logger.debug('removeCb - IndexError exception', exc_info=True)

    # select axes based on indexing, used only in add()
    def select_plot(self, index):
        self.logger.info('select_plot - index: %s', index)
        cp = self._get_current_plot()
        for i, axis in enumerate(cp.figure.axes):
            if (i == index and axis is not None):
                return axis

    def plotPosition(self, index, canvas_layout):
        """Map a flat slot index to (row, col) in `canvas_layout` — the
        current tab's [nRow, nCol], supplied by the MainWindow adapter."""
        self.logger.info('plotPosition - index: %s', index)
        cntr = 0
        canvasLayout = canvas_layout
        for i in range(canvasLayout[0]):
            for j in range(canvasLayout[1]):
                if index == cntr:
                    return i, j
                else:
                    cntr += 1

    # ------------------------------------------------------------------
    # Plot setup and rendering
    # ------------------------------------------------------------------

    # set up histogram limits according to the ReST info.
    # Called from add(), when the plot is first added
    def setupPlot(self, axis, index):
        self.logger.debug('setupPlot - index: %s', index)
        self._layout_dirty = True
        if self._name_from_index(index):
            name = self._name_from_index(index)
            dim  = self._spectra.get(name, "dim")
            minx = self._get_spectrum_info("minx", index=index)
            maxx = self._get_spectrum_info("maxx", index=index)
            binx = self._get_spectrum_info("binx", index=index)
            biny = self._get_spectrum_info("biny", index=index)
            w    = self._cutoff_masked_data(index)

            if dim == 1:
                axis.set_xlim(minx, maxx)
                line, = axis.plot([], [], drawstyle='steps')
                spec_name = self._get_spectrum_info("name", index=index)
                axis.set_title("{}".format(spec_name))
                self._set_spectrum_info(spectrum=line, index=index)
                if len(w) > 0:
                    # Bin edges must span the spectrum's true axis range (REST
                    # store), like plotPlot and the 2D imshow extent below.
                    # The per-tab minx/maxx used for set_xlim above hold the
                    # current VIEW range (updatePlotLimits/loadGeo write zoom
                    # limits there), so building edges from them compresses the
                    # whole spectrum into the zoomed window and fit overlays
                    # (drawn in true coordinates) no longer sit on the data.
                    minxREST = self._spectra.get(name, "minx")
                    maxxREST = self._spectra.get(name, "maxx")
                    binxREST = self._spectra.get(name, "binx")
                    X = np.array(self.createRange(binxREST, minxREST, maxxREST))
                    line.set_data(X, w)
                    self._set_spectrum_info(spectrum=line, index=index)
            else:
                minxREST = self._spectra.get(name, "minx")
                maxxREST = self._spectra.get(name, "maxx")
                minyREST = self._spectra.get(name, "miny")
                maxyREST = self._spectra.get(name, "maxy")

                if w is None:
                    w = np.zeros((int(binx), int(biny)))

                self.palette = copy(plt.cm.plasma)
                w = np.ma.masked_where(w == 0, w)
                self.palette.set_bad(color='white')

                spectrum = axis.imshow(w,
                                       interpolation='none',
                                       extent=[float(minxREST), float(maxxREST),
                                               float(minyREST), float(maxyREST)],
                                       aspect='auto',
                                       origin='lower',
                                       vmin=float(self.minZ), vmax=float(self.maxZ),
                                       cmap=self.palette)
                self._set_spectrum_info(spectrum=spectrum, index=index)

                spec_name = self._get_spectrum_info("name", index=index)
                axis.set_title("{}".format(spec_name))

                if w is not None:
                    spectrum.set_data(w)
                    self._set_spectrum_info(spectrum=spectrum, index=index)

                if self._get_enlarged_spectrum() is None:
                    divider = make_axes_locatable(axis)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    label = "colorbar_" + str(index)
                    cax.set_label(label)
                    self._get_current_plot().figure.colorbar(spectrum, cax=cax, orientation='vertical')

    # geometrically add plots in the right place and call the plotting.
    # Should be called only by addPlot and by on_dblclick when entering or leaving enlarged mode
    def add(self, index):
        self.logger.info('add - index: %s', index)
        cp = self._get_current_plot()
        a  = None
        if cp.isEnlarged:
            a = cp.figure.get_axes()[0]
        else:
            a = self.select_plot(index)
        try:
            self.removeCb(a)
        except Exception:
            pass
        a.clear()
        self.setupPlot(a, index)

    def addPlot(self, selected_name=None, tab_click_bound=True):
        """Place the selected spectrum (`selected_name` is the histo_list
        selection — None when the list is empty — and `tab_click_bound` is the
        current tab's click-binding state, both supplied by the adapter)."""
        self.logger.info('addPlot')
        if not self.geometry_applied:
            self.logger.warning('addPlot - geometry not applied yet; press Apply first')
            return

        cp = self._get_current_plot()
        if selected_name is None:
            QMessageBox.about(self._parent_widget, "Warning",
                              'Please click on "Connection" and fill in the information')

        try:
            if cp.isLoaded:
                self.logger.debug('addPlot - isLoaded TRUE - getGeo: %s', self._get_geo())
                cp.histo_autoscale.setChecked(True)
                for key, value in self._get_geo().items():
                    if self._spectra.get(value, "dim") is None:
                        continue
                    self.add(key)
                    self.autoScaleAxisBox(key)
            else:
                index = self._next_index()
                name  = str(selected_name)
                self.logger.debug('addPlot - isLoaded FALSE - index, name: %s, %s', index, name)

                if self._spectra.get(name, "dim") is None:
                    self.logger.debug('addPlot - isLoaded FALSE - dim is None')
                    return

                self._set_spectrum_info(cutoff=None, index=index)
                self._set_geo(index, name)
                self.add(index)

                cp.histo_autoscale.setChecked(True)
                self.autoScaleAxisBox(index)

                dim = self._spectra.get(self._name_from_index(index), "dim")
                ax  = self._get_spectrum_info("axis", index=index)
                xmin, xmax = ax.get_xlim()

                if dim == 1:
                    ymin = self._get_spectrum_info("miny", index=index)
                    ymax = self.getMinMaxInRange(index, xmin=xmin, xmax=xmax)
                    ax.set_ylim(ymin, ymax)
                else:
                    ymin, ymax = ax.get_ylim()
                    zmin, zmax = self.getMinMaxInRange(index, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
                    self._set_spectrum_info(maxz=zmax, index=index)
                    self._set_spectrum_info(minz=zmin, index=index)
                    spectrum = self._get_spectrum_info("spectrum", index=index)
                    spectrum.set_clim(vmin=zmin, vmax=zmax)
                    self._set_spectrum_info(spectrum=spectrum, index=index)

                cp.logButton.setDown(False)
                self._set_spectrum_info(log=False, index=index)
                self._draw_gate(index)

                self.removeRectangle()
                cp.recDashed = self.createDashedRectangle(cp.figure.axes[cp.next_plot_index])

                try:
                    cp.figure.tight_layout()
                    self._layout_dirty = False
                except ValueError:
                    self.logger.debug('addPlot - ValueError exception', exc_info=True)

                cp.canvas.draw_idle()
                cp.isSelected = False
        except NameError:
            raise

        if self._stop_auto_update_thread.is_set():
            self._auto_update_start()

        if not tab_click_bound:
            self._bind_dynamic_signal()

    @staticmethod
    def createRange(bins, vmin, vmax):
        return np.linspace(float(vmin), float(vmax), int(bins) + 1)

    def _cutoff_masked_data(self, index, raw=None):
        """Return the spectrum's data with its per-slot cutoff applied (a
        masked array). The canonical array lives in the SpectrumStore; the
        cutoff is a per-tab / per-slot display setting."""
        name = self._name_from_index(index)
        w = raw if raw is not None else self._spectra.get(name, "data")
        if w is None:
            return w
        cutoff = self._get_spectrum_info("cutoff", index=index)
        if cutoff and len(cutoff) >= 2:
            minCutoff, maxCutoff = cutoff[0], cutoff[1]
            if minCutoff is not None:
                w = np.ma.masked_where(w < minCutoff, w)
            if maxCutoff is not None:
                w = np.ma.masked_where(w > maxCutoff, w)
        return w

    # fill the spectrum with new data. Called from addPlot and updatePlot;
    # does not draw the plot itself
    def plotPlot(self, index, cmap=None):
        self.logger.debug('plotPlot - index: %s', index)
        name     = self._name_from_index(index)
        dim      = self._spectra.get(name, "dim")
        minx     = self._spectra.get(name, "minx")
        maxx     = self._spectra.get(name, "maxx")
        binx     = self._spectra.get(name, "binx")
        spectrum = self._get_spectrum_info("spectrum", index=index)
        w        = self._cutoff_masked_data(index)

        if w is None or len(w) <= 0:
            self.logger.debug('plotPlot - w is None or len(w) <= 0')
            return

        if dim == 1:
            X = np.array(self.createRange(binx, minx, maxx))
            spectrum.set_data(X, w)
        else:
            w = np.ma.masked_where(w == 0, w)
            spectrum.set_data(w)
            if cmap is not None:
                spectrum.set_cmap(cmap)
            elif self.old_cmap is not None:
                spectrum.set_cmap(self.old_cmap)
            elif self.palette is not None:
                spectrum.set_cmap(self.palette)
            else:
                spectrum.set_cmap(plt.get_cmap("viridis"))

        self._set_spectrum_info(spectrum=spectrum, index=index)

    @pyqtSlot()
    def _updatePlotOnGui(self):
        # The auto-update timer's tick — the only unforced path, so the
        # change-driven skip applies here (all other callers force a redraw).
        self.updatePlot(force=False)

    def _tick_signature(self, cp, auto_scale_status):
        """Cheap fingerprint of everything the auto-update tick would render,
        so a redundant redraw can be skipped when nothing changed since the
        previous tick (change-driven redraw). SpecTcl spectra are cumulative
        counters, so a per-pad data sum is a reliable change signal — any
        increment moves it, and a clear resets it to 0 (also a change)."""
        try:
            indices = [0] if cp.isEnlarged else list(self._get_geo().keys())
            pad_sigs = {}
            for index in indices:
                name   = self._name_from_index(index)
                log    = self._get_spectrum_info("log", index=index)
                cutoff = self._get_spectrum_info("cutoff", index=index)
                w = self._spectra.get(name, "data") if name is not None else None
                data_sig = float(np.ma.sum(w)) if w is not None and len(w) > 0 else None
                pad_sigs[index] = (
                    name,
                    tuple(log)    if isinstance(log, list)    else log,
                    tuple(cutoff) if isinstance(cutoff, list) else cutoff,
                    data_sig,
                )
            signature = (id(cp), bool(cp.isEnlarged), bool(auto_scale_status),
                         tuple(sorted(pad_sigs.items(), key=lambda kv: repr(kv[0]))))
            return signature, pad_sigs
        except Exception:
            self.logger.debug('_tick_signature - exception; forcing redraw', exc_info=True)
            return None, None

    # callback for the histo_geo_update button, also used in various functions.
    # Redraws the spectrum, updates the axis scales and redraws the gates
    def updatePlot(self, force=True):
        cp = self._get_current_plot()
        auto_scale_status = cp.histo_autoscale.isChecked()
        cp.histo_autoscale.setChecked(auto_scale_status)
        self.logger.debug('updatePlot')

        # on an unforced (auto-update timer) tick, skip the whole redraw when
        # the frame is byte-identical to the last one we drew. Forced calls (the
        # hide-gates toggle, gate/sum-region/geometry updates) always render.
        signature, pad_sigs = self._tick_signature(cp, auto_scale_status)
        if (not force
                and signature is not None
                and signature == self._last_tick_signature):
            self.logger.debug('updatePlot - unchanged since last tick; skipping redraw')
            return

        # Whatever changed, it was not every pad. On an unforced tick,
        # re-render only the pads whose fingerprint moved: a live session
        # usually has one filling spectrum beside several idle ones, and
        # re-masking + re-setting an idle 2-D pad's data costs as much as the
        # live one's.
        last_pads = None if force else self._last_pad_signatures

        self._clean_popup_exit(False)

        try:
            if cp.isEnlarged:
                index = self._auto_index()
                name  = self._name_from_index(index)
                if index is None or self._spectra.get(name, "dim") is None:
                    self.logger.debug('updatePlot - index is None or dim is None')
                    return
                self.logger.debug('updatePlot - self.currentPlot.isEnlarged TRUE')
                ax = self._get_spectrum_info("axis", index=0)
                if ax is None:
                    # No axis built for the enlarged pad yet (polling started
                    # before Add/geometry, or the geo file drifted). Skip the
                    # tick quietly — never a modal here: this path is driven by
                    # the auto-update timer, so a blocking dialog stacks a new
                    # one every tick and freezes the GUI.
                    self.logger.debug('updatePlot - ax is None (enlarged); skipping tick')
                    return
                self.plotPlot(index)
                dim = self._spectra.get(self._name_from_index(0), "dim")
                if auto_scale_status:
                    if dim == 1:
                        self.setAxisScale(ax, 0, "x", "y")
                if dim == 2:
                    self.setAxisScale(ax, 0, "x", "y", "z")
                self._draw_gate(0)
            else:
                self.logger.debug('updatePlot - self.currentPlot.isEnlarged FALSE')
                for index, value in self._get_geo().items():
                    ax = self._get_spectrum_info("axis", index=index)
                    if ax is None:
                        # This geometry slot has no built axis yet — skip it and
                        # keep refreshing the valid pads. Never a modal here (see
                        # the enlarged branch above): the auto-update timer drives
                        # this loop, so a per-tick blocking dialog freezes the GUI.
                        self.logger.debug('updatePlot - ax is None for index %s; skipping slot', index)
                        continue
                    unchanged = (last_pads is not None and pad_sigs is not None
                                 and index in last_pads
                                 and pad_sigs.get(index) == last_pads[index])
                    if not unchanged:
                        self.plotPlot(index)
                        dim = self._spectra.get(self._name_from_index(index), "dim")
                        if auto_scale_status:
                            if dim == 1:
                                self.setAxisScale(ax, index, "x", "y")
                            elif dim == 2:
                                self.setAxisScale(ax, index, "x", "y", "z")
                    # gates are redrawn for every pad either way: their cost is
                    # not what this skip measured, and a gate edit routes through
                    # a forced tick whose artists must land on every pad
                    self._draw_gate(index)

            if self._layout_dirty:
                cp.figure.tight_layout()
                self._layout_dirty = False
            cp.canvas.draw_idle()
            self._last_tick_signature = signature
            self._last_pad_signatures = pad_sigs
        except Exception:
            self.logger.debug('updatePlot - exception', exc_info=True)

    def onColormapChange(self, cmap_name: str):
        self.logger.info("onColormapChange - cmap: %s", cmap_name)

        if not self._get_geo():
            self.logger.debug("onColormapChange - no active plots")
            return

        try:
            cp = self._get_current_plot()
            if cmap_name.lower() == "custom":
                QMessageBox.information(
                    self._parent_widget,
                    "Custom Colormap Format from a .txt File",
                    "Each line should be:\n<low> <high> <red> <green> <blue>\n"
                    "Example:\n0.0 0.5 0 1 0\n0.5 0.7 1 0 0\n0.7 1.0 0 0 1\n\n"
                    "<low> and <high> represent count percentiles.\n"
                    "The <low> value of the first line must be 0.0, and the <low> of each line must match the <high> of the previous line.\n\n"
                    "<red>, <green>, and <blue> are RGB values between 0 and 1."
                )
                filename, _ = QFileDialog.getOpenFileName(
                    self._parent_widget, "Open Custom Colormap", "", "Text Files (*.txt)"
                )
                if not filename:
                    self.logger.debug("No file selected for custom cmap")
                    return

                bounds     = []
                color_list = []
                last = None
                with open(filename) as f:
                    for line in f:
                        parts = line.split()
                        if not parts or len(parts) < 5:
                            continue
                        lo, hi = float(parts[0]), float(parts[1])
                        r, g, b = map(float, parts[2:5])
                        bounds.append(lo)
                        color_list.append((r, g, b))
                        last = (hi, r, g, b)
                if last is None:
                    QMessageBox.warning(
                        self._parent_widget, "Custom Colormap",
                        "No valid '<low> <high> <r> <g> <b>' lines found in the file.")
                    return
                hi, r, g, b = last
                bounds.append(hi)
                color_list.append((r, g, b))

                self.palette = colors.LinearSegmentedColormap.from_list(
                    "custom_cmap", list(zip(bounds, color_list)), N=256
                )
                self.palette.set_bad(color="white")
            else:
                self.palette = copy(plt.get_cmap(cmap_name))
                self.palette.set_bad(color="white")

            for index, _ in self._get_geo().items():
                if self._spectra.get(self._name_from_index(index), "dim") != 2:
                    continue
                spectrum = self._get_spectrum_info("spectrum", index=index)
                if spectrum is None:
                    continue
                spectrum.set_cmap(self.palette)
                self._set_spectrum_info(spectrum=spectrum, index=index)
                if cp.isEnlarged:
                    self.old_cmap = spectrum.get_cmap()
                ax = self._get_spectrum_info("axis", index=index)
                if ax is not None:
                    cb_label = "colorbar_" + str(index)
                    cax = next((a for a in cp.figure.axes if a.get_label() == cb_label), None)
                    if cax is None:
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        cax.set_label(cb_label)
                    cax.cla()
                    cp.figure.colorbar(spectrum, cax=cax, orientation="vertical")

            cp.canvas.draw_idle()

        except Exception as e:
            self.logger.error("onColormapChange - exception: %s", str(e), exc_info=True)

    # ------------------------------------------------------------------
    # Rectangle overlays (plot-selection highlight)
    # ------------------------------------------------------------------

    def createRectangle(self, plot):
        self.logger.info('createRectangle')
        rec = matplotlib.patches.Rectangle((0, 0), 1, 1, ls="-", lw=2, ec="red", fc="none", transform=plot.transAxes)
        rec = plot.add_patch(rec)
        rec.set_clip_on(False)
        return rec

    def createDashedRectangle(self, plot):
        self.logger.info('createDashedRectangle')
        rec = matplotlib.patches.Rectangle((0, 0), 1, 1, ls=":", lw=2, ec="red", fc="none", transform=plot.transAxes)
        rec = plot.add_patch(rec)
        rec.set_clip_on(False)
        return rec

    def removeRectangle(self):
        self.logger.info('removeRectangle')
        try:
            cp = self._get_current_plot()
            for ax in cp.figure.axes:
                for child in ax.get_children():
                    if type(child) == matplotlib.patches.Rectangle:
                        if child.get_ls() == ":" and child.get_lw() == 2:
                            if cp.recDashed is not None:
                                cp.recDashed.remove()
                                cp.recDashed = None
                        elif child.get_ls() == "-" and child.get_lw() == 2:
                            if cp.rec is not None:
                                cp.rec.remove()
                                cp.rec = None
        except NameError:
            raise

