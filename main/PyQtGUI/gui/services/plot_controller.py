import logging
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from copy import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMenu, QMessageBox, QFileDialog


class centeredNorm(colors.Normalize):
    def __init__(self, data, vcenter=0, halfrange=None, clip=False):
        if halfrange is None:
            halfrange = np.max(np.abs(data - vcenter))
        super().__init__(vmin=vcenter - halfrange, vmax=vcenter + halfrange, clip=clip)


class PlotController:
    """Owns all plot rendering, axis control, zoom, and canvas management."""

    def __init__(self, wTab, wConf, spectra,
                 get_current_plot, get_geo, set_geo,
                 get_spectrum_info, set_spectrum_info, get_spectrum_info_dict,
                 name_from_index, get_enlarged_spectrum,
                 auto_index, next_index, bind_dynamic_signal,
                 draw_gate, clean_popup_exit, auto_update_start,
                 stop_auto_update_thread, cutoff_popup,
                 min_y, max_y, min_z, max_z,
                 parent_widget=None, logger=None):
        self._wTab                    = wTab
        self._wConf                   = wConf
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
        self._cutoffp                 = cutoff_popup
        self.minY                     = min_y
        self.maxY                     = max_y
        self.minZ                     = min_z
        self.maxZ                     = max_z
        self._parent_widget           = parent_widget
        self.logger                   = logger or logging.getLogger(__name__)

        self.palette          = None
        self.old_cmap         = None
        self.geometry_applied = False

    # ------------------------------------------------------------------
    # Canvas / layout
    # ------------------------------------------------------------------

    def setCanvasLayout(self):
        self.logger.info('setCanvasLayout')
        indexTab = self._wTab.currentIndex()
        nRow = int(self._wConf.histo_geo_row.currentText())
        nCol = int(self._wConf.histo_geo_col.currentText())
        self._wTab.layout[indexTab] = [nRow, nCol]
        self._wTab.wPlot[indexTab].InitializeCanvas(nRow, nCol)
        self._wTab.selected_plot_index_bak[indexTab] = None
        cp = self._get_current_plot()
        cp.selected_plot_index = None
        cp.next_plot_index     = -1
        self.geometry_applied  = True

    # ------------------------------------------------------------------
    # Axis scaling
    # ------------------------------------------------------------------

    def setAxisScale(self, ax, index, *scale):
        self.logger.info('setAxisScale - index: %s', index)

        cp            = self._get_current_plot()
        axisIsLog     = self._get_spectrum_info("log", index=index)
        axisIsAutoScale = cp.histo_autoscale.isChecked()
        name          = self._name_from_index(index)

        if self._spectra.get(name, "dim") == 1:
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

    def setCmapNorm(self, scale, index):
        self.logger.info('setCmapNorm')
        validScales = ["linear", "log", "linearCentered"]
        if scale not in validScales or index is None:
            self.logger.debug('setCmapNorm - scale not in validScales or index is None')
            return
        spectrum = self._get_spectrum_info("spectrum", index=index)
        zmin, zmax = spectrum.get_clim()

        if scale is validScales[0]:
            if zmin > zmax:
                self.logger.warning('setCmapNorm - zmin > zmax')
                spectrum.set_norm(colors.Normalize(vmin=self.minZ, vmax=self.maxZ))
            else:
                spectrum.set_norm(colors.Normalize(vmin=zmin, vmax=zmax))
        elif scale is validScales[1]:
            if zmin and zmin <= 0:
                zmin = 0.001
                self.logger.warning('setCmapNorm - LogNorm with zmin<=0, may want to use CenteredNorm')
            spectrum.set_norm(colors.LogNorm(vmin=zmin, vmax=zmax))
            if zmin > zmax:
                self.logger.warning('setCmapNorm - zmin > zmax')
                spectrum.set_norm(colors.LogNorm(vmin=self.minZ, vmax=self.maxZ))
        elif scale is validScales[2]:
            palette = copy(plt.cm.jet)
            palette.set_bad(color='white')
            data = self._get_spectrum_info("data", index=index)
            spectrum.set_cmap(palette)
            spectrum.set_norm(centeredNorm(data, 50000))
        if self._get_enlarged_spectrum() is None:
            ax = spectrum.axes
            self.removeCb(ax)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            label = "colorbar_" + str(index)
            cax.set_label(label)
            self._get_current_plot().figure.colorbar(spectrum, cax=cax, orientation='vertical')

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
            pass

    # ------------------------------------------------------------------
    # Range / min-max helpers
    # ------------------------------------------------------------------

    def getMinMaxInRange(self, index, **limits):
        self.logger.info('getMinMaxInRange - limits: %s', limits)
        result = None
        if not limits:
            self.logger.warning('getMinMaxInRange - limits identifier not valid - expect xmin=val, xmax=val etc. for y with 2D')
            return
        if "xmin" and "xmax" in limits:
            xmin = limits["xmin"]
            xmax = limits["xmax"]
        if "ymin" and "ymax" in limits:
            ymin = limits["ymin"]
            ymax = limits["ymax"]

        name  = self._name_from_index(index)
        dim   = self._spectra.get(name, "dim")
        minx  = self._spectra.get(name, "minx")
        maxx  = self._spectra.get(name, "maxx")
        binx  = self._spectra.get(name, "binx")
        data  = self._get_spectrum_info("data", index=index)
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

    def customMinMax(self, data):
        self.logger.info('customMinMax')
        minimum = None
        maximum = None
        nbCol = data.shape[1]
        nbRow = data.shape[0]

        if not data.any():
            return minimum, maximum
        if nbCol < 200 and nbRow < 200:
            maximum = data.max()
            minimum = np.min(data[np.nonzero(data)])
            return minimum, maximum
        else:
            stepX = nbCol if nbCol < 200 else 200
            stepY = nbRow if nbRow < 200 else 200
            rangeX = list(range(0, data.shape[1], stepX))
            rangeY = list(range(0, data.shape[0], stepY))
            subMax = []
            subMin = []
            yprev = data.shape[0] + 1
            xprev = data.shape[1] + 1
            for x in rangeX[::-1]:
                for y in rangeY[::-1]:
                    subData = data[y:yprev, x:xprev]
                    nonZeroIndices = np.where(subData > 0)
                    filteredSubData = subData[nonZeroIndices]
                    if filteredSubData is not None and filteredSubData.size > 0:
                        subMax.append(filteredSubData.max())
                        subMin.append(filteredSubData.min())
                    yprev = y
                xprev = x
            if len(subMin) == 0:
                minimum = self.minZ
            if len(subMax) == 0:
                minimum = self.maxZ
            elif len(subMin) > 0 and len(subMax) > 0:
                minimum = min(subMin)
                maximum = max(subMax)
            return minimum, maximum

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
            pass

    # ------------------------------------------------------------------
    # Zoom / toolbar callbacks
    # ------------------------------------------------------------------

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

    def customHomeButtonCallback(self, index=None):
        self.logger.info('customHomeButtonCallback - index: %s', index)
        index_list = [idx for idx, name in self._get_geo().items() if index is None]
        if index is not None:
            index_list = [index]
        for idx in index_list:
            ax       = None
            spectrum = self._get_spectrum_info("spectrum", index=idx)
            if spectrum is None:
                return
            ax   = spectrum.axes
            name = self._name_from_index(idx)
            dim  = self._spectra.get(name, "dim")
            xmin = self._spectra.get(name, "minx")
            xmax = self._spectra.get(name, "maxx")
            ymin = self._spectra.get(name, "miny")
            ymax = self._spectra.get(name, "maxy")

            ax.set_xlim(xmin, xmax)
            if dim == 1:
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

    def okCutoff(self):
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
        rangeXmin  = self._cutoffp.lineeditXMin.text()
        rangeXmax  = self._cutoffp.lineeditXMax.text()
        rangeYmin  = self._cutoffp.lineeditYMin.text()
        rangeYmax  = self._cutoffp.lineeditYMax.text()

        cutoffVal  = [None, None]
        cutoffMin  = self._cutoffp.lineeditZMin.text()
        cutoffMax  = self._cutoffp.lineeditZMax.text()

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
            if self._cutoffp.lineeditZMin.text() != "" and self._cutoffp.lineeditZMin.text().isdigit():
                cutoffVal[0] = float(cutoffMin)
                self._set_spectrum_info(cutoff=cutoffVal, index=index)
            if self._cutoffp.lineeditZMax.text() != "" and self._cutoffp.lineeditZMax.text().isdigit():
                cutoffVal[1] = float(cutoffMax)
                self._set_spectrum_info(cutoff=cutoffVal, index=index)
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
        except NameError as err:
            self.logger.debug('okCutoff - NameError', exc_info=True)
            pass

        self._cutoffp.close()

    def cancelCutoff(self):
        self._cutoffp.close()

    def resetCutoff(self, doUpdate):
        self.logger.info('resetCutoff - doUpdate: %s', doUpdate)
        index = self._get_current_plot().selected_plot_index
        if index is None:
            return
        cutoffVal = [None, None]
        self._set_spectrum_info(cutoff=cutoffVal, index=index)
        if doUpdate:
            self.updatePlot()
        self._cutoffp.close()

    def cutoffButtonCallback(self, *arg):
        self.logger.info('cutoffButtonCallback')
        cp    = self._get_current_plot()
        cp.histo_autoscale.setChecked(False)
        index = cp.selected_plot_index
        if index is None:
            return QMessageBox.about(self._parent_widget, "Warning!", "Please Add/Select a Spectrum")
        name = self._name_from_index(index)
        if name is not None:
            self._cutoffp.setWindowTitle("Set zoom range for: " + name)
        else:
            self._cutoffp.setWindowTitle("Set zoom range for: ???")
        self._cutoffp.setGeometry(300, 100, 300, 100)
        if self._cutoffp.isVisible():
            self._cutoffp.close()
        if self._get_spectrum_info("cutoff", index=index) is not None and len(self._get_spectrum_info("cutoff", index=index)) > 0:
            ax  = self._get_spectrum_info("axis", index=index)
            dim = self._spectra.get(name, "dim")
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            self._cutoffp.lineeditXMin.setText(f"{xmin:.1f}")
            self._cutoffp.lineeditXMax.setText(f"{xmax:.1f}")
            self._cutoffp.lineeditYMin.setText(f"{ymin:.1f}")
            self._cutoffp.lineeditYMax.setText(f"{ymax:.1f}")
            if dim == 2:
                spectrum = self._get_spectrum_info("spectrum", index=index)
                zmin, zmax = spectrum.get_clim()
                self._cutoffp.lineeditZMin.setText(f"{zmin:.1f}")
                self._cutoffp.lineeditZMax.setText(f"{zmax:.1f}")
            if dim == 1:
                self._cutoffp.layout1d()
            elif dim == 2:
                self._cutoffp.layout2d()
            self._cutoffp.show()
        else:
            QMessageBox.about(self._parent_widget, "Warning!", "Please Add/Select a Spectrum")
            self.logger.warning('cutoffButtonCallback - you broke something really bad - spectrum dict: %s',
                                self._get_spectrum_info("cutoff", index=index))

    def updatePlotLimits(self, sleepTime=0):
        self.logger.info('updatePlotLimits - sleepTime: %s', sleepTime)
        cp    = self._get_current_plot()
        index = cp.selected_plot_index
        ax    = self._get_spectrum_info("axis", index=index)
        if ax is None:
            self.logger.debug('updatePlotLimits - ax is None')
            return

        time.sleep(sleepTime)

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
        except NameError as err:
            self.logger.debug('updatePlotLimits - NameError', exc_info=True)
            print(err)
            pass

    # ------------------------------------------------------------------
    # Colorbar / canvas helpers
    # ------------------------------------------------------------------

    def removeCb(self, axis):
        im = axis.images
        if im is not None and len(im) > 0:
            try:
                cb = im[-1].colorbar
                cb.remove()
            except Exception:
                self.logger.debug('removeCb - IndexError exception', exc_info=True)
                pass

    def select_plot(self, index):
        self.logger.info('select_plot - index: %s', index)
        cp = self._get_current_plot()
        for i, axis in enumerate(cp.figure.axes):
            if (i == index and axis is not None):
                return axis

    def plotPosition(self, index):
        self.logger.info('plotPosition - index: %s', index)
        cntr = 0
        canvasLayout = self._wTab.layout[self._wTab.currentIndex()]
        for i in range(canvasLayout[0]):
            for j in range(canvasLayout[1]):
                if index == cntr:
                    return i, j
                else:
                    cntr += 1

    # ------------------------------------------------------------------
    # Plot setup and rendering
    # ------------------------------------------------------------------

    def setupPlot(self, axis, index):
        self.logger.info('setupPlot - index: %s', index)
        if self._name_from_index(index):
            name = self._name_from_index(index)
            dim  = self._spectra.get(name, "dim")
            minx = self._get_spectrum_info("minx", index=index)
            maxx = self._get_spectrum_info("maxx", index=index)
            binx = self._get_spectrum_info("binx", index=index)
            biny = self._get_spectrum_info("biny", index=index)
            w    = self._spectra.get(name, "data")

            if self._get_spectrum_info("cutoff", index=index) is not None:
                if len(self._get_spectrum_info("cutoff", index=index)) > 0:
                    minCutoff = self._get_spectrum_info("cutoff", index=index)[0]
                    maxCutoff = self._get_spectrum_info("cutoff", index=index)[1]
                    if minCutoff is not None:
                        if dim == 1:
                            w = np.ma.masked_where(w < minCutoff, w)
                        if dim == 2:
                            w = np.ma.masked_where(w < minCutoff, w)
                    if maxCutoff is not None:
                        if dim == 1:
                            w = np.ma.masked_where(w < maxCutoff, w)
                        if dim == 2:
                            w = np.ma.masked_where(w < maxCutoff, w)
            self._set_spectrum_info(data=w, index=index)

            if dim == 1:
                axis.set_xlim(minx, maxx)
                line, = axis.plot([], [], drawstyle='steps')
                spec_name = self._get_spectrum_info("name", index=index)
                axis.set_title("{}".format(spec_name))
                self._set_spectrum_info(spectrum=line, index=index)
                if len(w) > 0:
                    X = np.array(self.createRange(binx, minx, maxx))
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

    def addPlot(self):
        self.logger.info('addPlot')
        if not self.geometry_applied:
            print("addPlot: Apply Geometry first!, return")
            return

        cp = self._get_current_plot()
        if self._wConf.histo_list.count() == 0:
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
                name  = str(self._wConf.histo_list.currentText())
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
                    ymin, ymax = ax.get_xlim()
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
                except ValueError:
                    self.logger.debug('addPlot - ValueError exception', exc_info=True)
                    pass

                cp.canvas.draw_idle()
                cp.isSelected = False
        except NameError:
            raise

        if self._stop_auto_update_thread.is_set():
            self._auto_update_start()

        if not self._wTab.countClickTab[self._wTab.currentIndex()]:
            self._bind_dynamic_signal()

    def createRange(self, bins, vmin, vmax):
        self.logger.info('createRange')
        x    = []
        step = (float(vmax) - float(vmin)) / float(bins)
        for i in np.arange(float(vmin), float(vmax), step):
            x.append(i + step)
        x.insert(0, float(vmin))
        return x

    def plotPlot(self, index, cmap=None):
        self.logger.info('plotPlot - index: %s', index)
        name     = self._name_from_index(index)
        dim      = self._spectra.get(name, "dim")
        minx     = self._spectra.get(name, "minx")
        maxx     = self._spectra.get(name, "maxx")
        binx     = self._spectra.get(name, "binx")
        spectrum = self._get_spectrum_info("spectrum", index=index)
        w        = self._spectra.get(name, "data")

        if self._get_spectrum_info("cutoff", index=index) is not None:
            if len(self._get_spectrum_info("cutoff", index=index)) > 0:
                minCutoff = self._get_spectrum_info("cutoff", index=index)[0]
                maxCutoff = self._get_spectrum_info("cutoff", index=index)[1]
                if minCutoff is not None:
                    if dim == 1:
                        w = np.ma.masked_where(w < minCutoff, w)
                    if dim == 2:
                        w = np.ma.masked_where(w < minCutoff, w)
                if maxCutoff is not None:
                    if dim == 1:
                        w = np.ma.masked_where(w > maxCutoff, w)
                    if dim == 2:
                        w = np.ma.masked_where(w > maxCutoff, w)
        if w is None or len(w) <= 0:
            self.logger.debug('plotPlot - w is None or len(w) <= 0')
            return
        self._set_spectrum_info(data=w, index=index)

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
        self.updatePlot()

    def updatePlot(self):
        cp = self._get_current_plot()
        auto_scale_status = cp.histo_autoscale.isChecked()
        cp.histo_autoscale.setChecked(auto_scale_status)
        self.logger.info('updatePlot')

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
                    self.logger.debug('updatePlot - ax is None')
                    if len(self._get_spectrum_info_dict()) == 0:
                        return QMessageBox.about(self._parent_widget, "Warning!", "Configuration file has probably changed, please reset the window geometry (add plots or load geo file)")
                    return
                self.plotPlot(index)
                dim = self._spectra.get(self._name_from_index(0), "dim")
                if auto_scale_status:
                    if dim == 1:
                        self.setAxisScale(ax, 0, "x", "y")
                if dim == 2:
                    self.setAxisScale(ax, 0, "x", "y", "z")

                    spectrum = self._get_spectrum_info("spectrum", index=index)
                    if spectrum is not None and dim == 2:
                        if ax:
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes("right", size="5%", pad=0.05)
                            cp.figure.colorbar(spectrum, cax=cax, orientation="vertical")
                            cp.figure.tight_layout(rect=[0, 0, 0.95, 1])
                            cp.canvas.draw_idle()

                try:
                    self.removeCb(ax)
                except Exception:
                    pass
                self._draw_gate(0)
            else:
                self.logger.debug('updatePlot - self.currentPlot.isEnlarged FALSE')
                for index, value in self._get_geo().items():
                    ax = self._get_spectrum_info("axis", index=index)
                    if ax is None:
                        self.logger.debug('updatePlot - ax is None')
                        if len(self._get_spectrum_info_dict()) == 0:
                            return QMessageBox.about(self._parent_widget, "Warning!", "Configuration file has probably changed, please reset the window geometry (add plots or load geo file)")
                        continue
                    self.plotPlot(index)
                    dim = self._spectra.get(self._name_from_index(index), "dim")
                    if auto_scale_status:
                        if dim == 1:
                            self.setAxisScale(ax, index, "x", "y")
                        elif dim == 2:
                            self.setAxisScale(ax, index, "x", "y", "z")
                    self._draw_gate(index)

            cp.figure.tight_layout()
            cp.canvas.draw()
        except NameError:
            self.logger.debug('updatePlot - NameError exception', exc_info=True)
            pass

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
                with open(filename) as f:
                    for line in f:
                        parts = line.split()
                        if not parts or len(parts) < 5:
                            continue
                        lo, hi = float(parts[0]), float(parts[1])
                        r, g, b = map(float, parts[2:])
                        bounds.append(lo)
                        color_list.append((r, g, b))
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
                    try:
                        self.removeCb(ax)
                    except Exception:
                        pass
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
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

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def axesChilds(self):
        try:
            cp = self._get_current_plot()
            for ax in cp.figure.axes:
                print("Simon - axes ---------------------------- ", ax)
                for child in ax.get_children():
                    print("Simon - axesChilds - ", child)
                    if type(child) == matplotlib.lines.Line2D:
                        print("Simon - axesChild get_c", child.get_c())
                        print("Simon - axesChild get_lw", child.get_lw())
                        print("Simon - axesChild get_ls", child.get_ls())
                        print("Simon - axesChild get_xdata:", child.get_xdata())
                        print("Simon - axesChild get_ydata:", child.get_ydata())
        except NameError:
            raise

    def axesChildsTest(self, axis=None):
        try:
            dumTypeList = []
            if type(axis) == type(dumTypeList):
                return
            print("Simon - axes ---------------------------- ", axis)
            for child in axis.get_children():
                print("Simon - axesChilds - ", child)
                if type(child) == matplotlib.lines.Line2D:
                    print("Simon - axesChild get_c", child.get_c())
                    print("Simon - axesChild get_lw", child.get_lw())
                    print("Simon - axesChild get_ls", child.get_ls())
                    print("Simon - axesChild get_xdata:", child.get_xdata())
                    print("Simon - axesChild get_ydata:", child.get_ydata())
        except NameError:
            raise
