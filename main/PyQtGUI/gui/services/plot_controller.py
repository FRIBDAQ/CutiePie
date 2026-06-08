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

    def __init__(self, window, wTab, wConf, logger=None):
        self._w     = window
        self._wTab  = wTab
        self._wConf = wConf
        self.logger = logger or logging.getLogger(__name__)

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
        self._w.currentPlot.selected_plot_index = None
        self._w.currentPlot.next_plot_index = -1
        self._w.geometry_applied = True

    # ------------------------------------------------------------------
    # Axis scaling
    # ------------------------------------------------------------------

    def setAxisScale(self, ax, index, *scale):
        self.logger.info('setAxisScale - index: %s', index)

        wPlot = self._w.currentPlot
        axisIsLog = self._w.getSpectrumInfo("log", index=index)
        axisIsAutoScale = wPlot.histo_autoscale.isChecked()

        if (self._w.getSpectrumInfoREST("dim", index=index) == 1):
            xmin = self._w.getSpectrumInfo("minx", index=index)
            xmax = self._w.getSpectrumInfo("maxx", index=index)
            if "x" in scale and xmin is not None and xmax is not None:
                ax.set_xlim(xmin, xmax)
            if "y" in scale or "log" in scale:
                ymin = self._w.getSpectrumInfo("miny", index=index)
                ymax = self._w.getSpectrumInfo("maxy", index=index)
                if (not ymin or ymin is None or ymin == 0) and (not ymax or ymax is None or ymax == 0):
                    ymin = self._w.minY
                    ymax = self._w.maxY
                if axisIsAutoScale:
                    xmin, xmax = ax.get_xlim()
                    ymax = self.getMinMaxInRange(index, xmin=xmin, xmax=xmax)
                if axisIsLog:
                    if ymin <= 0:
                        ymin = 0.001
                    if ymax <= 0:
                        self.logger.warning('setAxisScale - all value <0 for : %s - cannot log scale', self._w.nameFromIndex(index))
                    else:
                        ax.set_ylim(ymin, ymax)
                        ax.set_yscale("log")
                else:
                    ax.set_ylim(ymin, ymax)
                    ax.set_yscale("linear")
                self._w.setSpectrumInfo(miny=ymin, index=index)
                self._w.setSpectrumInfo(maxy=ymax, index=index)
        else:
            xmin = self._w.getSpectrumInfo("minx", index=index)
            xmax = self._w.getSpectrumInfo("maxx", index=index)
            ymin = self._w.getSpectrumInfo("miny", index=index)
            ymax = self._w.getSpectrumInfo("maxy", index=index)

            if "x" in scale and xmin is not None and xmax is not None:
                ax.set_xlim(xmin, xmax)
            if "y" in scale and ymin is not None and ymax is not None:
                ax.set_ylim(ymin, ymax)
            if "z" in scale or "log" in scale:
                zmin = self._w.getSpectrumInfo("minz", index=index)
                zmax = self._w.getSpectrumInfo("maxz", index=index)
                spectrum = self._w.getSpectrumInfo("spectrum", index=index)
                if spectrum is None:
                    return
                if (not zmin or zmin is None or zmin == 0) and (not zmax or zmax is None or zmax == 0):
                    zmin = self._w.minZ
                    zmax = self._w.maxZ
                if axisIsAutoScale:
                    xmin, xmax = ax.get_xlim()
                    ymin, ymax = ax.get_ylim()
                    zmin, zmax = self.getMinMaxInRange(index, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
                    self._w.setSpectrumInfo(maxz=zmax, index=index)
                    self._w.setSpectrumInfo(minz=zmin, index=index)
                spectrum.set_clim(vmin=zmin, vmax=zmax)
                if axisIsLog:
                    self.setCmapNorm("log", index)
                else:
                    self.setCmapNorm("linear", index)
                self._w.setSpectrumInfo(spectrum=spectrum, index=index)

    def setCmapNorm(self, scale, index):
        self.logger.info('setCmapNorm')
        validScales = ["linear", "log", "linearCentered"]
        if scale not in validScales or index is None:
            self.logger.debug('setCmapNorm - scale not in validScales or index is None')
            return
        spectrum = self._w.getSpectrumInfo("spectrum", index=index)
        zmin, zmax = spectrum.get_clim()

        if scale is validScales[0]:
            if zmin > zmax:
                self.logger.warning('setCmapNorm - zmin > zmax')
                spectrum.set_norm(colors.Normalize(vmin=self._w.minZ, vmax=self._w.maxZ))
            else:
                spectrum.set_norm(colors.Normalize(vmin=zmin, vmax=zmax))
        elif scale is validScales[1]:
            if zmin and zmin <= 0:
                zmin = 0.001
                self.logger.warning('setCmapNorm - LogNorm with zmin<=0, may want to use CenteredNorm')
            spectrum.set_norm(colors.LogNorm(vmin=zmin, vmax=zmax))
            if zmin > zmax:
                self.logger.warning('setCmapNorm - zmin > zmax')
                spectrum.set_norm(colors.LogNorm(vmin=self._w.minZ, vmax=self._w.maxZ))
        elif scale is validScales[2]:
            palette = copy(plt.cm.jet)
            palette.set_bad(color='white')
            data = self._w.getSpectrumInfo("data", index=index)
            spectrum.set_cmap(palette)
            spectrum.set_norm(centeredNorm(data, 50000))
        if self._w.getEnlargedSpectrum() is None:
            ax = spectrum.axes
            self.removeCb(ax)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            label = "colorbar_" + str(index)
            cax.set_label(label)
            self._w.currentPlot.figure.colorbar(spectrum, cax=cax, orientation='vertical')

    def autoScaleAxisBox(self, forIndex):
        self.logger.info('autoScaleAxisBox - forIndex: %s', forIndex)
        try:
            ax = None
            if self._w.currentPlot.isEnlarged:
                ax = self._w.getSpectrumInfo("axis", index=0)
                dim = self._w.getSpectrumInfoREST("dim", index=0)
                if ax is None:
                    self.logger.debug('autoScaleAxisBox - isEnlarged TRUE - ax is None ')
                    return
                if dim == 1:
                    self.setAxisScale(ax, 0, "y")
                elif dim == 2:
                    self.setAxisScale(ax, 0, "z")
                self._w.drawGate(0)
            elif forIndex is not None:
                ax = self._w.getSpectrumInfo("axis", index=forIndex)
                dim = self._w.getSpectrumInfoREST("dim", index=forIndex)
                if ax is None:
                    self.logger.debug('autoScaleAxisBox - forIndex - ax is None ')
                    return
                if dim == 1:
                    self.setAxisScale(ax, forIndex, "y")
                elif dim == 2:
                    self.setAxisScale(ax, forIndex, "z")
            else:
                for index, name in self._w.getGeo().items():
                    if name:
                        ax = self._w.getSpectrumInfo("axis", index=index)
                        dim = self._w.getSpectrumInfoREST("dim", index=index)
                        if ax is None:
                            self.logger.debug('autoScaleAxisBox - isEnlarged FALSE - ax is None ')
                            return
                        if dim == 1:
                            self.setAxisScale(ax, index, "y")
                        elif dim == 2:
                            self.setAxisScale(ax, index, "z")
                        self._w.drawGate(index)
            self._w.currentPlot.canvas.draw()
        except:
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

        dim = self._w.getSpectrumInfoREST("dim", index=index)
        minx = self._w.getSpectrumInfoREST("minx", index=index)
        maxx = self._w.getSpectrumInfoREST("maxx", index=index)
        binx = self._w.getSpectrumInfoREST("binx", index=index)
        data = self._w.getSpectrumInfo("data", index=index)
        stepx = (float(maxx) - float(minx)) / float(binx)
        binminx = int((xmin - minx) / stepx)
        binmaxx = int((xmax - minx) / stepx)
        if dim == 1:
            try:
                result = data[binminx + 1:binmaxx + 2].max() * 1.1
            except:
                self.logger.debug('getMinMaxInRange - dim == 1 - exception occured', exc_info=True)
                return self._w.maxY
        elif dim == 2:
            try:
                miny = self._w.getSpectrumInfoREST("miny", index=index)
                maxy = self._w.getSpectrumInfoREST("maxy", index=index)
                biny = self._w.getSpectrumInfoREST("biny", index=index)
                stepy = (float(maxy) - float(miny)) / float(biny)
                binminy = int((ymin - miny) / stepy)
                binmaxy = int((ymax - miny) / stepy)
                minimum, maximum = self.customMinMax(data[binminy:binmaxy + 1, binminx:binmaxx + 1])
                result = minimum, maximum
            except:
                self.logger.debug('getMinMaxInRange - dim == 2 - exception occured', exc_info=True)
                return self._w.minZ, self._w.maxZ
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
                minimum = self._w.minZ
            if len(subMax) == 0:
                minimum = self._w.maxZ
            elif len(subMin) > 0 and len(subMax) > 0:
                minimum = min(subMin)
                maximum = max(subMax)
            return minimum, maximum

    def getAxisProperties(self, index):
        self.logger.info('getAxisProperties')
        try:
            ax = self._w.getSpectrumInfo("axis", index=index)
            if ax is None:
                return None
            else:
                return list(ax.get_xlim()), list(ax.get_ylim())
        except:
            self.logger.debug('getAxisProperties - exception occured', exc_info=True)
            pass

    # ------------------------------------------------------------------
    # Zoom / toolbar callbacks
    # ------------------------------------------------------------------

    def zoomInOut(self, arg):
        self._w.currentPlot.histo_autoscale.setChecked(False)
        self.logger.info('zoomInOut - arg: %s', arg)
        index = self._w.autoIndex()
        ax = None
        spectrum = self._w.getSpectrumInfo("spectrum", index=index)
        if spectrum is None:
            return
        ax = spectrum.axes
        dim = self._w.getSpectrumInfoREST("dim", index=index)
        if dim == 1:
            ymin, ymax = ax.get_ylim()
            if arg == "in":
                ymax = ymax * 0.5
            elif arg == "out":
                ymax = ymax * 2
            ax.set_ylim(ymin, ymax)
            self._w.setSpectrumInfo(miny=ymin, index=index)
            self._w.setSpectrumInfo(maxy=ymax, index=index)
            self._w.setSpectrumInfo(spectrum=spectrum, index=index)
        elif dim == 2:
            zmin, zmax = spectrum.get_clim()
            if arg == "in":
                zmax = zmax * 0.5
            elif arg == "out":
                zmax = zmax * 2
            spectrum.set_clim(zmin, zmax)
            self._w.setSpectrumInfo(minz=zmin, index=index)
            self._w.setSpectrumInfo(maxz=zmax, index=index)
            self._w.setSpectrumInfo(spectrum=spectrum, index=index)
        self._w.drawGate(index)
        self._w.currentPlot.canvas.draw()

    def zoomCallback(self, event):
        self.logger.info('zoomCallback')
        self._w.currentPlot.zoomPress = True

    def customZoomButtonCallback(self):
        self.logger.info('customZoomButtonCallback')
        self._w.currentPlot.histo_autoscale.setChecked(False)
        if self._w.currentPlot.zoomPress:
            self._w.currentPlot.zoom_action.triggered.emit()
            self._w.currentPlot.zoom_action.setChecked(False)
            self._w.currentPlot.customZoomButton.setDown(False)
            self._w.currentPlot.zoomPress = False
        else:
            self._w.currentPlot.zoom_action.triggered.emit()
            self._w.currentPlot.zoom_action.setChecked(True)
            self._w.currentPlot.customZoomButton.setDown(True)

    def customHomeButtonCallback(self, index=None):
        self.logger.info('customHomeButtonCallback - index: %s', index)
        index_list = [idx for idx, name in self._w.getGeo().items() if index is None]
        if index is not None:
            index_list = [index]
        for idx in index_list:
            ax = None
            spectrum = self._w.getSpectrumInfo("spectrum", index=idx)
            if spectrum is None:
                return
            ax = spectrum.axes
            dim = self._w.getSpectrumInfoREST("dim", index=idx)
            xmin = self._w.getSpectrumInfoREST("minx", index=idx)
            xmax = self._w.getSpectrumInfoREST("maxx", index=idx)
            ymin = self._w.getSpectrumInfoREST("miny", index=idx)
            ymax = self._w.getSpectrumInfoREST("maxy", index=idx)

            ax.set_xlim(xmin, xmax)
            if dim == 1:
                ymax = self.getMinMaxInRange(idx, xmin=xmin, xmax=xmax)
                ax.set_ylim(ymin, ymax)
                if self._w.getSpectrumInfo("log", index=idx):
                    ax.set_yscale("linear")
            if dim == 2:
                ax.set_ylim(ymin, ymax)
                zmin, zmax = self.getMinMaxInRange(idx, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
                spectrum.set_clim(vmin=zmin, vmax=zmax)
                self.setCmapNorm("linear", idx)
                self._w.setSpectrumInfo(maxz=zmax, index=idx)
                self._w.setSpectrumInfo(minz=zmin, index=idx)
            self._w.drawGate(idx)

            self._w.setSpectrumInfo(log=None, index=idx)
            self._w.setSpectrumInfo(minx=xmin, index=idx)
            self._w.setSpectrumInfo(maxx=xmax, index=idx)
            self._w.setSpectrumInfo(miny=ymin, index=idx)
            self._w.setSpectrumInfo(maxy=ymax, index=idx)
            self._w.setSpectrumInfo(spectrum=spectrum, index=idx)
        self._w.currentPlot.canvas.draw()

    def logButtonCallback(self, *arg):
        self.logger.info('logButtonCallback - arg: %s', arg)
        index = None
        logAllPlot = False
        unlogAllPlot = False
        if "logAll" in arg:
            logAllPlot = True
        elif "unlogAll" in arg:
            unlogAllPlot = True
        else:
            index = arg[0]

        wPlot = self._w.currentPlot
        index_list = [idx for idx, name in self._w.getGeo().items() if logAllPlot or unlogAllPlot]

        if index is not None:
            index_list = [index]
        for idx in index_list:
            ax = None
            spectrum = self._w.getSpectrumInfo("spectrum", index=idx)
            if spectrum is None:
                continue
            ax = spectrum.axes
            if logAllPlot:
                self._w.setSpectrumInfo(index=idx, log=True)
            elif unlogAllPlot:
                self._w.setSpectrumInfo(index=idx, log=False)
            elif self._w.getSpectrumInfo("log", index=idx) and not logAllPlot and not unlogAllPlot:
                self._w.setSpectrumInfo(index=idx, log=False)
            elif not self._w.getSpectrumInfo("log", index=idx) and not logAllPlot and not unlogAllPlot:
                self._w.setSpectrumInfo(index=idx, log=True)

            self.setAxisScale(ax, idx, "log")
        wPlot.canvas.draw()

    def zoom_handle_right_click(self):
        self.logger.info('zoom_handle_right_click')
        menu = QMenu()
        item1 = menu.addAction("Set Zoom Range Manually")
        index = self._w.currentPlot.selected_plot_index
        if index is None:
            return QMessageBox.about(self._w, "Warning!", "Please add/select a spectrum")
        else:
            item1.triggered.connect(self.cutoffButtonCallback)
            plotgui = self._w.currentPlot
            menuPosX = plotgui.mapToGlobal(QtCore.QPoint(0, 0)).x() + plotgui.customZoomButton.geometry().topLeft().x()
            menuPosY = plotgui.mapToGlobal(QtCore.QPoint(0, 0)).y() + plotgui.customZoomButton.geometry().topLeft().y()
            menuPos = QtCore.QPoint(menuPosX, menuPosY)
            menu.exec_(menuPos)

    def handle_right_click(self):
        self.logger.info('handle_right_click')
        menu = QMenu()
        item1 = menu.addAction("Reset all")
        item1.triggered.connect(lambda: self.customHomeButtonCallback())
        plotgui = self._w.currentPlot
        menuPosX = plotgui.mapToGlobal(QtCore.QPoint(0, 0)).x() + plotgui.customHomeButton.geometry().topLeft().x()
        menuPosY = plotgui.mapToGlobal(QtCore.QPoint(0, 0)).y() + plotgui.customHomeButton.geometry().topLeft().y()
        menuPos = QtCore.QPoint(menuPosX, menuPosY)
        menu.exec_(menuPos)

    def log_handle_right_click(self):
        self.logger.info('log_handle_right_click')
        menu = QMenu()
        item1 = menu.addAction("Log all")
        item2 = menu.addAction("unLog all")
        item1.triggered.connect(lambda: self.logButtonCallback("logAll"))
        item2.triggered.connect(lambda: self.logButtonCallback("unlogAll"))
        plotgui = self._w.currentPlot
        menuPosX = plotgui.mapToGlobal(QtCore.QPoint(0, 0)).x() + plotgui.logButton.geometry().topLeft().x()
        menuPosY = plotgui.mapToGlobal(QtCore.QPoint(0, 0)).y() + plotgui.logButton.geometry().topLeft().y()
        menuPos = QtCore.QPoint(menuPosX, menuPosY)
        menu.exec_(menuPos)

    # ------------------------------------------------------------------
    # Cutoff popup
    # ------------------------------------------------------------------

    def okCutoff(self):
        self.logger.info('okCutoff')
        index = self._w.currentPlot.selected_plot_index
        if index is None:
            self.logger.debug('okCutoff - index is None')
            return
        spectrum = self._w.getSpectrumInfo("spectrum", index=index)
        if spectrum is None:
            return
        ax = self._w.getSpectrumInfo("axis", index=index)
        if ax is None:
            self.logger.debug('okCutoff - ax is None')
            return

        dim = self._w.getSpectrumInfoREST("dim", index=index)
        rangeXmin = self._w.cutoffp.lineeditXMin.text()
        rangeXmax = self._w.cutoffp.lineeditXMax.text()
        rangeYmin = self._w.cutoffp.lineeditYMin.text()
        rangeYmax = self._w.cutoffp.lineeditYMax.text()

        cutoffVal = [None, None]
        cutoffMin = self._w.cutoffp.lineeditZMin.text()
        cutoffMax = self._w.cutoffp.lineeditZMax.text()

        try:
            rangeXmin = float(rangeXmin)
            rangeXmax = float(rangeXmax)
            rangeYmin = float(rangeYmin)
            rangeYmax = float(rangeYmax)
        except ValueError:
            self.logger.warning("okCutoff Range - Invalid input format for zoom range(s). Please enter valid numbers.")
            return

        if rangeXmin is not None and rangeXmax is not None and rangeXmax < rangeXmin:
            buff = rangeXmin
            rangeXmin = rangeXmax
            rangeXmax = buff
            self.logger.warning('okCutoff Range - new range X values swapped because min > max')
        if rangeYmin is not None and rangeYmax is not None and rangeYmax < rangeYmin:
            buff = rangeYmin
            rangeYmin = rangeYmax
            rangeYmax = buff
            self.logger.warning('okCutoff Range - new range Y values swapped because min > max')
        if dim == 2:
            if self._w.cutoffp.lineeditZMin.text() != "" and self._w.cutoffp.lineeditZMin.text().isdigit():
                cutoffVal[0] = float(cutoffMin)
                self._w.setSpectrumInfo(cutoff=cutoffVal, index=index)
            if self._w.cutoffp.lineeditZMax.text() != "" and self._w.cutoffp.lineeditZMax.text().isdigit():
                cutoffVal[1] = float(cutoffMax)
                self._w.setSpectrumInfo(cutoff=cutoffVal, index=index)
            if cutoffVal[0] is not None and cutoffVal[1] is not None and cutoffVal[1] < cutoffVal[0]:
                cutoffVal = [cutoffVal[1], cutoffVal[0]]
                self._w.setSpectrumInfo(cutoff=cutoffVal, index=index)
        try:
            spectrum = self._w.getSpectrumInfo("spectrum", index=index)
            ax.set_xlim(float(rangeXmin), float(rangeXmax))
            ax.set_ylim(float(rangeYmin), float(rangeYmax))
            self._w.setSpectrumInfo(minx=rangeXmin, index=index)
            self._w.setSpectrumInfo(maxx=rangeXmax, index=index)
            self._w.setSpectrumInfo(miny=rangeYmin, index=index)
            self._w.setSpectrumInfo(maxy=rangeYmax, index=index)
            if dim == 2:
                spectrum.set_clim(cutoffVal[0], cutoffVal[1])
                self._w.setSpectrumInfo(minz=cutoffVal[0], index=index)
                self._w.setSpectrumInfo(maxz=cutoffVal[1], index=index)
                self._w.setSpectrumInfo(spectrum=spectrum, index=index)
            self._w.setSpectrumInfo(spectrum=spectrum, index=index)
            self._w.drawGate(index)
            self._w.currentPlot.canvas.draw()
        except NameError as err:
            self.logger.debug('okCutoff - NameError', exc_info=True)
            pass

        self._w.cutoffp.close()

    def cancelCutoff(self):
        self._w.cutoffp.close()

    def resetCutoff(self, doUpdate):
        self.logger.info('resetCutoff - doUpdate: %s', doUpdate)
        index = self._w.currentPlot.selected_plot_index
        if index is None:
            return
        cutoffVal = [None, None]
        self._w.setSpectrumInfo(cutoff=cutoffVal, index=index)
        if doUpdate:
            self.updatePlot()
        self._w.cutoffp.close()

    def cutoffButtonCallback(self, *arg):
        self.logger.info('cutoffButtonCallback')
        self._w.currentPlot.histo_autoscale.setChecked(False)
        index = self._w.currentPlot.selected_plot_index
        if index is None:
            return QMessageBox.about(self._w, "Warning!", "Please Add/Select a Spectrum")
        name = self._w.nameFromIndex(index)
        if name is not None:
            self._w.cutoffp.setWindowTitle("Set zoom range for: " + name)
        else:
            self._w.cutoffp.setWindowTitle("Set zoom range for: ???")
        self._w.cutoffp.setGeometry(300, 100, 300, 100)
        if self._w.cutoffp.isVisible():
            self._w.cutoffp.close()
        if self._w.getSpectrumInfo("cutoff", index=index) is not None and len(self._w.getSpectrumInfo("cutoff", index=index)) > 0:
            ax = self._w.getSpectrumInfo("axis", index=index)
            dim = self._w.getSpectrumInfoREST("dim", index=index)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            self._w.cutoffp.lineeditXMin.setText(f"{xmin:.1f}")
            self._w.cutoffp.lineeditXMax.setText(f"{xmax:.1f}")
            self._w.cutoffp.lineeditYMin.setText(f"{ymin:.1f}")
            self._w.cutoffp.lineeditYMax.setText(f"{ymax:.1f}")
            if dim == 2:
                spectrum = self._w.getSpectrumInfo("spectrum", index=index)
                zmin, zmax = spectrum.get_clim()
                self._w.cutoffp.lineeditZMin.setText(f"{zmin:.1f}")
                self._w.cutoffp.lineeditZMax.setText(f"{zmax:.1f}")
            if dim == 1:
                self._w.cutoffp.layout1d()
            elif dim == 2:
                self._w.cutoffp.layout2d()
            self._w.cutoffp.show()
        else:
            QMessageBox.about(self._w, "Warning!", "Please Add/Select a Spectrum")
            self.logger.warning('cutoffButtonCallback - you broke something really bad - spectrum dict: %s', self._w.getSpectrumInfo("cutoff", index=index))

    def updatePlotLimits(self, sleepTime=0):
        self.logger.info('updatePlotLimits - sleepTime: %s', sleepTime)
        ax = None
        index = self._w.currentPlot.selected_plot_index
        ax = self._w.getSpectrumInfo("axis", index=index)
        if ax is None:
            self.logger.debug('updatePlotLimits - ax is None')
            return

        time.sleep(sleepTime)

        try:
            x_range, y_range = self.getAxisProperties(index)
            self._w.setSpectrumInfo(minx=x_range[0], index=index)
            self._w.setSpectrumInfo(maxx=x_range[1], index=index)
            self._w.setSpectrumInfo(miny=y_range[0], index=index)
            self._w.setSpectrumInfo(maxy=y_range[1], index=index)

            spectrum = self._w.getSpectrumInfo("spectrum", index=index)
            ax.set_xlim(x_range[0], x_range[1])
            ax.set_ylim(y_range[0], y_range[1])

            self._w.setSpectrumInfo(spectrum=spectrum, index=index)
            self._w.drawGate(index)
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
            except:
                self.logger.debug('removeCb - IndexError exception', exc_info=True)
                pass

    def select_plot(self, index):
        self.logger.info('select_plot - index: %s', index)
        for i, axis in enumerate(self._w.currentPlot.figure.axes):
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
        if self._w.nameFromIndex(index):

            dim = self._w.getSpectrumInfoREST("dim", index=index)
            minx = self._w.getSpectrumInfo("minx", index=index)
            maxx = self._w.getSpectrumInfo("maxx", index=index)
            binx = self._w.getSpectrumInfo("binx", index=index)
            biny = self._w.getSpectrumInfo("biny", index=index)

            w = self._w.getSpectrumInfoREST("data", index=index)

            if self._w.getSpectrumInfo("cutoff", index=index) is not None:
                if len(self._w.getSpectrumInfo("cutoff", index=index)) > 0:
                    minCutoff = self._w.getSpectrumInfo("cutoff", index=index)[0]
                    maxCutoff = self._w.getSpectrumInfo("cutoff", index=index)[1]
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
            self._w.setSpectrumInfo(data=w, index=index)

            if dim == 1:
                axis.set_xlim(minx, maxx)
                line, = axis.plot([], [], drawstyle='steps')
                name = self._w.getSpectrumInfo("name", index=index)
                axis.set_title("{}".format(name))
                self._w.setSpectrumInfo(spectrum=line, index=index)
                if len(w) > 0:
                    X = np.array(self.createRange(binx, minx, maxx))
                    line.set_data(X, w)
                    self._w.setSpectrumInfo(spectrum=line, index=index)
            else:
                minxREST = self._w.getSpectrumInfoREST("minx", index=index)
                maxxREST = self._w.getSpectrumInfoREST("maxx", index=index)
                minyREST = self._w.getSpectrumInfoREST("miny", index=index)
                maxyREST = self._w.getSpectrumInfoREST("maxy", index=index)

                if w is None:
                    w = np.zeros((int(binx), int(biny)))

                self._w.palette = copy(plt.cm.plasma)
                w = np.ma.masked_where(w == 0, w)
                self._w.palette.set_bad(color='white')

                spectrum = axis.imshow(w,
                                       interpolation='none',
                                       extent=[float(minxREST), float(maxxREST), float(minyREST), float(maxyREST)],
                                       aspect='auto',
                                       origin='lower',
                                       vmin=float(self._w.minZ), vmax=float(self._w.maxZ),
                                       cmap=self._w.palette)
                self._w.setSpectrumInfo(spectrum=spectrum, index=index)

                name = self._w.getSpectrumInfo("name", index=index)
                axis.set_title("{}".format(name))

                if w is not None:
                    spectrum.set_data(w)
                    self._w.setSpectrumInfo(spectrum=spectrum, index=index)

                if self._w.getEnlargedSpectrum() is None:
                    divider = make_axes_locatable(axis)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    label = "colorbar_" + str(index)
                    cax.set_label(label)
                    self._w.currentPlot.figure.colorbar(spectrum, cax=cax, orientation='vertical')

    def add(self, index):
        self.logger.info('add - index: %s', index)
        a = None
        if self._w.currentPlot.isEnlarged:
            a = self._w.currentPlot.figure.get_axes()[0]
        else:
            a = self.select_plot(index)
        try:
            self.removeCb(a)
        except:
            pass
        a.clear()
        self.setupPlot(a, index)

    def addPlot(self):
        self.logger.info('addPlot')
        if not self._w.geometry_applied:
            print("addPlot: Apply Geometry first!, return")
            return

        if self._wConf.histo_list.count() == 0:
            QMessageBox.about(self._w, "Warning", 'Please click on "Connection" and fill in the information')

        try:
            if self._w.currentPlot.isLoaded:
                self.logger.debug('addPlot - isLoaded TRUE - getGeo: %s', self._w.getGeo())
                self._w.currentPlot.histo_autoscale.setChecked(True)
                for key, value in self._w.getGeo().items():
                    if self._w.getSpectrumInfoREST("dim", name=value) is None:
                        continue
                    self.add(key)
                    self.autoScaleAxisBox(key)
            else:
                index = self._w.nextIndex()
                name = str(self._wConf.histo_list.currentText())
                self.logger.debug('addPlot - isLoaded FALSE - index, name: %s, %s', index, name)

                if self._w.getSpectrumInfoREST("dim", name=name) is None:
                    self.logger.debug('addPlot - isLoaded FALSE - dim is None')
                    return

                self._w.setSpectrumInfo(cutoff=None, index=index)
                self._w.setGeo(index, name)
                self.add(index)

                self._w.currentPlot.histo_autoscale.setChecked(True)
                self.autoScaleAxisBox(index)

                dim = self._w.getSpectrumInfoREST("dim", index=index)
                ax = self._w.getSpectrumInfo("axis", index=index)
                xmin, xmax = ax.get_xlim()

                if dim == 1:
                    ymin = self._w.getSpectrumInfo("miny", index=index)
                    ymax = self.getMinMaxInRange(index, xmin=xmin, xmax=xmax)
                    ax.set_ylim(ymin, ymax)
                else:
                    ymin, ymax = ax.get_xlim()
                    zmin, zmax = self.getMinMaxInRange(index, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
                    self._w.setSpectrumInfo(maxz=zmax, index=index)
                    self._w.setSpectrumInfo(minz=zmin, index=index)
                    spectrum = self._w.getSpectrumInfo("spectrum", index=index)
                    spectrum.set_clim(vmin=zmin, vmax=zmax)
                    self._w.setSpectrumInfo(spectrum=spectrum, index=index)

                self._w.currentPlot.logButton.setDown(False)
                self._w.setSpectrumInfo(log=False, index=index)
                self._w.drawGate(index)

                self.removeRectangle()
                self._w.currentPlot.recDashed = self.createDashedRectangle(
                    self._w.currentPlot.figure.axes[self._w.currentPlot.next_plot_index])

                try:
                    self._w.currentPlot.figure.tight_layout()
                except ValueError:
                    self.logger.debug('addPlot - ValueError exception', exc_info=True)
                    pass

                self._w.currentPlot.canvas.draw_idle()
                self._w.currentPlot.isSelected = False
        except NameError:
            raise

        if self._w.stopAutoUpdateThread.is_set():
            self._w.autoUpdateStart()

        if not self._wTab.countClickTab[self._wTab.currentIndex()]:
            self._w.bindDynamicSignal()

    def createRange(self, bins, vmin, vmax):
        self.logger.info('createRange')
        x = []
        step = (float(vmax) - float(vmin)) / float(bins)
        for i in np.arange(float(vmin), float(vmax), step):
            x.append(i + step)
        x.insert(0, float(vmin))
        return x

    def plotPlot(self, index, cmap=None):
        self.logger.info('plotPlot - index: %s', index)
        currentPlot = self._w.currentPlot

        dim = self._w.getSpectrumInfoREST("dim", index=index)
        minx = self._w.getSpectrumInfoREST("minx", index=index)
        maxx = self._w.getSpectrumInfoREST("maxx", index=index)
        binx = self._w.getSpectrumInfoREST("binx", index=index)
        spectrum = self._w.getSpectrumInfo("spectrum", index=index)
        w = self._w.getSpectrumInfoREST("data", index=index)

        if self._w.getSpectrumInfo("cutoff", index=index) is not None:
            if len(self._w.getSpectrumInfo("cutoff", index=index)) > 0:
                minCutoff = self._w.getSpectrumInfo("cutoff", index=index)[0]
                maxCutoff = self._w.getSpectrumInfo("cutoff", index=index)[1]
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
        self._w.setSpectrumInfo(data=w, index=index)

        if dim == 1:
            X = np.array(self.createRange(binx, minx, maxx))
            spectrum.set_data(X, w)
        else:
            w = np.ma.masked_where(w == 0, w)
            spectrum.set_data(w)
            if cmap is not None:
                spectrum.set_cmap(cmap)
            elif self._w.old_cmap is not None:
                spectrum.set_cmap(self._w.old_cmap)
            elif hasattr(self._w, "palette") and self._w.palette is not None:
                spectrum.set_cmap(self._w.palette)
            else:
                spectrum.set_cmap(plt.get_cmap("viridis"))

        self._w.setSpectrumInfo(spectrum=spectrum, index=index)
        self._w.currentPlot = currentPlot

    @pyqtSlot()
    def _updatePlotOnGui(self):
        self.updatePlot()

    def updatePlot(self):
        auto_scale_status = self._w.currentPlot.histo_autoscale.isChecked()
        self._w.currentPlot.histo_autoscale.setChecked(auto_scale_status)
        self.logger.info('updatePlot')

        self._w.cleanPopupExit(False)

        try:
            if self._w.currentPlot.isEnlarged:
                index = self._w.autoIndex()
                name = self._w.nameFromIndex(index)
                if index is None or self._w.getSpectrumInfoREST("dim", name=name) is None:
                    self.logger.debug('updatePlot - index is None or dim is None')
                    return
                self.logger.debug('updatePlot - self.currentPlot.isEnlarged TRUE')
                ax = self._w.getSpectrumInfo("axis", index=0)
                if ax is None:
                    self.logger.debug('updatePlot - ax is None')
                    if len(self._w.getSpectrumInfoDict()) == 0:
                        return QMessageBox.about(self._w, "Warning!", "Configuration file has probably changed, please reset the window geometry (add plots or load geo file)")
                    return
                self.plotPlot(index)
                dim = self._w.getSpectrumInfoREST("dim", index=0)
                if auto_scale_status:
                    if dim == 1:
                        self.setAxisScale(ax, 0, "x", "y")
                if dim == 2:
                    self.setAxisScale(ax, 0, "x", "y", "z")

                    spectrum = self._w.getSpectrumInfo("spectrum", index=index)
                    if spectrum is not None and dim == 2:
                        if ax:
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes("right", size="5%", pad=0.05)
                            self._w.currentPlot.figure.colorbar(spectrum, cax=cax, orientation="vertical")
                            self._w.currentPlot.figure.tight_layout(rect=[0, 0, 0.95, 1])
                            self._w.currentPlot.canvas.draw_idle()

                try:
                    self.removeCb(ax)
                except:
                    pass
                self._w.drawGate(0)
            else:
                self.logger.debug('updatePlot - self.currentPlot.isEnlarged FALSE')
                for index, value in self._w.getGeo().items():
                    ax = self._w.getSpectrumInfo("axis", index=index)
                    if ax is None:
                        self.logger.debug('updatePlot - ax is None')
                        if len(self._w.getSpectrumInfoDict()) == 0:
                            return QMessageBox.about(self._w, "Warning!", "Configuration file has probably changed, please reset the window geometry (add plots or load geo file)")
                        continue
                    self.plotPlot(index)
                    dim = self._w.getSpectrumInfoREST("dim", index=index)
                    if auto_scale_status:
                        if dim == 1:
                            self.setAxisScale(ax, index, "x", "y")
                        elif dim == 2:
                            self.setAxisScale(ax, index, "x", "y", "z")
                    self._w.drawGate(index)

            self._w.currentPlot.figure.tight_layout()
            self._w.currentPlot.canvas.draw()
        except NameError:
            self.logger.debug('updatePlot - NameError exception', exc_info=True)
            pass

    def onColormapChange(self, cmap_name: str):
        self.logger.info("onColormapChange - cmap: %s", cmap_name)

        if not self._w.getGeo():
            self.logger.debug("onColormapChange - no active plots")
            return

        try:
            if cmap_name.lower() == "custom":
                QMessageBox.information(
                    self._w,
                    "Custom Colormap Format from a .txt File",
                    "Each line should be:\n<low> <high> <red> <green> <blue>\n"
                    "Example:\n0.0 0.5 0 1 0\n0.5 0.7 1 0 0\n0.7 1.0 0 0 1\n\n"
                    "<low> and <high> represent count percentiles.\n"
                    "The <low> value of the first line must be 0.0, and the <low> of each line must match the <high> of the previous line.\n\n"
                    "<red>, <green>, and <blue> are RGB values between 0 and 1."
                )
                filename, _ = QFileDialog.getOpenFileName(
                    self._w, "Open Custom Colormap", "", "Text Files (*.txt)"
                )
                if not filename:
                    self.logger.debug("No file selected for custom cmap")
                    return

                bounds = []
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

                self._w.palette = colors.LinearSegmentedColormap.from_list(
                    "custom_cmap", list(zip(bounds, color_list)), N=256
                )
                self._w.palette.set_bad(color="white")
            else:
                self._w.palette = copy(plt.get_cmap(cmap_name))
                self._w.palette.set_bad(color="white")

            for index, _ in self._w.getGeo().items():
                if self._w.getSpectrumInfoREST("dim", index=index) != 2:
                    continue
                spectrum = self._w.getSpectrumInfo("spectrum", index=index)
                if spectrum is None:
                    continue
                spectrum.set_cmap(self._w.palette)
                self._w.setSpectrumInfo(spectrum=spectrum, index=index)
                if self._w.currentPlot.isEnlarged:
                    self._w.old_cmap = spectrum.get_cmap()
                ax = self._w.getSpectrumInfo("axis", index=index)
                if ax is not None:
                    try:
                        self.removeCb(ax)
                    except Exception:
                        pass
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    self._w.currentPlot.figure.colorbar(spectrum, cax=cax, orientation="vertical")

            self._w.currentPlot.canvas.draw_idle()

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
            for ax in self._w.currentPlot.figure.axes:
                for child in ax.get_children():
                    if type(child) == matplotlib.patches.Rectangle:
                        if child.get_ls() == ":" and child.get_lw() == 2:
                            if self._w.currentPlot.recDashed is not None:
                                self._w.currentPlot.recDashed.remove()
                                self._w.currentPlot.recDashed = None
                        elif child.get_ls() == "-" and child.get_lw() == 2:
                            if self._w.currentPlot.rec is not None:
                                self._w.currentPlot.rec.remove()
                                self._w.currentPlot.rec = None
        except NameError:
            raise

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def axesChilds(self):
        try:
            for ax in self._w.currentPlot.figure.axes:
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
