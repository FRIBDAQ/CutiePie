import logging

import matplotlib
import matplotlib.lines as mlines

from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt, QObject, pyqtSignal


class SumRegionManager(QObject):
    """Owns sum-region CRUD, integration calculations, and result display.

    H2: this service no longer holds the histogram combo or the integrate popup.
    Widget writes leave as signals (region readout, name-combo population, popup
    show/close, integration result rows); widget reads arrive as method
    arguments gathered by the MainWindow adapter. The ONE injected widget that
    remains is the sum-region popup, and only for its `listRegionLine` /
    `prevPoint` drawing buffer, which is CO-OWNED with `gate_manager`
    (`addLine`/`removePrevLine`, mode="sum_region"); eliminating it is deferred
    to the gate_manager inversion. QMessageBox prompts stay as service-created
    dialogs (the same residual fit_manager/plot_controller keep).
    """

    canvasDrawRequested          = pyqtSignal()
    figureTightLayoutRequested   = pyqtSignal()
    updatePlotRequested          = pyqtSignal()
    sumRegionStarted             = pyqtSignal(int)   # sets currentPlot.toCreateSumRegion = True
    sumRegionEnded               = pyqtSignal()      # sets currentPlot.toCreateSumRegion = False
    gateSignalsDisconnectRequested = pyqtSignal()

    # H2 output signals (MainWindow renders the widgets)
    regionReadoutChanged         = pyqtSignal(str)    # the region-point text box content
    sumRegionCreatePrepared      = pyqtSignal(list)   # region names -> populate name combo + show popup
    sumRegionSelectionChanged    = pyqtSignal(str)    # set the name combo's current text
    sumRegionPopupCloseRequested = pyqtSignal()
    integrationResultsReady      = pyqtSignal(list)   # table rows (each a 7-item list); [] = nothing to integrate
    integratePopupCloseRequested = pyqtSignal()

    def __init__(self, spectra, name_from_index, get_spectrum_info,
                 get_histo_names, skip_auto,
                 add_line, remove_prev_line, get_rest,
                 check_and_cancel_gate,
                 sum_popup, parent_widget=None, logger=None):
        super().__init__()
        self._spectra               = spectra
        self._name_from_index       = name_from_index      # (index) -> str
        self._get_spectrum_info     = get_spectrum_info    # (key, index=) -> value
        self._get_histo_names       = get_histo_names      # () -> [str]
        self._skip_auto             = skip_auto            # threading.Event
        self._add_line              = add_line             # (x, y, index[, label]) -> Line2D|None
        self._remove_prev_line      = remove_prev_line     # () -> None
        self._get_rest              = get_rest             # () -> PyREST|None
        self._check_and_cancel_gate = check_and_cancel_gate  # (doClose) -> None
        self._popup                 = sum_popup            # co-owned listRegionLine/prevPoint buffer only
        self._parent_widget         = parent_widget
        self.logger                 = logger or logging.getLogger(__name__)
        self.sumRegionDict          = {}
        self._creating_sum_region   = False
        # scratch state formerly parked on the popup (NOT shared with gate_manager)
        self._saved_region_names    = []
        self._active_sum_index      = None

    # ------------------------------------------------------------------
    # Sum region mouse handlers
    # ------------------------------------------------------------------

    def on_singleclick_sumRegion(self, event, index, name):
        self.logger.info('on_singleclick_sumRegion - index: %s', index)
        dim = self._spectra.get(name, "dim")
        if dim == 1:
            l = self._add_line(float(event.xdata), 0, index)
            self._popup.listRegionLine.append(l)
            if len(self._popup.listRegionLine) > 2:
                self._remove_prev_line()

            lineText = ""
            for nbLine in range(len(self._popup.listRegionLine)):
                if nbLine == 0:
                    lineText = lineText + (f"{nbLine}: X= {self._popup.listRegionLine[nbLine].get_xdata()[0]:.5f}")
                else:
                    lineText = lineText + (f"\n{nbLine}: X= {self._popup.listRegionLine[nbLine].get_xdata()[0]:.5f}")
            self.regionReadoutChanged.emit(lineText)

        elif dim == 2:
            tempLine = [line for line in self._popup.listRegionLine if line.get_label() == "closing_segment"]
            if len(tempLine) == 1:
                tempLine[0].remove()
                self._popup.listRegionLine.pop()

            l = self._add_line(float(event.xdata), float(event.ydata), index)
            if l is not None:
                self._popup.listRegionLine.append(l)

            lineNb = len(self._popup.listRegionLine)
            lineText = ""
            for nbLine in range(lineNb):
                if nbLine == 0:
                    lineText = lineText + (f"{nbLine}: X= {self._popup.listRegionLine[nbLine].get_xdata()[0]:.5f}   Y= {self._popup.listRegionLine[nbLine].get_ydata()[0]:.5f}")
                else:
                    lineText = lineText + (f"\n{nbLine}: X= {self._popup.listRegionLine[nbLine].get_xdata()[0]:.5f}   Y= {self._popup.listRegionLine[nbLine].get_ydata()[0]:.5f}")
            if lineNb == 0:
                lineText = lineText + (f"{lineNb}: X= {float(event.xdata):.5f}   Y= {float(event.ydata):.5f}")
            else:
                lineText = lineText + (f"\n{lineNb}: X= {float(event.xdata):.5f}   Y= {float(event.ydata):.5f}")
            self.regionReadoutChanged.emit(lineText)

            if lineNb > 1:
                label = "closing_segment"
                l = self._add_line(self._popup.listRegionLine[0].get_xdata()[0],
                                   self._popup.listRegionLine[0].get_ydata()[0], index, label)
                if l is not None:
                    self._popup.listRegionLine.append(l)

        self.canvasDrawRequested.emit()


    def on_singleclick_sumRegion_right(self, index, name):
        self.logger.info('on_singleclick_sumRegion_right - index: %s', index)
        dim = self._spectra.get(name, "dim")

        if dim == 2:
            lineNb = len(self._popup.listRegionLine)
            try:
                self._popup.prevPoint = [self._popup.listRegionLine[-2].get_xdata()[0],
                                         self._popup.listRegionLine[-2].get_ydata()[0]]
            except IndexError:
                self.logger.debug('on_singleclick_sumRegion_right - IndexError', exc_info=True)
                return
            if lineNb == 3:
                for i in range(2):
                    self._popup.listRegionLine[-1].remove()
                    self._popup.listRegionLine.pop(-1)
            elif lineNb > 3:
                tempLine = self._popup.listRegionLine[-3]
                self._popup.listRegionLine[-2].remove()
                self._popup.listRegionLine.pop(-2)
                self._popup.listRegionLine[-1].set_xdata([tempLine.get_xdata()[1], self._popup.listRegionLine[0].get_xdata()[0]])
                self._popup.listRegionLine[-1].set_ydata([tempLine.get_ydata()[1], self._popup.listRegionLine[0].get_ydata()[0]])

            lineNb = len(self._popup.listRegionLine)
            lineText = ""
            for nbLine in range(lineNb):
                if nbLine == 0:
                    lineText = lineText + (f"{nbLine}: X= {self._popup.listRegionLine[nbLine].get_xdata()[0]:.5f}   Y= {self._popup.listRegionLine[nbLine].get_ydata()[0]:.5f}")
                else:
                    lineText = lineText + (f"\n{nbLine}: X= {self._popup.listRegionLine[nbLine].get_xdata()[0]:.5f}   Y= {self._popup.listRegionLine[nbLine].get_ydata()[0]:.5f}")
            if lineNb == 1:
                lineText = lineText + (f"\n{lineNb}: X= {self._popup.listRegionLine[0].get_xdata()[1]:.5f}   Y= {self._popup.listRegionLine[0].get_ydata()[1]:.5f}")
            self.regionReadoutChanged.emit(lineText)

        self.canvasDrawRequested.emit()

    # ------------------------------------------------------------------
    # Sum region dictionary management
    # ------------------------------------------------------------------

    def setSumRegion(self, index, line, name):
        self.logger.info('setSumRegion')
        if index is None:
            self.logger.debug('setSumRegion - index: %s', index)
            return
        labelSplit = line.get_label().split("_-_")
        if len(labelSplit) == 3 and labelSplit[0] == "sumReg":
            if name not in self.sumRegionDict:
                self.sumRegionDict[name] = [line]
            else:
                self.sumRegionDict[name].append(line)


    def getSumRegion(self, index, name):
        self.logger.info('getSumRegion - index: %s', index)
        if index is None:
            return
        result = self.sumRegionDict.get(name)
        self.logger.info('getSumRegion - spectrumName, result: %s, %s', name, result)
        return result


    def deleteSumRegionDict(self, labelSumRegion, figure_axes):
        self.logger.info('deleteSumRegionDict - labelSumRegion: %s', labelSumRegion)
        labelSplit = labelSumRegion.split("_-_")
        if len(labelSplit) == 3 and labelSplit[0] == "sumReg":
            for spectrumName, regionList in self.sumRegionDict.items():
                for line in regionList:
                    if line.get_label() == labelSumRegion:
                        for ax in figure_axes:
                            toRemove = [child for child in ax.get_children()
                                        if isinstance(child, matplotlib.lines.Line2D)
                                        and child.get_label() == labelSumRegion]
                            for linetoRemove in toRemove:
                                linetoRemove.remove()
                        regionList.remove(line)


    def refreshSpectrumSumRegionDict(self):
        self.logger.info('refreshSpectrumSumRegionDict')
        histoList = list(self._get_histo_names())
        keyToDelete = [name for name in self.sumRegionDict if name not in histoList]
        for key in keyToDelete:
            del self.sumRegionDict[key]

    # ------------------------------------------------------------------
    # Sum region CRUD
    # ------------------------------------------------------------------

    def saveSumRegion(self, index, name, region_name):
        self.logger.info('saveSumRegion - index: %s', index)
        spectrum = self._get_spectrum_info("spectrum", index=index)
        ax = spectrum.axes
        dim = self._spectra.get(name, "dim")
        if ax is None:
            self.logger.debug('saveSumRegion - ax is None')
            return
        regionName = region_name
        if regionName is None or regionName == "None":
            self.logger.debug('saveSumRegion - regionName is None or "None"')
            return
        self.logger.debug('saveSumRegion - spectrumName, dim, regionName: %s, %s, %s',
                          name, dim, regionName)

        if dim == 1:
            ylim = ax.get_ybound()
            if len(self._popup.listRegionLine) != 2:
                return
            for iLine in range(2):
                xlim = self._popup.listRegionLine[iLine].get_xdata()
                lineLabel = "sumReg_-_" + regionName + "_-_" + str(iLine)
                line = mlines.Line2D([xlim[0], xlim[0]], [ylim[0], ylim[1]],
                                     picker=True, color='blue', label=lineLabel)
                line.set_pickradius(5)
                self.setSumRegion(index, line, name)
                ax.add_line(line)

        elif dim == 2:
            xPoints = []
            yPoints = []
            for iLine, line in enumerate(self._popup.listRegionLine):
                if iLine == 0:
                    for iPoint in range(2):
                        xPoints.append(self._popup.listRegionLine[iLine].get_xdata()[iPoint])
                        yPoints.append(self._popup.listRegionLine[iLine].get_ydata()[iPoint])
                else:
                    xPoints.append(self._popup.listRegionLine[iLine].get_xdata()[1])
                    yPoints.append(self._popup.listRegionLine[iLine].get_ydata()[1])
            lineLabel = "sumReg_-_" + regionName + "_-_"
            line = mlines.Line2D(xPoints, yPoints, picker=True, color='blue', label=lineLabel)
            line.set_pickradius(5)
            self.setSumRegion(index, line, name)
            ax.add_line(line)


    def createSumRegion(self, index, name, ax):
        self.logger.info('createSumRegion CallBack')
        self._skip_auto.set()

        if index is None:
            return QMessageBox.about(self._parent_widget, "Warning!", "Please add/select a spectrum")

        dim = self._spectra.get(name, "dim")
        if ax is None:
            self.logger.debug('createSumRegion - ax is None')
            return

        self.refreshSpectrumSumRegionDict()

        regionNames = []
        sumRegionLabels = [child.get_label() for child in ax.get_children()
                           if isinstance(child, matplotlib.lines.Line2D)
                           and "_-_" in child.get_label()]
        for label in sumRegionLabels:
            parts = label.split("_-_")
            if dim == 1:
                if parts[0] == "sumReg" and parts[2] == '0':
                    regionNames.append(parts[1])
            elif dim == 2:
                if parts[0] == "sumReg":
                    regionNames.append(parts[1])

        self._saved_region_names = list(regionNames)
        self._active_sum_index   = index
        self._creating_sum_region = True
        self.sumRegionStarted.emit(index)
        # MainWindow populates the name combo (editable, completer, "None"
        # default) from these names and shows the popup.
        self.sumRegionCreatePrepared.emit(regionNames)


    def okSumRegion(self, sum_region_name):
        self.logger.info('okSumRegion')
        sumRegionName = sum_region_name
        spec_index = self._active_sum_index
        name = self._name_from_index(spec_index)

        if sumRegionName in self._saved_region_names:
            self.logger.debug('okSumRegion - sumRegionName: %s already exist', sumRegionName)
            msgBox = QMessageBox(self._parent_widget)
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setWindowFlag(Qt.WindowStaysOnTopHint, True)
            msgBox.setText("Summing region name already exists.")
            msgBox.setInformativeText(
                'Do you want to overwrite "' + sumRegionName + '" summing region definition?')
            msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
            msgBox.setDefaultButton(QMessageBox.Cancel)
            ret = msgBox.exec()
            if ret == QMessageBox.Yes:
                ax = self._get_spectrum_info("axis", index=spec_index)
                self.deleteSumRegion(spec_index, name, ax, sumRegionName)
                self.sumRegionSelectionChanged.emit(sumRegionName)
            elif ret == QMessageBox.Cancel:
                return

        elif "_-_" in sumRegionName:
            self.logger.debug('okSumRegion - sumRegionName has _-_ in name')
            msgBox = QMessageBox(self._parent_widget)
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setWindowFlag(Qt.WindowStaysOnTopHint, True)
            msgBox.setText('Region name must not include "_-_"')
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.setDefaultButton(QMessageBox.Ok)
            ret = msgBox.exec()
            if ret == QMessageBox.Ok:
                return

        self.saveSumRegion(spec_index, name, sumRegionName)
        self.cancelSumRegion()


    def cancelSumRegion(self, doClose=True):
        self.logger.info('cancelSumRegion')
        self._creating_sum_region = False
        self.sumRegionEnded.emit()
        if doClose:
            self.sumRegionPopupCloseRequested.emit()
        self.updatePlotRequested.emit()


    def cleanPopupExit(self, doClose=True, popup_visible=False):
        self._check_and_cancel_gate(doClose)
        if self._creating_sum_region and not popup_visible:
            self.cancelSumRegion(doClose)


    def deleteSumRegion(self, index, name, ax, sum_region_name):
        self.logger.info('deleteSumRegion')
        dim = self._spectra.get(name, "dim")
        if ax is None:
            return
        sumRegionName = sum_region_name

        if sumRegionName not in self._saved_region_names:
            self.logger.debug('deleteSumRegion - sumRegionName: %s doesnt exist', sumRegionName)
            msgBox = QMessageBox(self._parent_widget)
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setWindowFlag(Qt.WindowStaysOnTopHint, True)
            msgBox.setText("Cannot delete summing region " + sumRegionName + " not found")
            msgBox.setStandardButtons(QMessageBox.Ok)
            ret = msgBox.exec()
            if ret == QMessageBox.Ok:
                return
        else:
            if dim == 1:
                for iLine in range(2):
                    label = "sumReg_-_" + sumRegionName + "_-_" + str(iLine)
                    self.deleteSumRegionDict(label, ax.figure.axes)
            elif dim == 2:
                label = "sumReg_-_" + sumRegionName + "_-_"
                self.deleteSumRegionDict(label, ax.figure.axes)

        self.sumRegionSelectionChanged.emit("None")
        self.figureTightLayoutRequested.emit()
        self.canvasDrawRequested.emit()

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def integrate(self, index, name, ax):
        self.logger.info('integrate')
        if ax is None or index is None:
            self.logger.debug('integrate - ax is None or index is None')
            return QMessageBox.about(self._parent_widget, "Warning!", "Please add/select one spectrum")

        self.gateSignalsDisconnectRequested.emit()

        resultsCombined = {}
        results = {}

        sumRegionLines = self.getSumRegion(index, name)
        if sumRegionLines is None or len(sumRegionLines) == 0:
            self.logger.debug('integrate - sumRegionLines is None or empty')
        else:
            results = self.integrateGateLocal(index, name, sumRegionLines)

        gateIdentifier = "gate_-_"
        gateLines = [child for child in ax.get_children()
                     if type(child) == matplotlib.lines.Line2D
                     and gateIdentifier in child.get_label()]
        resultsGate = self.integrateGateLocal(index, name, gateLines)

        if results is not None and len(results) > 0 and resultsGate is not None and len(resultsGate) > 0:
            resultsCombined = results
            for specName, listGateResults in resultsGate.items():
                if specName in resultsCombined:
                    for listItem in listGateResults:
                        resultsCombined[specName].append(listItem)
                else:
                    resultsCombined[specName] = listGateResults
        elif results is not None and len(results) > 0:
            resultsCombined = results
        else:
            resultsCombined = resultsGate

        if resultsCombined is None or len(resultsCombined) == 0:
            self.integrationResultsReady.emit([])   # MainWindow renders "Nothing to integrate"
            return

        self.integrationResultsReady.emit(self._format_integration_rows(resultsCombined))


    def okIntegrate(self):
        self.logger.info('okIntegrate')
        self.gateSignalsDisconnectRequested.emit()
        self.integratePopupCloseRequested.emit()


    def _format_integration_rows(self, results):
        """Turn the combined integration results into table rows (each a list of
        7 strings). Repeated spectrum names blank column 0, matching the old
        table-level dedup. Pure: no widget access."""
        self.logger.info('_format_integration_rows')
        rows = []
        seen = set()
        for spectrumName, resultList in results.items():
            for result in resultList:
                if len(result) == 0 or "centroid" not in result.keys():
                    continue
                result = self.setPrecisionIntegrationResult(result)
                if isinstance(result["centroid"], list):
                    row = [spectrumName, result["regionName"], str(result["counts"]),
                           str(result["centroid"][0]), str(result["centroid"][1]),
                           str(result["fwhm"][0]), str(result["fwhm"][1])]
                else:
                    row = [spectrumName, result["regionName"], str(result["counts"]),
                           str(result["centroid"]), '', str(result["fwhm"]), '']
                if row[0] in seen:
                    row = [''] + row[1:]
                else:
                    seen.add(row[0])
                rows.append(row)
        return rows


    def setPrecisionIntegrationResult(self, toRound):
        self.logger.info('setPrecisionIntegrationResult - toRound: %s', toRound)
        order = 3
        sciFormat = "{:." + str(order) + "E}"
        if type(['dumList']) == type(toRound["fwhm"]):
            for i, val in enumerate(toRound["fwhm"]):
                if val is None:
                    continue
                toRound["centroid"][i] = sciFormat.format(toRound["centroid"][i])
                toRound["fwhm"][i] = sciFormat.format(toRound["fwhm"][i])
        else:
            if toRound["centroid"] is not None:
                toRound["centroid"] = sciFormat.format(toRound["centroid"])
            if toRound["fwhm"] is not None:
                toRound["fwhm"] = sciFormat.format(toRound["fwhm"])
        toRound["counts"] = int(toRound["counts"])
        return toRound


    def integrateGateLocal(self, index, name, gateLines):
        self.logger.info('integrateGateLocal - index: %s', index)
        if gateLines is None or len(gateLines) == 0:
            self.logger.debug('integrateGateLocal - gateLines is None or empty')
            return

        rest = self._get_rest()
        if rest is None:
            self.logger.warning('integrateGateLocal - REST client not available')
            return

        dim = self._spectra.get(name, "dim")
        step = 2 if dim == 1 else 1
        resultsList = []

        for iLine in range(0, len(gateLines), step):
            try:
                if dim == 1:
                    gateName = gateLines[iLine].get_label().split("_-_")[1]
                    boundaries = sorted([gateLines[iLine].get_xdata()[0],
                                         gateLines[iLine + 1].get_xdata()[0]])
                    results = rest.integrate1D(name, boundaries[0], boundaries[1])
                elif dim == 2:
                    gateName = gateLines[iLine].get_label().split("_-_")[1]
                    points = gateLines[iLine].get_xydata()
                    if points[0][0] != points[-1][0] or points[0][1] != points[-1][1]:
                        continue
                    results = rest.integrate2D(name, points)
                else:
                    continue

                defaultResult = {'centroid': None, 'fwhm': None, 'counts': 0}
                if dim == 2:
                    defaultResult = {'centroid': [None, None], 'fwhm': [None, None], 'counts': 0}
                if type(results) == type('dumString'):
                    results = defaultResult
                results['regionName'] = gateName
                resultsList.append(results)
            except Exception:
                self.logger.debug('integrateGateLocal - exception', exc_info=True)
                continue

        return {name: resultsList}
