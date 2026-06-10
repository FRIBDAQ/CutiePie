import logging

import matplotlib
import matplotlib.lines as mlines

from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QCompleter, QMessageBox, QTableWidgetItem,
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal


class SumRegionManager(QObject):
    """Owns sum-region CRUD, integration calculations, and result display."""

    canvasDrawRequested            = pyqtSignal()
    figureTightLayoutRequested     = pyqtSignal()
    updatePlotRequested            = pyqtSignal()
    sumRegionStarted               = pyqtSignal(int)   # sets currentPlot.toCreateSumRegion = True
    sumRegionEnded                 = pyqtSignal()      # sets currentPlot.toCreateSumRegion = False
    gateSignalsDisconnectRequested = pyqtSignal()
    sidTableConnectionUpdated      = pyqtSignal(object)

    def __init__(self, spectra, name_from_index, get_spectrum_info,
                 histo_list, integrate_popup, skip_auto,
                 add_line, remove_prev_line, get_rest,
                 check_and_cancel_gate,
                 sum_popup, parent_widget=None, logger=None):
        super().__init__()
        self._spectra               = spectra
        self._name_from_index       = name_from_index      # (index) -> str
        self._get_spectrum_info     = get_spectrum_info    # (key, index=) -> value
        self._histo_list            = histo_list           # QComboBox
        self._integrate_popup       = integrate_popup      # OutputIntegratePopup widget
        self._skip_auto             = skip_auto            # threading.Event
        self._add_line              = add_line             # (x, y, index[, label]) -> Line2D|None
        self._remove_prev_line      = remove_prev_line     # () -> None
        self._get_rest              = get_rest             # () -> PyREST|None
        self._check_and_cancel_gate = check_and_cancel_gate  # (doClose) -> None
        self._popup                 = sum_popup
        self._parent_widget         = parent_widget
        self.logger                 = logger or logging.getLogger(__name__)
        self.sumRegionDict          = {}
        self._creating_sum_region   = False

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
            self._popup.regionPoint.clear()
            self._popup.regionPoint.insertPlainText(lineText)

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
            self._popup.regionPoint.clear()
            self._popup.regionPoint.insertPlainText(lineText)

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
            self._popup.regionPoint.clear()
            self._popup.regionPoint.insertPlainText(lineText)

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
        histoList = [self._histo_list.itemText(i)
                     for i in range(self._histo_list.count())]
        keyToDelete = [name for name in self.sumRegionDict if name not in histoList]
        for key in keyToDelete:
            del self.sumRegionDict[key]

    # ------------------------------------------------------------------
    # Sum region CRUD
    # ------------------------------------------------------------------

    def saveSumRegion(self, index, name):
        self.logger.info('saveSumRegion - index: %s', index)
        spectrum = self._get_spectrum_info("spectrum", index=index)
        ax = spectrum.axes
        dim = self._spectra.get(name, "dim")
        if ax is None:
            self.logger.debug('saveSumRegion - ax is None')
            return
        regionName = self._popup.sumRegionNameList.currentText()
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
        self._popup.sumRegionNameList.setEditable(True)
        self._popup.sumRegionNameList.setInsertPolicy(QComboBox.NoInsert)

        if index is None:
            return QMessageBox.about(self._parent_widget, "Warning!", "Please add/select a spectrum")

        self._popup.clearInfo()

        dim = self._spectra.get(name, "dim")
        if ax is None:
            self.logger.debug('createSumRegion - ax is None')
            return

        self.refreshSpectrumSumRegionDict()

        sumRegionLabels = [child.get_label() for child in ax.get_children()
                           if isinstance(child, matplotlib.lines.Line2D)
                           and "_-_" in child.get_label()]
        for label in sumRegionLabels:
            if dim == 1:
                parts = label.split("_-_")
                if parts[0] == "sumReg" and parts[2] == '0':
                    self._popup.sumRegionNameList.addItem(parts[1])
            elif dim == 2:
                parts = label.split("_-_")
                if parts[0] == "sumReg":
                    self._popup.sumRegionNameList.addItem(parts[1])
        self._popup.sumRegionNameList.setCurrentText("None")
        self._popup.sumRegionNameList.completer().setCompletionMode(QCompleter.PopupCompletion)
        self._popup.sumRegionNameList.completer().setFilterMode(QtCore.Qt.MatchContains)
        self._popup.sumRegionNameListSaved = [
            self._popup.sumRegionNameList.itemText(i)
            for i in range(self._popup.sumRegionNameList.count())
            if self._popup.sumRegionNameList.itemText(i) != "None"
        ]

        self._creating_sum_region = True
        self.sumRegionStarted.emit(index)
        self._popup.sumRegionSpectrumIndex = index
        self._popup.show()


    def okSumRegion(self):
        self.logger.info('okSumRegion')
        sumRegionName = self._popup.sumRegionNameList.currentText()
        spec_index = self._popup.sumRegionSpectrumIndex
        name = self._name_from_index(spec_index)

        if sumRegionName in self._popup.sumRegionNameListSaved:
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
                self.deleteSumRegion(spec_index, name, ax)
                self._popup.sumRegionNameList.setCurrentText(sumRegionName)
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

        self.saveSumRegion(spec_index, name)
        self.cancelSumRegion()


    def cancelSumRegion(self, doClose=True):
        self.logger.info('cancelSumRegion')
        self._creating_sum_region = False
        self.sumRegionEnded.emit()
        if doClose:
            self._popup.close()
        self.updatePlotRequested.emit()


    def cleanPopupExit(self, doClose=True):
        self._check_and_cancel_gate(doClose)
        if self._creating_sum_region and not self._popup.isVisible():
            self.cancelSumRegion(doClose)


    def deleteSumRegion(self, index, name, ax):
        self.logger.info('deleteSumRegion')
        dim = self._spectra.get(name, "dim")
        if ax is None:
            return
        sumRegionName = self._popup.sumRegionNameList.currentText()

        if sumRegionName not in self._popup.sumRegionNameListSaved:
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

        self._popup.sumRegionNameList.setCurrentText("None")
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

        self._integrate_popup.clearInfo()
        self.gateSignalsDisconnectRequested.emit()

        colHeader = ['Spectrum', 'Region', 'Counts', 'Centroid X', 'Centroid Y', 'FWHM X', 'FWHM Y']
        for col, header in enumerate(colHeader):
            headerItem = QTableWidgetItem(header)
            font = self._integrate_popup.resultsText.font()
            font.setBold(True)
            headerItem.setFont(font)
            self._integrate_popup.resultsText.setHorizontalHeaderItem(col, headerItem)

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
            self._integrate_popup.resultsText.insertRow(0)
            newItem = QTableWidgetItem("Nothing to integrate")
            self._integrate_popup.resultsText.setItem(0, 0, newItem)
            self._integrate_popup.show()
            return

        self.formatResultsIntegrate(resultsCombined)
        conn = self._integrate_popup.resultsText.itemSelectionChanged.connect(
            self.copySelectionIntegrateTable)
        self.sidTableConnectionUpdated.emit(conn)
        self._integrate_popup.show()


    def okIntegrate(self):
        self.logger.info('okIntegrate')
        self.gateSignalsDisconnectRequested.emit()
        self._integrate_popup.close()


    def copySelectionIntegrateTable(self):
        self.logger.info('copySelectionIntegrateTable')
        resultTable = self._integrate_popup.resultsText
        selectedItems = resultTable.selectedItems()
        if not selectedItems:
            return
        allValues = []
        for irow in range(resultTable.rowCount()):
            rowValues = []
            for icol in range(resultTable.columnCount()):
                item = resultTable.item(irow, icol)
                rowValues.append(item.text() if item else "")
            if rowValues:
                allValues.append("\t".join(rowValues))
        QApplication.clipboard().setText("\n".join(allValues))


    def formatResultsIntegrate(self, results):
        self.logger.info('formatResultsIntegrate')
        dum = []
        irow = 0
        for spectrumName, resultList in results.items():
            for result in resultList:
                if len(result) == 0:
                    continue
                elif "centroid" in result.keys():
                    result = self.setPrecisionIntegrationResult(result)
                    if type(dum) == type(result["centroid"]):
                        row = [spectrumName, result["regionName"], str(result["counts"]),
                               str(result["centroid"][0]), str(result["centroid"][1]),
                               str(result["fwhm"][0]), str(result["fwhm"][1])]
                    else:
                        row = [spectrumName, result["regionName"], str(result["counts"]),
                               str(result["centroid"]), '', str(result["fwhm"]), '']

                self._integrate_popup.resultsText.insertRow(irow)
                for icol, cell in enumerate(row):
                    if icol == 0 and self._integrate_popup.resultsText.findItems(
                            cell, QtCore.Qt.MatchFixedString):
                        continue
                    newItem = QTableWidgetItem(cell)
                    self._integrate_popup.resultsText.setItem(irow, icol, newItem)
                irow += 1

        self._integrate_popup.show()


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
