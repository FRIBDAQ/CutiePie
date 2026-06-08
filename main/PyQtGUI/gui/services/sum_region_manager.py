import logging

import matplotlib
import matplotlib.lines as mlines

from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QCompleter, QMessageBox, QTableWidgetItem,
)
from PyQt5.QtCore import Qt


class SumRegionManager:
    """Owns sum-region CRUD, integration calculations, and result display."""

    def __init__(self, window, sum_popup, logger=None):
        self._w      = window       # bridge: rest, wTab, currentPlot, gatePopup, integratePopup, etc.
        self._popup  = sum_popup    # MenuSumRegion widget
        self.logger  = logger or logging.getLogger(__name__)

        self.sumRegionDict = {}     # {spectrumName: [Line2D, ...]}

    # ------------------------------------------------------------------
    # Sum region mouse handlers
    # ------------------------------------------------------------------

    def on_singleclick_sumRegion(self, event, index):
        self.logger.info('on_singleclick_sumRegion - index: %s', index)
        dim = self._w.getSpectrumInfoREST("dim", index=index)
        if dim == 1:
            l = self._w.addLine(float(event.xdata), 0, index)
            self._popup.listRegionLine.append(l)
            if len(self._popup.listRegionLine) > 2:
                self._w.removePrevLine()

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

            l = self._w.addLine(float(event.xdata), float(event.ydata), index)
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
                l = self._w.addLine(self._popup.listRegionLine[0].get_xdata()[0],
                                    self._popup.listRegionLine[0].get_ydata()[0], index, label)
                if l is not None:
                    self._popup.listRegionLine.append(l)

        self._w.currentPlot.canvas.draw()


    def on_singleclick_sumRegion_right(self, index):
        self.logger.info('on_singleclick_sumRegion_right - index: %s', index)
        dim = self._w.getSpectrumInfoREST("dim", index=index)

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

        self._w.currentPlot.canvas.draw()

    # ------------------------------------------------------------------
    # Sum region dictionary management
    # ------------------------------------------------------------------

    def setSumRegion(self, index, line):
        self.logger.info('setSumRegion')
        if index is None:
            self.logger.debug('setSumRegion - index: %s', index)
            return
        labelSplit = line.get_label().split("_-_")
        if len(labelSplit) == 3 and labelSplit[0] == "sumReg":
            spectrumName = self._w.nameFromIndex(index)
            if spectrumName not in self.sumRegionDict:
                self.sumRegionDict[spectrumName] = [line]
            else:
                self.sumRegionDict[spectrumName].append(line)


    def getSumRegion(self, index):
        self.logger.info('getSumRegion - index: %s', index)
        result = None
        if index is None:
            return
        spectrumName = self._w.nameFromIndex(index)
        if spectrumName not in self.sumRegionDict:
            pass
        else:
            result = self.sumRegionDict[spectrumName]
        self.logger.info('getSumRegion - spectrumName, result: %s, %s', spectrumName, result)
        return result


    def deleteSumRegionDict(self, labelSumRegion):
        self.logger.info('deleteSumRegionDict - labelSumRegion: %s', labelSumRegion)
        labelSplit = labelSumRegion.split("_-_")
        if len(labelSplit) == 3 and labelSplit[0] == "sumReg":
            for spectrumName, regionList in self.sumRegionDict.items():
                for line in regionList:
                    if line.get_label() == labelSumRegion:
                        for ax in self._w.currentPlot.figure.axes:
                            toRemove = [child for child in ax.get_children()
                                        if isinstance(child, matplotlib.lines.Line2D)
                                        and child.get_label() == labelSumRegion]
                            for linetoRemove in toRemove:
                                linetoRemove.remove()
                        regionList.remove(line)


    def refreshSpectrumSumRegionDict(self):
        self.logger.info('refreshSpectrumSumRegionDict')
        histoList = [self._w.wConf.histo_list.itemText(i)
                     for i in range(self._w.wConf.histo_list.count())]
        keyToDelete = []
        for spectrumNameSumReg in self.sumRegionDict.keys():
            if spectrumNameSumReg not in histoList:
                keyToDelete.append(spectrumNameSumReg)
        if len(keyToDelete) > 0:
            for key in keyToDelete:
                del self.sumRegionDict[key]

    # ------------------------------------------------------------------
    # Sum region CRUD
    # ------------------------------------------------------------------

    def saveSumRegion(self, index):
        self.logger.info('saveSumRegion - index: %s', index)
        spectrumName = self._w.nameFromIndex(index)
        spectrum = self._w.getSpectrumInfo("spectrum", index=index)
        ax = spectrum.axes
        dim = self._w.getSpectrumInfoREST("dim", name=spectrumName)
        if ax is None:
            self.logger.debug('saveSumRegion - ax is None')
            return
        regionName = self._popup.sumRegionNameList.currentText()
        if regionName is None or regionName == "None":
            self.logger.debug('saveSumRegion - regionName is None or regionName == "None"')
            return
        self.logger.debug('saveSumRegion - spectrumName, dim, regionName: %s, %s, %s',
                          spectrumName, dim, regionName)

        if dim == 1:
            ylim = ax.get_ybound()
            if len(self._popup.listRegionLine) != 2:
                return
            for iLine in range(2):
                xlim = self._popup.listRegionLine[iLine].get_xdata()
                lineLabel = "sumReg_-_" + regionName + "_-_" + str(iLine)
                line = mlines.Line2D([xlim[0], xlim[0]], [ylim[0], ylim[1]],
                                     picker=5, color='blue', label=lineLabel)
                self.setSumRegion(index, line)
                ax.add_artist(line)

        elif dim == 2:
            lineLabel = "sumReg_-_" + regionName + "_-_"
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
            line = mlines.Line2D(xPoints, yPoints, picker=5, color='blue', label=lineLabel)
            self.setSumRegion(index, line)
            ax.add_artist(line)


    def createSumRegion(self):
        self.logger.info('createSumRegion CallBack')
        self._w.skipAutoUpdateThread.set()
        self._popup.sumRegionNameList.setEditable(True)
        self._popup.sumRegionNameList.setInsertPolicy(QComboBox.NoInsert)

        if self._w.currentPlot.selected_plot_index is None:
            return QMessageBox.about(self._w, "Warning!", "Please add/select a spectrum")
        else:
            self._popup.clearInfo()

            spectrumName = self._w.nameFromIndex(self._w.currentPlot.selected_plot_index)
            dim = self._w.getSpectrumInfoREST("dim", name=spectrumName)
            ax = self._w.getSpectrumInfo("axis", index=self._w.currentPlot.selected_plot_index)
            if ax is None:
                self.logger.debug('createSumRegion - ax is None')
                return

            self.refreshSpectrumSumRegionDict()

            sumRegionLabels = [child.get_label() for child in ax.get_children()
                               if isinstance(child, matplotlib.lines.Line2D)
                               and "_-_" in child.get_label()]
            for label in sumRegionLabels:
                if dim == 1:
                    label = label.split("_-_")
                    if label[0] == "sumReg" and label[2] == '0':
                        self._popup.sumRegionNameList.addItem(label[1])
                elif dim == 2:
                    label = label.split("_-_")
                    if label[0] == "sumReg":
                        self._popup.sumRegionNameList.addItem(label[1])
            self._popup.sumRegionNameList.setCurrentText("None")
            self._popup.sumRegionNameList.completer().setCompletionMode(QCompleter.PopupCompletion)
            self._popup.sumRegionNameList.completer().setFilterMode(QtCore.Qt.MatchContains)
            self._popup.sumRegionNameListSaved = [
                self._popup.sumRegionNameList.itemText(i)
                for i in range(self._popup.sumRegionNameList.count())
                if self._popup.sumRegionNameList.itemText(i) != "None"
            ]

            self._w.currentPlot.toCreateSumRegion = True
            self._popup.sumRegionSpectrumIndex = self._w.currentPlot.selected_plot_index
        self._popup.show()


    def okSumRegion(self):
        self.logger.info('okSumRegion')
        sumRegionName = self._popup.sumRegionNameList.currentText()
        spec_index = self._popup.sumRegionSpectrumIndex

        if sumRegionName in self._popup.sumRegionNameListSaved:
            self.logger.debug('okSumRegion - sumRegionName: %s already exist', sumRegionName)
            msgBox = QMessageBox(self._w)
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setWindowFlag(Qt.WindowStaysOnTopHint, True)
            msgBox.setText("Summing region name already exists.")
            msgBox.setInformativeText(
                'Do you want to overwrite "' + sumRegionName + '" summing region definition?')
            msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
            msgBox.setDefaultButton(QMessageBox.Cancel)
            ret = msgBox.exec()
            if ret == QMessageBox.Yes:
                self.deleteSumRegion()
                self._popup.sumRegionNameList.setCurrentText(sumRegionName)
            elif ret == QMessageBox.Cancel:
                return

        elif "_-_" in sumRegionName:
            self.logger.debug('okSumRegion - sumRegionName has _-_ in its name')
            msgBox = QMessageBox(self._w)
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setWindowFlag(Qt.WindowStaysOnTopHint, True)
            msgBox.setText('Region name must not include "_-_"')
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.setDefaultButton(QMessageBox.Ok)
            ret = msgBox.exec()
            if ret == QMessageBox.Ok:
                return

        self.saveSumRegion(spec_index)
        self.cancelSumRegion()


    def cancelSumRegion(self, doClose=True):
        self.logger.info('cancelSumRegion')
        self._w.currentPlot.toCreateSumRegion = False
        if doClose:
            self._popup.close()
        self._w.updatePlot()


    def cleanPopupExit(self, doClose=True):
        if (self._w.currentPlot.toCreateGate or self._w.currentPlot.toEditGate) \
                and not self._w.gatePopup.isVisible():
            self._w.cancelGate(doClose)
        if self._w.currentPlot.toCreateSumRegion and not self._popup.isVisible():
            self.cancelSumRegion(doClose)


    def deleteSumRegion(self):
        self.logger.info('deleteSumRegion')
        spectrumName = self._w.nameFromIndex(self._w.currentPlot.selected_plot_index)
        dim = self._w.getSpectrumInfoREST("dim", name=spectrumName)
        ax = self._w.getSpectrumInfo("axis", index=self._w.currentPlot.selected_plot_index)
        if ax is None:
            return
        sumRegionName = self._popup.sumRegionNameList.currentText()

        if sumRegionName not in self._popup.sumRegionNameListSaved:
            self.logger.debug('deleteSumRegion - sumRegionName: %s doesnt exists', sumRegionName)
            msgBox = QMessageBox(self._w)
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
                    self.deleteSumRegionDict(label)
            elif dim == 2:
                label = "sumReg_-_" + sumRegionName + "_-_"
                self.deleteSumRegionDict(label)

        self._popup.sumRegionNameList.setCurrentText("None")
        self._w.currentPlot.figure.tight_layout()
        self._w.currentPlot.canvas.draw()

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def integrate(self):
        self.logger.info('integrate')
        ax = self._w.getSpectrumInfo("axis", index=self._w.currentPlot.selected_plot_index)
        if ax is None or self._w.currentPlot.selected_plot_index is None:
            self.logger.debug('integrate - ax is None or selected_plot_index is None')
            return QMessageBox.about(self._w, "Warning!", "Please add/select one spectrum")
        else:
            index = self._w.currentPlot.selected_plot_index

            self._w.integratePopup.clearInfo()
            self._w.disconnectGateSignals()

            colHeader = ['Spectrum', 'Region', 'Counts', 'Centroid X', 'Centroid Y', 'FWHM X', 'FWHM Y']
            for col, header in enumerate(colHeader):
                headerItem = QTableWidgetItem(header)
                font = self._w.integratePopup.resultsText.font()
                font.setBold(True)
                headerItem.setFont(font)
                self._w.integratePopup.resultsText.setHorizontalHeaderItem(col, headerItem)

            resultsCombined = {}
            results = {}

            sumRegionLines = self.getSumRegion(index)
            if sumRegionLines is None or len(sumRegionLines) == 0:
                self.logger.debug('integrate - sumRegionLines is None or empty')
                pass
            else:
                results = self.integrateGateLocal(index, sumRegionLines)

            gateIdentifier = "gate_-_"
            gateLines = [child for child in ax.get_children()
                         if type(child) == matplotlib.lines.Line2D
                         and gateIdentifier in child.get_label()]
            resultsGate = self.integrateGateLocal(index, gateLines)

            if results is not None and len(results) > 0 and resultsGate is not None and len(resultsGate) > 0:
                resultsCombined = results
                for specName, listGateResults in resultsGate.items():
                    if specName in resultsCombined.keys():
                        for listItem in listGateResults:
                            resultsCombined[specName].append(listItem)
                    else:
                        resultsCombined[specName] = listGateResults
            elif results is not None and len(results) > 0:
                resultsCombined = results
            else:
                resultsCombined = resultsGate

            if resultsCombined is None or len(resultsCombined) == 0:
                self._w.integratePopup.resultsText.insertRow(0)
                newItem = QTableWidgetItem("Nothing to integrate")
                self._w.integratePopup.resultsText.setItem(0, 0, newItem)
                self._w.integratePopup.show()
                return
            else:
                self.formatResultsIntegrate(resultsCombined)
                # sidTableIntegrateCopy stored on _w so gate_manager.disconnectGateSignals can find it
                self._w.sidTableIntegrateCopy = self._w.integratePopup.resultsText.itemSelectionChanged.connect(
                    self.copySelectionIntegrateTable)
                self._w.integratePopup.show()


    def okIntegrate(self):
        self.logger.info('okIntegrate')
        self._w.disconnectGateSignals()
        self._w.integratePopup.close()


    def copySelectionIntegrateTable(self):
        self.logger.info('copySelectionIntegrateTable')
        resultTable = self._w.integratePopup.resultsText
        selectedItems = resultTable.selectedItems()
        if not selectedItems:
            return
        allValues = []
        for irow in range(resultTable.rowCount()):
            rowValues = []
            for icol in range(resultTable.columnCount()):
                item = resultTable.item(irow, icol)
                if item:
                    rowValues.append(item.text())
                else:
                    rowValues.append("")
            if len(rowValues) > 0:
                allValues.append("\t".join(rowValues))
        formattedText = "\n".join(allValues)
        QApplication.clipboard().setText(formattedText)


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

                self._w.integratePopup.resultsText.insertRow(irow)
                for icol, cell in enumerate(row):
                    if icol == 0 and self._w.integratePopup.resultsText.findItems(
                            cell, QtCore.Qt.MatchFixedString):
                        continue
                    newItem = QTableWidgetItem(cell)
                    self._w.integratePopup.resultsText.setItem(irow, icol, newItem)
                irow += 1

        self._w.integratePopup.show()


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


    def integrateGateLocal(self, index, gateLines):
        self.logger.info('integrateGateLocal - index: %s', index)
        resultsList = []
        resultsDict = {}
        index = self._w.currentPlot.selected_plot_index

        lineList = gateLines
        if lineList is None or len(lineList) == 0:
            self.logger.debug('integrateGateLocal - lineList is None or empty')
            return

        dim = self._w.getSpectrumInfoREST("dim", index=index)
        spectrumName = self._w.nameFromIndex(index)

        step = 1
        if dim == 1:
            step = 2

        for iLine in range(0, len(lineList), step):
            boundaries = []
            if dim == 1:
                gateName = lineList[iLine].get_label().split("_-_")[1]
                boundaries = [lineList[iLine].get_xdata()[0], lineList[iLine + 1].get_xdata()[0]]
                if boundaries[0] > boundaries[1]:
                    boundaries.sort()
                results = self._w.rest.integrate1D(spectrumName, boundaries[0], boundaries[1])
            elif dim == 2:
                gateName = lineList[iLine].get_label().split("_-_")[1]
                points = lineList[iLine].get_xydata()
                if points[0][0] != points[-1][0] or points[0][1] != points[-1][1]:
                    continue
                results = self._w.rest.integrate2D(spectrumName, points)

            try:
                defaultResult = {'centroid': None, 'fwhm': None, 'counts': 0}
                if dim == 2:
                    defaultResult = {'centroid': [None, None], 'fwhm': [None, None], 'counts': 0}
                if type(results) == type('dumString'):
                    results = defaultResult
                results['regionName'] = gateName
                resultsList.append(results)
            except:
                self.logger.debug('integrateGateLocal - exception', exc_info=True)
                continue
        resultsDict[spectrumName] = resultsList
        return resultsDict
