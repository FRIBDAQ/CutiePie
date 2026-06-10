import logging
import math
import re
import time

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QCompleter, QComboBox, QMessageBox, QShortcut,
)


class GateManager(QObject):
    """Owns gate CRUD, gate drawing, REST push, and mouse interactions."""

    canvasDrawRequested     = pyqtSignal()
    canvasDrawIdleRequested = pyqtSignal()
    updatePlotRequested     = pyqtSignal()
    gateCreationStarted     = pyqtSignal(int)   # sets currentPlot.toCreateGate=True, toEditGate=False
    gateEditingStarted      = pyqtSignal()      # sets toEditGate=True, toCreateGate=False
    gateEnded               = pyqtSignal()      # sets both flags to False

    def __init__(self, spectra, name_from_index, get_spectrum_info,
                 get_is_enlarged, get_geo, get_sum_region, get_current_canvas,
                 integrate_popup, get_integrate_copy,
                 gate_hide_cb, gate_annotation_cb, gate_edit_disable_cb,
                 sum_region_popup, skip_auto, get_rest,
                 gate_popup, parent_widget=None, logger=None):
        super().__init__()
        self._spectra              = spectra
        self._name_from_index      = name_from_index        # (index) -> str
        self._get_spectrum_info    = get_spectrum_info      # (key, index=) -> value
        self._get_is_enlarged      = get_is_enlarged        # () -> bool
        self._get_geo              = get_geo                # () -> {index: name}
        self._get_sum_region       = get_sum_region         # (index, name) -> list|None
        self._get_current_canvas   = get_current_canvas     # () -> canvas
        self._integrate_popup      = integrate_popup
        self._get_integrate_copy   = get_integrate_copy     # () -> connection|None
        self._gate_hide_cb         = gate_hide_cb           # QCheckBox
        self._gate_annotation_cb   = gate_annotation_cb     # QCheckBox
        self._gate_edit_disable_cb = gate_edit_disable_cb   # QCheckBox
        self._sum_region_popup     = sum_region_popup
        self._skip_auto            = skip_auto              # threading.Event
        self._get_rest             = get_rest               # () -> PyREST|None
        self._popup                = gate_popup
        self._parent_widget        = parent_widget
        self.logger                = logger or logging.getLogger(__name__)

        self.gateColor      = {}
        self.gateAnnotation = {}
        self.epsilon        = 5

        self._creating_gate = False
        self._editing_gate  = False

        self._gate_cache     = []
        self._gate_cache_ts  = 0.0
        self._GATE_CACHE_TTL = 30.0

    # ------------------------------------------------------------------
    # Gate drawing
    # ------------------------------------------------------------------

    def drawGate(self, index):
        self.logger.debug('drawGate - index: %s', index)
        spectrumName = self._name_from_index(index)
        if not spectrumName:
            self.logger.debug("drawGate: no name for index %s; skipping", index)
            return
        spectrumType = self._spectra.get(spectrumName, "type")
        dim          = self._spectra.get(spectrumName, "dim")
        parameters   = self._spectra.get(spectrumName, "parameters")
        if not spectrumType or parameters is None:
            self.logger.debug("drawGate: blank canvas/missing metadata for '%s'; skipping", spectrumName)
            return
        if spectrumType == "gd":
            parametersFormat = []
            for item in parameters:
                pars = item.split(' ')
                if len(pars) == 2:
                    parametersFormat.append(pars[0])
                    parametersFormat.append(pars[1])
            parameters = parametersFormat
        self.logger.debug('drawGate - spectrumName, spectrumType, dim, paramters: %s, %s, %s, %s',
                          spectrumName, spectrumType, dim, parameters)
        ax = self._get_spectrum_info("axis", index=index)
        if ax is None:
            self.logger.debug('drawGate - ax is None')
            return
        drawableTypes = {
            "b":  ["s"],
            "1v": ["vs+", "vs*"],
            "1":  ["s"],
            "g1": ["gs"],
            "2":  ["c", "b"],
            "g2": ["gc", "gb"],
            "gd": ["gc", "gb"],
            "m2": ["NotDefinedYet"],
            "s":  ["NotDefinedYet"],
        }
        rest = self._get_rest()
        if rest is None:
            return
        now = time.monotonic()
        if now - self._gate_cache_ts > self._GATE_CACHE_TTL:
            self._gate_cache    = rest.listGate()
            self._gate_cache_ts = now
        gateList = [d for d in self._gate_cache
                    if "type" in d and "parameters" in d
                    and d["type"] in drawableTypes[spectrumType]
                    and d["parameters"] == parameters]

        for gate in gateList:
            if dim == 1:
                xlim = [gate["low"], gate["high"]]
                ylim = ax.get_ybound()
                for iLine in range(2):
                    lineLabel = "gate_-_" + gate["name"] + "_-_" + str(iLine)
                    toRemove = [gl for gl in ax.lines if gl.get_label() == lineLabel]
                    for lr in toRemove:
                        lr.remove()
                    if self._gate_hide_cb.isChecked():
                        continue
                    line = mlines.Line2D([xlim[iLine], xlim[iLine]],
                                        [ylim[0], ylim[1]],
                                        picker=True, color='red', label=lineLabel)
                    line.set_pickradius(5)
                    ax.add_line(line)

            elif dim == 2:
                lineLabel = "gate_-_" + gate["name"] + "_-_"
                toRemove = [gl for gl in ax.lines if lineLabel in gl.get_label()]
                for lr in toRemove:
                    lr.remove()
                if self._gate_hide_cb.isChecked():
                    continue
                if spectrumType not in ["s"]:
                    xPoints = [pd["x"] for pd in gate["points"]]
                    yPoints = [pd["y"] for pd in gate["points"]]
                    if gate["type"] not in ["b", "gb"]:
                        xPoints.append(gate["points"][0]["x"])
                        yPoints.append(gate["points"][0]["y"])
                    line = mlines.Line2D(xPoints, yPoints,
                                        picker=True, color='red', label=lineLabel)
                    line.set_pickradius(5)
                    ax.add_line(line)

            if (self._gate_annotation_cb.isChecked()
                    and not self._gate_hide_cb.isChecked()):
                self.setGateAnnotation(index, True)
            else:
                self.setGateAnnotation(index, False)

        lineListSumReg = self._get_sum_region(index, spectrumName)
        if lineListSumReg is None:
            return
        for sumRegionLine in lineListSumReg:
            if dim == 1 or dim == 2:
                lineLabel = sumRegionLine.get_label()
                toRemove = [ln for ln in ax.lines if ln.get_label() == lineLabel]
                for lr in toRemove:
                    lr.remove()
                xlim = sumRegionLine.get_xdata()
                ylim = sumRegionLine.get_ydata()
                if dim == 1:
                    ylim = ax.get_ylim()
                line = mlines.Line2D(xlim, ylim, picker=True, color='blue', label=lineLabel)
                line.set_pickradius(5)
                ax.add_artist(line)

    # ------------------------------------------------------------------
    # Gate annotation
    # ------------------------------------------------------------------

    def gateAnnotationCallBack(self):
        self.logger.info('gateAnnotationCallBack - isChecked: %s',
                         self._gate_annotation_cb.isChecked())
        doAnnotate = self._gate_annotation_cb.isChecked()
        if self._get_is_enlarged():
            self.setGateAnnotation(0, doAnnotate)
        else:
            for index, name in self._get_geo().items():
                if name:
                    self.setGateAnnotation(index, doAnnotate)
        self.canvasDrawRequested.emit()

    def setGateAnnotation(self, index, doAnnotate):
        self.logger.debug('setGateAnnotation - index, doAnnotate: %s, %s', index, doAnnotate)
        ax = self._get_spectrum_info("axis", index=index)
        if ax is None:
            self.logger.debug('setGateAnnotation - ax is None')
            return
        dim = self._spectra.get(self._name_from_index(index), "dim")

        for child in ax.lines:
            if isinstance(child, matplotlib.lines.Line2D):
                label     = child.get_label()
                labelSplit = label.split("_-_")
                if len(labelSplit) == 3 and labelSplit[0] == "gate":
                    gateName      = labelSplit[1]
                    gateSegmentNum = labelSplit[2]
                    labelBuff     = None

                    if dim == 1:
                        if gateSegmentNum == "0":
                            color     = self.getGateColor(gateName)
                            labelBuff = gateName + "_low"
                        elif gateSegmentNum == "1":
                            labelBuff = gateName + "_high"

                        toRemove = [an for an in ax.get_children()
                                    if type(an) == matplotlib.text.Annotation
                                    and an.get_text() == labelBuff]
                        if len(toRemove) == 1:
                            toRemove[0].remove()

                        positionX = child.get_xdata()[0]
                        positionY = 0.95
                        offsetX   = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.002

                        if doAnnotate:
                            xy = self.getXYAnnotation(
                                self._name_from_index(index),
                                gateName,
                                (positionX + offsetX, positionY),
                            )
                            ax.annotate(int(positionX), xy=xy,
                                        xycoords=("data", "axes fraction"),
                                        color=color, fontsize=8, clip_on=True)
                            child.set_color(color)
                        else:
                            child.set_color('red')

                    elif dim == 2:
                        toRemove = [an for an in ax.get_children()
                                    if type(an) == matplotlib.text.Annotation
                                    and an.get_text() == gateName]
                        if len(toRemove) == 1:
                            toRemove[0].remove()

                        if doAnnotate:
                            color     = self.getGateColor(gateName)
                            positionX = child.get_xdata()[0]
                            positionY = child.get_ydata()[0]
                            child.set_color(color)
                            child.set_gid(gateName)
                        else:
                            child.set_color('red')
                            child.set_gid(gateName)

        if doAnnotate:
            handles, labels = [], []
            for line in ax.get_lines():
                lbl = line.get_gid()
                if lbl and not lbl.startswith("_"):
                    handles.append(line)
                    labels.append(lbl)
            if handles:
                ax.legend(handles, labels, loc="upper right", fontsize=16, frameon=False)
            else:
                if ax.get_legend():
                    ax.get_legend().remove()

    def getGateColor(self, gateName):
        self.logger.info('getGateColor - gateName : %s', gateName)
        colorList = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown',
                     'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        nextColor = colorList[len(self.gateColor) % len(colorList)]
        if gateName not in self.gateColor:
            self.gateColor[gateName] = nextColor
            return nextColor
        return self.gateColor[gateName]

    def getXYAnnotation(self, spectrumName, gateName, xy):
        self.logger.info('getXYAnnotation - spectrumName, gateName, xy : %s, %s, %s',
                         spectrumName, gateName, xy)
        if spectrumName not in self.gateAnnotation:
            self.gateAnnotation[spectrumName] = {gateName: xy}
            return xy
        for gateDict in self.gateAnnotation[spectrumName]:
            if gateName in gateDict:
                return xy
            return (xy[0], xy[1] - 0.05)

    # ------------------------------------------------------------------
    # REST push
    # ------------------------------------------------------------------

    def pushGateToREST(self, gateName, gateType):
        self.logger.info('pushGateToREST')
        if gateName is None or gateName == "None":
            self.logger.debug('pushGateToREST - gateName is None or "None"')
            return

        rest = self._get_rest()
        if rest is None:
            self.logger.warning('pushGateToREST - REST client not available')
            return

        dim          = self._get_spectrum_info("dim", index=self._popup.gateSpectrumIndex)
        name         = self._name_from_index(self._popup.gateSpectrumIndex)
        parameters   = self._spectra.get(name, "parameters")
        spectrumType = self._spectra.get(name, "type")

        if spectrumType == "gd":
            parametersFormat = []
            for item in parameters:
                pars = item.split(' ')
                if len(pars) == 2:
                    parametersFormat.append(pars[0])
                    parametersFormat.append(pars[1])
            parameters = parametersFormat

        if spectrumType == "m2":
            parametersFormat = {}
            if len(parameters) % 2 != 0:
                parameters = parameters[:-1]
            countPar = 0
            for ipar in range(0, len(parameters), 2):
                par1 = parameters[ipar]
                par2 = parameters[ipar + 1]
                dumNum1 = str(10 + countPar + 1)
                dumNum2 = str(10 + countPar + 2)
                countPar += 2
                nameSubGate = "_" + gateName + "." + dumNum1 + "." + dumNum2 + ".000"
                parametersFormat[nameSubGate] = [par1, par2]

        if len(parameters) == 0:
            return

        boundaries = []
        if self._editing_gate:
            points = self.formatGatePopupPointText(dim)
            if points is None:
                return
            if dim == 1:
                boundaries = points
            elif dim == 2:
                for point in points:
                    boundaries.append({"x": point[0], "y": point[1]})
        else:
            if len(self._popup.listRegionLine) == 0:
                return
            if dim == 1:
                if gateType in ["s", "vs+", "vs*", "gs"]:
                    boundaries = [self._popup.listRegionLine[0].get_xdata()[0],
                                  self._popup.listRegionLine[1].get_xdata()[0]]
                    if boundaries[0] > boundaries[1]:
                        boundaries.sort()
                        self.logger.warning(
                            'pushGateToREST - 1d - found boundaries[0] > boundaries[1] so sorted')
            else:
                for iline, line in enumerate(self._popup.listRegionLine):
                    if line.get_label() != "closing_segment":
                        if iline == 0:
                            for ipoint in range(2):
                                boundaries.append({"x": line.get_xdata()[ipoint],
                                                   "y": line.get_ydata()[ipoint]})
                        else:
                            boundaries.append({"x": line.get_xdata()[1],
                                               "y": line.get_ydata()[1]})

        if spectrumType == "m2":
            for subname, par in parametersFormat.items():
                rest.createGate(subname, gateType, par, boundaries)
            subGateNameList = list(parametersFormat.keys())
            rest.createGate(gateName, "+", subGateNameList, None)
        else:
            rest.createGate(gateName, gateType, parameters, boundaries)

    def formatGatePopupPointText(self, dim):
        self.logger.info('formatGatePopupPointText - dim: %s', dim)
        points    = []
        pointDict = {}
        textBlockLines = self._popup.regionPoint.toPlainText().split("\n")
        for line in textBlockLines:
            if not line:
                continue
            pointId = re.search(r'\d+', line)
            if not pointId:
                self.logger.warning(
                    'formatGatePopupPointText - Expect a point number before X (and Y) at beginning of a line')
                return None
            if int(pointId.group()) in pointDict.keys():
                self.logger.warning(
                    'formatGatePopupPointText - Expect unique point number: %s', int(pointId.group()))
                return None
            pointId = int(pointId.group())
            posX = re.search(r'X(\s*[=]\s*)([-+]?(?:\d*\.*\d+))', line)
            if dim == 1 and not posX:
                self.logger.warning(
                    'formatGatePopupPointText - 1d spectrum - Line format is: i: X=f1')
                return None
            try:
                posX = float(posX.group(2))
            except Exception:
                self.logger.debug('formatGatePopupPointText - exception ', exc_info=True)
                return None
            pointDict[pointId] = posX

            posY = re.search(r'Y(\s*[=]\s*)([-+]?(?:\d*\.*\d+))', line)
            if dim == 2 and not posY:
                self.logger.warning(
                    'formatGatePopupPointText - 2d spectrum - Line format is: i: X=f1 Y=f2')
                return None
            elif dim == 2 and posY:
                posY = float(posY.group(2))
                pointDict[pointId] = [posX, posY]

        pointDict = {key: pointDict[key] for key in sorted(pointDict)}
        if dim == 1:
            points = [val for val in pointDict.values()]
            points = points[-2:]
        elif dim == 2:
            for point in pointDict.values():
                points.append([point[0], point[1]])
        return points

    # ------------------------------------------------------------------
    # Gate popup ok / cancel
    # ------------------------------------------------------------------

    def okGate(self):
        self.logger.info('okGate')
        rest = self._get_rest()
        if rest is None:
            return
        gateName     = self._popup.gateNameList.currentText()
        gateNameList = [gate["name"] for gate in rest.listGate()]
        if not self._editing_gate:
            if gateName in gateNameList:
                self.logger.debug('okGate - gateName: %s already exists', gateName)
                msgBox = QMessageBox(self._parent_widget)
                msgBox.setIcon(QMessageBox.Warning)
                msgBox.setWindowFlag(Qt.WindowStaysOnTopHint, True)
                msgBox.setText("Gate name already exists.")
                msgBox.setInformativeText(
                    'Do you want to overwrite "' + gateName + '" gate definition?')
                msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
                msgBox.setDefaultButton(QMessageBox.Cancel)
                ret = msgBox.exec()
                if ret == QMessageBox.Yes:
                    pass
                elif ret == QMessageBox.Cancel:
                    return
            elif "_-_" in gateName:
                self.logger.debug('okGate - gateName has _-_ in name')
                msgBox = QMessageBox(self._parent_widget)
                msgBox.setIcon(QMessageBox.Warning)
                msgBox.setWindowFlag(Qt.WindowStaysOnTopHint, True)
                msgBox.setText('Gate name must not include "_-_"')
                msgBox.setStandardButtons(QMessageBox.Ok)
                msgBox.setDefaultButton(QMessageBox.Ok)
                ret = msgBox.exec()
                if ret == QMessageBox.Ok:
                    pass
        self.pushGateToREST(gateName, self._popup.listGateType.currentText())
        self._gate_cache_ts = 0.0   # invalidate so new/edited gate shows on next draw
        self._popup.clearInfo()
        self.cancelGate()

    def cancelGate(self, doClose=True):
        self.logger.info('cancelGate')
        self._creating_gate = False
        self._editing_gate  = False
        self.gateEnded.emit()
        self.disconnectGateSignals()
        if doClose:
            self._popup.close()
        self.updatePlotRequested.emit()

    def disconnectGateSignals(self):
        self.logger.info('disconnectGateSignals')
        canvas = self._get_current_canvas()
        try:
            if hasattr(self, 'gateReleaser'):
                canvas.mpl_disconnect(self.gateReleaser)
        except TypeError:
            pass
        try:
            if hasattr(self, 'gateFollower'):
                canvas.mpl_disconnect(self.gateFollower)
        except TypeError:
            pass
        try:
            if hasattr(self, 'sid'):
                canvas.mpl_disconnect(self.sid)
        except TypeError:
            pass
        try:
            if hasattr(self, 'sidGateNameListChanged'):
                self._popup.gateNameList.currentTextChanged.disconnect(self.sidGateNameListChanged)
        except TypeError:
            pass
        try:
            if hasattr(self, 'sidGateTypeListChanged'):
                self._popup.listGateType.currentIndexChanged.disconnect(self.sidGateTypeListChanged)
        except TypeError:
            pass
        try:
            if hasattr(self, 'gatePopupPreview'):
                self._popup.preview.clicked.disconnect(self.gatePopupPreview)
        except TypeError:
            pass
        try:
            if hasattr(self, 'shortcutInsertRegionPoint'):
                self.shortcutInsertRegionPoint.setEnabled(False)
        except TypeError:
            pass
        conn = self._get_integrate_copy()
        if conn is not None:
            try:
                self._integrate_popup.resultsText.itemSelectionChanged.disconnect(conn)
            except TypeError:
                pass

    # ------------------------------------------------------------------
    # Gate creation
    # ------------------------------------------------------------------

    def createGate(self, index):
        self.logger.info('createGate')
        self._skip_auto.set()
        self._popup.gateActionCreate.setChecked(True)
        if self._gate_edit_disable_cb.isChecked():
            self._popup.gateActionEdit.setChecked(False)
            self._popup.gateActionEdit.setEnabled(False)
        else:
            self._popup.gateActionEdit.setEnabled(True)

        try:
            if hasattr(self, 'sidGateNameListChanged'):
                self._popup.gateNameList.currentTextChanged.disconnect(self.sidGateNameListChanged)
                self.logger.debug('createGate - disconnected sidGateNameListChanged')
        except TypeError:
            pass

        self._popup.gateNameList.setEditable(True)
        self._popup.gateNameList.setInsertPolicy(QComboBox.NoInsert)
        self._popup.gateNameList.setCurrentText("gate-001")

        if index is None:
            return QMessageBox.about(self._parent_widget, "Warning!", "Please add at least one spectrum")

        gateTypesDict = {
            "b":  ["NotDefinedYet"],
            "1v": ["vs*", "vs+"],
            "1":  ["s"],
            "g1": ["gs"],
            "2":  ["c", "b"],
            "g2": ["gc", "gb"],
            "gd": ["gc", "gb"],
            "m2": ["c", "b"],
            "s":  ["NotDefinedYet"],
        }
        spectrumType = self._spectra.get(self._name_from_index(index), "type")
        if spectrumType is None:
            return

        self._popup.clearInfo()
        self.disconnectGateSignals()

        gateTypesList = gateTypesDict[spectrumType]
        for gtype in gateTypesList:
            if gtype == "NotDefinedYet":
                self.logger.debug('createGate - gate type NotDefinedYet')
                msgBox = QMessageBox(self._parent_widget)
                msgBox.setIcon(QMessageBox.Warning)
                msgBox.setWindowFlag(Qt.WindowStaysOnTopHint, True)
                msgBox.setText('No gate type available for "' + spectrumType + '" spectrum')
                msgBox.setStandardButtons(QMessageBox.Ok)
                msgBox.setDefaultButton(QMessageBox.Ok)
                msgBox.exec()
                return
            self._popup.listGateType.addItem(gtype)

        self._creating_gate = True
        self._editing_gate  = False
        self.gateCreationStarted.emit(index)

        self.sidGateTypeListChanged = self._popup.listGateType.currentIndexChanged.connect(
            self.gateTypeListChanged)

        self._popup.gateSpectrumIndex = index

        self._populateGateNameListFromAxis(index)
        self._popup.gateNameList.setCurrentText(self._nextGateName())

        self._popup.show()

    def _populateGateNameListFromAxis(self, spec_index):
        ax   = self._get_spectrum_info("axis", index=spec_index)
        name = self._name_from_index(spec_index)
        dim  = self._spectra.get(name, "dim")
        cb   = self._popup.gateNameList
        cb.clear()
        if ax is None or dim is None:
            return
        seen = set()
        for child in ax.lines:
                label = child.get_label()
                if "_-_" not in label:
                    continue
                parts = label.split("_-_")
                if dim == 1 and parts[0] == "gate" and parts[2] == "0":
                    if parts[1] not in seen:
                        cb.addItem(parts[1]); seen.add(parts[1])
                elif dim == 2 and parts[0] == "gate":
                    if parts[1] not in seen:
                        cb.addItem(parts[1]); seen.add(parts[1])

    def _nextGateName(self):
        rx = re.compile(r"^gate-(\d+)$")
        mx = 0
        cb = self._popup.gateNameList
        for i in range(cb.count()):
            m = rx.match(cb.itemText(i))
            if m:
                mx = max(mx, int(m.group(1)))
        return f"gate-{mx+1:03d}"

    def onGatePopupPreview(self):
        self.logger.info('onGatePopupPreview')
        ax = self._get_spectrum_info("axis", index=self._popup.gateSpectrumIndex)
        if ax is None:
            self.logger.debug('onGatePopupPreview - ax is None')
            return
        name   = self._name_from_index(self._popup.gateSpectrumIndex)
        dim    = self._spectra.get(name, "dim")
        points = self.formatGatePopupPointText(dim)
        if points is None:
            self.logger.debug('onGatePopupPreview - points is None')
            return
        if not hasattr(self, 'editThisGateLine') or self.editThisGateLine is None:
            try:
                gateIdentifier    = "gate_-_" + self._popup.gateNameList.currentText() + "_-_"
                self.editThisGateLine = [
                    child for child in ax.lines
                    if gateIdentifier in child.get_label()
                ][0]
            except Exception:
                self.logger.debug('onGatePopupPreview - exception ', exc_info=True)
                return
        if dim == 1:
            label         = self.editThisGateLine.get_label()
            labelSplit    = label.split("_-_")
            gateIdentifier = "gate_-_" + labelSplit[1] + "_-_"
            lines = [child for child in ax.lines if gateIdentifier in child.get_label()]
            if len(lines) == len(points):
                for iline, line in enumerate(lines):
                    line.set_xdata([points[iline], points[iline]])
        elif dim == 2:
            lineX, lineY = [], []
            for point in points:
                lineX.append(point[0])
                lineY.append(point[1])
            specialTypes = ["c", "gc"]
            if self._popup.listGateType.currentText() in specialTypes:
                lineX.append(points[0][0])
                lineY.append(points[0][1])
            self.editThisGateLine.set_data(lineX, lineY)
        self.canvasDrawRequested.emit()

    def checkConnections(self):
        canvas = self._get_current_canvas()
        if canvas.callbacks.callbacks:
            for event_name, callbacks_dict in canvas.callbacks.callbacks.items():
                print(f"Event: {event_name}, Callbacks: {callbacks_dict}")

    def find_callbacks(self):
        canvas = self._get_current_canvas()
        callbacks_for_gate_manager = []
        for event_name, callbacks_dict in canvas.callbacks.callbacks.items():
            for callback_id, callback_func in callbacks_dict.items():
                if hasattr(callback_func, '__self__') and callback_func.__self__ is self:
                    callbacks_for_gate_manager.append((event_name, callback_func))
        return callbacks_for_gate_manager

    # ------------------------------------------------------------------
    # Gate editing
    # ------------------------------------------------------------------

    def editGate(self):
        self.logger.info('editGate')
        self._popup.gateActionCreate.setChecked(False)

        try:
            if hasattr(self, 'sidGateTypeListChanged'):
                self._popup.listGateType.currentIndexChanged.disconnect(self.sidGateTypeListChanged)
                self.logger.debug('editGate - disconnected sidGateTypeListChanged')
        except TypeError:
            pass

        self.sid = self._get_current_canvas().mpl_connect(
            'pick_event', self.clickOnGateLine)

        self.shortcutInsertRegionPoint = QShortcut(
            QKeySequence("Alt+E"), self._parent_widget)
        self.shortcutInsertRegionPoint.activated.connect(self.onKeyActivateEditGate)

        if self._popup.gateSpectrumIndex is None:
            self.logger.debug('editGate - gateSpectrumIndex is None')
            return QMessageBox.about(self._parent_widget, "Warning!", "Please add at least one spectrum")

        spectrumName = self._name_from_index(self._popup.gateSpectrumIndex)
        dim = self._spectra.get(spectrumName, "dim")
        ax  = self._get_spectrum_info("axis", index=self._popup.gateSpectrumIndex)
        if ax is None:
            self.logger.debug('editGate - ax is None')
            return

        self._popup.clearInfo()

        gateLabels = [child.get_label() for child in ax.lines if "_-_" in child.get_label()]
        for label in gateLabels:
            if dim == 1:
                label = label.split("_-_")
                if label[0] == 'gate' and label[2] == '0':
                    self._popup.gateNameList.addItem(label[1])
            elif dim == 2:
                label = label.split("_-_")
                if label[0] == 'gate':
                    self._popup.gateNameList.addItem(label[1])
        self._popup.gateNameList.setCurrentText("-- select a gate --")
        self._popup.gateNameList.completer().setCompletionMode(QCompleter.PopupCompletion)
        self._popup.gateNameList.completer().setFilterMode(Qt.MatchContains)
        self.sidGateNameListChanged = self._popup.gateNameList.currentTextChanged.connect(
            self.gateNameListChanged)

        self._creating_gate = False
        self._editing_gate  = True
        self.gateEditingStarted.emit()
        self.altPressed = False
        self._popup.regionPoint.setReadOnly(False)

    def gateTypeListChanged(self):
        self.logger.info('gateTypeListChanged')
        self._popup.gateNameList.clear()
        self._popup.gateNameList.setCurrentText("gate-001")
        for line in self._popup.listRegionLine:
            line.remove()
        self._popup.listRegionLine.clear()
        self._popup.prevPoint.clear()
        self._popup.regionPoint.clear()
        self._popup.gateSpectrumIndex  = 0
        self._popup.gateEditOption     = None

    def gateNameListChanged(self):
        self.logger.info('gateNameListChanged')
        ax = self._get_spectrum_info("axis", index=self._popup.gateSpectrumIndex)
        if ax is None:
            self.logger.debug('gateNameListChanged - ax is None')
            return
        gateName       = self._popup.gateNameList.currentText()
        gateFoundAtIdx = self._popup.gateNameList.findText(gateName)
        if gateFoundAtIdx == -1:
            self._popup.regionPoint.clear()
            self._popup.listGateType.clear()
            self.logger.debug('gateNameListChanged - gateFoundAtIdx == -1')
            return
        gateIdentifier = "gate_-_" + gateName + "_-_"
        lines = [child for child in ax.lines if gateIdentifier in child.get_label()]
        rest = self._get_rest()
        if rest is None:
            return
        gate = [d for d in rest.listGate() if d["name"] == gateName]
        self._popup.gateNameList.setCurrentText(gateName)
        self._popup.listGateType.addItem(gate[0]["type"])
        self.updateTextGatePopup(lines)
        for line in lines:
            line.set_marker(marker='o')
            line.set_color("green")
        self.canvasDrawRequested.emit()

    # ------------------------------------------------------------------
    # Mouse interaction helpers
    # ------------------------------------------------------------------

    def on_singleclick_gate(self, event, index):
        self.logger.info('on_singleclick_gate - index: %s', index)
        if not self._get_is_enlarged():
            return
        dim      = self._spectra.get(self._name_from_index(index), "dim")
        gateType = self._popup.listGateType.currentText()
        if dim == 1:
            l = self.addLine(float(event.xdata), 0, index)
            self._popup.listRegionLine.append(l)
            if len(self._popup.listRegionLine) > 2:
                self.removePrevLine()
            lineText = ""
            for nbLine in range(len(self._popup.listRegionLine)):
                prefix = "" if nbLine == 0 else "\n"
                lineText += prefix + f"{nbLine}: X= {self._popup.listRegionLine[nbLine].get_xdata()[0]:.3f}"
            self._popup.regionPoint.clear()
            self._popup.regionPoint.insertPlainText(lineText)

        elif dim == 2:
            tempLine = [ln for ln in self._popup.listRegionLine
                        if ln.get_label() == "closing_segment"]
            if gateType not in ["b", "gb"] and len(tempLine) == 1:
                tempLine[0].remove()
                self._popup.listRegionLine.pop()

            l = self.addLine(float(event.xdata), float(event.ydata), index)
            if l is not None:
                self._popup.listRegionLine.append(l)

            lineNb   = len(self._popup.listRegionLine)
            lineText = ""
            for nbLine in range(lineNb):
                prefix = "" if nbLine == 0 else "\n"
                lineText += (prefix +
                             f"{nbLine}: X= {self._popup.listRegionLine[nbLine].get_xdata()[0]:.3f}"
                             f"   Y= {self._popup.listRegionLine[nbLine].get_ydata()[0]:.3f}")
            if lineNb == 0:
                lineText += f"{lineNb}: X= {float(event.xdata):.3f}   Y= {float(event.ydata):.3f}"
            else:
                lineText += f"\n{lineNb}: X= {float(event.xdata):.3f}   Y= {float(event.ydata):.3f}"
            self._popup.regionPoint.clear()
            self._popup.regionPoint.insertPlainText(lineText)

            if gateType not in ["b", "gb"] and lineNb > 1:
                label = "closing_segment"
                l = self.addLine(self._popup.listRegionLine[0].get_xdata()[0],
                                 self._popup.listRegionLine[0].get_ydata()[0],
                                 index, label)
                if l is not None:
                    self._popup.listRegionLine.append(l)

        self.canvasDrawRequested.emit()

    def on_singleclick_gate_right(self, index):
        self.logger.info('on_singleclick_gate_right - index: %s', index)
        if not self._get_is_enlarged():
            return
        dim          = self._spectra.get(self._name_from_index(index), "dim")
        gateType     = self._popup.listGateType.currentText()
        gateTypeList1 = ["c", "gc"]
        gateTypeList2 = ["b"]

        if dim == 2:
            if gateType in gateTypeList1:
                lineNb = len(self._popup.listRegionLine)
                try:
                    self._popup.prevPoint = [
                        self._popup.listRegionLine[-2].get_xdata()[0],
                        self._popup.listRegionLine[-2].get_ydata()[0],
                    ]
                except IndexError:
                    self.logger.debug('on_singleclick_gate_right - IndexError', exc_info=True)
                    return
                if lineNb == 3:
                    for _ in range(2):
                        self._popup.listRegionLine[-1].remove()
                        self._popup.listRegionLine.pop(-1)
                elif lineNb > 3:
                    tempLine = self._popup.listRegionLine[-3]
                    self._popup.listRegionLine[-2].remove()
                    self._popup.listRegionLine.pop(-2)
                    self._popup.listRegionLine[-1].set_xdata(
                        [tempLine.get_xdata()[1], self._popup.listRegionLine[0].get_xdata()[0]])
                    self._popup.listRegionLine[-1].set_ydata(
                        [tempLine.get_ydata()[1], self._popup.listRegionLine[0].get_ydata()[0]])

            if gateType in gateTypeList2:
                lineNb = len(self._popup.listRegionLine)
                if lineNb >= 2:
                    self._popup.prevPoint = [
                        self._popup.listRegionLine[-2].get_xdata()[1],
                        self._popup.listRegionLine[-2].get_ydata()[1],
                    ]
                    self._popup.listRegionLine[-1].remove()
                    self._popup.listRegionLine.pop(-1)

            lineNb   = len(self._popup.listRegionLine)
            lineText = ""
            for nbLine in range(lineNb):
                prefix = "" if nbLine == 0 else "\n"
                lineText += (prefix +
                             f"{nbLine}: X= {self._popup.listRegionLine[nbLine].get_xdata()[0]:.3f}"
                             f"   Y= {self._popup.listRegionLine[nbLine].get_ydata()[0]:.3f}")
            if lineNb == 1:
                lineText += (f"\n{lineNb}: X= {self._popup.listRegionLine[0].get_xdata()[1]:.3f}"
                             f"   Y= {self._popup.listRegionLine[0].get_ydata()[1]:.3f}")
            self._popup.regionPoint.clear()
            self._popup.regionPoint.insertPlainText(lineText)

        self.canvasDrawRequested.emit()

    def on_singleclick_gate_edit(self, event):
        self.logger.info('on_singleclick_gate_edit')
        name = self._name_from_index(self._popup.gateSpectrumIndex)
        dim  = self._spectra.get(name, "dim")
        ax   = self._get_spectrum_info("axis", index=self._popup.gateSpectrumIndex)
        if ax is None:
            self.logger.debug('on_singleclick_gate_edit - ax is None')
            return
        if dim == 2:
            self.xyRef = np.array([event.xdata, event.ydata])
        if event.button == 1:
            if hasattr(self, 'altPressed') and self.altPressed:
                self.insertPointGate(event)
        if event.button == 3:
            if hasattr(self, 'altPressed') and self.altPressed:
                self.deletePointGate(event)
            self.canvasDrawRequested.emit()
        self.altPressed = False

    def on_dblclick_gate_edit(self, event, index):
        self.logger.info('on_dblclick_gate_edit')
        name = self._name_from_index(self._popup.gateSpectrumIndex)
        dim  = self._spectra.get(name, "dim")
        if dim == 2:
            self._popup.gateEditOption = "2d_move_all"
            self.gateReleaser = self._get_current_canvas().mpl_connect(
                "button_press_event", self.releaseonclick)

    def addLine(self, posx, posy, index, label=None, mode="gate"):
        self.logger.info('addLine - posx, posy, index, label: %s, %s, %s, %s',
                         posx, posy, index, label)
        spectrum = self._get_spectrum_info("spectrum", index=index)
        dim      = self._get_spectrum_info("dim", index=index)
        if spectrum is None:
            self.logger.debug('addLine - spectrum is None')
            return
        ax = spectrum.axes
        l  = None

        if dim == 1:
            ymin, ymax = ax.get_ybound()
            l = mlines.Line2D([posx, posx], [ymin, ymax], picker=True, label=label)
            l.set_pickradius(5)
            ax.add_line(l)
        elif dim == 2:
            if mode == "gate":
                xyPrev = self._popup.prevPoint
                self._popup.prevPoint = [posx, posy]
                if label == "closing_segment":
                    self._popup.prevPoint = xyPrev
            elif mode == "sum_region":
                xyPrev = self._sum_region_popup.prevPoint
                self._sum_region_popup.prevPoint = [posx, posy]
                if label == "closing_segment":
                    self._sum_region_popup.prevPoint = xyPrev
            else:
                return
            if xyPrev is None or len(xyPrev) == 0:
                return
            l = mlines.Line2D([xyPrev[0], posx], [xyPrev[1], posy], picker=True, label=label)
            l.set_pickradius(5)
            ax.add_line(l)

        if l is None:
            self.logger.debug('addLine - l is None')
            return
        l.set_color('b' if mode == "sum_region" else 'r')
        return l

    def removePrevLine(self, mode="gate"):
        self.logger.info('removePrevLine')
        if mode == "gate":
            popup = self._popup
        elif mode == "sum_region":
            popup = self._sum_region_popup
        else:
            return
        if popup is not None:
            l = popup.listRegionLine[0]
            l.remove()
            popup.listRegionLine.pop(0)

    def releaseonclick(self, event):
        self.logger.info('releaseonclick')
        canvas = self._get_current_canvas()
        try:
            if hasattr(self, 'gateReleaser'):
                canvas.mpl_disconnect(self.gateReleaser)
        except TypeError:
            pass
        try:
            if hasattr(self, 'gateFollower'):
                canvas.mpl_disconnect(self.gateFollower)
        except TypeError:
            pass
        self._popup.gateEditOption = None
        self.xyRef        = None
        self.movingMarker = []

    def onKeyActivateEditGate(self):
        self.logger.info('onKeyActivateEditGate - _editing_gate: %s', self._editing_gate)
        if not self._editing_gate:
            return
        self.altPressed = True

    def insertPointGate(self, event):
        self.logger.info('insertPointGate')
        name = self._name_from_index(self._popup.gateSpectrumIndex)
        dim  = self._spectra.get(name, "dim")
        ax   = self._get_spectrum_info("axis", index=self._popup.gateSpectrumIndex)
        if ax is None or dim != 2:
            self.logger.debug('insertPointGate - ax is None or dim!=2: %s', dim)
            return

        lineX  = self.editThisGateLine.get_xdata()
        lineY  = self.editThisGateLine.get_ydata()
        lineXY = self.editThisGateLine.get_transform().transform(self.editThisGateLine.get_xydata())

        p        = event.x, event.y
        insertAt = None
        for i in range(len(lineXY) - 1):
            d = self.dist_point_to_segment(p, lineXY[i], lineXY[i + 1])
            if d <= self.epsilon:
                insertAt = i
                break

        if insertAt is not None:
            lineX.insert(insertAt + 1, ax.transData.inverted().transform((event.x, event.y))[0])
            lineY.insert(insertAt + 1, ax.transData.inverted().transform((event.x, event.y))[1])
            self.editThisGateLine.set_data(lineX, lineY)
            self.canvasDrawIdleRequested.emit()

    def deletePointGate(self, event):
        self.logger.info('deletePointGate')
        name = self._name_from_index(self._popup.gateSpectrumIndex)
        dim  = self._spectra.get(name, "dim")
        ax   = self._get_spectrum_info("axis", index=self._popup.gateSpectrumIndex)
        if ax is None or dim != 2:
            self.logger.debug('deletePointGate - ax is None or dim!=2: %s', dim)
            return

        lineX     = self.editThisGateLine.get_xdata()
        lineY     = self.editThisGateLine.get_ydata()
        markerPos = np.array([lineX, lineY])
        p         = ax.transData.inverted().transform((event.x, event.y))
        distances = np.linalg.norm(markerPos - p.reshape(2, -1), axis=0)
        dataRadius = abs(ax.transData.inverted().transform((self.epsilon, 0))[0])
        specialGateTypes = ["c", "gc"]

        markerIdx = np.where(distances <= dataRadius)[0]
        if markerIdx.size > 0:
            if markerIdx[0] == 0 and self._popup.listGateType.currentText() in specialGateTypes:
                lineX[-1] = lineX[1]
                lineY[-1] = lineY[1]
            lineX.pop(markerIdx[0])
            lineY.pop(markerIdx[0])
            self.editThisGateLine.set_data(lineX, lineY)
            self.updateTextGatePopup([self.editThisGateLine])
            self.canvasDrawIdleRequested.emit()

    def followmouse(self, event):
        if self._popup.gateEditOption == "1d_move_line":
            self.editThisGateLine.set_color("green")
            self.editThisGateLine.set_xdata([event.xdata, event.xdata])
        elif self._popup.gateEditOption == "2d_move_all":
            lineX  = self.editThisGateLine.get_xdata()
            lineY  = self.editThisGateLine.get_ydata()
            shiftX = event.xdata - lineX[0]
            shiftY = event.ydata - lineY[0]
            lineX  = [item + shiftX for item in lineX]
            lineY  = [item + shiftY for item in lineY]
            self.editThisGateLine.set_data(lineX, lineY)
        elif self._popup.gateEditOption == "2d_move_point":
            lineX = self.editThisGateLine.get_xdata()
            lineY = self.editThisGateLine.get_ydata()
            if hasattr(self, 'movingMarker') and len(self.movingMarker) > 0:
                for mIdx in self.movingMarker:
                    lineX[mIdx] = event.xdata
                    lineY[mIdx] = event.ydata
                self.editThisGateLine.set_data(lineX, lineY)
            else:
                markerPos = np.array([lineX, lineY])
                try:
                    ax = self._get_spectrum_info("axis", index=self._popup.gateSpectrumIndex)
                    if ax is None:
                        return
                    distances      = np.linalg.norm(markerPos - self.xyRef.reshape(2, -1), axis=0)
                    xlims          = ax.get_xlim()
                    ylims          = ax.get_ylim()
                    figTest        = plt.gcf()
                    plottingAreaWidth, plottingAreaHeight = figTest.get_size_inches() * figTest.dpi
                    radX           = self.pixel_to_data_distance(self.epsilon, xlims, plottingAreaWidth)
                    radY           = self.pixel_to_data_distance(self.epsilon, ylims, plottingAreaHeight)
                    dataRadius     = math.sqrt(radX * radX + radY * radY)
                    markerIdx      = np.where(distances <= dataRadius)[0]
                    if markerIdx.size > 0:
                        self.movingMarker = []
                        for mIdx in markerIdx:
                            lineX[mIdx] = event.xdata
                            lineY[mIdx] = event.ydata
                            self.movingMarker.append(mIdx)
                        self.editThisGateLine.set_data(lineX, lineY)
                        self.gateReleaser = self._get_current_canvas().mpl_connect(
                            "button_press_event", self.releaseonclick)
                except NameError:
                    raise
        self.canvasDrawIdleRequested.emit()

    def pixel_to_data_distance(self, pixel_distance, axis_limits, plotting_area_size):
        axis_range = axis_limits[1] - axis_limits[0]
        return pixel_distance * (axis_range / plotting_area_size)

    def updateTextGatePopup(self, gateList):
        name = self._name_from_index(self._popup.gateSpectrumIndex)
        dim  = self._spectra.get(name, "dim")
        if dim == 1:
            lineText = ""
            for nbLine in range(len(gateList)):
                prefix = "" if nbLine == 0 else "\n"
                lineText += prefix + f"{nbLine}: X= {gateList[nbLine].get_xdata()[0]:.3f}"
        elif dim == 2:
            lines    = gateList[0].get_xydata()
            lineText = ""
            specialTypes = ["c", "gc"]
            nbPoints = len(lines)
            if self._popup.listGateType.currentText() in specialTypes:
                nbPoints = len(lines) - 1
            for nbLine in range(nbPoints):
                prefix = "" if nbLine == 0 else "\n"
                lineText += prefix + f"{nbLine}: X= {lines[nbLine][0]:.3f}   Y= {lines[nbLine][1]:.3f}"
        self._popup.regionPoint.clear()
        self._popup.regionPoint.insertPlainText(lineText)

    def clickOnGateLine(self, event):
        self.logger.info('clickOnGateLine')
        if event.mouseevent.button != 1:
            return
        self.editThisGateLine = None
        name = self._name_from_index(self._popup.gateSpectrumIndex)
        dim  = self._spectra.get(name, "dim")
        ax   = self._get_spectrum_info("axis", index=self._popup.gateSpectrumIndex)
        if ax is None:
            self.logger.debug('clickOnGateLine - ax is None')
            return

        lineCandidate = [child for child in ax.lines if child == event.artist]
        if len(lineCandidate) > 0:
            self.editThisGateLine = lineCandidate[0]
        else:
            self.logger.debug('clickOnGateLine - len(lineCandidate) <= 0')
            return

        labelSplit = lineCandidate[0].get_label().split("_-_")
        if len(labelSplit) != 3 or labelSplit[0] != "gate":
            self.logger.debug('clickOnGateLine - lineLabel has not the expected format')
            return
        gateName = labelSplit[1]
        rest = self._get_rest()
        if rest is None:
            return
        gate = [d for d in rest.listGate() if d["name"] == gateName]

        self._popup.gateNameList.setCurrentText(gateName)
        self._popup.listGateType.clear()
        self._popup.listGateType.addItem(gate[0]["type"])

        gateIdentifier = "gate_-_" + gateName + "_-_"
        gateLines = [child for child in ax.lines if gateIdentifier in child.get_label()]
        self.updateTextGatePopup(gateLines)

        canvas = self._get_current_canvas()
        self.gateFollower = canvas.mpl_connect(
            "motion_notify_event", self.followmouse)
        if dim == 1:
            self._popup.gateEditOption = "1d_move_line"
            self.gateReleaser = canvas.mpl_connect(
                "button_press_event", self.releaseonclick)
        elif dim == 2:
            self.editThisGateLine.set_marker(marker='o')
            self.editThisGateLine.set_color("green")
            self.canvasDrawRequested.emit()
            self._popup.gateEditOption = "2d_move_point"

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def dist(self, x, y):
        d = x - y
        return np.sqrt(np.dot(d, d))

    def dist_point_to_segment(self, p, s0, s1):
        v  = s1 - s0
        w  = p  - s0
        c1 = np.dot(w, v)
        if c1 <= 0:
            return self.dist(p, s0)
        c2 = np.dot(v, v)
        if c2 <= c1:
            return self.dist(p, s1)
        b  = c1 / c2
        pb = s0 + b * v
        return self.dist(p, pb)
