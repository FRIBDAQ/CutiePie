from PyQt5.QtWidgets import QComboBox, QCompleter
from PyQt5 import QtCore


class GateAdapter:
    """Bridges GateManager signals to the gate popup and currentPlot flags.
    Created by MainWindow after the service and popup exist."""

    def __init__(self, gate_manager, gate_popup, get_current_plot, logger):
        self._popup = gate_popup
        self._get_plot = get_current_plot
        self.logger = logger

        # widget -> service (direct, no MainWindow state)
        gate_popup.gateActionEdit.clicked.connect(gate_manager.editGate)
        gate_popup.listGateType.currentIndexChanged.connect(
            gate_manager.gateTypeListChanged)
        gate_popup.gateNameList.currentTextChanged.connect(
            gate_manager.gateNameListChanged)

        # service -> adapter slots
        gate_manager.canvasDrawRequested.connect(self._on_gm_canvas_draw)
        gate_manager.canvasDrawIdleRequested.connect(
            self._on_gm_canvas_draw_idle)
        gate_manager.gateCreationStarted.connect(
            self._on_gate_creation_started)
        gate_manager.gateEditingStarted.connect(
            self._on_gate_editing_started)
        gate_manager.gateEnded.connect(self._on_gate_ended)
        gate_manager.gateReadoutChanged.connect(
            self._on_gate_readout_changed)
        gate_manager.gateReadoutEditable.connect(
            self._on_gate_readout_editable)
        gate_manager.gateTypeCleared.connect(self._on_gate_type_cleared)
        gate_manager.gateTypeItemAdded.connect(
            self._on_gate_type_item_added)
        gate_manager.gateNamesPrepared.connect(
            self._on_gate_names_prepared)
        gate_manager.gateNameSelected.connect(
            self._on_gate_name_selected)
        gate_manager.gateNameListEditable.connect(
            self._on_gate_name_list_editable)
        gate_manager.gateNameCompleterConfigured.connect(
            self._on_gate_name_completer_configured)

        # service -> popup direct
        gate_manager.gateClearInfoRequested.connect(gate_popup.clearInfo)
        gate_manager.gatePopupShowRequested.connect(gate_popup.show)
        gate_manager.gatePopupCloseRequested.connect(gate_popup.close)
        gate_manager.gateActionCreateChecked.connect(
            gate_popup.gateActionCreate.setChecked)
        gate_manager.gateActionEditChecked.connect(
            gate_popup.gateActionEdit.setChecked)
        gate_manager.gateActionEditEnabled.connect(
            gate_popup.gateActionEdit.setEnabled)

    def _on_gm_canvas_draw(self):
        self._get_plot().canvas.draw()

    def _on_gm_canvas_draw_idle(self):
        self._get_plot().canvas.draw_idle()

    def _on_gate_creation_started(self, index):
        self._get_plot().toCreateGate = True
        self._get_plot().toEditGate   = False

    def _on_gate_editing_started(self):
        self._get_plot().toEditGate   = True
        self._get_plot().toCreateGate = False

    def _on_gate_ended(self):
        self._get_plot().toCreateGate = False
        self._get_plot().toEditGate   = False

    def _on_gate_readout_changed(self, text):
        self._popup.regionPoint.clear()
        self._popup.regionPoint.insertPlainText(text)

    def _on_gate_readout_editable(self, editable):
        self._popup.regionPoint.setReadOnly(not editable)

    def _on_gate_type_cleared(self):
        self._popup.listGateType.clear()

    def _on_gate_type_item_added(self, text):
        self._popup.listGateType.addItem(text)

    def _on_gate_names_prepared(self, names, current):
        cb = self._popup.gateNameList
        cb.clear()
        for name in names:
            cb.addItem(name)
        cb.setCurrentText(current)

    def _on_gate_name_selected(self, name):
        self._popup.gateNameList.setCurrentText(name)

    def _on_gate_name_list_editable(self):
        cb = self._popup.gateNameList
        cb.setEditable(True)
        cb.setInsertPolicy(QComboBox.NoInsert)

    def _on_gate_name_completer_configured(self):
        cb = self._popup.gateNameList
        cb.completer().setCompletionMode(QCompleter.PopupCompletion)
        cb.completer().setFilterMode(QtCore.Qt.MatchContains)
