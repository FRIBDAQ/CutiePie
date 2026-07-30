from PyQt5.QtWidgets import QComboBox, QCompleter, QTableWidgetItem
from PyQt5 import QtCore


class SumRegionAdapter:
    """Bridges SumRegionManager signals to the sum-region and integrate
    popup widgets plus currentPlot flags.  Created by MainWindow after
    the service and popups exist."""

    def __init__(self, sum_region_manager, get_current_plot,
                 sum_region_popup, integrate_popup,
                 copy_selection_callback, logger):
        self._get_plot = get_current_plot
        self._sr_popup = sum_region_popup
        self._int_popup = integrate_popup
        self._copy_selection = copy_selection_callback
        self.logger = logger
        self.sid_table_integrate_copy = None

        sum_region_manager.canvasDrawRequested.connect(
            self._on_srm_canvas_draw)
        sum_region_manager.figureTightLayoutRequested.connect(
            self._on_srm_tight_layout)
        sum_region_manager.sumRegionStarted.connect(
            self._on_sum_region_started)
        sum_region_manager.sumRegionEnded.connect(
            self._on_sum_region_ended)
        sum_region_manager.regionReadoutChanged.connect(
            self._on_region_readout_changed)
        sum_region_manager.sumRegionCreatePrepared.connect(
            self._on_sum_region_create_prepared)
        sum_region_manager.sumRegionSelectionChanged.connect(
            self._on_sum_region_selection_changed)
        sum_region_manager.sumRegionPopupCloseRequested.connect(
            self._on_sum_region_popup_close)
        sum_region_manager.integrationResultsReady.connect(
            self._on_integration_results)
        sum_region_manager.integratePopupCloseRequested.connect(
            self._on_integrate_popup_close)

        integrate_popup.ok.clicked.connect(
            sum_region_manager.okIntegrate)

    def _on_srm_canvas_draw(self):
        self._get_plot().canvas.draw()

    def _on_srm_tight_layout(self):
        self._get_plot().figure.tight_layout()
        self._get_plot().canvas.draw()

    def _on_sum_region_started(self, index):
        self._get_plot().toCreateSumRegion = True

    def _on_sum_region_ended(self):
        self._get_plot().toCreateSumRegion = False

    def _on_region_readout_changed(self, text):
        self._sr_popup.regionPoint.clear()
        self._sr_popup.regionPoint.insertPlainText(text)

    def _on_sum_region_create_prepared(self, names):
        combo = self._sr_popup.sumRegionNameList
        self._sr_popup.clearInfo()
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.NoInsert)
        for name in names:
            combo.addItem(name)
        combo.setCurrentText("None")
        combo.completer().setCompletionMode(QCompleter.PopupCompletion)
        combo.completer().setFilterMode(QtCore.Qt.MatchContains)
        self._sr_popup.show()

    def _on_sum_region_selection_changed(self, text):
        self._sr_popup.sumRegionNameList.setCurrentText(text)

    def _on_sum_region_popup_close(self):
        self._sr_popup.close()

    def _on_integration_results(self, rows):
        table = self._int_popup.resultsText
        self._int_popup.clearInfo()
        colHeader = ['Spectrum', 'Region', 'Counts', 'Centroid X',
                     'Centroid Y', 'FWHM X', 'FWHM Y']
        for col, header in enumerate(colHeader):
            headerItem = QTableWidgetItem(header)
            font = table.font()
            font.setBold(True)
            headerItem.setFont(font)
            table.setHorizontalHeaderItem(col, headerItem)
        if not rows:
            table.insertRow(0)
            table.setItem(0, 0, QTableWidgetItem("Nothing to integrate"))
            self._int_popup.show()
            return
        for irow, row in enumerate(rows):
            table.insertRow(irow)
            for icol, cell in enumerate(row):
                table.setItem(irow, icol, QTableWidgetItem(cell))
        self.sid_table_integrate_copy = table.itemSelectionChanged.connect(
            self._copy_selection)
        self._int_popup.show()

    def _on_integrate_popup_close(self):
        self._int_popup.close()
