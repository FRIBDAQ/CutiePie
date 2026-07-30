class FitAdapter:
    """Bridges FitManager/PlotController signals to the SpecialFunctions and
    Cutoff popup widgets.  Created by MainWindow after the services and
    popups exist; owns the signal wiring and the five _on_* slot methods
    that were formerly in MainWindow."""

    def __init__(self, fit_manager, plot_controller, extra_popup,
                 cutoff_popup, logger):
        self._extra = extra_popup
        self._cutoff = cutoff_popup
        self.logger = logger

        fit_manager.fitBusyChanged.connect(self._on_fit_busy)
        fit_manager.abortEnabledChanged.connect(self._on_abort_enabled)
        fit_manager.fitResultsAppended.connect(self._append_fit_results)
        fit_manager.fitLabelsTextChanged.connect(self._set_fit_labels_text)
        plot_controller.cutoffPopupPrepared.connect(self._show_cutoff_popup)
        plot_controller.cutoffPopupCloseRequested.connect(cutoff_popup.close)

        extra_popup.plot_csv_button.clicked.connect(
            fit_manager.on_plot_csv_clicked)
        extra_popup.abort_button.clicked.connect(
            fit_manager.on_abort_clicked)

    def _on_fit_busy(self, busy):
        self._extra.fit_button.setEnabled(not busy)
        self._extra.abort_button.setEnabled(busy)

    def _on_abort_enabled(self, enabled):
        self._extra.abort_button.setEnabled(enabled)

    def _append_fit_results(self, text):
        self._extra.fit_results.append(text)

    def _set_fit_labels_text(self, text):
        self._extra.delete_fitIdx_list.setText(text)

    def _show_cutoff_popup(self, info):
        name = info.get("name")
        self._cutoff.setWindowTitle(
            "Set zoom range for: " + (name if name is not None else "???"))
        self._cutoff.setGeometry(300, 100, 300, 100)
        if self._cutoff.isVisible():
            self._cutoff.close()
        self._cutoff.lineeditXMin.setText(f"{info['xmin']:.1f}")
        self._cutoff.lineeditXMax.setText(f"{info['xmax']:.1f}")
        self._cutoff.lineeditYMin.setText(f"{info['ymin']:.1f}")
        self._cutoff.lineeditYMax.setText(f"{info['ymax']:.1f}")
        if info["dim"] == 2:
            self._cutoff.lineeditZMin.setText(f"{info['zmin']:.1f}")
            self._cutoff.lineeditZMax.setText(f"{info['zmax']:.1f}")
            self._cutoff.layout2d()
        elif info["dim"] == 1:
            self._cutoff.layout1d()
        self._cutoff.show()
