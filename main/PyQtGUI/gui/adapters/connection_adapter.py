from PyQt5.QtWidgets import QMessageBox, QComboBox
from PyQt5.QtWidgets import QCompleter
from PyQt5 import QtCore


class ConnectionAdapter:
    """Bridges ConnectionManager signals to the connect button, spectrum
    list combo, tab reset logic, and error dialogs.  Created by MainWindow
    after the service, wTab, and toolbar exist."""

    def __init__(self, connection_manager, wTab, connect_button,
                 histo_list, remove_cb, parent_widget, logger):
        self._wTab = wTab
        self._connect_button = connect_button
        self._histo_list = histo_list
        self._remove_cb = remove_cb
        self._parent = parent_widget
        self.logger = logger

        connection_manager.spectrumRemoved.connect(
            self._on_spectrum_removed_rest)
        connection_manager.shmViewsInvalidated.connect(
            self._on_shm_views_invalidated)
        connection_manager.connectionRefused.connect(
            self._on_connection_refused)
        connection_manager.connectFailed.connect(
            self._on_connect_failed_dialog)
        connection_manager.spectrumDiscarded.connect(
            self._on_spectrum_discarded)
        connection_manager.connectionStateChanged.connect(
            self._render_connect_state)
        connection_manager.connectAttemptBusy.connect(
            self._on_connect_attempt_busy)
        connection_manager.spectrumListUpdated.connect(
            self._render_spectrum_list)

    def _on_shm_views_invalidated(self):
        for tabIdx in self._wTab.sessions.indices():
            plotVal = self._wTab.plot(tabIdx)
            try:
                nRow, nCol = self._wTab.tabLayout(tabIdx)
                plotVal.InitializeCanvas(nRow, nCol)
                plotVal.isEnlarged = False
                plotVal.selected_plot_index = None
                plotVal.next_plot_index     = -1
                self._wTab.setSelectedPad(tabIdx, None)
                self._wTab.setZoomInfo(tabIdx, None)
            except Exception:
                self.logger.exception(
                    '_on_shm_views_invalidated - tab %s reset failed', tabIdx)
        for tabIdx in self._wTab.sessions.indices():
            for info in self._wTab.tabSlots(tabIdx).values():
                info["spectrum"] = None
                info["axis"]     = None

    def _on_connection_refused(self, msg):
        QMessageBox.warning(self._parent, "Connection refused", msg)

    def _on_connect_failed_dialog(self, msg):
        QMessageBox.critical(self._parent, "Connection failed", msg)

    def _on_spectrum_discarded(self, msg):
        QMessageBox.warning(self._parent, "Spectrum skipped", msg)

    def _render_connect_state(self, state):
        button = self._connect_button
        if state == "connected":
            button.setStyleSheet("background-color:#bcee68;")
            button.setText("Connected")
        elif state == "connecting":
            button.setStyleSheet("background-color:rgb(255, 200, 0);")
            button.setText("Connecting to mirror…")
        else:
            button.setStyleSheet("background-color:rgb(252, 48, 3);")
            button.setText("Disconnected")

    def _on_connect_attempt_busy(self, busy):
        self._connect_button.setEnabled(not busy)

    def _render_spectrum_list(self, names, init):
        self._histo_list.blockSignals(True)
        self._histo_list.clear()
        self._histo_list.addItems(names)
        self._histo_list.blockSignals(False)
        if init:
            self._histo_list.setEditable(True)
            self._histo_list.setInsertPolicy(QComboBox.NoInsert)
            self._histo_list.completer().setCompletionMode(
                QCompleter.PopupCompletion)
            self._histo_list.completer().setFilterMode(
                QtCore.Qt.MatchContains)

    def _on_spectrum_removed_rest(self, name):
        for tabIdx in self._wTab.sessions.indices():
            plotVal = self._wTab.plot(tabIdx)
            to_delete = [
                key for key, value in plotVal.h_dict_geo.items()
                if name in value]
            for key in to_delete:
                if key in self._wTab.tabSlots(tabIdx):
                    spectrum = self._wTab.tabSlots(tabIdx)[key].spectrum
                    if hasattr(spectrum, 'axes'):
                        ax = spectrum.axes
                        self._remove_cb(ax)
                        ax.clear()
                    plotVal.h_dict_geo[key] = "empty"
                    del spectrum
