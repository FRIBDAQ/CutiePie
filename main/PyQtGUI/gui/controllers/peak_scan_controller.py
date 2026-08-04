"""Peak Finder 1: scan a pad, list what was found, draw the markers.

Lifted out of MainWindow (ARCH.md §7, D5). The finding itself already lives in
`services/peak_finder.py`; what is here is the layer above it — the widget
reads, the checkable list, and the four red artists each peak owns.

Two things are load-bearing.

The x axis is built from the STORE tier (`binx`/`minx`/`maxx`) and only the
window to clip to comes from the pad's axes. Taking the bin count from the
per-tab view tier instead is what drew 1-D spectra at the wrong x coordinates
once already, and here it would put every marker in the wrong place.

The bulk check/uncheck blocks the list's `itemChanged` while it flips the
states, then syncs the markers in one pass with a single redraw. Unblocked,
every flip re-enters the item handler, which draws the canvas itself: n
redraws where one will do, on a path the user reaches with one button.
"""

import logging

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QListWidgetItem

from services.peak_finder import (PEAK_ALGORITHMS, find_peaks_in_range,
                                  format_peak_labels, format_peak_output)


class PeakScanController:

    def __init__(self, peak_tab, get_current_plot, get_selected_index,
                 get_store_info, get_view_info, plot_controller, logger=None):
        self._peak           = peak_tab            # extraPopup.peak
        self._get_plot       = get_current_plot
        self._get_index      = get_selected_index
        self._get_store_info = get_store_info
        self._get_view_info  = get_view_info
        self._plot_controller = plot_controller
        self.logger = logger or logging.getLogger(__name__)

        # the scan's result, and the artists standing on the pad for it
        self.datax      = None
        self.datay      = None
        self.peaks      = None
        self.properties = None
        self.isChecked  = {}
        self.peak_pos   = {}
        self.peak_vl    = {}
        self.peak_hl    = {}
        self.peak_txt   = {}

    # ------------------------------------------------------------------
    # list <-> marker synchronisation
    # ------------------------------------------------------------------

    def _syncPeakMarker(self, row, checked):
        """Draw or remove one peak's markers to match its list check state."""
        if not checked and self.isChecked.get(row, False):
            try:
                self.removePeak(row)
            except Exception:
                self.logger.debug('_syncPeakMarker - peak artist cleanup failed', exc_info=True)
            self.isChecked[row] = False
        elif checked and not self.isChecked.get(row, False):
            self.drawSinglePeaks(self.peaks, self.properties, self.datay, row)
            self.isChecked[row] = True

    def peakItemChanged(self, item):
        self.logger.info('peakItemChanged')
        row = self._peak.peak_list.row(item)
        self._syncPeakMarker(row, item.checkState() == Qt.Checked)
        self._get_plot().canvas.draw()

    def setAllPeaksChecked(self, checked):
        self.logger.info('setAllPeaksChecked - checked: %s', checked)
        peak_list = self._peak.peak_list
        state = Qt.Checked if checked else Qt.Unchecked
        # block itemChanged while flipping the states, then sync the markers
        # in one pass with a single canvas redraw
        peak_list.blockSignals(True)
        try:
            for row in range(peak_list.count()):
                peak_list.item(row).setCheckState(state)
        finally:
            peak_list.blockSignals(False)
        for row in range(peak_list.count()):
            self._syncPeakMarker(row, checked)
        self._get_plot().canvas.draw()

    def populatePeakList(self):
        """Rebuild the checkable peak list, one row per found peak (no cap —
        replaces the fixed 12-checkbox grid), and draw every marker checked."""
        self.logger.info('populatePeakList')
        # drop markers left over from a previous Scan (the old grid redrew
        # over its stale artist handles, leaking them onto the canvas)
        self.setAllPeaksChecked(False)
        peak_list = self._peak.peak_list
        peak_list.blockSignals(True)
        try:
            peak_list.clear()
            labels = format_peak_labels(self.peaks, self.properties, self.datax)
            for i, label in enumerate(labels):
                item = QListWidgetItem(label)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                peak_list.addItem(item)
                self.isChecked[i] = False
        finally:
            peak_list.blockSignals(False)
        for i in range(len(self.peaks)):
            self._syncPeakMarker(i, True)
        self._get_plot().canvas.draw()

    # ------------------------------------------------------------------
    # artists
    # ------------------------------------------------------------------

    def peakAnalClear(self):
        self.logger.info('peakAnalClear')
        self._peak.peak_results.clear()
        self.removeAllPeaks()
        peak_list = self._peak.peak_list
        peak_list.blockSignals(True)
        try:
            peak_list.clear()
        finally:
            peak_list.blockSignals(False)
        self.resetPeakDict()

    def removePeak(self, i):
        self.logger.info('removePeak')
        self.peak_pos[i][0].remove()
        del self.peak_pos[i]
        self.peak_vl[i].remove()
        del self.peak_vl[i]
        self.peak_hl[i].remove()
        del self.peak_hl[i]
        self.peak_txt[i].remove()
        del self.peak_txt[i]

    def resetPeakDict(self):
        self.logger.info('resetPeakDict')
        self.peak_pos = {}
        self.peak_vl = {}
        self.peak_hl = {}
        self.peak_txt = {}

    def removeAllPeaks(self):
        self.logger.info('removeAllPeaks')
        self.setAllPeaksChecked(False)

    def drawSinglePeaks(self, peaks, properties, data, index):
        self.logger.info('drawSinglePeaks - index, properties: %s, %s', index, properties)
        ax = self._get_view_info("axis", index=self._get_index())
        x = self.datax.tolist()
        self.peak_pos[index] = ax.plot(x[peaks[index]], int(data[peaks[index]]), "v", color="red")
        self.peak_vl[index] = ax.vlines(x=x[peaks[index]], ymin=data[peaks[index]] - properties["prominences"][index], ymax = data[peaks[index]], color = "red")
        self.peak_hl[index] = ax.hlines(y=properties["width_heights"][index], xmin=properties["left_ips"][index], xmax=properties["right_ips"][index], color = "red")
        self.peak_txt[index] = ax.text(x[peaks[index]], int(data[peaks[index]]*1.1), str(int(x[peaks[index]])))

    # ------------------------------------------------------------------
    # the scan
    # ------------------------------------------------------------------

    def update_peak_output(self, peaks, properties):
        self.logger.info('update_peak_output - len(peaks), properties: %s, %s', len(peaks), properties)
        for s in format_peak_output(peaks, properties, self.datax):
            self._peak.peak_results.append(s)

    def analyzePeak(self):
        self.logger.info('analyzePeak')
        try:
            index = self._get_index()
            ax = self._get_view_info("axis", index=index)
            # input points for peak finding
            width = int(self._peak.peak_width.text())
            binx = self._get_store_info("binx", index=index)
            minxREST = self._get_store_info("minx", index=index)
            maxxREST = self._get_store_info("maxx", index=index)

            xtmp = self._plot_controller.createRange(binx, minxREST, maxxREST)
            ytmp = (self._get_store_info("data", index=index)).tolist()

            xmin, xmax = ax.get_xlim()
            algo_name = self._peak.peak_algo.currentText()
            finder = PEAK_ALGORITHMS.get(algo_name, find_peaks_in_range)
            self.logger.debug('analyzePeak - algo, xmin, xmax: %s, %s, %s',
                              algo_name, xmin, xmax)

            self.datax, self.datay, self.peaks, self.properties = \
                finder(xtmp, ytmp, xmin, xmax, width)

            self.update_peak_output(self.peaks, self.properties)
            self.populatePeakList()

        except Exception:
            # Peak analysis is best-effort: a bad width entry, an empty view,
            # or a find_peaks failure must not crash the GUI — but must not be
            # silent either (the user would see nothing happen with no clue why).
            self.logger.exception('analyzePeak - peak analysis failed')
