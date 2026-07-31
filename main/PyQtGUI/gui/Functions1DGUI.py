import time
import matplotlib
matplotlib.use("Qt5Agg")

from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QListWidget, QPushButton, QTableWidget,
    QTextEdit, QVBoxLayout,
)

from services.peak_finder import PEAK_ALGORITHMS, PEAK2_TABLE_COLUMNS

class Fncts1D(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

    def create_peakChecks(self):
        # checkable list replaces the old fixed 12-QCheckBox grid: one row per
        # found peak (no cap), rebuilt by MainWindow on every Scan; itemChanged
        # is wired ONCE in GUI.py (the grid reconnected stateChanged per scan)
        pCheck = QGroupBox("Peak Selection")

        self.peak_all = QPushButton("All", self)
        self.peak_none = QPushButton("None", self)
        self.peak_list = QListWidget(self)

        buttons = QHBoxLayout()
        buttons.addWidget(self.peak_all)
        buttons.addWidget(self.peak_none)
        buttons.addStretch(1)

        deflayout = QVBoxLayout()
        deflayout.addLayout(buttons)
        deflayout.addWidget(self.peak_list)

        pCheck.setLayout(deflayout)

        return pCheck

    def create_peakBox(self):
        # The older scan-based finder, hidden in SpecialFunctionsGUI. The
        # "(scan)" suffix keeps it apart from the click-to-fit box, which now
        # holds the plain "Peak Finder" title.
        peakBox = QGroupBox("Peak Finder (scan)")

        self.peak_width_label = QLabel("Peak Width (in bins)")
        self.peak_width = QLineEdit()
        self.peak_width.setText("20")
        self.peak_algo_label = QLabel("Algorithm")
        self.peak_algo = QComboBox()
        # names come from the service dispatch table so the combo and
        # analyzePeak can never drift apart; index 0 (Mariscotti) is the default
        self.peak_algo.addItems(list(PEAK_ALGORITHMS.keys()))
        self.peak_analysis = QPushButton("Scan", self)
        self.peak_analysis.setStyleSheet("background-color:#bcee68;")
        self.peak_analysis_clear = QPushButton("Clear", self)

        self.peak_results_label = QLabel("Output")
        self.peak_results = QTextEdit()
        self.peak_results.setReadOnly(True)

        layy = QHBoxLayout()
        layy.addWidget(self.peak_width_label)
        layy.addWidget(self.peak_width)

        layalgo = QHBoxLayout()
        layalgo.addWidget(self.peak_algo_label)
        layalgo.addWidget(self.peak_algo)

        lay = QHBoxLayout()
        lay.addWidget(self.peak_analysis)
        lay.addWidget(self.peak_analysis_clear)

        layout = QVBoxLayout()
        layout.addLayout(layy)
        layout.addLayout(layalgo)
        layout.addLayout(lay)
        layout.addWidget(self.peak_results_label)
        layout.addWidget(self.peak_results)
        layout.addStretch(1)
        peakBox.setLayout(layout)

        return peakBox

    def create_peakBox2(self):
        # The click-to-fit peak finder. Start toggles an armed mode (wired in
        # GUI.py) where each pad click fits gaussian+linear around the click.
        # Titled just "Peak Finder" in the UI now that the older scan-based box
        # is hidden; the attribute names keep the peak2/peakBox2 prefix.
        peakBox2 = QGroupBox("Peak Finder")

        self.peak2_start = QPushButton("Start", self)
        self.peak2_start.setCheckable(True)
        self.peak2_start.setStyleSheet("background-color:#bcee68;")
        self.peak2_start.setToolTip(
            "Arm the selected pad: each left-click fits a gaussian + linear "
            "background around the click (fit window chosen automatically)")
        self.peak2_fix = QPushButton("Fix Peak", self)
        self.peak2_fix.setCheckable(True)
        self.peak2_fix.setStyleSheet("background-color:#bcee68;")
        self.peak2_fix.setToolTip(
            "Arm fixed-μ fitting: each left-click fits with the peak centre "
            "pinned exactly at the clicked x (window centred on the click). "
            "Use for a small peak next to a bigger one. Mutually exclusive "
            "with Start")
        self.peak2_delete = QPushButton("Delete", self)
        self.peak2_delete.setToolTip("Remove the selected fit (row) and its curve")
        self.peak2_clear = QPushButton("Clear", self)
        self.peak2_clear.setToolTip("Remove all fitted peaks and clear the table")
        self.peak2_config = QPushButton("Config", self)
        self.peak2_config.setToolTip(
            "Set the max fit window (in bins); empty = no cap. With a cap set, "
            "clicks that can't be fitted are skipped silently")

        # signal + background model for NEW fits (wired in GUI.py). The tail-side
        # box only matters when the signal is Crystal ball. Selecting a fit row
        # syncs these back to that fit's model; changing one re-fits the
        # selected fit and sets the default for the next new fit.
        self.peak2_signal = QComboBox(self)
        self.peak2_signal.addItems(["Gaussian", "Crystal ball"])
        self.peak2_signal.setToolTip("Signal shape used for new fits")
        self.peak2_bg = QComboBox(self)
        self.peak2_bg.addItems(["Linear", "Quadratic", "Cubic"])
        self.peak2_bg.setToolTip("Background shape used for new fits")
        self.peak2_cb_tail = QComboBox(self)
        self.peak2_cb_tail.addItems(["low", "high"])
        self.peak2_cb_tail.setToolTip(
            "Crystal Ball tail side (ignored for a Gaussian signal)")

        # one row per fitted peak; sortable by header; hover a row for the full
        # σ/A/background/window detail. Selecting a row highlights its curve on
        # the pad (wired in GUI.py).
        self.peak2_results_label = QLabel("Fitted peaks")
        self.peak2_table = QTableWidget(0, len(PEAK2_TABLE_COLUMNS))
        self.peak2_table.setHorizontalHeaderLabels(list(PEAK2_TABLE_COLUMNS))
        self.peak2_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.peak2_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.peak2_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.peak2_table.setSortingEnabled(True)
        self.peak2_table.verticalHeader().setVisible(False)
        self.peak2_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # latest status/feedback line ([armed]/[config]/[skip]/[failed]/…)
        self.peak2_status = QLabel("")
        self.peak2_status.setWordWrap(True)

        layb = QHBoxLayout()
        layb.addWidget(self.peak2_start)
        layb.addWidget(self.peak2_fix)
        layb.addWidget(self.peak2_delete)
        layb.addWidget(self.peak2_clear)
        layb.addWidget(self.peak2_config)

        lays = QHBoxLayout()
        lays.addWidget(QLabel("Signal"))
        lays.addWidget(self.peak2_signal)
        lays.addWidget(QLabel("Bg"))
        lays.addWidget(self.peak2_bg)
        lays.addWidget(QLabel("CB tail"))
        lays.addWidget(self.peak2_cb_tail)

        layout = QVBoxLayout()
        layout.addLayout(layb)
        layout.addLayout(lays)
        layout.addWidget(self.peak2_results_label)
        layout.addWidget(self.peak2_table)
        layout.addWidget(self.peak2_status)
        layout.addStretch(1)
        peakBox2.setLayout(layout)

        return peakBox2

    def create_jupBox(self):
        jupBox = QGroupBox("Jupyter Notebook")        

        self.jup_start = QPushButton("Start", self)
        self.jup_start.setStyleSheet("background-color:#bcee68;")
        self.jup_stop = QPushButton("Stop", self)
        self.jup_save = QPushButton("Save", self)
        self.jup_df_filename = QLineEdit()
        filename = "df-"+time.strftime("%Y%m%d-%H%M%S")+".gzip"
        self.jup_df_filename.setText(filename)

        self.jup_stop.setEnabled(False)

        layout = QHBoxLayout()
        layout.addWidget(self.jup_start)
        layout.addWidget(self.jup_stop)
        layout.addWidget(self.jup_save)

        layoutC = QVBoxLayout()
        layoutC.addLayout(layout)
        layoutC.addWidget(self.jup_df_filename)
        jupBox.setLayout(layoutC)
        
        return jupBox        

