import sys, csv, io, time
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PyQt5.QtWidgets import (
    QComboBox, QDialog, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QPushButton, QTextEdit, QVBoxLayout,
)

from services.peak_finder import PEAK_ALGORITHMS

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
        peakBox = QGroupBox("Peak Finder")

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
        # Peak Finder 2: click-to-fit. Start toggles an armed mode (wired in
        # GUI.py) where each pad click fits gaussian+linear around the click.
        peakBox2 = QGroupBox("Peak Finder 2")

        self.peak2_start = QPushButton("Start", self)
        self.peak2_start.setCheckable(True)
        self.peak2_start.setStyleSheet("background-color:#bcee68;")
        self.peak2_start.setToolTip(
            "Arm the selected pad: each left-click fits a gaussian + linear "
            "background around the click")
        self.peak2_clear = QPushButton("Clear", self)
        self.peak2_clear.setToolTip("Remove all fitted peaks and clear the output")

        self.peak2_window_label = QLabel("Window (in bins)")
        self.peak2_window = QLineEdit()
        self.peak2_window.setText("40")

        self.peak2_results_label = QLabel("Output")
        self.peak2_results = QTextEdit()
        self.peak2_results.setReadOnly(True)

        layw = QHBoxLayout()
        layw.addWidget(self.peak2_window_label)
        layw.addWidget(self.peak2_window)

        layb = QHBoxLayout()
        layb.addWidget(self.peak2_start)
        layb.addWidget(self.peak2_clear)

        layout = QVBoxLayout()
        layout.addLayout(layw)
        layout.addLayout(layb)
        layout.addWidget(self.peak2_results_label)
        layout.addWidget(self.peak2_results)
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

