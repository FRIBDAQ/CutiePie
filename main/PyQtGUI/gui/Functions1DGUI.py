import sys, csv, io, time
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
import CPyConverter as cpy

class Fncts1D(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

    def create_peakChecks(self, npeaks):
        pCheck = QGroupBox("Peak Selection")
        row = 0
        col = 0
        counter = 0
        self.peak_cbox = []
        deflayout = QGridLayout()
        for i in range(npeaks):
            title = "Peak "+str(i+1)
            self.peak_cbox.append(QCheckBox(title,self))
            if (i%4 == 0) and i != 0:
                row += 1
                col = 0
                counter += 1
            else:
                col = i-counter*4
            deflayout.addWidget(self.peak_cbox[i], row, col)

        pCheck.setLayout(deflayout)

        return pCheck

    def create_peakBox(self):
        peakBox = QGroupBox("Peak Finder")

        self.peak_width_label = QLabel("Peak Width (in bins)")
        self.peak_width = QLineEdit()
        self.peak_width.setText("20")
        self.peak_analysis = QPushButton("Scan", self)
        self.peak_analysis.setStyleSheet("background-color:#bcee68;")
        self.peak_analysis_clear = QPushButton("Clear", self)

        self.peak_results_label = QLabel("Output")
        self.peak_results = QTextEdit()
        self.peak_results.setReadOnly(True)

        layy = QHBoxLayout()
        layy.addWidget(self.peak_width_label)
        layy.addWidget(self.peak_width)

        lay = QHBoxLayout()
        lay.addWidget(self.peak_analysis)
        lay.addWidget(self.peak_analysis_clear)

        layout = QVBoxLayout()
        layout.addLayout(layy)
        layout.addLayout(lay)
        layout.addWidget(self.peak_results_label)
        layout.addWidget(self.peak_results)
        layout.addStretch(1)
        peakBox.setLayout(layout)

        return peakBox

    def create_jupBox(self):    
        jupBox = QGroupBox("Jupyter Notebook")        

        self.jup_start = QPushButton("Start", self)
        self.jup_start.setStyleSheet("background-color:#bcee68;")
        self.jup_stop = QPushButton("Stop", self)
        self.jup_df_filename = QLineEdit()
        filename = "df-"+time.strftime("%Y%m%d-%H%M%S")+".gzip"
        self.jup_df_filename.setText(filename)

        self.jup_stop.setEnabled(False)

        layout = QHBoxLayout()
        layout.addWidget(self.jup_start)
        layout.addWidget(self.jup_stop)

        layoutC = QVBoxLayout()
        layoutC.addLayout(layout)
        layoutC.addWidget(self.jup_df_filename)
        jupBox.setLayout(layoutC)
        
        return jupBox        

