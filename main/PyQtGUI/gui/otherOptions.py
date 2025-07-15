import sys, csv, io, time
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
import CPyConverter as cpy
from PyQt5.QtCore import Qt

class options(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

    def create_options(self):
        pCheck = QGroupBox("Other options")

        self.gateAnnotation = QCheckBox("Gate annotation",self)
        self.gateEditDisable = QCheckBox("Disable gate edition",self)
        self.gateHide = QCheckBox("Hide gate",self)
        self.debugMode = QCheckBox("Debug mode",self)
        self.debugMode.setToolTip("When checked, print debug info into ./debugCutiePie.log")  

               
        self.autoUpdate = QSlider(QtCore.Qt.Horizontal, self)        
        self.autoUpdate.setMinimum(0)
        self.autoUpdate.setMaximum(7)
        self.autoUpdate.setTickInterval(1)
        self.autoUpdate.setValue(7)
        # Proposed update intervals : 
        self.autoUpdateIntervalsUser = ["5 secs", "10 secs", "30 secs", "1 min", "3 mins", "5 mins", "10 mins", "Inf."]
        self.autoUpdateIntervals = [5, 10, 30, 60, 180, 300, 600, 9e9]
        # Interval read by autoUpdateThread
        self.autoUpdateInterval = self.autoUpdateIntervals[self.autoUpdate.value()]
        self.autoUpdateIntervalUser = self.autoUpdateIntervalsUser[self.autoUpdate.value()]
        self.autoUpdateLabel = QLabel("Update every: {}".format(self.autoUpdateIntervalUser))

        layout = QGridLayout()
        layout.addWidget(self.gateAnnotation, 1, 1, 1, 1)
        layout.addWidget(self.gateEditDisable, 2, 1, 1, 1)
        layout.addWidget(self.gateHide, 3, 1, 1, 1)
        layout.addWidget(self.debugMode, 4, 1, 1, 1)
        layout.addWidget(self.autoUpdateLabel, 5, 1, 1, 1)
        layout.addWidget(self.autoUpdate, 6, 1, 1, 1)
        layout.setAlignment(Qt.AlignTop)
        pCheck.setLayout(layout)

        return pCheck


