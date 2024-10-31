import time
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import CPyConverter as cpy
from Functions1DGUI import Fncts1D # popup peak finder window
from Functions2DGUI import Fncts2D # popup clustering and overlaying an image window
from otherOptions import options # popup clustering and overlaying an image window

class SpecialFunctions(QWidget):
    def __init__(self, *args, **kwargs):
            super(SpecialFunctions, self).__init__(*args, **kwargs)

            self.setWindowTitle("Special Functions")
            
            self.peak = Fncts1D()
            self.imaging = Fncts2D() 
            self.options = options()       
            
            self.layout = QHBoxLayout()
            self.layout.addWidget(self.create_fitBox())

            self.v = QVBoxLayout()
            self.v.addWidget(self.peak.create_peakBox())
            self.v.addWidget(self.peak.create_peakChecks(12))            
            self.v.addWidget(self.peak.create_jupBox())            
            
            self.layout.addLayout(self.v)
            self.layout.addWidget(self.imaging.create_liseBox())
            self.layout.addWidget(self.options.create_options())
            # fillpoints has issues with speed
            clustering = self.imaging.create_clusterBox()
            #clustering.setEnabled(False)
            #self.layout.addWidget(clustering)
            self.setLayout(self.layout)

    def create_fitBox(self):
        fitBox = QGroupBox("Fitting Functions")
        
        self.fit_label = QLabel("Fitting Functions 1D")
        self.fit_label.setToolTip("Gauss\np0*exp(-(x-p1)^2/(2*p2^2))\n\nExpo\np0+p1*exp(x*p2)\n\nPol1\np0+p1*x\n\nPol2\np0+p1*x+p2*x^2\n\nG+Pol1\np5*p0*exp(-(x-p1)^2/(2*p2^2))+(1-p5)*(p3+p4*x)\n\nG+Pol2\np6*p0*exp(-(x-p1)^2/(2*p2^2))+(1-p6)*(p3+p4*x+p5*x^2)\n\nSkeleton\nUser-defined function (needs implementation)")
        self.fit_list = QComboBox()
        self.fit_button = QPushButton("Fit", self)
        self.fit_button.setStyleSheet("background-color:#bcee68;")
        self.fit_range_label = QLabel("Fitting Range")
        self.fit_range_label_min = QLabel("Min X")
        self.fit_range_label_max = QLabel("Max X")
        self.fit_range_min = QLineEdit(self)
        self.fit_range_max = QLineEdit(self)
        self.fit_p0_label = QLabel("p0")
        self.fit_p0 = QLineEdit(self)
        self.fit_p1_label = QLabel("p1")
        self.fit_p1 = QLineEdit(self)
        self.fit_p2_label = QLabel("p2")
        self.fit_p2 = QLineEdit(self)                
        self.fit_p3_label = QLabel("p3")
        self.fit_p3 = QLineEdit(self)
        self.fit_p4_label = QLabel("p4")
        self.fit_p4 = QLineEdit(self)
        self.fit_p5_label = QLabel("p5")
        self.fit_p5 = QLineEdit(self)
        self.fit_p6_label = QLabel("p6")
        self.fit_p6 = QLineEdit(self)                
        self.fit_p7_label = QLabel("p7")
        self.fit_p7 = QLineEdit(self)  
        self.delete_fitIdx_list = QLineEdit(self) 
        self.all_fitIdx_button = QPushButton("Sel. All", self) 
        self.all_fitIdx_button.setToolTip("Return the indexes of all fitted lines on the current plot" )
        self.delete_button = QPushButton("Delete", self)
        self.delete_button.setToolTip("Delete the listed fitted lines" )    

        self.fit_p0.setText("0")
        self.fit_p1.setText("0")
        self.fit_p2.setText("0")
        self.fit_p3.setText("0")
        self.fit_p4.setText("0")
        self.fit_p5.setText("0")
        self.fit_p6.setText("0")
        self.fit_p7.setText("0")                
        
        self.fit_results_label = QLabel("Fit output")
        self.fit_results = QTextEdit()
        self.fit_results.setReadOnly(True)

        # fitting
        v1a = QHBoxLayout()
        v1a.addWidget(self.fit_range_label_min)
        v1a.addWidget(self.fit_range_label_max)
        
        v1b = QHBoxLayout()
        v1b.addWidget(self.fit_range_min)
        v1b.addWidget(self.fit_range_max)

        deflayout = QGridLayout()
        deflayout.addWidget(self.fit_p0_label, 0, 0)
        deflayout.addWidget(self.fit_p0, 0, 1)        
        deflayout.addWidget(self.fit_p1_label, 0, 2)
        deflayout.addWidget(self.fit_p1, 0, 3)        
        deflayout.addWidget(self.fit_p2_label, 1, 0)
        deflayout.addWidget(self.fit_p2, 1, 1)        
        deflayout.addWidget(self.fit_p3_label, 1, 2)       
        deflayout.addWidget(self.fit_p3, 1, 3)        
        deflayout.addWidget(self.fit_p4_label, 2, 0)
        deflayout.addWidget(self.fit_p4, 2, 1)                
        deflayout.addWidget(self.fit_p5_label, 2, 2)
        deflayout.addWidget(self.fit_p5, 2, 3)                
        deflayout.addWidget(self.fit_p6_label, 3, 0)
        deflayout.addWidget(self.fit_p6, 3, 1)                
        deflayout.addWidget(self.fit_p7_label, 3, 2)
        deflayout.addWidget(self.fit_p7, 3, 3)                
        
        v2 = QVBoxLayout()
        v2.addWidget(self.fit_label)
        v2.addWidget(self.fit_list)
        # v2.addWidget(self.fit_button)
        v2.addWidget(self.fit_range_label)
        v2.addLayout(v1a)
        v2.addLayout(v1b)
        v2.addLayout(deflayout)
        v2.addWidget(self.fit_button)
        
        v3 = QVBoxLayout()
        v3.addWidget(self.fit_results_label)
        v3.addWidget(self.fit_results)

        deleteLayout = QGridLayout()
        deleteLayout.addWidget(self.delete_fitIdx_list, 0, 0)
        deleteLayout.addWidget(self.all_fitIdx_button, 0, 1)   
        deleteLayout.addWidget(self.delete_button, 0, 2)   

        vv = QVBoxLayout()
        vv.addLayout(v2)
        vv.addLayout(v3)
        vv.addLayout(deleteLayout)
        
        fitBox.setLayout(vv)

        return fitBox

