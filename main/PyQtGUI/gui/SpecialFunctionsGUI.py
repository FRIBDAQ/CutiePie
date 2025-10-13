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
        # self.fit_label.setToolTip("Gauss\np0*exp(-(x-p1)^2/(2*p2^2))\n\nExpo\np0+p1*exp(x*p2)\n\nPol1\np0+p1*x\n\nPol2\np0+p1*x+p2*x^2\n\nG+Pol1\np5*p0*exp(-(x-p1)^2/(2*p2^2))+(1-p5)*(p3+p4*x)\n\nG+Pol2\np6*p0*exp(-(x-p1)^2/(2*p2^2))+(1-p6)*(p3+p4*x+p5*x^2)\n\nSkeleton\nUser-defined function (needs implementation)")
        self.fit_label.setToolTip(
        """Gauss
        p0*exp(-(x-p1)^2/(2*p2^2))

        Expo
        p0+p1*exp(x*p2)

        Pol1
        p0+p1*x

        Pol2
        p0+p1*x+p2*x^2

        G+Pol1
        p5*p0*exp(-(x-p1)^2/(2*p2^2)) + (1-p5)*(p3+p4*x)

        G+Pol2
        p6*p0*exp(-(x-p1)^2/(2*p2^2)) + (1-p6)*(p3+p4*x+p5*x^2)

        AlphaEMG1 (single peak, single tail)
        Model: Exponentially-Modified Gaussian (EMG) integrated over bin width (counts/bin).
        Seeds: p0=A, p1=mu, p2=sigma, p3=tau.  (p4..p7 ignored)
        Extras (appended by GUI): bw (bin width), wmode.
        wmode: 0=unweighted, 1=Poisson(data), 2=Poisson(model, IRLS)

        AlphaEMG2 (two peaks)
        Model: Sum of two AlphaEMG1 peaks (same bin integration).
        Seeds: p0=A1, p1=mu1, p2=sig1, p3=tau11,  p4=A2, p5=mu2, p6=sig2, p7=tau12
        Notes: mu2 = mu1 + dmu with a minimum separation to avoid merging peaks.
            (tau12 can be linked to tau11 via config)
        
        AlphaEMG3 (three peaks)
        Model: Sum of three AlphaEMG1 peaks (with bin integration).
        Seeds: p0..p3 → peak1, p4..p7 → peak2; peak3 is auto-guessed unless provided
        via extended seeds (p8..p11 = A3,mu3,sig3,tau13). Same separation rule.

        AlphaEMG12 (single peak, two tails)
        Model: Double-tail Exponentially-Modified Gaussian (EMG) integrated over bin width

        AlphaEMG22 (two peaks, two tails)
        Model: Two AlphaEMG12 peaks (with bin integration).

        AlphaEMG32 (three peaks, two tails)
        Model: Three AlphaEMG12 peaks (with bin integration). 
        """
        )

        self.fit_list = QComboBox()


        ############################################################
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
        self.fit_p8_label = QLabel("p8")  # Bashir added for alpha22
        self.fit_p8 = QLineEdit(self)
        self.fit_p9_label = QLabel("p9")
        self.fit_p9 = QLineEdit(self)
        self.fit_p10_label = QLabel("p10")
        self.fit_p10 = QLineEdit(self)
        self.fit_p11_label = QLabel("p11")
        self.fit_p11 = QLineEdit(self)
        self.fit_p12_label = QLabel("p12") 
        self.fit_p12 = QLineEdit(self)
        self.fit_p13_label = QLabel("p13")  
        self.fit_p13 = QLineEdit(self)
        self.fit_p14_label = QLabel("p14")
        self.fit_p14 = QLineEdit(self)
        self.fit_p15_label = QLabel("p15")  
        self.fit_p15 = QLineEdit(self)
        self.fit_p16_label = QLabel("p16")  
        self.fit_p16 = QLineEdit(self)
        self.fit_p17_label = QLabel("p17")  
        self.fit_p17 = QLineEdit(self)
        self.fit_p18_label = QLabel("p18")  
        self.fit_p18 = QLineEdit(self)
        self.fit_p19_label = QLabel("p19")  
        self.fit_p19 = QLineEdit(self)
        self.delete_fitIdx_list = QLineEdit(self) 
        self.all_fitIdx_button = QPushButton("Sel. All", self) 
        self.all_fitIdx_button.setToolTip("Return the indexes of all fitted lines on the current plot" )
        self.delete_button = QPushButton("Delete", self)
        self.delete_button.setToolTip("Delete the listed fitted lines" )

        ### Bashir added for alpha spectra ########################
                #### Bashir added for alpha spectra ########################
        # models in the dropdown
        self.fit_list.addItems(["AlphaEMG22", "AlphaEMG32"])
        self.fit_list.currentTextChanged.connect(self._on_model_changed)
        # set initial labeling
        self._on_model_changed("AlphaEMG22")  # or "AlphaEMG32"
        ############################################################

        self.fit_p0.setText("0")
        self.fit_p1.setText("0")
        self.fit_p2.setText("0")
        self.fit_p3.setText("0")
        self.fit_p4.setText("0")
        self.fit_p5.setText("0")
        self.fit_p6.setText("0")
        self.fit_p7.setText("0")
        self.fit_p8.setText("0")  # Bashir added for alpha22
        self.fit_p9.setText("0")
        self.fit_p10.setText("0")
        self.fit_p11.setText("0")                
        self.fit_p12.setText("0") 
        self.fit_p13.setText("0")
        self.fit_p14.setText("0")
        self.fit_p15.setText("0")
        self.fit_p16.setText("0")
        self.fit_p17.setText("0")
        self.fit_p18.setText("0")
        self.fit_p19.setText("0")

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
        deflayout.addWidget(self.fit_p8_label, 4, 0)  # Bashir added for alpha22
        deflayout.addWidget(self.fit_p8, 4, 1)                
        deflayout.addWidget(self.fit_p9_label, 4, 2)
        deflayout.addWidget(self.fit_p9, 4, 3)                
        deflayout.addWidget(self.fit_p10_label, 5, 0)
        deflayout.addWidget(self.fit_p10, 5, 1)                
        deflayout.addWidget(self.fit_p11_label, 5, 2)
        deflayout.addWidget(self.fit_p11, 5, 3)
        deflayout.addWidget(self.fit_p12_label, 6, 0) 
        deflayout.addWidget(self.fit_p12, 6, 1) 
        deflayout.addWidget(self.fit_p13_label, 6, 2)  
        deflayout.addWidget(self.fit_p13, 6, 3)
        deflayout.addWidget(self.fit_p14_label, 7, 0)
        deflayout.addWidget(self.fit_p14, 7, 1)
        deflayout.addWidget(self.fit_p15_label, 7, 2)  
        deflayout.addWidget(self.fit_p15, 7, 3)
        deflayout.addWidget(self.fit_p16_label, 8, 0)  
        deflayout.addWidget(self.fit_p16, 8, 1)
        deflayout.addWidget(self.fit_p17_label, 8, 2)  
        deflayout.addWidget(self.fit_p17, 8, 3)
        deflayout.addWidget(self.fit_p18_label, 9, 0)  
        deflayout.addWidget(self.fit_p18, 9, 1)
        deflayout.addWidget(self.fit_p19_label, 9, 2)  
        deflayout.addWidget(self.fit_p19, 9, 3)              
        
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

    ### Bashir added for alpha spectra ########################
    def _clear_fields(self, indices):
        edits = [
            self.fit_p0, self.fit_p1, self.fit_p2, self.fit_p3, self.fit_p4,
            self.fit_p5, self.fit_p6, self.fit_p7, self.fit_p8, self.fit_p9,
            self.fit_p10, self.fit_p11, self.fit_p12, self.fit_p13, self.fit_p14,
            self.fit_p15, self.fit_p16, self.fit_p17, self.fit_p18, self.fit_p19
        ]
        for i in indices:
            if 0 <= i < len(edits) and edits[i] is not None:
                edits[i].clear()
        
    def _on_model_changed(self, name: str):
        # Map model → nice parameter names (left-to-right: p0..p19)
        maps = {
            "AlphaEMG12": ["A","mu","sigma","tau1","tau2","eta"],

            "AlphaEMG22": [
                "A1","mu1","sigma1","tau11","tau12","eta1",
                "ratio (A2/A1)","mu2","sigma2","tau21","tau22","eta2",
                "bw", "wmode"
            ],
            "AlphaEMG32": [
                "A1","mu1","sigma1","tau11","tau12","eta1",
                "ratio2 (A2/A1)","mu2","sigma2","tau21","tau22","eta2",
                "ratio3 (A3/A1)","mu3","sigma3","tau31","tau32","eta3"
            ],
        }

        labels = [
            self.fit_p0_label, self.fit_p1_label, self.fit_p2_label, self.fit_p3_label, self.fit_p4_label,
            self.fit_p5_label, self.fit_p6_label, self.fit_p7_label, self.fit_p8_label, self.fit_p9_label,
            self.fit_p10_label, self.fit_p11_label, self.fit_p12_label, self.fit_p13_label, self.fit_p14_label,
            self.fit_p15_label, self.fit_p16_label, self.fit_p17_label, self.fit_p18_label, self.fit_p19_label,
        ]
        edits = [
            self.fit_p0, self.fit_p1, self.fit_p2, self.fit_p3, self.fit_p4,
            self.fit_p5, self.fit_p6, self.fit_p7, self.fit_p8, self.fit_p9,
            self.fit_p10, self.fit_p11, self.fit_p12, self.fit_p13, self.fit_p14,
            self.fit_p15, self.fit_p16, self.fit_p17, self.fit_p18, self.fit_p19,
        ]

        names = maps.get(name, [])
        # Update text and visibility
        for i, lab in enumerate(labels):
            use = i < len(names)
            lab.setVisible(use); edits[i].setVisible(use)
            if use:
                lab.setText(names[i])

        # Helpful tooltips for the eta’s:
        tt = "Slow-tail weight η in [0,1] (fixed by user)"
        if name == "AlphaEMG22":
            self._clear_fields([6, 12, 13, 14, 15])
            self.fit_p5_label.setToolTip(tt)    # eta1
            self.fit_p11_label.setToolTip(tt)   # eta2
        elif name == "AlphaEMG32":
            self._clear_fields([6, 12, 13, 14, 15, 16, 17])
            self.fit_p5_label.setToolTip(tt)    # eta1
            self.fit_p11_label.setToolTip(tt)   # eta2
            self.fit_p17_label.setToolTip(tt)   # eta3
