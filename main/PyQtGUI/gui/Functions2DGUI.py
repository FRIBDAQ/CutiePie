import sys, csv, io
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
import CPyConverter as cpy

# import joystick widget for moving an image
from JoystickGUI import Joystick

class Fncts2D(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

    def create_clusterBox(self):
        clusterBox = QGroupBox("Clustering 2D")
        
        self.clusterN_label = QLabel("Number of Clusters")
        self.clusterN = QComboBox()
        for i in range(20):
            self.clusterN.addItem(str(i+1))
        self.clusterAlgo_label = QLabel("Clustering Algorithm")
        self.clusterAlgo = QComboBox()        
        self.threshold_label = QLabel("Threshold Level (0)")
        self.threshold_slider = QSlider(QtCore.Qt.Horizontal, self)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(300)
        self.threshold_slider.setTickInterval(1)
        self.threshold_slider.setValue(0)
        self.analyzerButton = QPushButton("Analyze", self)
        
        layoutC = QVBoxLayout()
        layoutC.addWidget(self.clusterN_label)
        layoutC.addWidget(self.clusterN)
        layoutC.addWidget(self.clusterAlgo_label)
        layoutC.addWidget(self.clusterAlgo)
        #layoutC.addWidget(self.threshold_label)
        #layoutC.addWidget(self.threshold_slider)        
        layoutC.addWidget(self.analyzerButton)       
        layoutC.addStretch(1)
        clusterBox.setLayout(layoutC)
        
        return clusterBox

    def create_liseBox(self):    
        liseBox = QGroupBox("Overlay an Image")        

        self.loadButton = QPushButton("Open File", self)
        self.loadLISE_name = QLineEdit(self)
        self.addButton = QPushButton("Add", self)
        self.addButton.setStyleSheet("background-color:#bcee68;")
        self.deleteButton = QPushButton("Delete", self)                        
        self.alpha_label = QLabel("Transparency Level (20 %)")        
        self.alpha_slider = QSlider(QtCore.Qt.Horizontal, self)
        self.zoomX_label = QLabel("Zoom X Level (20 %)")        
        self.zoomX_slider = QSlider(QtCore.Qt.Horizontal, self)
        self.zoomY_label = QLabel("Zoom Y Level (20 %)")        
        self.zoomY_slider = QSlider(QtCore.Qt.Horizontal, self)                
        
        self.alpha_slider.setMinimum(0)
        self.alpha_slider.setMaximum(10)
        self.alpha_slider.setTickInterval(1)
        self.alpha_slider.setValue(2)
        self.zoomX_slider.setMinimum(0)
        self.zoomX_slider.setMaximum(10)
        self.zoomX_slider.setTickInterval(1)
        self.zoomX_slider.setValue(2)
        self.zoomY_slider.setMinimum(0)
        self.zoomY_slider.setMaximum(10)
        self.zoomY_slider.setTickInterval(1)
        self.zoomY_slider.setValue(2)                                        

        # joystick layout
        self.joystick = Joystick()
        self.upButton = QToolButton(self)        
        self.upButton.setArrowType(QtCore.Qt.UpArrow)
        self.upButton.setFixedSize(QtCore.QSize(120,20))
        self.downButton = QToolButton(self)        
        self.downButton.setArrowType(QtCore.Qt.DownArrow)
        self.downButton.setFixedSize(QtCore.QSize(120,20))
        self.leftButton = QToolButton(self)        
        self.leftButton.setArrowType(QtCore.Qt.LeftArrow)
        self.leftButton.setFixedSize(QtCore.QSize(20,120))
        self.rightButton = QToolButton(self)        
        self.rightButton.setArrowType(QtCore.Qt.RightArrow)        
        self.rightButton.setFixedSize(QtCore.QSize(20,120))    

        leftLay = QGridLayout()
        leftLay.addWidget(self.leftButton, 0, 0, 3, 1)

        rightLay = QGridLayout()
        rightLay.addWidget(self.rightButton, 0, 0, 3, 1)        

        centerLay = QGridLayout()
        centerLay.addWidget(self.upButton, 0, 0)
        centerLay.addWidget(self.joystick, 1, 0)
        centerLay.addWidget(self.downButton, 2, 0)        

        joyLay = QHBoxLayout()
        joyLay.addLayout(leftLay)
        joyLay.addLayout(centerLay)
        joyLay.addLayout(rightLay)                
        
        lay1 = QVBoxLayout()        
        lay1.addWidget(self.loadButton)
        lay1.addWidget(self.loadLISE_name)
        lay1.addLayout(joyLay)
        lay1.addWidget(self.alpha_label)
        lay1.addWidget(self.alpha_slider)
        lay1.addWidget(self.zoomX_label)
        lay1.addWidget(self.zoomX_slider)
        lay1.addWidget(self.zoomY_label)
        lay1.addWidget(self.zoomY_slider)                

        lay2 = QHBoxLayout()
        lay2.addWidget(self.addButton)
        lay2.addWidget(self.deleteButton)
        
        lstlayout = QVBoxLayout()
        lstlayout.addLayout(lay1)
        lstlayout.addLayout(lay2)
        lstlayout.addStretch(1)
        liseBox.setLayout(lstlayout)
        
        return liseBox        

