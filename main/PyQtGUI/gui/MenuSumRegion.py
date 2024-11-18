import os
import getpass
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class MenuSumRegion(QWidget):

    clearInfoSignal = pyqtSignal()

    def __init__(self, parent=None):
        super(MenuSumRegion, self).__init__(parent)

        self.setWindowTitle("Summing Region Editor")
        self.resize(350, 200);
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        # self.sumRegionCreateLabel = QLabel()
        # self.sumRegionCreateLabel.setText('Create: ')
        # self.sumRegionCreateLabel.setFixedWidth(50) 
        # self.sumRegionCreate = QRadioButton()
        # self.sumRegionCreate.setFixedWidth(15)

        #Maybe for later...
        #See a summing region as a temporary region and not as important as a gate
        #Instead of modifying the region give the possibility to delete (and draw a new one)
        # self.gateActionEditLabel = QLabel()
        # self.gateActionEditLabel.setText('Edit: ')
        # self.gateActionEditLabel.setFixedWidth(45) 
        # self.gateActionEdit = QRadioButton()
        # self.gateActionEdit.setFixedWidth(15)
 
        self.sumRegionNameLabel = QLabel()
        self.sumRegionNameLabel.setText('Name: ')
        self.sumRegionNameList = QComboBox()
        self.sumRegionNameList.setFixedWidth(180) 
        self.sumRegionNameList.setCurrentText('None')

        self.regionPointLabel = QLabel("Points: ")
        self.regionPoint = QTextEdit()
        self.regionPoint.setReadOnly(True)

        self.ok = QPushButton("Ok", self)       
        self.cancel = QPushButton("Cancel", self)         
        self.delete = QPushButton("Delete", self)    

        # Holds previous points [x,y] for drawing 2d region 
        self.prevPoint = []   
        # Temporary holds summing region lines - to control edition with on_singleclick and on_dblclick (reset once gate is pushed to ReST)
        self.listRegionLine = []
        # Holds the spectrum index on which createGate is called (if one uses the plot index 
        # it is possible to loose the spectrum (e.g. parameters) info if user selects another plot before saving the region)
        self.sumRegionSpectrumIndex = 0
        #list that holds the region names when popup openned so that one can compare the new name with saved names
        self.sumRegionNameListSaved = []


        layout = QGridLayout()

        # layout.addWidget(self.gateActionCreateLabel, 1, 1, 1, 1)
        # layout.addWidget(self.gateActionCreate, 1, 2, 1, 1)
        # layout.addWidget(self.gateActionEditLabel, 1, 3, 1, 1)
        # layout.addWidget(self.gateActionEdit, 1, 4, 1, 1)

        layout.addWidget(self.sumRegionNameLabel, 1, 1, 1, 1)
        layout.addWidget(self.sumRegionNameList, 1, 2, 1, 1)
        # layout.addWidget(self.gateTypeLabel, 2, 3, 1, 1)
        # layout.addWidget(self.listGateType, 2, 4, 1, 1)  

        layout.addWidget(self.regionPointLabel, 2, 1, 1, 1)           
        layout.addWidget(self.regionPoint, 3, 1, 1, 4)           

        self.lay = QHBoxLayout()
        self.lay.addWidget(self.ok)
        self.lay.addWidget(self.cancel)
        self.lay.addWidget(self.delete)            
        layout.addLayout(self.lay, 4, 1, 1, 4)    

        self.setLayout(layout)


    def clearInfo(self):
        for line in self.listRegionLine:
            line.remove()
        self.listRegionLine.clear()
        self.prevPoint.clear()
        self.regionPoint.clear()
        self.regionPoint.setReadOnly(True)
        self.sumRegionSpectrumIndex = 0
        self.sumRegionNameList.clear()
        self.sumRegionNameList.setCurrentText('None')
        self.sumRegionNameListSaved.clear()


    #override close method, want to reset all info if close with [X]
    def closeEvent(self, event):
        # Use a signal to call clearInfo from main thread and 
        # avoid threading error (due to self.sumRegionNameList.setCurrentText('None'))
        self.clearInfoSignal.emit()
        event.accept()



