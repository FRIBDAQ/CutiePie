import os
import getpass
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class MenuGate(QWidget):

    clearInfoSignal = pyqtSignal()

    def __init__(self, parent=None):
        super(MenuGate, self).__init__(parent)

        self.setWindowTitle("Gate Editor")
        self.resize(350, 200);
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        self.gateActionCreateLabel = QLabel()
        self.gateActionCreateLabel.setText('Create: ')
        self.gateActionCreateLabel.setFixedWidth(50) 
        self.gateActionCreate = QRadioButton()
        self.gateActionCreate.setFixedWidth(15)
        # self.gateActionCreate.clicked.connect(self.onGateActionCreate)

        self.gateActionEditLabel = QLabel()
        self.gateActionEditLabel.setText('Edit: ')
        self.gateActionEditLabel.setFixedWidth(45) 
        self.gateActionEdit = QRadioButton()
        self.gateActionEdit.setFixedWidth(15)
        # self.gateActionEdit.clicked.connect(self.onGateActionEdit)
 
        self.gateNameLabel = QLabel()
        self.gateNameLabel.setText('Name: ')
        self.gateNameList = QComboBox()
        self.gateNameList.setFixedWidth(180) 
        # self.gateName = QLineEdit()
        # self.gateName.setFixedWidth(180)
        ### Bashir changed the gate name
        # self.gateNameList.setCurrentText('None')
        self.gateNameList.setCurrentText('gate-001')

        self.gateTypeLabel = QLabel()
        self.gateTypeLabel.setText('Type: ')
        self.listGateType = QComboBox()
        self.listGateType.setFixedWidth(45)

        self.regionPointLabel = QLabel("Points: ")
        self.regionPoint = QTextEdit()
        self.regionPoint.setReadOnly(True)

        self.ok = QPushButton("Ok", self)       
        self.cancel = QPushButton("Cancel", self)
        ## Bashir added
        self.cancel.clicked.connect(self.close)   # QWidget.close()         
        ################
        ### Bashir commented out preview
        # self.preview = QPushButton("Preview", self)      
        # self.preview.setEnabled(False)
        ###################################################
        # Holds previous points [x,y] for drawing 2d gate 
        self.prevPoint = []   
        # Temporary holds gate lines - to control edition with on_singleclick and on_dblclick (reset once gate is pushed to ReST)
        self.listRegionLine = []
        # Holds the spectrum index on which createGate is called (if one uses the plot index 
        # it is possible to loose the spectrum (e.g. parameters) info if user selects another plot before saving the gate)
        self.gateSpectrumIndex = 0


        layout = QGridLayout()

        layout.addWidget(self.gateActionCreateLabel, 1, 1, 1, 1)
        layout.addWidget(self.gateActionCreate, 1, 2, 1, 1)
        layout.addWidget(self.gateActionEditLabel, 1, 3, 1, 1)
        layout.addWidget(self.gateActionEdit, 1, 4, 1, 1)

        layout.addWidget(self.gateNameLabel, 2, 1, 1, 1)
        layout.addWidget(self.gateNameList, 2, 2, 1, 1)
        layout.addWidget(self.gateTypeLabel, 2, 3, 1, 1)
        layout.addWidget(self.listGateType, 2, 4, 1, 1)  

        layout.addWidget(self.regionPointLabel, 3, 1, 1, 1)           
        layout.addWidget(self.regionPoint, 4, 1, 1, 4)           

        self.lay = QHBoxLayout()
        ### Bashir commented out preview
        # self.lay.addWidget(self.preview)            
        self.lay.addWidget(self.ok)
        self.lay.addWidget(self.cancel)
        layout.addLayout(self.lay, 5, 1, 1, 4)    

        self.setLayout(layout)

    # def onGateActionCreate(self):
    #     self.preview.setEnabled(False)
    #     self.gateActionEdit.setChecked(False)

    # def onGateActionEdit(self):
    #     self.preview.setEnabled(True)
    #     self.gateActionCreate.setChecked(False)


    def clearInfo(self):
        for line in self.listRegionLine:
            line.remove()
        self.listRegionLine.clear()
        self.prevPoint.clear()
        self.listGateType.clear()
        self.regionPoint.clear()
        self.gateSpectrumIndex = 0
        self.gateEditOption = None
        self.gateNameList.clear()
        ### Bashir changed the gate name
        self.gateNameList.setCurrentText('gate-001')
        # self.gateNameList.setCurrentText('None')
        self.regionPoint.setReadOnly(True)


    #override close method, want to reset all info if close with [X]
    def closeEvent(self, event):
        # Use a signal to call clearInfo from main thread and 
        # avoid threading error (due to self.gateNameList.setCurrentText('None'))
        self.clearInfoSignal.emit()
        event.accept()



