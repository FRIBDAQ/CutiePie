import time
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QCheckBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QWidget,
)

class CopyProperties(QWidget):
    def __init__(self, *args, **kwargs):
            super(CopyProperties, self).__init__(*args, **kwargs)

            self.setWindowTitle("Copy Attributes")
            self.resize(400, 300);
            
            self.histoLabel = QLabel("histo_name", self)                                    
            self.histoLabel.setAlignment(QtCore.Qt.AlignCenter)
            self.axisLimitX = QCheckBox("Axis limit X")
            self.axisLimitY = QCheckBox("Axis limit Y")            
            self.axisLimLabelX = QLabel("[x1,x2]", self)
            self.axisLimLabelY = QLabel("[y1,y2]", self)            
            self.axisScale = QCheckBox("Axis scale")
            self.axisSLabel = QLabel("Linear", self)            
            self.histoScaleminZ = QCheckBox("Min Z")
            self.histoScaleValueminZ = QLineEdit("0", self)
            self.histoScalemaxZ = QCheckBox("Max Z")
            self.histoScaleValuemaxZ = QLineEdit("256", self)            
            self.histoAll = QCheckBox("Select all properties")
            self.copyLabel = QLabel("Copy to", self)                        
            self.copyLabel.setAlignment(QtCore.Qt.AlignCenter)
            self.copy_log = QFormLayout()
            self.copyWidget = QWidget()
            self.copyWidget.setLayout(self.copy_log)

            self.selectAll = QPushButton("Select all", self)
            self.okAttr = QPushButton("Ok", self)
            self.applyAttr = QPushButton("Apply", self)
            self.cancelAttr = QPushButton("Cancel", self)            

            self.index_og = 0
            self.xlim_og = []
            self.ylim_og = []
            self.scale_og = False
            
            lay1 = QHBoxLayout()
            lay1.addWidget(self.axisLimitX)
            lay1.addWidget(self.axisLimLabelX)

            lay2 = QHBoxLayout()
            lay2.addWidget(self.axisLimitY)
            lay2.addWidget(self.axisLimLabelY)

            lay3 = QHBoxLayout()
            lay3.addWidget(self.axisScale)
            lay3.addWidget(self.axisSLabel)

            lay4 = QHBoxLayout()
            lay4.addWidget(self.okAttr)            
            lay4.addWidget(self.applyAttr)
            lay4.addWidget(self.cancelAttr)

            lay5 = QHBoxLayout()
            lay5.addWidget(self.histoScaleminZ)
            lay5.addWidget(self.histoScaleValueminZ)
            lay5.addWidget(self.histoScalemaxZ)
            lay5.addWidget(self.histoScaleValuemaxZ)                        
            
            layout = QVBoxLayout()
            layout.addWidget(self.histoLabel)
            layout.addLayout(lay1)
            layout.addLayout(lay2)
            layout.addLayout(lay3)
            layout.addLayout(lay5)            
            layout.addWidget(self.histoAll)            
            layout.addWidget(self.copyLabel)
            layout.addWidget(self.copyWidget)
            layout.addWidget(self.selectAll)
            layout.addLayout(lay4)                        
            self.setLayout(layout)

    def closeEvent(self, event):
        discard = ["Ok", "Cancel", "Apply", "Select all", "Deselect all"]
        for instance in self.findChildren(QPushButton):
            if instance.text() not in discard:
                instance.deleteLater()
        self.index_og = 0
        self.xlim_og = []
        self.ylim_og = []
        self.scale_og = False
        event.accept()