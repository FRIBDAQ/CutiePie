"""The plain dialog widgets, kept out of GUI.py.

Three Qt shells with no MainWindow reference: the tab-rename dialog, the
per-pad cutoff dialog, and the Qt log sink the notebook view subscribes to.
"""

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import (QDialog, QGridLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QVBoxLayout)


class QtLogger(QObject):
    newlog = pyqtSignal(str)

    def __init__(self, parent):
        super(QtLogger, self).__init__(parent)


class TabPopup(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.lineedit = QLineEdit(self)
        self.okButton = QPushButton("Ok", self)
        self.cancelButton = QPushButton("Cancel", self)

        layButt = QHBoxLayout()
        layButt.addWidget(self.okButton)
        layButt.addWidget(self.cancelButton)

        layout = QVBoxLayout()
        layout.addWidget(self.lineedit)
        layout.addLayout(layButt)
        self.setLayout(layout)


class cutoffPopup(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.lineeditXMin = QLineEdit(self)
        self.lineeditXMax = QLineEdit(self)
        self.labelXMin = QLabel(self)
        self.labelXMin.setText("X Min")
        self.labelXMax = QLabel(self)
        self.labelXMax.setText("X Max")
        
        self.lineeditYMin = QLineEdit(self)
        self.lineeditYMax = QLineEdit(self)
        self.labelYMin = QLabel(self)
        self.labelYMin.setText("Y Min")
        self.labelYMax = QLabel(self)
        self.labelYMax.setText("Y Max")
        
        self.lineeditZMin = QLineEdit(self)
        self.lineeditZMax = QLineEdit(self)        
        self.labelZMin = QLabel(self)
        self.labelZMin.setText("Z Min")
        self.labelZMax = QLabel(self)
        self.labelZMax.setText("Z Max")
        
        self.okButton = QPushButton("Ok", self)
        self.cancelButton = QPushButton("Cancel", self)
        self.resetButton = QPushButton("Reset", self)

        self.mainLayout = QGridLayout()

    def setVisibleFields(self, value=True):
        self.labelZMin.setVisible(value)
        self.labelZMax.setVisible(value)
        self.lineeditZMin.setVisible(value)
        self.lineeditZMax.setVisible(value)
        
    def layout1d(self):
        self.setVisibleFields(False)
        fieldsLayoutX = QHBoxLayout()
        fieldsLayoutX.addWidget(self.labelXMin)
        fieldsLayoutX.addWidget(self.lineeditXMin)
        fieldsLayoutX.addWidget(self.labelXMax)
        fieldsLayoutX.addWidget(self.lineeditXMax)
        fieldsLayoutY = QHBoxLayout()
        fieldsLayoutY.addWidget(self.labelYMin)
        fieldsLayoutY.addWidget(self.lineeditYMin)
        fieldsLayoutY.addWidget(self.labelYMax)
        fieldsLayoutY.addWidget(self.lineeditYMax)

        buttonsLayout = QHBoxLayout()        
        buttonsLayout.addWidget(self.okButton)
        buttonsLayout.addWidget(self.resetButton)
        buttonsLayout.addWidget(self.cancelButton)        
        
        self.mainLayout.addLayout(fieldsLayoutX, 1, 0, 1, 0)
        self.mainLayout.addLayout(fieldsLayoutY, 2, 0, 1, 0)
        self.mainLayout.addLayout(buttonsLayout, 3, 0, 1, 0)
        self.setLayout(self.mainLayout)

    def layout2d(self):
        self.setVisibleFields(True)
        fieldsLayoutX = QHBoxLayout()
        fieldsLayoutX.addWidget(self.labelXMin)
        fieldsLayoutX.addWidget(self.lineeditXMin)
        fieldsLayoutX.addWidget(self.labelXMax)
        fieldsLayoutX.addWidget(self.lineeditXMax)
        fieldsLayoutY = QHBoxLayout()
        fieldsLayoutY.addWidget(self.labelYMin)
        fieldsLayoutY.addWidget(self.lineeditYMin)
        fieldsLayoutY.addWidget(self.labelYMax)
        fieldsLayoutY.addWidget(self.lineeditYMax)        
        fieldsLayoutZ = QHBoxLayout()
        fieldsLayoutZ.addWidget(self.labelZMin)
        fieldsLayoutZ.addWidget(self.lineeditZMin)
        fieldsLayoutZ.addWidget(self.labelZMax)
        fieldsLayoutZ.addWidget(self.lineeditZMax)        

        buttonsLayout = QHBoxLayout()        
        buttonsLayout.addWidget(self.okButton)
        buttonsLayout.addWidget(self.resetButton)
        buttonsLayout.addWidget(self.cancelButton)        
        
        self.mainLayout.addLayout(fieldsLayoutX, 1, 0, 1, 0)
        self.mainLayout.addLayout(fieldsLayoutY, 2, 0, 1, 0)
        self.mainLayout.addLayout(fieldsLayoutZ, 3, 0, 1, 0)        
        self.mainLayout.addLayout(buttonsLayout, 4, 0, 1, 0)
        self.setLayout(self.mainLayout)
