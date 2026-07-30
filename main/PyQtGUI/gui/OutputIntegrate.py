import matplotlib
matplotlib.use("Qt5Agg")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QGridLayout, QHBoxLayout, QPushButton, QTableWidget, QWidget,
)

class OutputIntegratePopup(QWidget):
    def __init__(self, parent=None):
        super(OutputIntegratePopup, self).__init__(parent)

        self.setWindowTitle("Integration results")
        self.resize(760, 250);
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        self.resultsText = QTableWidget()

        self.ok = QPushButton("Ok", self)       

        # Holds the spectrum index
        self.regionSpectrumIndex = 0

        layout = QGridLayout()

        layout.addWidget(self.resultsText, 1, 1, 1, 4)           

        self.lay = QHBoxLayout()
        self.lay.addWidget(self.ok)
        layout.addLayout(self.lay, 2, 2, 1, 2)    

        self.setLayout(layout)


    def clearInfo(self):
        self.resultsText.clear()
        self.resultsText.setColumnCount(7)
        self.resultsText.setRowCount(0)
        self.regionSpectrumIndex = 0


    #override close method, want to reset all info if close with [X]
    def closeEvent(self, event):
        self.clearInfo()
        event.accept()
