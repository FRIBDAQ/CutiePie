import csv
import io
import matplotlib
matplotlib.use("Qt5Agg")

from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)

class OutputPopup(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.fit_results_label = QLabel("Output")

        self.closeButton = QPushButton("Dismiss", self)
        self.saveButton = QPushButton("Save", self)
        self.loadButton = QPushButton("Load", self)
        self.copyButton = QPushButton("Copy", self)                

        layButt = QHBoxLayout()
        layButt.addWidget(self.closeButton)
        layButt.addWidget(self.copyButton)                
        layButt.addWidget(self.loadButton)        
        layButt.addWidget(self.saveButton)

        self.createTable()
        
        layout = QVBoxLayout()
        layout.addWidget(self.fit_results_label)
        layout.addWidget(self.tableWidget)
        layout.addLayout(layButt)
        self.setLayout(layout)

        self.closeButton.clicked.connect(self.close)
        self.saveButton.clicked.connect(self.save)
        self.loadButton.clicked.connect(self.load)
        self.copyButton.clicked.connect(self.copy)                

    def copy(self):
        selection = self.tableWidget.selectedIndexes()
        if selection:
            rows = sorted(index.row() for index in selection)
            columns = sorted(index.column() for index in selection)
            rowcount = rows[-1] - rows[0] + 1
            colcount = columns[-1] - columns[0] + 1
            table = [[''] * colcount for _ in range(rowcount)]
            for index in selection:
                row = index.row() - rows[0]
                column = index.column() - columns[0]
                table[row][column] = index.data()
            stream = io.StringIO()
            csv.writer(stream).writerows(table)
            qApp.clipboard().setText(stream.getvalue())
        
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Open file...", "","CSV Files (*.csv);;All Files (*)", options=options)
        if fileName:
            return fileName
        
    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"Save file...","","CSV Files (*.csv);;All Files (*)", options=options)
        if fileName:
            return fileName

    def load(self):
        fileName = self.openFileNameDialog()
        try:        
            with open(fileName, newline='') as csvfile:
                self.tableWidget.setRowCount(0)
                self.tableWidget.setColumnCount(0)
                reader = csv.reader(csvfile)
                for rowdata in reader:
                    row = self.tableWidget.rowCount()
                    self.tableWidget.insertRow(row)
                    self.tableWidget.setColumnCount(len(rowdata))
                    for column, data in enumerate(rowdata):
                        item = QTableWidgetItem(data)
                        self.tableWidget.setItem(row, column, item)
        except TypeError:
            pass

                
    def save(self):
        fileName = self.saveFileDialog()
        try:
            with open(fileName, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in range(self.tableWidget.rowCount()):
                    rowdata = []
                    for column in range(self.tableWidget.columnCount()):
                        item = self.tableWidget.item(row, column)
                        if item is not None:
                            rowdata.append(str(item.text()))
                        else:
                            rowdata.append('')
                    writer.writerow(rowdata)
        except TypeError:
            pass

    def createTable(self):
        # Create table
        self.tableWidget = QTableWidget()
        self.headers = ["   ID   ", "   Spectrum   ", "   Name   ", "   centroid X   ", "   centroid Y   ", "   FWHM  X   ", "   FWHM  Y   ", "   Area   "]
        self.tableWidget = QTableWidget(0, len(self.headers))
        self.tableWidget.setHorizontalHeaderLabels(self.headers)
        self.tableWidget.move(0,0)
