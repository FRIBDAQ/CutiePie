from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from services.error_report import report_canvas_errors


class CsvPlotWindow(QWidget):
    """The window the Fit CSV plot lives in. It exists so that plot has a Qt
    owner: made through pyplot instead, the figure would carry a second window
    of its own and stay alive in pyplot's registry after this one closed."""

    def __init__(self, parent=None):
        super(CsvPlotWindow, self).__init__(parent)
        # a window of its own: with a parent and no window flag it would be a
        # small widget inside the main window, in its top-left corner
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle("CSV")
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        report_canvas_errors(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def showWindow(self):
        """Show, and raise an already-open one rather than opening a second."""
        self.show()
        self.raise_()
