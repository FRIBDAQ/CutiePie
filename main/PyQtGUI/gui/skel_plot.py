import matplotlib
matplotlib.use("Qt5Agg")

from PyQt5.QtWidgets import (QDialog, QVBoxLayout)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure





class skel_plot(QDialog):
    def __init__(self, *args, **kwargs):
        super(skel_plot, self).__init__(*args, **kwargs)

        # a figure instance to plot on
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # set the layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

    def plot(self):

        # create an axis
        self.ax = self.figure.add_subplot(111)
        # discards the old graph
        self.ax.clear()

        # refresh canvas
        self.canvas.draw_idle()
        
