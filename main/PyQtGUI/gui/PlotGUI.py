import random
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
#### Bashir imports
import time
#####################

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

debug = False

class Tabs(QTabWidget):
    def __init__(self, loggerMain):
        QTabWidget.__init__(self)
        self.wPlot = {}
        self.logger = loggerMain
        #spectrum_dict {key:geo_index_spectrum, value:{info_spectrum - including spectrum name}}, user can change spectrum_info.
        self.spectrum_dict = {} #dict of dict
        self.spectrum_dict[0] = {}
        self.zoomPlotInfo = {} #Histo [index origin, name] in the zoomed/expanded mode
        self.zoomPlotInfo[0] = [] #dict of list
        self.countClickTab = {} #dict of flag to know if widgets already have dynamic bind
        self.createTabs()
        self.setTabsClosable(True)


    def createTabs(self):
        self.wPlot[0] = Plot(self.logger)
        self.countClickTab[0] = False
        self.setUpdatesEnabled(True)
        self.insertTab(0, self.wPlot[0], "Tab" )
        self.insertTab(1, QWidget(),'  +  ')
        self.selected_plot_index_bak = []
        self.selected_plot_index_bak.append(None)
        #layout is a list that keeps for each tab [numberOfRow, numberOfColumn]
        self.layout = []
        self.layout.append([1,1])


    def addTab(self, index):
        self.wPlot[index] = Plot(self.logger)
        self.logger.debug('addTab -- Inserting tab at index: %d',index)
        # last tab was clicked. add tab
        self.insertTab(index, self.wPlot[index], "Tab %d" %(index+1))
        self.resetTabText()
        self.setCurrentIndex(index)
        self.selected_plot_index_bak.append(None)
        self.layout.append([1,1])
        self.spectrum_dict[index] = {}
        self.zoomPlotInfo[index] = []
        self.countClickTab[index] = False
        # remove the default close button
        self.tabBar().setTabButton(index, QTabBar.RightSide, None)        


    # keep the default naming ordered, when add/delete a tab
    def resetTabText(self):
        # First check if tab + is not the last and relocate.
        tabPlusIndex = None
        for i in range(0, self.count()):
            if "  +  " == self.tabText(i):
                tabPlusIndex = i
                self.logger.debug('resetTabText - Found tab + at index: %d among %d tabs', i, self.count())
                break
        if tabPlusIndex is None: 
            self.logger.warning('resetTabText - Tab + not found')
        else:
            self.tabBar().moveTab(tabPlusIndex,self.count()-1)
            # Make sure none of the tabs have close button visible 
            for i in range(0, self.count()):
                self.tabBar().setTabButton(i, QTabBar.RightSide, None)
        self.orderDefaultName()


    # Order default tab name only ! might be confusing not sure if useful
    def orderDefaultName(self):
        for i in range(0, self.count() - 1):
            # Skip user defined tab name
            if "Tab" != self.tabText(i)[:3]:
                continue
            elif i == 0:
                self.setTabText(i, "Tab")
            else:
                self.setTabText(i, "Tab %d" %(i))


    # Deletes the specified key from a dictionary and reassigns remaining keys.
    # Assuming keys are successive numbers.
    def deleteDictEntry(self, dict, keyToDelete):
        del dict[keyToDelete]
        newDict = {}
        for i, (k, v) in enumerate(dict.items()):
            newDict[i] = v
        return newDict


    def deleteTab(self, index):
        self.logger.info('deleteTab -- at index %d', index)
        # -1 in upper bound to avoid the last "+" tab 
        if index >= 0 and index < self.count()-1:
            # Remove the tab from the QTabWidget
            self.removeTab(index)
            # Update related data structures
            self.wPlot = self.deleteDictEntry(self.wPlot, index)
            self.spectrum_dict = self.deleteDictEntry(self.spectrum_dict, index)
            self.zoomPlotInfo = self.deleteDictEntry(self.zoomPlotInfo, index)
            self.countClickTab = self.deleteDictEntry(self.countClickTab, index)
            del self.selected_plot_index_bak[index]
            del self.layout[index]
            return True
        elif index == self.count() - 1:
            self.logger.warning('deleteTab -- Trying to delete last + tab')
            return False
        

    # Swap tab dict/list items
    def swapItems(self, dictionary, indexFrom, indexTo):
        valTo = dictionary[indexTo]
        dictionary[indexTo] = dictionary[indexFrom]
        dictionary[indexFrom] = valTo
        return dictionary
        

    # Swap all tab dicts
    def swapTabDict(self, indexFrom, indexTo):
        self.logger.info('swapTabDict - indexFrom %d indexTo %d', indexFrom, indexTo)
        if indexTo < self.count()-1:
            self.wPlot = self.swapItems(self.wPlot, indexFrom, indexTo)
            self.spectrum_dict = self.swapItems(self.spectrum_dict, indexFrom, indexTo)
            self.zoomPlotInfo = self.swapItems(self.zoomPlotInfo, indexFrom, indexTo)
            self.countClickTab = self.swapItems(self.countClickTab, indexFrom, indexTo)
            self.selected_plot_index_bak = self.swapItems(self.selected_plot_index_bak, indexFrom, indexTo)
            self.layout = self.swapItems(self.layout, indexFrom, indexTo)
        else :
            self.logger.warning('swapTabDict -- Trying to swap tab dictionaries with last tab') 


class Plot(QWidget):
    def __init__(self, loggerMain, *args, **kwargs):
        super(Plot, self).__init__(*args, **kwargs)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.copyButton = QPushButton("Copy Properties", self)
        self.histoLabel = QLabel(self)
        self.histoLabel.setText("Spectrum: \nX: Y:")
        self.gateLabel = QLabel(self)
        self.gateLabel.setText("Gate applied: \n")
        self.pointerLabel = QLabel(self)
        self.pointerLabel.setText("Pointer: \nX: Y: Count:")
        # self.createSumRegionButton = QPushButton("Summing region", self)
        self.histo_autoscale = QCheckBox("Autoscale",self)
        self.customZoomButton = QPushButton("", self)
        self.customZoomButton.setFixedWidth(30)
        self.customZoomButton.setStyleSheet("QPushButton { background-color: light gray }"
            "QPushButton:pressed { background-color: grey }" )
        self.logButton = QPushButton("Log", self)
        self.logButton.setFixedWidth(50)
        self.logButton.setStyleSheet("QPushButton { background-color: light gray }"
            "QPushButton:pressed { background-color: grey }" )
        self.plusButton = QPushButton("+", self)
        self.plusButton.setFixedWidth(30)
        self.minusButton = QPushButton("-", self)
        self.minusButton.setFixedWidth(30)
        self.cutoffButton = QPushButton("Zoom Range", self)
        self.cutoffButton.setFixedWidth(100)
        self.cutoffButton.setStyleSheet("QPushButton { background-color: light gray }"
            "QPushButton:pressed { background-color: grey }" )
        self.customHomeButton = QPushButton("Reset", self)
        self.customHomeButton.setFixedWidth(70)


        self.logger = loggerMain



        spacer1 = QWidget()
        spacer1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        spacer2 = QWidget()
        spacer2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        

        # save wanted actions before remove all 
        allActions = self.toolbar.actions()
        for idx, x in enumerate(allActions):
            if idx == 5:
                zoomAction = x 
            if idx == 9:
                saveAction = x 
            self.toolbar.removeAction(x)

        # set actions in desired order
        self.toolbar.addWidget(self.histo_autoscale)
        # zoom action triggered by customZoomButton so setVisible(False)
        self.toolbar.addAction(zoomAction)
        zoomAction.setVisible(False)
        # Copy zoom icon for customZoomButton
        zoomIcon = zoomAction.icon()
        self.customZoomButton.setIcon(zoomIcon)
        self.toolbar.addWidget(self.customZoomButton)
        self.toolbar.addWidget(self.cutoffButton)
        self.toolbar.addWidget(self.plusButton)
        self.toolbar.addWidget(self.minusButton)
        self.toolbar.addWidget(self.logButton)
        self.toolbar.addWidget(self.customHomeButton)
        self.toolbar.addWidget(spacer1)
        self.toolbar.addWidget(self.pointerLabel)
        self.toolbar.addWidget(self.histoLabel)
        self.toolbar.addWidget(self.gateLabel)
        self.toolbar.addWidget(spacer2)
        self.toolbar.addWidget(self.copyButton)
        self.toolbar.addAction(saveAction)


        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # plotting variables for each tab
        self.old_row = 1
        self.old_col = 1
        self.old_row_idx = 0
        self.old_col_idx = 0

        self.h_dict = {}
        self.h_dict_geo = {}
        self.axbkg = {}
        self.h_log = {} # bool dict for linear/log axes
        self.h_dim = []
        self.h_lst = []
        self.axis_lst = []
        self.cbar = {}
        self.next_plot_index = -1


        self.selected_plot_index = None
        self.index = 0

        self.autoScale = False
        self.logScale = False
        # drawing tools
        self.isLoaded = False
        self.isFull = False
        self.isEnlarged = False #Tells if the canvas is in single pad mode
        self.isSelected = False
        self.rec = None
        self.recDashed = None
        self.recDashedZoom = None
        #Simon - added flag
        self.isZoomCallback = False
        # self.isZoomInOut = False

        self.zoomPress = False #true when mouse press and drag rectangle, false at release

        # gates
        self.gate_dict = {}
        self.style_dict = {}
        self.artist_dict = {}
        self.artist_list = []
        self.artist1D = {}
        self.artist2D = {}
        self.gateTypeDict = {}
        self.region_dict = {}
        self.regionTypeDict = {}
        self.region_name = ""
        self.region_type = ""
        self.counter = 0
        self.counter_sr = 0
        self.toCreateGate = False
        self.toEditGate = False
        self.toCreateSumRegion = False
        self.xs = []
        self.ys = []
        # #temporary holds gate lines - to control edition with on_singleclick and on_dblclick (reset once gate is pushed to ReST)
        # self.listGateLine = []

        # default canvas
        self.InitializeCanvas(self.old_row,self.old_col)


    def InitializeCanvas(self, row, col, flag = True):
        self.logger.debug('InitializeCanvas -- with dimensions (row, col): %d %d', row, col)

        t0 = time.time()
        # self.axis_grid = [[None for _ in range(col)] for _ in range(row)]

        if flag:
            self.h_dict.clear()
            self.h_dict_geo.clear()

            self.index = 0
            self.idx = 0
        # t1 = time.time()
        # self.figure.clear()
        while self.figure.axes:
            self.figure.delaxes(self.figure.axes[0])

        # t2 = time.time()
        self.InitializeFigure(self.CreateFigure(row, col), row, col, flag)
        # t3 = time.time()

        if row * col <= 16:
            self.figure.tight_layout()

        # t4 = time.time()
        self.canvas.draw()
        # t5 = time.time()

        # print("InitializeCanvas: flag: %2f, clear: %.2f, InitializeFigure: %.2f, tight_layout: %.2f, draw: %.2f" % (t1-t0, t2-t1, t3-t2, t4-t3, t5-t4))


    def CreateFigure(self, row, col):
        self.grid = gridspec.GridSpec(ncols=col, nrows=row, figure=self.figure)
        return self.grid

    def InitializeHistogram(self):
        return {"name": "empty", "dim": 1, "xmin": 0, "xmax": 1, "xbin": 1,
                "ymin": 0, "ymax": 1, "ybin": 1, "parameters": [], "type": "", "scale": False}

    # get value for a dictionary at index x with key y
    def get_histo_key_value(self, d, index, key):
        if key in d[index]:
            return d[index][key]

    # get a list of elements identified by the key for a dictionary
    def get_histo_key_list(self, d, keys):
        lst = []
        for key, value in d.items():
            lst.append(self.get_histo_key_value(d, key, keys))
        return lst

    def InitializeFigure(self, grid, row, col, flag = True):
        self.logger.debug('InitializeFigure')

        for i in range(row):
            for j in range(col):
                # if self.axis_grid[i][j] is None:
                    # self.axis_grid[i][j] = self.figure.add_subplot(self.grid[i, j])

                a = self.figure.add_subplot(grid[i,j])

        if flag:
            self.h_dim.clear()
            self.h_lst.clear()
            self.axis_lst.clear()

            if not self.isLoaded:
                self.old_row = row
                self.old_col = col

            for z in range(self.old_row*self.old_col):
                self.h_dict[z] = self.InitializeHistogram()
                self.h_dict_geo[z] = "empty"
                self.h_log[z] = False
                self.h_lst.append(None)
                self.axis_lst.append(None)
            self.h_dim = self.get_histo_key_list(self.h_dict, "dim")
