#!/usr/bin/env python3os
# import modules and packages

import sys, os, ast, subprocess
import logging, copy, cv2
import threading, time, math, re
from ctypes import *
from copy import copy, deepcopy
import pandas as pd
import numpy as np
import logging, logging.handlers
import CPyConverter as cpy


# import importlib
# import io, pickle, traceback, sys, os, subprocess, ast, csv, gzip
# import signal, logging, ctypes, copy, json, httplib2, cv2
# import threading, itertools, time, multiprocessing, math, re
# from ctypes import *
# from copy import copy, deepcopy
# from itertools import chain, compress, zip_longest
# import pandas as pd
# import numpy as np
# import logging, logging.handlers

sys.path.append(os.getcwd())

# use preprocessor macro __file__ to get the installation directory
# caveat : expects a particular format of installation directory (N.NN-NNN)
instPath = ""
fileDir = os.path.dirname(os.path.abspath(__file__))
subDirList = fileDir.split("/")
for subDir in subDirList:
    if subDir != "":
        instPath += "/"+subDir
        if re.search(r'\d+\.\d{2}-\d{3}', subDir ):
            # specVersion = subDir
            break

sys.path.append(instPath + "/lib")

# removes the webproxy from spdaq machines
os.environ['NO_PROXY'] = ""
os.environ['XDG_RUNTIME_DIR'] = os.getcwd()


from PyQt5 import QtCore, QtNetwork
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtTest import *


from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.mlab as mlab
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import matplotlib.colorbar as mcolorbar
import matplotlib.colors as colors
from matplotlib.backend_bases import *
from matplotlib.artist import Artist

from matplotlib.patches import Polygon, Circle, Ellipse
from matplotlib.path import Path
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib.text as mtext



# List of implementation topics
# 0) Class definition
# 1) Main layout GUI
# 2) Signals
# 3) Implementation of Signals
# 4) GUI on startup
# 5) Connection to REST for gates
# 6) Accessing the ShMem
# 7) Load/save geometry window
# 8) Zoom operations
# 9) Histogram operations
# 10) Gates
# 11) 1D/2D region integration
# 12) Fitting
# 13) Peak Finding
# 14) Clustering
# 15) Overlaying pic
# 16) Jupyter Notebook
# 17) Misc Tools

# import widgets
from MenuAndConfigGUI import * #All widgets of the top menu section
from SpecialFunctionsGUI import SpecialFunctions # all the extra functions we defined
# from OutputGUI import OutputPopup # popup output window
from PlotGUI import Plot # area defined for the histograms
from PlotGUI import Tabs # area defined for the Tabs
from PyREST import PyREST # class interface for SpecTcl REST plugin
from CopyPropertiesGUI import CopyProperties
from connectConfigGUI import ConnectConfiguration #class for the connection configuration popup
from MenuGate import MenuGate #class for the gate creation/edition popup
from MenuSumRegion import MenuSumRegion #class for the gate creation/edition popup
from OutputIntegrate import OutputIntegratePopup #popup for gate/summing region integrate outputs
# from OutputIntegrate import TableModel #table model for popup for gate/summing region

from logger import log, setup_logging, set_logger
from notebook_process import testnotebook, startnotebook, stopnotebook
from WebWindow import WebWindow

#from collapseMenu import Spoiler

SETTING_BASEDIR = "workdir"
SETTING_EXECUTABLE = "exec"
DEBUG = False
DEBOUNCE_DUR = 0.25
t = None

# 0) Class definition
class MainWindow(QMainWindow):

    def __init__(self, factory, fit_factory, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # initialize debug logging 
        logging.basicConfig(datefmt='%d-%b-%y %H:%M:%S')        
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # define streamHandler for logging
        formatterStreamHandler = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.streamHandler = logging.StreamHandler()
        self.streamHandler.setLevel(logging.WARNING)
        self.streamHandler.setFormatter(formatterStreamHandler)

        # define fileHandler for logging
        formatterFileHandler   = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        #when='h', interval=1, backupCount=0 means overwrite log file every 1h
        # 1 backup log file created, 0 gives infinite backup
        self.fileHandler = logging.handlers.TimedRotatingFileHandler(
            filename="debugCutiePie.log", when='m', interval=10, backupCount=1)
        self.fileHandler.setLevel(logging.DEBUG)
        self.fileHandler.setFormatter(formatterFileHandler)

        #add only stream handler here, the fileHandler is added/removed by user action
        self.logger.addHandler(self.streamHandler)
        #following line to avoid main logger printing log in addition to its handlers
        self.logger.propagate = False


        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)

        self.factory = factory
        self.fit_factory = fit_factory

        self.setWindowTitle("CutiePie(QtPy) - It's not a bug, it's a feature (cit.) Qt5 and PyQty5 used under open source terms.")
        self.setMouseTracking(True)


        self.stopAutoUpdateThread = threading.Event() 
        self.skipAutoUpdateThread = threading.Event() 
        self.threadAutoUpdate = False

        self.stopRestThread = threading.Event() 
        self.threadRest = False

        # self.pidPairedSpecTcl  = False


        #######################
        # 1) Main layout GUI
        #######################

        mainLayout = QGridLayout()
        fullLayout = QGridLayout()
        mainLayout.setContentsMargins(10, 10, 10, 0)
        fullLayout.setContentsMargins(0, 0, 0, 0)


        # config menu
        self.wConf = Configuration()
        # self.wConf.setFixedHeight(40)

        # plot widget
        self.wTab = Tabs(self.logger)
        self.wTab.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.wTab.customContextMenuRequested.connect(self.tab_handle_right_click)
        self.wTab.setMovable(True)
        # with setTabsClosable comes a close button, don't want/need it...
        for i in range(self.wTab.count()):
            self.wTab.tabBar().setTabButton(i, QTabBar.RightSide, None)

        self.currentPlot = None

        # gui composition
        self.wConf.setContentsMargins(10, 10, 10, 0)

        fullLayout.addLayout(self.wConf, 1, 0, 1, 0)
        fullLayout.addWidget(self.wTab, 2, 0, 1, 0)


        widget = QWidget()
        widget.setLayout(fullLayout)
        # widget.setLayout(mainLayout)
        self.setCentralWidget(widget)


        # extra popup window
        self.extraPopup = SpecialFunctions()

        # Tab editing popup
        self.tabp = TabPopup()

        # cutoff editing popup
        self.cutoffp = cutoffPopup()

        # summing region popup
        self.sumRegionPopup = MenuSumRegion()

        # integrate gate and summing region popup
        self.integratePopup = OutputIntegratePopup()

        # connection configuration windows
        self.connectConfig = ConnectConfiguration(self.logger)

        # gate window
        self.gatePopup = MenuGate()

        # copy attributes windows
        self.copyAttr = CopyProperties()

        # copy attributes windows
        self.zoomSetRangePopup = zoomPopup()

        # initialize factory from algo_creator
        #self.factory.initialize(self.extraPopup.imaging.clusterAlgo)
        # initialize factory from fit_creator
        self.fit_factory.initialize(self.extraPopup.fit_list)

        # global variables
        #spectrum_dict_rest {key:name_spectrum, value:{info_spectrum}}, created when connect and updated continuously with traces (if any), hidden to user.
        self.spectrum_dict_rest = {}

        #list of summing region line labels of the displayed spectrum {spectrumName : [sumRegionLabel, ...]}
        self.sumRegionDict = {} 

        # default min/max for x,y
        self.minX = 0
        self.maxX = 1024
        self.minY = 0.001
        self.maxY = 1024
        # default gradient for 2d plots
        self.minZ = 0.001
        self.maxZ = 256

        # for peak finding
        self.datax = None
        self.datay = None
        self.peaks = None
        self.properties = None
        self.peak_pos = {}
        self.peak_vl = {}
        self.peak_hl = {}
        self.peak_txt = {}
        self.isChecked = {}

        # overlay
        self.onFigure = False

        # editing gates
        self.epsilon = 5  # max pixel distance to count as a vertex hit

        #dictionary used in case user wants to have one color per gate {gateName: color}
        self.gateColor = {}
        #dictionary used to know the annotation positions to prevent overlay {spectrumName: lowestPosition}
        self.gateAnnotation = {}

        # Bool set by extra options -> differentiate gates 
        # self.annotateGate = False

        #################
        # 2) Signals
        #################

        # top menu signals
        self.wConf.connectButton.clicked.connect(self.connectPopup)
        self.connectConfig.ok.clicked.connect(self.okConnect)
        self.connectConfig.cancel.clicked.connect(self.closeConnect)

        self.wConf.saveButton.clicked.connect(self.saveGeo)
        self.wConf.loadButton.clicked.connect(self.loadGeo)
        self.wConf.exitButton.clicked.connect(self.closeAll)

        # new tab creation
        self.wTab.tabBarClicked.connect(self.clickedTab)
        self.wTab.tabBar().tabMoved.connect(self.movedTab)

        # config menu signals
        self.wConf.histo_geo_add.clicked.connect(self.addPlot)
        self.wConf.histo_geo_update.clicked.connect(self.updatePlot)
        self.wConf.extraButton.clicked.connect(self.spfunPopup)

        self.wConf.histo_geo_row.activated.connect( self.setCanvasLayout )
        self.wConf.histo_geo_col.activated.connect( self.setCanvasLayout )

        self.wConf.createGate.clicked.connect(self.createGate)
        self.wConf.createGate.setEnabled(False)
        self.gatePopup.ok.clicked.connect(self.okGate)
        self.gatePopup.cancel.clicked.connect(self.cancelGate)
        self.gatePopup.gateActionCreate.clicked.connect(self.createGate)
        self.gatePopup.gateActionEdit.clicked.connect(self.editGate)
        self.gatePopup.clearInfoSignal.connect(self.gatePopup.clearInfo)
        self.gatePopup.clearInfoSignal.connect(self.autoUpdateResume) 

        # summing region
        self.wConf.createSumRegionButton.clicked.connect(self.createSumRegion)
        self.sumRegionPopup.ok.clicked.connect(self.okSumRegion)
        self.sumRegionPopup.cancel.clicked.connect(self.cancelSumRegion)
        self.sumRegionPopup.delete.clicked.connect(self.deleteSumRegion)  
        self.sumRegionPopup.clearInfoSignal.connect(self.sumRegionPopup.clearInfo)
        self.sumRegionPopup.clearInfoSignal.connect(self.autoUpdateResume)      


        # self.wConf.editGate.setToolTip("Key bindings for Modify->Edit:\n"
        #                               "'i' insert vertex\n"
        #                               "'d' delete vertex\n")

        #integrate gate and summing region
        self.wConf.integrateGateAndRegion.clicked.connect(self.integrate)
        self.integratePopup.ok.clicked.connect(self.okIntegrate)

        self.tabp.okButton.clicked.connect(self.okTab)
        self.tabp.cancelButton.clicked.connect(self.cancelTab)

        self.cutoffp.okButton.clicked.connect(self.okCutoff)
        self.cutoffp.cancelButton.clicked.connect(self.cancelCutoff)
        self.cutoffp.resetButton.clicked.connect(lambda: self.resetCutoff(True))

        # zoom callback
        self.wTab.wPlot[self.wTab.currentIndex()].canvas.toolbar.actions()[1].triggered.connect(self.zoomCallback)
        # zoom set range popup
        self.zoomSetRangePopup.okButton.clicked.connect(self.okZoomSetRange)
        self.zoomSetRangePopup.cancelButton.clicked.connect(self.cancelZoomSetRange)
        # copy properties
        self.wTab.wPlot[self.wTab.currentIndex()].copyButton.clicked.connect(self.copyPopup)
        # autoscale
        self.wTab.wPlot[self.wTab.currentIndex()].histo_autoscale.clicked.connect(lambda: self.autoScaleAxisBox(None))
        # Custom Zoom button
        self.wTab.wPlot[self.wTab.currentIndex()].customZoomButton.clicked.connect(self.customZoomButtonCallback)
        self.wTab.wPlot[self.wTab.currentIndex()].customZoomButton.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.wTab.wPlot[self.wTab.currentIndex()].customZoomButton.customContextMenuRequested.connect(self.zoom_handle_right_click)
        # plus button
        self.wTab.wPlot[self.wTab.currentIndex()].plusButton.clicked.connect(lambda: self.zoomInOut("in"))
        # minus button
        self.wTab.wPlot[self.wTab.currentIndex()].minusButton.clicked.connect(lambda: self.zoomInOut("out"))
        # cutoff button
        self.wTab.wPlot[self.wTab.currentIndex()].cutoffButton.clicked.connect(self.cutoffButtonCallback)
        self.wTab.wPlot[self.wTab.currentIndex()].cutoffButton.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.wTab.wPlot[self.wTab.currentIndex()].cutoffButton.customContextMenuRequested.connect(self.cutoff_handle_right_click)
        # copy attributes
        self.copyAttr.histoAll.clicked.connect(lambda: self.histAllAttr(self.copyAttr.histoAll))
        self.copyAttr.okAttr.clicked.connect(self.okCopy)
        self.copyAttr.applyAttr.clicked.connect(self.applyCopy)
        self.copyAttr.cancelAttr.clicked.connect(self.closeCopy)
        self.copyAttr.selectAll.clicked.connect(self.selectAll)
        # Custom Home button
        self.wTab.wPlot[self.wTab.currentIndex()].customHomeButton.clicked.connect(lambda: self.customHomeButtonCallback(self.currentPlot.selected_plot_index))
        self.wTab.wPlot[self.wTab.currentIndex()].customHomeButton.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.wTab.wPlot[self.wTab.currentIndex()].customHomeButton.customContextMenuRequested.connect(self.handle_right_click)
        #log button
        self.wTab.wPlot[self.wTab.currentIndex()].logButton.clicked.connect(lambda: self.logButtonCallback(self.currentPlot.selected_plot_index))
        self.wTab.wPlot[self.wTab.currentIndex()].logButton.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.wTab.wPlot[self.wTab.currentIndex()].logButton.customContextMenuRequested.connect(self.log_handle_right_click)

        self.wTab.countClickTab[self.wTab.currentIndex()] = True

        # extra popup
        self.extraPopup.fit_button.clicked.connect(self.fit)
        self.extraPopup.all_fitIdx_button.clicked.connect(self.printFitLineLabels)
        self.extraPopup.delete_button.clicked.connect(self.deleteFit)

        self.extraPopup.peak.peak_analysis.clicked.connect(self.analyzePeak)
        self.extraPopup.peak.peak_analysis_clear.clicked.connect(self.peakAnalClear)

        self.extraPopup.peak.jup_start.clicked.connect(self.jupyterStart)
        self.extraPopup.peak.jup_stop.clicked.connect(self.jupyterStop)

        self.extraPopup.options.gateAnnotation.clicked.connect(self.gateAnnotationCallBack)
        self.extraPopup.options.gateHide.clicked.connect(self.updatePlot)
        self.extraPopup.options.debugMode.clicked.connect(self.debugModeCallBack)
        self.extraPopup.options.autoUpdate.valueChanged.connect(self.autoUpdateStart)

        self.extraPopup.imaging.loadButton.clicked.connect(self.loadFigure)
        self.extraPopup.imaging.addButton.clicked.connect(self.addFigure)
        self.extraPopup.imaging.deleteButton.clicked.connect(self.deleteFigure)
        self.extraPopup.imaging.alpha_slider.valueChanged.connect(self.transFigure)
        self.extraPopup.imaging.zoomX_slider.valueChanged.connect(self.zoomFigureX)
        self.extraPopup.imaging.zoomY_slider.valueChanged.connect(self.zoomFigureY)
        self.extraPopup.imaging.joystick.mousemoved.connect(self.moveFigure)
        self.extraPopup.imaging.upButton.clicked.connect(self.fineUpMove)
        self.extraPopup.imaging.downButton.clicked.connect(self.fineDownMove)
        self.extraPopup.imaging.leftButton.clicked.connect(self.fineLeftMove)
        self.extraPopup.imaging.rightButton.clicked.connect(self.fineRightMove)

        # key press event
        self.wTab.wPlot[self.wTab.currentIndex()].canvas.setFocusPolicy( QtCore.Qt.ClickFocus )
        self.wTab.wPlot[self.wTab.currentIndex()].canvas.setFocus()

        # other signals
        self.resizeID = self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_connect("resize_event", self.on_resize)
        self.pressID = self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_connect("button_press_event", self.on_press)
        self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_connect("button_release_event", self.on_release)

        self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_connect("motion_notify_event", self.histoHover)

        # create helpers
        self.wConf.histo_list.installEventFilter(self)

        # Hotkeys
        # zoom (click-drag)
        self.shortcutZoomDrag = QShortcut(QKeySequence("Alt+Z"), self)
        # self.shortcutZoomDrag.activated.connect(self.zoomKeyCallback)
        self.shortcutZoomDrag.activated.connect(self.customZoomButtonCallback)


        self.currentPlot = self.wTab.wPlot[self.wTab.currentIndex()] # definition of current plot

    ################################
    # 3) Implementation of Signals
    ################################

    #So that signals work for each tab, called in clickedTab()
    def bindDynamicSignal(self):
        self.logger.info('bindDynamicSignal')
        for index, val in self.wTab.countClickTab.items():
            if val:
                self.wTab.wPlot[index].logButton.disconnect()
                self.wTab.wPlot[index].cutoffButton.disconnect()
                self.wTab.wPlot[self.wTab.currentIndex()].histo_autoscale.disconnect()
                self.wTab.wPlot[self.wTab.currentIndex()].customZoomButton.disconnect()
                self.wTab.wPlot[self.wTab.currentIndex()].plusButton.disconnect()
                self.wTab.wPlot[self.wTab.currentIndex()].minusButton.disconnect()
                self.wTab.wPlot[self.wTab.currentIndex()].copyButton.disconnect()
                self.wTab.wPlot[self.wTab.currentIndex()].customHomeButton.disconnect()
                self.wTab.countClickTab[index] = False

        self.wTab.wPlot[self.wTab.currentIndex()].canvas.toolbar.actions()[1].triggered.connect(self.zoomCallback)
        self.wTab.wPlot[self.wTab.currentIndex()].histo_autoscale.clicked.connect(lambda: self.autoScaleAxisBox(None))
        self.wTab.wPlot[self.wTab.currentIndex()].customZoomButton.clicked.connect(self.customZoomButtonCallback)
        self.wTab.wPlot[self.wTab.currentIndex()].customZoomButton.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.wTab.wPlot[self.wTab.currentIndex()].customZoomButton.customContextMenuRequested.connect(self.zoom_handle_right_click)
        self.wTab.wPlot[self.wTab.currentIndex()].plusButton.clicked.connect(lambda: self.zoomInOut("in"))
        self.wTab.wPlot[self.wTab.currentIndex()].minusButton.clicked.connect(lambda: self.zoomInOut("out"))
        self.wTab.wPlot[self.wTab.currentIndex()].cutoffButton.clicked.connect(self.cutoffButtonCallback)
        self.wTab.wPlot[self.wTab.currentIndex()].cutoffButton.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.wTab.wPlot[self.wTab.currentIndex()].cutoffButton.customContextMenuRequested.connect(self.cutoff_handle_right_click)
        self.wTab.wPlot[self.wTab.currentIndex()].copyButton.clicked.connect(self.copyPopup)
        self.wTab.wPlot[self.wTab.currentIndex()].customHomeButton.clicked.connect(lambda: self.customHomeButtonCallback(self.currentPlot.selected_plot_index))
        self.wTab.wPlot[self.wTab.currentIndex()].customHomeButton.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.wTab.wPlot[self.wTab.currentIndex()].customHomeButton.customContextMenuRequested.connect(self.handle_right_click)
        self.wTab.wPlot[self.wTab.currentIndex()].logButton.clicked.connect(lambda: self.logButtonCallback(self.currentPlot.selected_plot_index))
        self.wTab.wPlot[self.wTab.currentIndex()].logButton.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.wTab.wPlot[self.wTab.currentIndex()].logButton.customContextMenuRequested.connect(self.log_handle_right_click)

        self.resizeID = self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_connect("resize_event", self.on_resize)
        self.pressID = self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_connect("button_press_event", self.on_press)
        self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_connect("button_release_event", self.on_release)

        self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_connect("motion_notify_event", self.histoHover)

        self.wTab.countClickTab[self.wTab.currentIndex()] = True


    def connect(self):
        self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_connect("button_press_event", self.on_press)


    def disconnect(self):
        self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_disconnect(self.pressID)


    #Event filter for search in histo_list widget
    def eventFilter(self, obj, event):
        if (obj == self.wConf.histo_list or self.gatePopup.gateNameList) and event.type() == QtCore.QEvent.HoverEnter:
            self.onHovered(obj)
        return super(MainWindow, self).eventFilter(obj, event)


    def onHovered(self, obj):
        if (obj == self.wConf.histo_list):
            self.wConf.histo_list.setToolTip(self.wConf.histo_list.currentText())
        elif (obj == self.gatePopup.gateNameList):
            self.gatePopup.gateNameList.setToolTip(self.gatePopup.gateNameList.currentText())


    #Display information when hover spectrum
    def histoHover(self, event):
        try:
            index = 0
            if not event.inaxes: return

            index = list(self.currentPlot.figure.axes).index(event.inaxes)
            xTitle = self.getSpectrumInfoREST("parameters", index=index)[0]
            coordinates = self.getPointerInfo(event, "coordinates", index)
            type = self.getSpectrumInfoREST("type", index=index)
            if self.getSpectrumInfoREST("dim", index=index) == 1:
                if type == "g1" :
                    xTitle = self.getSpectrumInfoREST("parameters", index=index)[0] + ", ..."
                self.currentPlot.histoLabel.setText("Spectrum: "+self.nameFromIndex(index)+"\nX: "+xTitle)
                self.currentPlot.pointerLabel.setText(f"Pointer:\nX: {coordinates[0]:.2f} Y: {coordinates[1]:.0f} Count: {coordinates[2]:.0f}")
            elif self.getSpectrumInfoREST("dim", index=index) == 2:
                yTitle = self.getSpectrumInfoREST("parameters", index=index)[1]
                if type == "g2" or type == "m2" or type == "gd":
                    xTitle = self.getSpectrumInfoREST("parameters", index=index)[0] + ", ..."
                    yTitle = self.getSpectrumInfoREST("parameters", index=index)[1] + ", ..."
                self.currentPlot.histoLabel.setText("Spectrum: "+self.nameFromIndex(index)+"\nX: "+xTitle+" Y: "+yTitle) 
                self.currentPlot.pointerLabel.setText(f"Pointer:\nX: {coordinates[0]:.2f} Y: {coordinates[1]:.2f}  Count: {coordinates[2]:.0f}")        
                if type == "s" :
                    xTitle = self.getSpectrumInfoREST("parameters", index=index)[0] + ", ..."
                    self.currentPlot.histoLabel.setText("Spectrum: "+self.nameFromIndex(index)+"\nX: "+xTitle) 
                    self.currentPlot.pointerLabel.setText(f"Pointer:\nX: {coordinates[0]:.2f} Y: {coordinates[1]:.2f}  Count: {coordinates[2]:.0f}") 
            gateName = self.getAppliedGateName(index=index)
            if gateName is not None:
                self.currentPlot.gateLabel.setText("Gate applied: "+gateName+"\n") 
            else :
                self.currentPlot.gateLabel.setText("Gate applied: \n") 
        except:
            # self.logger.debug('histoHover - exception', exc_info=True)
            self.currentPlot.histoLabel.setText("Spectrum: \nX: Y:")
            self.currentPlot.pointerLabel.setText(f"Pointer:\nX: Y: Count: ")
            self.currentPlot.gateLabel.setText("Gate applied: \n")


    #called in histoHover, return the bin position under mouse pointer
    def getPointerInfo(self, event, info, index):
        result = ['','','']
        if self.getEnlargedSpectrum():
            index = self.getEnlargedSpectrum()[0]
        try:
            ax = self.getSpectrumInfo("axis", index=index)
            dim = self.getSpectrumInfoREST("dim", index=index)
            minx = self.getSpectrumInfoREST("minx", index=index)
            maxx = self.getSpectrumInfoREST("maxx", index=index)
            binx = self.getSpectrumInfoREST("binx", index=index)
            data = self.getSpectrumInfoREST("data", index=index)
            if ax is None or len(data) <= 0:
                return result
            x, y = ax.transData.inverted().transform([event.x, event.y])
            stepx = (float(maxx)-float(minx))/float(binx)
            binminx = int((x-minx)/stepx)
            if dim == 1:
                if "coordinates" == info:
                    count = data[binminx+1:binminx+2]
                    # y = data[binminx:binminx+1]
                    result = [x,y,count[0]]
                elif "bins" == info:
                    result = [binminx,'','']
            elif dim == 2:
                if "coordinates" == info:
                    miny = self.getSpectrumInfoREST("miny", index=index)
                    maxy = self.getSpectrumInfoREST("maxy", index=index)
                    biny = self.getSpectrumInfoREST("biny", index=index)
                    stepy = (float(maxy)-float(miny))/float(biny)
                    binminy = int((y-miny)/stepy)
                    #ndarray [row][column]
                    z = data[binminy:binminy+1, binminx:binminx+1]
                    result = [x,y,z[0][0]]
                elif "bins" == info:
                    result = [binminx, binminy,'']
        except NameError:
            self.logger.error('getPointerInfo - NameError exception', exc_info=True)
            raise
        return result
     

    def on_resize(self, event):
        self.logger.info('on_resize')
        self.currentPlot.figure.tight_layout()
        self.currentPlot.canvas.draw()


    # Introduced for endding zoom action (toolbar) on release
    # For this purpose dont need to check if release outside of axis (it can happen)
    # small delay introduced such that updatePlotLimits is executed after on_release.
    def on_release(self, event):
        self.logger.info('on_release - self.currentPlot.zoomPress: %s',self.currentPlot.zoomPress)
        # print("Simon - on_release - self.currentPlot.zoomPress",self.currentPlot.zoomPress,self.currentPlot.canvas.toolbar.actions()[1].isChecked())
        if self.currentPlot.zoomPress:
            self.currentPlot.canvas.toolbar.actions()[1].triggered.emit()
            self.currentPlot.canvas.toolbar.actions()[1].setChecked(False)
            self.currentPlot.customZoomButton.setDown(False)
            threadLimits = threading.Thread(target=self.updatePlotLimits, args=(0.1,))
            threadLimits.start()
            self.currentPlot.zoomPress = False


    # when mouse pressed in main window
    # Introduced for endding zoom action (toolbar) see on_release and on_press too
    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.logger.info('mousePressEvent - self.currentPlot.zoomPress: %s',self.currentPlot.zoomPress)
        # print("Simon - mousePressEvent - ",self.currentPlot.zoomPress,self.currentPlot.canvas.toolbar.actions()[1].isChecked())
        if not self.currentPlot.zoomPress : return 
        #height and width determined empirically... better if overestimated because event handled by on_press in that case
        #these values doesnt change with window resizing but could change if decide to change the layout.
        footH = 15 
        headerH = 130 
        sidesW = 10 

        leftLimit = sidesW
        rightLimit = self.geometry().width() - sidesW
        topLimit = headerH
        bottomLimit = self.geometry().height() - footH

        withinLimits = True if (event.x() in range(leftLimit,rightLimit)) and (event.y() in range(topLimit,bottomLimit)) else False

        if not withinLimits or event.button == 3:
            self.currentPlot.canvas.toolbar.actions()[1].triggered.emit()
            self.currentPlot.canvas.toolbar.actions()[1].setChecked(False)
            self.currentPlot.customZoomButton.setDown(False)
            self.currentPlot.zoomPress = False


    #callback for button_press_event
    def on_press(self, event):
        self.logger.info('on_press')
        # print("Simon - on_press - ",self.currentPlot.zoomPress,self.currentPlot.canvas.toolbar.actions()[1].isChecked())
        #if initate zoom (magnifying glass) but dont press in axes, reset the action
        if self.currentPlot.zoomPress and not event.inaxes: 
            self.currentPlot.canvas.toolbar.actions()[1].triggered.emit()
            self.currentPlot.canvas.toolbar.actions()[1].setChecked(False)
            self.currentPlot.customZoomButton.setDown(False)
            self.currentPlot.zoomPress = False

        if not event.inaxes: return

        # if gatePopup or sumRegionPopup exit with [X], want to reset everything as if gate/SumReg Editor hasn't been openned
        self.cleanPopupExit(False)

        # selected a colorbar, dont want to interact with it
        if "colorbar_" in event.inaxes.get_label():
            return
        
        index = list(self.currentPlot.figure.axes).index(event.inaxes)

        if self.currentPlot.isEnlarged:
            index = self.wTab.selected_plot_index_bak[self.wTab.currentIndex()]
        self.currentPlot.selected_plot_index = index

        if event.dblclick:
            if self.currentPlot.toCreateGate or self.currentPlot.toCreateSumRegion :
                #dont need to do anything special here, the 2d line is closed when click okGate
                # self.on_dblclick_gate(event, index)
                pass
            elif self.currentPlot.toEditGate:
                self.on_dblclick_gate_edit(event, index)
            else :
                self.on_dblclick(index)
        else :
            if self.currentPlot.toCreateGate :
                #1: left mouse button, 3: right mouse button
                if event.button == 1:
                    self.on_singleclick_gate(event, index)
                if event.button == 3:
                    self.on_singleclick_gate_right(index)
            elif self.currentPlot.toEditGate:
                self.on_singleclick_gate_edit(event)
            elif self.currentPlot.toCreateSumRegion :
                #1: left mouse button, 3: right mouse button
                if event.button == 1:
                    self.on_singleclick_sumRegion(event, index)
                if event.button == 3:
                    self.on_singleclick_sumRegion_right(index)
            else :
                self.on_singleclick(index)


    #not used for now
    # def on_dblclick_gate(self, event, index):
    #     pass


    #called by on_press when not in create/edit Gate mode
    def on_singleclick(self, index):            
        self.logger.info('on_singleclick - index: %s', index)
        # change the log button status manually only here according to spectrum info
        # so that when one clicks on a spectrum the button shows if log or not
        axisIsLog = self.getSpectrumInfo("log", index=index)
        wPlot = self.currentPlot
        logBut = wPlot.logButton
        if axisIsLog :
            logBut.setDown(True)
        else :
            logBut.setDown(False) 
        # similar to log button, cutoff button change status according to spectrum info
        cutoffVal = self.getSpectrumInfo("cutoff", index=index)
        if cutoffVal is not None and len(cutoffVal)>1 and (cutoffVal[0] is not None or cutoffVal[1] is not None):
            wPlot.cutoffButton.setDown(True)
        else :
            wPlot.cutoffButton.setDown(False)
            self.resetCutoff(False)

        # If we are not zooming on one histogram we can select one histogram
        # and a red rectangle will contour the plot
        if self.currentPlot.isEnlarged == False:
            self.logger.debug('on_singleclick - isEnlarged FALSE')
            self.removeRectangle()
            self.currentPlot.isSelected = True
            self.currentPlot.next_plot_index = self.currentPlot.selected_plot_index
            self.currentPlot.rec = self.createRectangle(self.currentPlot.figure.axes[index])
            #tried to blit here but not successful (?) important delay for canvas with many plots
            self.currentPlot.canvas.draw()


    # find the closest bin edge position to the input position
    def closestBinPos(self, index=None, x=None, y=None):
        result = None 
        if index is None :
            self.logger.debug('closestBinPos - index is None')
            return result

        dim = self.getSpectrumInfoREST("dim", index=index)
        minx = self.getSpectrumInfoREST("minx", index=index)
        maxx = self.getSpectrumInfoREST("maxx", index=index)
        binx = self.getSpectrumInfoREST("binx", index=index)
        stepx = (float(maxx)-float(minx))/float(binx)

        if dim == 1 and x is not None:
            # to round int essential, will give the closest bin edge
            nXbin = round((x-minx)/stepx, 0)
            xbinPos = minx + nXbin*stepx
            result = xbinPos
            # print("Simon - closestBinPos - dim1 - ", minx, maxx, stepx, nXbin, xbinPos )
        if dim == 2 and x is not None and y is not None:
            miny = self.getSpectrumInfoREST("miny", index=index)
            maxy = self.getSpectrumInfoREST("maxy", index=index)
            biny = self.getSpectrumInfoREST("biny", index=index)
            stepy = (float(maxy)-float(miny))/float(biny)
            # to round int essential, will give the closest bin edge
            nXbin = round((x-minx)/stepx, 0)
            xbinPos = minx + nXbin*stepx
            nYbin = round((y-miny)/stepy, 0)
            ybinPos = miny + nYbin*stepy
            result = xbinPos, ybinPos
        else:
            pass

        return result 


    #called by on_press when create gate mode
    #add new 2DLines in ax and set text points of gatePopup
    def on_singleclick_gate(self, event, index):
        self.logger.info('on_singleclick_gate - index: %s', index)
        if self.currentPlot.isEnlarged == True:
            dim = self.getSpectrumInfoREST("dim", index=index)
            gateType = self.gatePopup.listGateType.currentText()
            if dim == 1:
                #second arg = 0 not used for 1d
                #l = self.addLine(self.closestBinPos(index, float(event.xdata)), 0, index)
                l = self.addLine(float(event.xdata), 0, index)
                self.gatePopup.listRegionLine.append(l)
                # removes the lines n-1 (n is the latest) from the plot
                if len(self.gatePopup.listRegionLine) > 2:
                    self.removePrevLine()
                
                lineText = ""
                for nbLine in range(len(self.gatePopup.listRegionLine)):
                    if nbLine == 0:
                        lineText = lineText + (f"{nbLine}: X= {self.gatePopup.listRegionLine[nbLine].get_xdata()[0]:.5f}") 
                    else :
                        lineText = lineText + (f"\n{nbLine}: X= {self.gatePopup.listRegionLine[nbLine].get_xdata()[0]:.5f}") 
                self.gatePopup.regionPoint.clear()
                self.gatePopup.regionPoint.insertPlainText(lineText)

            elif dim == 2:
                #If exist, remove line between the last and first points of the contour (closing_segment), except for band gate type
                tempLine = [line for line in self.gatePopup.listRegionLine if line.get_label() == "closing_segment"]
                if gateType not in ["b", "gb"] and len(tempLine) == 1 :
                    tempLine[0].remove()
                    self.gatePopup.listRegionLine.pop()

                # Add a new line based on clicked location
                #xBinned, yBinned= self.closestBinPos(index, float(event.xdata), float(event.ydata))
                #l = self.addLine(xBinned, yBinned, index)
                #xBinned, yBinned= self.closestBinPos(index, float(event.xdata), float(event.ydata))
                l = self.addLine(float(event.xdata), float(event.ydata), index)
                if l is not None :
                    self.gatePopup.listRegionLine.append(l)

                # Add text in gatePopup
                lineNb = len(self.gatePopup.listRegionLine)
                lineText = ""
                for nbLine in range(lineNb):
                    if nbLine == 0:
                            lineText = lineText + (f"{nbLine}: X= {self.gatePopup.listRegionLine[nbLine].get_xdata()[0]:.5f}   Y= {self.gatePopup.listRegionLine[nbLine].get_ydata()[0]:.5f}") 
                    else :
                        lineText = lineText + (f"\n{nbLine}: X= {self.gatePopup.listRegionLine[nbLine].get_xdata()[0]:.5f}   Y= {self.gatePopup.listRegionLine[nbLine].get_ydata()[0]:.5f}") 
                if lineNb == 0 :
                    lineText = lineText + (f"{lineNb}: X= {float(event.xdata):.5f}   Y= {float(event.ydata):.5f}") 
                else :
                    lineText = lineText + (f"\n{lineNb}: X= {float(event.xdata):.5f}   Y= {float(event.ydata):.5f}") 
                self.gatePopup.regionPoint.clear()
                self.gatePopup.regionPoint.insertPlainText(lineText)

                # Close contour, draw a line between the last and first points, except for band gate type
                if gateType not in ["b", "gb"] and lineNb > 1 :
                    # Special label to identify this segment elsewhere in the code
                    label = "closing_segment"
                    #xBinned, yBinned= self.closestBinPos(index, self.gatePopup.listRegionLine[0].get_xdata()[0], self.gatePopup.listRegionLine[0].get_ydata()[0])
                    #l = self.addLine(xBinned, yBinned, index, label)
                    l = self.addLine(self.gatePopup.listRegionLine[0].get_xdata()[0], self.gatePopup.listRegionLine[0].get_ydata()[0], index, label)
                    if l is not None :
                        self.gatePopup.listRegionLine.append(l)

                # for line in self.gatePopup.listRegionLine:
                #     print("Simon on_singleclick_gate - line -", line, line.get_label())

            self.currentPlot.canvas.draw()


    # called by on_press, right click when creating a gate is removing the last point and so change contour lines and update point text in gatePopup
    def on_singleclick_gate_right(self, index):
        self.logger.info('on_singleclick_gate_right - index: %s',index)
        if self.currentPlot.isEnlarged == True:
            
            dim = self.getSpectrumInfoREST("dim", index=index)
            gateType = self.gatePopup.listGateType.currentText()
            gateTypeList1 = ["c", "gc"]
            gateTypeList2 = ["b"]
            
            if dim == 2:
                if gateType in gateTypeList1:
                    lineNb = len(self.gatePopup.listRegionLine) 
                    # Reset prevPoint for next new line to first point of line n-1 
                    try:
                        self.gatePopup.prevPoint = [self.gatePopup.listRegionLine[-2].get_xdata()[0], self.gatePopup.listRegionLine[-2].get_ydata()[0]]
                    except IndexError:
                        self.logger.debug('on_sinon_singleclick_gate_rightgleclick_sumRegion_right - IndexError', exc_info=True)
                        return
                    # if contour has only 3 lines, remove the two last lines 
                    if lineNb == 3 :
                        for i in range(2):
                            self.gatePopup.listRegionLine[-1].remove()
                            self.gatePopup.listRegionLine.pop(-1)
                    # save n-2 line, remove line n-1 and change first point of the closing_segment to last point of saved n-2 line
                    elif lineNb > 3:
                        tempLine = self.gatePopup.listRegionLine[-3]
                        self.gatePopup.listRegionLine[-2].remove()
                        self.gatePopup.listRegionLine.pop(-2)
                        self.gatePopup.listRegionLine[-1].set_xdata([tempLine.get_xdata()[1], self.gatePopup.listRegionLine[0].get_xdata()[0]]) 
                        self.gatePopup.listRegionLine[-1].set_ydata([tempLine.get_ydata()[1], self.gatePopup.listRegionLine[0].get_ydata()[0]]) 
                
                # For band type we dont want to reconnect the extremities
                if gateType in gateTypeList2:
                    lineNb = len(self.gatePopup.listRegionLine) 
                    # remove the last line
                    if lineNb >= 2 :                   
                        # Reset prevPoint for next new line to last point of line n-2
                        self.gatePopup.prevPoint = [self.gatePopup.listRegionLine[-2].get_xdata()[1], self.gatePopup.listRegionLine[-2].get_ydata()[1]]
                        self.gatePopup.listRegionLine[-1].remove()
                        self.gatePopup.listRegionLine.pop(-1)

                # Add text in gatePopup
                lineNb = len(self.gatePopup.listRegionLine)
                lineText = ""
                for nbLine in range(lineNb):
                    if nbLine == 0:
                            lineText = lineText + (f"{nbLine}: X= {self.gatePopup.listRegionLine[nbLine].get_xdata()[0]:.5f}   Y= {self.gatePopup.listRegionLine[nbLine].get_ydata()[0]:.5f}") 
                    else :
                        lineText = lineText + (f"\n{nbLine}: X= {self.gatePopup.listRegionLine[nbLine].get_xdata()[0]:.5f}   Y= {self.gatePopup.listRegionLine[nbLine].get_ydata()[0]:.5f}") 
                if lineNb == 1 :
                    lineText = lineText + (f"\n{lineNb}: X= {self.gatePopup.listRegionLine[0].get_xdata()[1]:.5f}   Y= {self.gatePopup.listRegionLine[0].get_ydata()[1]:.5f}") 
                self.gatePopup.regionPoint.clear()
                self.gatePopup.regionPoint.insertPlainText(lineText)

            self.currentPlot.canvas.draw()



    #called by on_press when create summing region mode
    #add new 2DLines in ax and set text points of sumRegionPopup
    #similar to on_singleclick_gate, should merge both (difference in isEnlarged, popup object and need type check for gate)
    def on_singleclick_sumRegion(self, event, index):
        self.logger.info('on_singleclick_sumRegion - index: %s',index)
        dim = self.getSpectrumInfoREST("dim", index=index)
        if dim == 1:
            # second arg = 0 not used for 1d
            #l = self.addLine(self.closestBinPos(index, float(event.xdata)), 0, index)
            l = self.addLine(float(event.xdata), 0, index)
            self.sumRegionPopup.listRegionLine.append(l)
            # removes the lines n-1 (n is the latest) from the plot
            if len(self.sumRegionPopup.listRegionLine) > 2:
                self.removePrevLine()
            
            lineText = ""
            for nbLine in range(len(self.sumRegionPopup.listRegionLine)):
                if nbLine == 0:
                    lineText = lineText + (f"{nbLine}: X= {self.sumRegionPopup.listRegionLine[nbLine].get_xdata()[0]:.5f}") 
                else :
                    lineText = lineText + (f"\n{nbLine}: X= {self.sumRegionPopup.listRegionLine[nbLine].get_xdata()[0]:.5f}") 
            self.sumRegionPopup.regionPoint.clear()
            self.sumRegionPopup.regionPoint.insertPlainText(lineText)
        elif dim == 2:
            #If exist, remove line between the last and first points of the contour (closing_segment)
            tempLine = [line for line in self.sumRegionPopup.listRegionLine if line.get_label() == "closing_segment"]
            if len(tempLine) == 1 :
                tempLine[0].remove()
                self.sumRegionPopup.listRegionLine.pop()

            # Add a new line based on clicked location
            #xBinned, yBinned= self.closestBinPos(index, float(event.xdata), float(event.ydata))
            #l = self.addLine(xBinned, yBinned, index)
            #xBinned, yBinned= self.closestBinPos(index, float(event.xdata), float(event.ydata))
            l = self.addLine(float(event.xdata), float(event.ydata), index)
            if l is not None :
                self.sumRegionPopup.listRegionLine.append(l)

            # Add text in sumRegionPopup
            lineNb = len(self.sumRegionPopup.listRegionLine)
            lineText = ""
            for nbLine in range(lineNb):
                if nbLine == 0:
                        lineText = lineText + (f"{nbLine}: X= {self.sumRegionPopup.listRegionLine[nbLine].get_xdata()[0]:.5f}   Y= {self.sumRegionPopup.listRegionLine[nbLine].get_ydata()[0]:.5f}") 
                else :
                    lineText = lineText + (f"\n{nbLine}: X= {self.sumRegionPopup.listRegionLine[nbLine].get_xdata()[0]:.5f}   Y= {self.sumRegionPopup.listRegionLine[nbLine].get_ydata()[0]:.5f}") 
            if lineNb == 0 :
                lineText = lineText + (f"{lineNb}: X= {float(event.xdata):.5f}   Y= {float(event.ydata):.5f}") 
            else :
                lineText = lineText + (f"\n{lineNb}: X= {float(event.xdata):.5f}   Y= {float(event.ydata):.5f}") 
            self.sumRegionPopup.regionPoint.clear()
            self.sumRegionPopup.regionPoint.insertPlainText(lineText)

            # Close contour, draw a line between the last and first points, except for band gate type
            if lineNb > 1 :
                # Special label to identify this segment elsewhere in the code
                label = "closing_segment"
                #xBinned, yBinned= self.closestBinPos(index, self.sumRegionPopup.listRegionLine[0].get_xdata()[0], self.sumRegionPopup.listRegionLine[0].get_ydata()[0])
                #l = self.addLine(xBinned, yBinned, index, label)
                l = self.addLine(self.sumRegionPopup.listRegionLine[0].get_xdata()[0], self.sumRegionPopup.listRegionLine[0].get_ydata()[0], index, label)
                if l is not None :
                    self.sumRegionPopup.listRegionLine.append(l)

            # for line in self.sumRegionPopup.listRegionLine:
            #     print("Simon on_singleclick_gate - line -", line, line.get_label())

        self.currentPlot.canvas.draw()


    # called by on_press, right click when creating a summing region is removing the last point and so change contour lines and update point text
    # Similar to on_singleclick_gate_right, should merge both (difference in isEnlarged, gateType,  )
    def on_singleclick_sumRegion_right(self, index):
        self.logger.info('on_singleclick_sumRegion_right - index: %s',index)
        dim = self.getSpectrumInfoREST("dim", index=index)
        
        if dim == 2:
            lineNb = len(self.sumRegionPopup.listRegionLine) 
            # Reset prevPoint for next new line to first point of line n-1 
            try:
                self.sumRegionPopup.prevPoint = [self.sumRegionPopup.listRegionLine[-2].get_xdata()[0], self.sumRegionPopup.listRegionLine[-2].get_ydata()[0]]
            except IndexError:
                self.logger.debug('on_singleclick_sumRegion_right - IndexError', exc_info=True)
                return
            # if contour has only 3 lines, remove the two last lines 
            if lineNb == 3 :
                for i in range(2):
                    self.sumRegionPopup.listRegionLine[-1].remove()
                    self.sumRegionPopup.listRegionLine.pop(-1)
            # save n-2 line, remove line n-1 and change first point of the closing_segment to last point of saved n-2 line
            elif lineNb > 3:
                tempLine = self.sumRegionPopup.listRegionLine[-3]
                self.sumRegionPopup.listRegionLine[-2].remove()
                self.sumRegionPopup.listRegionLine.pop(-2)
                self.sumRegionPopup.listRegionLine[-1].set_xdata([tempLine.get_xdata()[1], self.sumRegionPopup.listRegionLine[0].get_xdata()[0]]) 
                self.sumRegionPopup.listRegionLine[-1].set_ydata([tempLine.get_ydata()[1], self.sumRegionPopup.listRegionLine[0].get_ydata()[0]]) 
        
            # Add text in sumRegionPopup
            lineNb = len(self.sumRegionPopup.listRegionLine)
            lineText = ""
            for nbLine in range(lineNb):
                if nbLine == 0:
                        lineText = lineText + (f"{nbLine}: X= {self.sumRegionPopup.listRegionLine[nbLine].get_xdata()[0]:.5f}   Y= {self.sumRegionPopup.listRegionLine[nbLine].get_ydata()[0]:.5f}") 
                else :
                    lineText = lineText + (f"\n{nbLine}: X= {self.sumRegionPopup.listRegionLine[nbLine].get_xdata()[0]:.5f}   Y= {self.sumRegionPopup.listRegionLine[nbLine].get_ydata()[0]:.5f}") 
            if lineNb == 1 :
                lineText = lineText + (f"\n{lineNb}: X= {self.sumRegionPopup.listRegionLine[0].get_xdata()[1]:.5f}   Y= {self.sumRegionPopup.listRegionLine[0].get_ydata()[1]:.5f}") 
            self.sumRegionPopup.regionPoint.clear()
            self.sumRegionPopup.regionPoint.insertPlainText(lineText)

        self.currentPlot.canvas.draw()


    #called by on_press when not in ceate/edit gate mode
    def on_dblclick(self, idx):
        self.logger.info('on_dblclick - idx, self.wTab.currentIndex(): %s, %s' ,idx, self.wTab.currentIndex())
        name = self.nameFromIndex(idx)
        index = self.wConf.histo_list.findText(name)
        self.wConf.histo_list.setCurrentIndex(index)

        if self.currentPlot.isEnlarged == False: # entering enlarged mode
            self.logger.debug('on_dblclick - isEnlarged TRUE')
            if name == "empty" or index == -1:
                self.logger.warning('on_dblclick - empty axes cannot enlarge')
                return
            self.removeRectangle()
            #important that zoomPlotInfo is set only while in zoom mode (not None only here)
            self.setEnlargedSpectrum(idx, name)
            self.currentPlot.next_plot_index = self.currentPlot.selected_plot_index
            self.currentPlot.isEnlarged = True
            # disabling adding histograms
            self.wConf.histo_geo_add.setEnabled(False)
            # disabling changing canvas layout
            self.wConf.histo_geo_row.setEnabled(False)
            self.wConf.histo_geo_col.setEnabled(False)
            # enabling gate creation
            self.wConf.createGate.setEnabled(True)
            # plot corresponding histogram
            self.wTab.selected_plot_index_bak[self.wTab.currentIndex()]= deepcopy(idx)

            # print("Simon - in dbl_click before reseet canvas - ", self.getSpectrumInfo("spectrum", index=idx).axes.get_xlim())
            # self.axesChilds()

            #setup single pad canvas
            self.currentPlot.InitializeCanvas(1,1,False)

            self.add(idx)
            self.updatePlot()
        else:
            self.logger.debug('on_dblclick - isEnlarged FALSE')
            # enabling adding histograms
            self.wConf.histo_geo_add.setEnabled(True)
            # enabling changing canvas layout
            self.wConf.histo_geo_row.setEnabled(True)
            self.wConf.histo_geo_col.setEnabled(True)

            # disabling gate creation
            self.wConf.createGate.setEnabled(False)

            #important that zoomPlotInfo is set only while in zoom mode (None only here)
            #tempIdxEnlargedSpectrum is used to draw back the dashed red rectangle, which pad was enlarged
            tempIdxEnlargedSpectrum = self.getEnlargedSpectrum()[0]
            self.setEnlargedSpectrum(None, None)
            self.currentPlot.isEnlarged = False

            canvasLayout = self.wTab.layout[self.wTab.currentIndex()]
            self.logger.debug('on_dblclick - canvasLayout: %s',canvasLayout)

            #draw back the original canvas
            self.currentPlot.InitializeCanvas(canvasLayout[0], canvasLayout[1], False)
            # self.currentPlot.selected_plot_index = None # this will allow to call drawGate and loop over all the gates
            for index, name in self.getGeo().items():
                if name is not None and name != "" and name != "empty":
                    self.add(index)
                    ax = self.getSpectrumInfo("axis", index=index)
                    self.plotPlot(index)
                    #reset the axis limits as it was before enlarge
                    #dont need to specify if log scale, it is checked inside setAxisScale, if 2D histo in log its z axis is set too.
                    dim = self.getSpectrumInfoREST("dim", index=index)
                    if dim == 1:
                        self.setAxisScale(ax, index, "x", "y")
                    elif dim == 2:
                        self.setAxisScale(ax, index, "x", "y", "z")
                    self.drawGate(index)
            #drawing back the dashed red rectangle on the unenlarged spectrum
            self.removeRectangle()
            self.currentPlot.recDashed = self.createDashedRectangle(self.currentPlot.figure.axes[tempIdxEnlargedSpectrum])
            #self.updatePlot() #replaced by the content of updatePlot in the above for loop (avoid looping twice)
            self.currentPlot.figure.tight_layout()
            # self.drawAllGates()
            self.currentPlot.canvas.draw()


    #obselete
    # def get_key(self, val):
    #     return next((key for key, value in self.currentPlot.h_dict.items() if any(val == value2 for value2 in value.values())), None)


    #callback for exitButton
    def closeAll(self):
        self.logger.info('closeAll')
        self.close()


     #callback when right click on tab
    def tab_handle_right_click(self):
        self.logger.info('tab_handle_right_click')
        if self.currentPlot.zoomPress: return
        menu = QMenu()
        item1 = menu.addAction("Rename")
        item2 = menu.addAction("Delete")
        item1.triggered.connect(lambda: self.renameTab(self.wTab.currentIndex()))
        item2.triggered.connect(lambda: self.closeTab(self.wTab.currentIndex()))

        tab = self.wTab.tabBar()
        tabRect = tab.rect()

        sizeSubTab = QtCore.QPoint(tabRect.width()/self.wTab.count(), tabRect.height()/2)
        pos = QtCore.QPoint(sizeSubTab.x()*self.wTab.currentIndex(), sizeSubTab.y())

        menuPos = tab.mapToGlobal(pos)
        menuPos = QtCore.QPoint(menuPos.x(), menuPos.y())
        # Shows menu at button position, need to calibrate with 0,0 position
        menu.exec_(menuPos)


    def closeTab(self, index):
        self.logger.info('closeTab')
        #For now, if change tab while working on gate, close any ongoing gate action
        if self.currentPlot.toCreateGate or self.currentPlot.toEditGate or self.gatePopup.isVisible():
            self.cancelGate()
        if self.currentPlot.toCreateSumRegion or self.sumRegionPopup.isVisible():
            self.cancelSumRegion()

        # Can't use removeTab of QTabWidget on current tab so change the current index to another tab
        newIndex = 0
        if index > 0:
            newIndex = index - 1
            self.wTab.setCurrentIndex(newIndex)
        elif index == 0 and self.wTab.count() > 2 :
            self.wTab.setCurrentIndex(1)

        if index == self.wTab.currentIndex():
            self.logger.warning('closeTab - trying to close current tab, please change tab and retry')
        else:
            # Delete Tab at user picked index
            if self.wTab.deleteTab(index):
                # Show a remaining tab
                self.clickedTab(newIndex)
                self.wTab.resetTabText()


    def renameTab(self, index):
        self.logger.info('renameTab')
        #For now, if change tab while working on gate, close any ongoing gate action
        if self.currentPlot.toCreateGate or self.currentPlot.toEditGate or self.gatePopup.isVisible():
            self.cancelGate()
        if self.currentPlot.toCreateSumRegion or self.sumRegionPopup.isVisible():
            self.cancelSumRegion()

        self.tabp.setWindowTitle("Rename tab...")
        self.tabp.setGeometry(200,350,100,50)
        if self.tabp.isVisible():
            self.tabp.close()

        self.tabp.show()


    def okTab(self):
        txt = self.wTab.tabText(self.wTab.currentIndex())
        if self.tabp.lineedit.text() != "" and self.tabp.lineedit.text() != "  +  ":
            txt = self.tabp.lineedit.text()
        else:
            self.logger.warning('okTab - Please choose another tab name')

        self.wTab.setTabText(self.wTab.currentIndex(), txt)
        self.tabp.lineedit.setText("")
        self.tabp.close()


    def cancelTab(self):
        self.tabp.close()

    
    def clickedTab(self, index):
        self.logger.info('clickedTab - index: %s',index)
        # End current auto update thread, to avoid thread issu, will start a new thread if/when tab is not empty 
        self.stopAutoUpdateThread.set()
        self.endThread(self.threadAutoUpdate)

        # For now, if change tab while working on gate, close any ongoing gate action
        if self.currentPlot.toCreateGate or self.currentPlot.toEditGate or self.gatePopup.isVisible():
            self.cancelGate()
            return
        if self.currentPlot.toCreateSumRegion or self.sumRegionPopup.isVisible():
            self.cancelSumRegion()
            return

        # Abord zoom action if click a tab
        if self.currentPlot.zoomPress:
            self.currentPlot.canvas.toolbar.actions()[1].triggered.emit()
            self.currentPlot.canvas.toolbar.actions()[1].setChecked(False)
            self.currentPlot.customZoomButton.setDown(False)
            self.currentPlot.zoomPress = False

        self.wTab.setCurrentIndex(index)

        # Check if create or switch/move existing tabs
        # First if when new tab
        if index == self.wTab.count()-1:
            self.wTab.addTab(index)
            self.currentPlot = self.wTab.wPlot[index]
            self.tabGeoWidgetAndFlags(index)            
        else:
            try:
                self.tabGeoWidgetAndFlags(index)
                self.removeRectangle()
                self.bindDynamicSignal()

                # If tab not empty, (re)start auto update
                for indexPlot, name in self.getGeo().items():
                    if name:
                        ax = self.getSpectrumInfo("axis", index=indexPlot)
                        if ax is not None :
                            self.autoUpdateStart()
                            break
            except:
                self.logger.debug('clickedTab - exception occured', exc_info=True)
                pass


    # Helper to set histo_geo widget and enable/disable buttons, when interact with tabs
    def tabGeoWidgetAndFlags(self, index):
        self.currentPlot = self.wTab.wPlot[index]
        nRow = self.wTab.layout[index][0]
        nCol = self.wTab.layout[index][1]
        self.logger.debug('tabGeoWidgetAndFlags - canvas layout: %s, %s',nRow, nCol)

        #nRow-1 because nRow (nCol) is the number of row (col) and the following sets an index starting at 0
        self.wConf.histo_geo_row.setCurrentIndex(nRow-1)
        self.wConf.histo_geo_col.setCurrentIndex(nCol-1)

        #enable/disable some widgets depending if enlarge mode of not, flags set also in on_dblclick 
        self.wConf.histo_geo_add.setEnabled(not self.currentPlot.isEnlarged)
        self.wConf.histo_geo_row.setEnabled(not self.currentPlot.isEnlarged)
        self.wConf.histo_geo_col.setEnabled(not self.currentPlot.isEnlarged)
        self.wConf.createGate.setEnabled(self.currentPlot.isEnlarged)


    def movedTab(self, indexFrom, indexTo):
        # Moving tab (indexFrom) is the one moving around currentTab
        self.logger.info('movedTab - index from: %s to: %s',indexFrom, indexTo)
        # if '+' tab not last, move to last
        if indexFrom == self.wTab.count()-1:
            self.wTab.tabBar().moveTab(indexTo, indexFrom)
        else: 
            self.wTab.swapTabDict(indexFrom, indexTo)
            self.tabGeoWidgetAndFlags(indexFrom)
            self.removeRectangle()
            self.bindDynamicSignal()

        # change only default tab text ! might be confusing not sure if useful
        self.wTab.orderDefaultName()


    #set geometry of the canvas
    def setCanvasLayout(self):
        self.logger.info('setCanvasLayout')
        indexTab = self.wTab.currentIndex()
        nRow = int(self.wConf.histo_geo_row.currentText())
        nCol = int(self.wConf.histo_geo_col.currentText())
        self.wTab.layout[indexTab] = [nRow, nCol]
        self.wTab.wPlot[indexTab].InitializeCanvas(nRow, nCol)
        self.wTab.selected_plot_index_bak[indexTab] = None
        self.currentPlot.selected_plot_index = None
        self.currentPlot.next_plot_index = -1



    ###############################################
    # 5) Connection to REST for gates
    ##
    #############################################

    # Change mirror and rest ports based on the selected SpecTcl PID 
    def specTclPIDListChanged(self):
        self.logger.info('specTclPIDListChanged')
        pid = self.connectConfig.spectclPID.currentText()
        self.connectConfig.rest.setText(str(self.connectConfig.getServiceInfo(self.connectConfig.user.text(), 'rest', 'port', 'first', pid)))
        self.connectConfig.mirror.setText(str(self.connectConfig.getServiceInfo(self.connectConfig.user.text(), 'mirror', 'port', 'first', pid)))


    # When server or user changed, change port manager or edit user of interest, and thus update following fields.
    def connectServerOrUserEdited(self, isEdited):
        self.logger.info('connectServerOrUserEdited')
        server = self.connectConfig.server.text()
        user = self.connectConfig.user.text()
        success = True
        self.connectConfig.spectclPID.clear()
        self.connectConfig.rest.clear()
        self.connectConfig.mirror.clear()
        if self.connectConfig.portManagerAvailable():
            success = self.connectConfig.setPortManager(server)
        # prompt a message to ask if really want to attempt connecting to another SpecTcl while it was openned paired with SpecTcl
        # return QMessageBox.about(self,"Warning!", "Please add/select a spectrum")
        self.connectConfig.updateFields(server, user, not success)




    # Callback for the configuration popup of REST/MIRROR ports
    def connectPopup(self):
        self.logger.info('callback connectPopup')
        self.disconnectConnectSignalsPopup()
        # Update spectclPID list if new SpecTcl locally or in port manager
        self.connectConfig.updateSpecTclPID()
        self.sidSpecTclPIDListChanged = self.connectConfig.spectclPID.currentIndexChanged.connect(self.specTclPIDListChanged)
        self.sidConnectServerChanged = self.connectConfig.server.editingFinished.connect(lambda: self.connectServerOrUserEdited("server"))
        self.sidConnectUserChanged = self.connectConfig.user.editingFinished.connect(lambda: self.connectServerOrUserEdited("user"))
        
        # host = "localhost"
        # self.connectConfig.setPortManager(host)

        self.connectConfig.show()


    # Make sure signals defined when popup is openned are disconnected.
    def disconnectConnectSignalsPopup(self):
        self.logger.info('disconnectConnectSignalsPopup')
        try:
            if hasattr(self, 'sidSpecTclPIDListChanged') :
                self.connectConfig.spectclPID.currentIndexChanged.disconnect(self.sidSpecTclPIDListChanged)
                self.logger.debug('disconnectConnectSignalsPopup - disconnected sidSpecTclPIDListChanged')
            if hasattr(self, 'sidConnectServerChanged') :
                self.connectConfig.server.editingFinished.disconnect(self.sidConnectServerChanged)
                self.logger.debug('disconnectConnectSignalsPopup - disconnected sidConnectServerChanged')
            if hasattr(self, 'sidConnectUserChanged') :
                self.connectConfig.user.editingFinished.disconnect(self.sidConnectUserChanged)
                self.logger.debug('disconnectConnectSignalsPopup - disconnected sidConnectUserChanged')
        except TypeError:
            pass


    # Callback to close configuration popup
    def closeConnect(self):
        self.logger.info('closeConnect callback')
        self.disconnectConnectSignalsPopup()
        self.connectConfig.close()


    # Callback to attempt a connection to REST/MIRROR
    def okConnect(self):
        self.logger.info('okConnect')
        pid = self.connectConfig.spectclPID.currentText()
        mirrorPort = self.connectConfig.mirror.text()
        restPort = self.connectConfig.rest.text()
        self.logger.info(f'okConnect - attempt connection to SpecTcl PID: {pid} with mirror port {mirrorPort} and rest port {restPort}')
        self.connectShMem()
        self.closeConnect()
        #check the ports matchs and not a uncknown combination has been choosen.
    

    #called in okGate, if was adding a new gate the pushed gate is gatePopup.listRegionLine
    #if was editing a gate, the pushed gate is based on the gatePopup.regionPoint text
    def pushGateToREST(self, gateName, gateType):
        self.logger.info('pushGateToREST')
        if gateName is None or gateName == "None" :
            self.logger.debug('pushGateToREST - gateName is None or gateName == "None"')
            return

        dim = self.getSpectrumInfo("dim", index=self.gatePopup.gateSpectrumIndex)
        parameters = self.getSpectrumInfoREST("parameters", index=self.gatePopup.gateSpectrumIndex)
        spectrumType = self.getSpectrumInfoREST("type", index=self.gatePopup.gateSpectrumIndex)

        # special case for gd the spectrum parameters do not match with the gate definition
        if spectrumType == "gd" :
            parametersFormat = []
            for item in parameters:
                pars = item.split(' ')
                if len(pars) == 2:
                    parametersFormat.append(pars[0])
                    parametersFormat.append(pars[1])
            parameters = parametersFormat

        # special case for m2 (passed as a + gate of sub gates), parameters are sub gates name made on pairs of spectrum parameters
        if spectrumType == "m2" :
            parametersFormat = {}
            if len(parameters) % 2 == 0:
                pass
            else:
                #discard last parameter if not even count (possible case?)
                parameters = parameters[:-1]
            countPar = 0
            for ipar in range(0, len(parameters), 2):
                par1 = parameters[ipar]
                par2 = parameters[ipar + 1]
                dumNum1 = str(10 + countPar + 1)
                dumNum2 = str(10 + countPar + 2)
                countPar += 2
                nameSubGate = "_" + gateName + "." + dumNum1 + "." + dumNum2 + ".000"
                parametersFormat[nameSubGate] = [par1, par2]
        if len(parameters) == 0:
            return

        boundaries = []
        # set boundaries from the point text in gatePopup 
        if self.currentPlot.toEditGate:
            #points for 1d: [x0, x1] and for 2d:  [[x0, y0], [x1, y1], ...]
            points = self.formatGatePopupPointText(dim)
            if points is None :
                return
            if dim == 1:
                boundaries = points 
            elif dim == 2:
                for point in points:
                    boundaries.append({"x": point[0], "y": point[1]})
        else:
            if len(self.gatePopup.listRegionLine) == 0:
                return
            if dim == 1:
                if gateType in ["s", "gs"] :
                    # gate 1Dgate_xamine s {aris.db1.ppac0.uc {1392.232056 1665.277466}}
                    boundaries = [self.gatePopup.listRegionLine[0].get_xdata()[0],self.gatePopup.listRegionLine[1].get_xdata()[0]]
                    # sort such that lowest first
                    if boundaries[0] > boundaries[1]:
                        boundaries.sort()
                        self.logger.warning('pushGateToREST - 1d - found boundaries[0] > boundaries[1] so sorted the boundaries')
            else:
                #{'2Dgate_xamine': {'name': '2Dgate_xamine', 'type': 'c',
                # 'parameters': ['aris.tof.tdc.db3scin_to_db5scin', 'aris.db5.pin.dE'],
                # 'points': [{'x': 126.876877, 'y': 29.429428}, {'x': 125.625626, 'y': 25.825825},
                #            {'x': 126.626625, 'y': 22.522522}, {'x': 129.879883, 'y': 22.522522}, {'x': 130.63063, 'y': 26.126125}, {'x': 129.629623, 'y': 29.129128}]
                #}}
                for iline, line in enumerate(self.gatePopup.listRegionLine):
                    if line.get_label() != "closing_segment":
                        #Add both points of the first line to boundaries
                        if iline == 0 :
                            for ipoint in range(2):
                                boundaries.append({"x": line.get_xdata()[ipoint], "y": line.get_ydata()[ipoint]})
                        #Add only the last point to boundaries
                        else :
                            boundaries.append({"x": line.get_xdata()[1], "y": line.get_ydata()[1]})

         # special case for m2 (passed as a + gate of sub gates), parameters are sub gates name made on pairs of spectrum parameters
        if spectrumType == "m2" :
            #push individual gates
            for name, par in parametersFormat.items():
                self.rest.createGate(name, gateType, par, boundaries)
            #push the combined gate 
            subGateNameList = []
            for subGateName in parametersFormat.keys() :
                subGateNameList.append(subGateName)
            self.rest.createGate(gateName, "+", subGateNameList, None)
        else :
            self.rest.createGate(gateName, gateType, parameters, boundaries)


    # format gatePopup.regionPoint into (1d) [x0, x1, ...] or (2d) [[x0, y0], [x1, y1], ...]
    def formatGatePopupPointText(self, dim):
        self.logger.info('formatGatePopupPointText - dim: %s',dim)
        points = []
        pointDict = {}
        #first split when new lines
        textBlockLines = self.gatePopup.regionPoint.toPlainText().split("\n")
        #loop on each line
        for line in textBlockLines:
            #pass on blank line
            if not line :
                continue
            #get first number from left in the lines 
            pointId = re.search(r'\d+', line)
            if not pointId:
                self.logger.warning('formatGatePopupPointText - Expect a point number before X (and Y) at beginning of a line')
                return None
            #pointId must be unique
            if int(pointId.group()) in pointDict.keys():
                self.logger.warning('formatGatePopupPointText - Expect unique point number: %s', int(pointId.group()))
                return None 
            pointId = int(pointId.group())
            #get first number between X and Y (X position) and first number from Y (Y position)
            posX = re.search(r'X(\s*[=]\s*)([-+]?(?:\d*\.*\d+))', line)
            if dim == 1 and not posX:
                self.logger.warning('formatGatePopupPointText - 1d spectrum - Line format is: i: X=f1 where i is integer and f1, f2 floats (no sci notation)')
                return None 
            try:
                posX = float(posX.group(2))
            except:
                self.logger.debug('formatGatePopupPointText - exception ', exc_info=True)
                return None
            pointDict[pointId] = posX

            posY = re.search(r'Y(\s*[=]\s*)([-+]?(?:\d*\.*\d+))', line)
            if dim == 2 and not posY:
                self.logger.warning('formatGatePopupPointText - 2d spectrum - Line format is: i: X=f1 Y=f2 where i is integer and f1, f2 floats (no sci notation)')
                return None 
            elif dim == 2 and posY:
                posY = float(posY.group(2))
                pointDict[pointId] = [posX, posY]

        #Order dict with pointId, so that pointId doesn't have to follow line index
        pointDict = {key:pointDict[key] for key in sorted(pointDict)}
        #Fill points with the format used in pushGateToRest
        if dim == 1:
            #Take only the two last values
            points = [val for key, val in pointDict.items()]
            points = points[-2:]
        elif dim == 2:
            for point in pointDict.values():
                points.append([point[0], point[1]])

        return points


    #Start ReST traces on a separated thread
    #Check for traces updates (poll) periodically, with period time < retention time.
    def restThread(self, retention):
        self.token = self.rest.startTraces(retention)
        if self.token is None:
            return
        self.stopRestThread.clear()
        self.wConf.connectButton.setStyleSheet("background-color:#bcee68;")
        self.wConf.connectButton.setText("Connected")

        while not self.stopRestThread.is_set():
            # Wait while stop event is false, if event true break loop
            # waits time < retention to avoid loosing information
            if self.stopRestThread.wait(retention/2):
                break
            try :
                tracesDetails = self.rest.pollTraces(self.token)
                if tracesDetails is None :
                    self.logger.debug('updateFromTraces - traceDetails is None')
                self.updateFromTraces(tracesDetails)
            except :
                self.logger.warning(f'updateFromTraces - could not poll from REST of SpecTcl (PID: {self.connectConfig.spectclPID.currentText()}) for '
                f'Server {self.connectConfig.server.text()} and User {self.connectConfig.user.text()}')
                break

        self.wConf.connectButton.setStyleSheet("background-color:rgb(252, 48, 3);")
        self.wConf.connectButton.setText("Disconnected")


    #Excecuted periodially (period defined in restThread)
    #Update various list and dict according to the trace.
    def updateFromTraces(self, tracesDetails):
        self.logger.info('updateFromTraces - tracesDetails: %s',tracesDetails)
        t_para = tracesDetails.get("parameter")
        t_spec = tracesDetails.get("spectrum")
        t_gate = tracesDetails.get("gate")
        t_bind = tracesDetails.get("binding")
        if len(t_bind) > 0:
            for str in t_bind:
                action, name, bindingIdx = str.split(" ") #list with 3 items: add/remove histoName bindingIndex
                if action == "remove" and name in self.getSpectrumInfoRESTDict():
                    #if only "remove" action, want to clear axis, update the spectrum and summing region dictionnaries
                    # if len(t_bind) == 1:
                    #     self.removeSpectrum(name=name, mode="definitive")
                    #     self.updateSpectrumList()
                    #     self.refreshSpectrumSumRegionDict()
                    # else:
                    #     self.removeSpectrum(name=name, mode="update")
                    #     self.updateSpectrumList()
                    #For now dont ditinguish if it is just an update or definitive removal
                    #had trouble redrawing efficiently the updated spectrum, see comments at end of action=="add"
                    self.removeSpectrum(name=name, mode="definitive")
                    self.updateSpectrumList()
                    self.refreshSpectrumSumRegionDict()
                elif action == "add" and name not in self.getSpectrumInfoRESTDict():
                    #get spectrum info from ReST but not the data, need to query shmem for that (connect button).
                    #dont need to check the binding, given the definition of this trace...
                    info = self.rest.listSpectrum(name)
                    #info[0] because expect only one element in the list with the name filter

                    #get the data in shmem ... maybe not the best to do it here
                    #server info are already checked at this point
                    hostname = self.connectConfig.server.text()
                    port = self.connectConfig.rest.text()
                    user = self.connectConfig.user.text()
                    mirror = self.connectConfig.mirror.text()
                    s = cpy.CPyConverter().getShMem(bytes(hostname, encoding='utf-8'), bytes(port, encoding='utf-8'), bytes(mirror, encoding='utf-8'), bytes(user, encoding='utf-8'))
                    data = []
                    binx = info[0]["axes"][0]["bins"]
                    minx = info[0]["axes"][0]["low"]
                    maxx = info[0]["axes"][0]["high"]
                    if "1" in info[0]["type"] or "b" in info[0]["type"] or "g1" in info[0]["type"] :
                        dim = 1
                        biny = miny = maxy = None
                        nameIndex = s[1].index(name)
                        try :
                            data = s[9][nameIndex][0:-1]
                            data[0] = 0
                        except:
                            debugstring = "updateFromTraces - nameIndex not in shmem np array for :" + name
                            self.logger.debug(debugstring, exc_info=True)
                        self.setSpectrumInfoREST(name, dim=dim, binx=binx, minx=minx, maxx=maxx, biny=biny, miny=miny, maxy=maxy, parameters=info[0]["parameters"], type=info[0]["type"], data=data)
                    elif "s" in info[0]["type"] :
                        dim = 2
                        # only the y axis in info[0]["axes"]
                        biny = binx
                        miny = minx
                        maxy = maxx
                        # get x axis from parameters list
                        binx = 0
                        minx = 9e+6
                        maxx = 0
                        for par in info[0]["parameters"]:
                            ipar = self.getLastDigitParam(par)
                            if ipar < minx :
                                minx = ipar
                            if ipar > maxx :
                                maxx = ipar
                        nameIndex = s[1].index(name)
                        maxx += 1
                        # Assuming 1 parameter per bin
                        # ! maybe different than the binning chosen by user !
                        # Because compared to shmem for summary spec the x binning is not provided.
                        # If want the same auto x axis definition at connection to shmem, see commented lines in connectShMem
                        binx = maxx - minx
                        try :
                            #print("Simon - data - min, max, binx ", s[9][nameIndex][:,:],minx,maxx, binx)
                            data = s[9][nameIndex][1:-1, 1:-1]
                            # data = s[9][nameIndex][0:-1]
                        except:
                            debugstring = "updateFromTraces - nameIndex not in shmem np array for :" + name
                            self.logger.debug(debugstring, exc_info=True)
                        self.setSpectrumInfoREST(name, dim=dim, binx=binx, minx=minx, maxx=maxx, biny=biny, miny=miny, maxy=maxy, parameters=info[0]["parameters"], type=info[0]["type"], data=data)
                    else :
                        dim = 2
                        biny = info[0]["axes"][1]["bins"]
                        miny = info[0]["axes"][1]["low"]
                        maxy = info[0]["axes"][1]["high"]
                        nameIndex = s[1].index(name)
                        try :
                            data = s[9][nameIndex][1:-1, 1:-1]
                        except:
                            debugstring = "updateFromTraces - nameIndex not in shmem np array for :" + name
                            self.logger.debug(debugstring, exc_info=True)
                            # print("updateFromTraces - nameIndex not in shmem np array for : ", name)
                        self.setSpectrumInfoREST(name, dim=dim, binx=binx, minx=minx, maxx=maxx, biny=biny, miny=miny, maxy=maxy, parameters=info[0]["parameters"], type=info[0]["type"], data=data)
                    self.updateSpectrumList()
                    #update an existing spectrum definition (binning, parameters...)
                    #this part is not working properly, need to correct if this is really needed
                    # if len(t_bind) == 2:
                    #     for plotVal in self.wTab.wPlot.values():
                    #         idx_to_update = [key for key, value in plotVal.h_dict_geo.items() if name in value]
                    #         print("Simon - removeSpectrum idx_to_update-",  idx_to_update)
                    #         # trick, set temporarily currentPlot to plotVal, because currentPlot is used in add 
                    #         self.currentPlot = plotVal
                    #         for plotIdx in idx_to_update:
                    #             self.add(plotIdx)
                    #     self.currentPlot = self.wTab.wPlot[self.wTab.currentIndex()]
                else:
                    return 


    #Set spectrum info from ReST in spectrum_dict_rest (identified by histo name and can update multiple info at once)
    #self.spectrum_dict_rest is used to keep track of the treegui definition (fixed) 
    def setSpectrumInfoREST(self, name, **info):
        self.logger.info('setSpectrumInfoREST - name, info: %s, %s',info, name)
        for key, value in info.items():
            if key in ("dim", "binx", "minx", "maxx", "biny", "miny", "maxy", "data", "parameters", "type"):
                if name not in self.spectrum_dict_rest:
                    self.spectrum_dict_rest[name] = {"dim":[],"binx":[],"minx":[],"maxx":[],"biny":[],"miny":[],"maxy":[],"data":[],"parameters":[],"type":[]}
                # print("Simon - setSpectrumInfoREST - name key value",name,key,value)
                self.spectrum_dict_rest[name][key] = value


    #Get spectrum info from spectrum_dict_rest (identified by histo name or index and info name)
    #template of expected arguments e.g.: ("dim", index=5) takes only the first info parameter (here "dim") (one per call)
    #Important that it gets only the info from self.spectrum_dict_rest here.
    def getSpectrumInfoREST(self, *info, **identifier):
        # self.logger.info('getSpectrumInfoREST - info, identifier: %s, %s',info, identifier)
        name = None
        if not identifier and self.getEnlargedSpectrum():
            name = self.getEnlargedSpectrum()[1]
        elif "index" in identifier:
            name = self.nameFromIndex(identifier["index"])
        elif "name" in identifier:
            name = identifier["name"]
        else:
            self.logger.debug('getSpectrumInfoREST - wrong identifier - expects name=histo_name or index=histo_index or shoud be in zoomed mode')
            # print("getSpectrumInfo - wrong identifier - expects name=histo_name or index=histo_index or shoud be in zoomed mode")
            return
        if name is not None and name in self.spectrum_dict_rest and info[0] in ("dim", "binx", "minx", "maxx", "biny", "miny", "maxy", "data", "parameters", "type"):
            # print("Simon - getSpectrumInfoREST - ",name,info[0],self.spectrum_dict_rest[name][info[0]])
            return self.spectrum_dict_rest[name][info[0]]
        else:
            return


    #Update spectrum info in spectrum_dict (identified by index and can update multiple info at once)
    #Important that only self.wTab.spectrum_dict is changed here
    #work in normal and enlarged mode
    def setSpectrumInfo(self, **info):
        self.logger.info('setSpectrumInfo - info: %s',info)
        name = None
        index = None
        if self.getEnlargedSpectrum():
            index = self.getEnlargedSpectrum()[0]
            name = self.getEnlargedSpectrum()[1]
        elif "index" in info:
            index = info["index"]
            name = self.nameFromIndex(info["index"])
        # commented the following option because can have several versions of the same plot in a window (name not a unique id)
        # elif "name" in info:
        #     name = info["name"]
        else:
            self.logger.debug('setSpectrumInfo - wrong identifier - expects index=histo_index or shoud be in zoomed mode')
            # print("setSpectrumInfo - wrong identifier - expects index=histo_index or shoud be in zoomed mode")
            return
        # print("Simon - setSpectrumInfo - ", index,name,info["index"])
        for key, value in info.items():
            if key in ("name", "dim", "binx", "minx", "maxx", "biny", "miny", "maxy", "data", "parameters", "type", "log", "minz", "maxz", "spectrum", "axis", "cutoff") and index is not None:
                if index not in self.wTab.spectrum_dict[self.wTab.currentIndex()]:
                    # print("setSpectrumInfo -",name,"not in spectrum_dict")
                    self.logger.debug('setSpectrumInfo - %s not in spectrum_dict', name)
                    return
                    # self.wTab.spectrum_dict[self.wTab.currentIndex()][name] = {"dim":[],"binx":[],"minx":[],"maxx":[],"biny":[],"miny":[],"maxy":[],"data":[],"parameters":[],"type":[],"log":[],"minz":[],"maxz":[]}
                self.wTab.spectrum_dict[self.wTab.currentIndex()][index][key] = value
                # print("Simon - setSpectrumInfo -",index, self.wTab.spectrum_dict[self.wTab.currentIndex()], self.wTab.spectrum_dict[self.wTab.currentIndex()][index])
                #set axes info at the same time than spectrum
                if key == "spectrum":
                    self.wTab.spectrum_dict[self.wTab.currentIndex()][index]["axis"] = value.axes


    #Get spectrum info from spectrum_dict (identified by index and info name)
    #template of expected arguments e.g.: ("dim", index=5) takes only the first info parameter (here "dim") (one per call)
    #Important that it gets only the info from self.wTab.spectrum_dict[self.wTab.currentIndex()] here.
    #work in normal and enlarged mode
    def getSpectrumInfo(self, *info, **identifier):
        self.logger.info('getSpectrumInfo - info, identifier: %s, %s', info, identifier)
        name = None
        index = None
        if self.getEnlargedSpectrum():
            index = self.getEnlargedSpectrum()[0]
            # name = self.getEnlargedSpectrum()[1]
        elif "index" in identifier:
            index = identifier["index"]
            # name = self.nameFromIndex(identifier["index"])
        # commented the following option because can have several versions of the same plot in a window (name not a unique id)
        # elif "name" in identifier:
        #     name = identifier["name"]
        else:
            self.logger.debug('getSpectrumInfo - wrong identifier - expects index=histo_index or shoud be in zoomed mode')
            # print("getSpectrumInfo - wrong identifier - expects index=histo_index or shoud be in zoomed mode")
            return
        if index is not None and index in self.wTab.spectrum_dict[self.wTab.currentIndex()] and info[0] in ("name", "dim", "binx", "minx", "maxx", "biny", "miny", "maxy", "data", "parameters", "type", "log", "minz", "maxz", "spectrum", "axis", "cutoff"):
        # if index is not None and info[0] in ("name", "dim", "binx", "minx", "maxx", "biny", "miny", "maxy", "data", "parameters", "type", "log", "minz", "maxz"):
            # print("Simon - in getSpectrumInfo - ",self.wTab.currentIndex(), index, info[0])
            return self.wTab.spectrum_dict[self.wTab.currentIndex()][index][info[0]]


    #Remove spectrum from self.wTab.spectrum_dict:
    #important: only functions that delete item in spectrum_dict and spectrum_dict_rest
    #important: should not be triggered by user, for now used only in updateFromTraces, because of the way it deletes local spectrumInfo entries.
    def removeSpectrum(self, **identifier):
        self.logger.info('removeSpectrum - identifier: %s', identifier)
        name = None
        mode = None
        if "index" in identifier:
            name = self.nameFromIndex(identifier["index"])
        elif "name" in identifier:
            name = identifier["name"]
        if "mode" in identifier:
            mode = identifier["mode"]
        if name is None :
            self.logger.debug('removeSpectrum - wrong identifier - expects name=histo_name or index=histo_index')
            # print("removeSpectrum - wrong identifier - expects name=histo_name or index=histo_index")
            return
        # in spectrumInfoReST dict can only have unique spectrum
        del self.spectrum_dict_rest[name]
    
        # in local spectrumInfo dict can have multiple spectra with the same name
        # dont use indexFromName because wont work properly when delete while in enlarged mode
        # also want to clear the corresponding axes in all tabs
        for tabIdx, plotVal in self.wTab.wPlot.items():
            to_delete = [key for key, value in plotVal.h_dict_geo.items() if name in value]
            for key in to_delete:
                if key in self.wTab.spectrum_dict[tabIdx]:
                    spectrum = self.wTab.spectrum_dict[tabIdx][key]["spectrum"]
                    #clear axis, remove colorbar and update in the geometry if mode="definitive"
                    if mode == "definitive":
                        ax = spectrum.axes
                        self.removeCb(ax)
                        ax.clear()
                        plotVal.h_dict_geo[key] = "empty"
                    del spectrum


    #get full spectrum dict self.wTab.spectrum_dict:
    def getSpectrumInfoDict(self):
        return self.wTab.spectrum_dict[self.wTab.currentIndex()]


    #get full spectrum dict self.spectrum_dict_rest:
    def getSpectrumInfoRESTDict(self):
        return self.spectrum_dict_rest
 

    #Find name with geo index:
    def nameFromIndex(self, index):
        # self.logger.info('nameFromIndex - index: %s', index)
        #Can call getSpectrumInfo and setSpectrumInfo with an identifier but still check if in zoom mode,
        #which is important for autoScaleAxis/setAxisScale
        if self.getEnlargedSpectrum():
            return self.getEnlargedSpectrum()[1]
        elif index in self.currentPlot.h_dict_geo:
            return self.currentPlot.h_dict_geo[index]


    #Find index(es) in geo for spectrum name (returns a list)
    def indexFromName(self, name):
        self.logger.info('indexFromName - name: %s', name)
        result = []
        #Have to be careful that zoomPlotInfo is well sets all the time.
        #Should have a value _only_while_ in enlarged mode
        if self.getEnlargedSpectrum():
            result = [0]
        else:
            result = [key for key, value in self.currentPlot.h_dict_geo.items() if name in value]
        return result


    #sets h_dict_geo {key=index, value=histoName}
    #Use only this function to set the geometry dict when add plot (against using it elsewhere because it initializes spectrum_dict)
    def setGeo(self, index, name):
        self.logger.info('setGeo - index, name: %s, %s', index, name)
        self.currentPlot.h_dict_geo[index] = name
        #Set also here the spectrum_dict with only the spectra defined in the geo
        if index not in self.wTab.spectrum_dict[self.wTab.currentIndex()]:
            self.wTab.spectrum_dict[self.wTab.currentIndex()][index] = {"name":[], "dim":[],"binx":[],"minx":[],"maxx":[],"biny":[],"miny":[],"maxy":[],"data":[],"parameters":[],"type":[],"log":[],"minz":[],"maxz":[], "spectrum":[], "axis":[], "cutoff":[]}
        self.wTab.spectrum_dict[self.wTab.currentIndex()][index]["name"] = name
        #Initialize with the same info than in spectrum_dict_rest.
        for key, value in self.spectrum_dict_rest[name].items():
            self.wTab.spectrum_dict[self.wTab.currentIndex()][index][key] = value


    #returns h_dict_geo {key=index, value=histoName}
    #Use only this function to get the geometry dict, to know the name at index there is nameFromIndex
    def getGeo(self):
        return self.currentPlot.h_dict_geo


    #attempt to use blit more generically (maybe for future)
    # def saveCanvasBkg(self, axis, spectrum, index):
    #     self.currentPlot.axbkg[index] = self.currentPlot.figure.canvas.copy_from_bbox(axis.bbox)
    #     # axis.draw_artist(spectrum)
    #     # self.currentPlot.figure.canvas.blit(axis.bbox)

    # def restoreCanvasBkg(self, axis, spectrum, index):
    #     axis.clear()
    #     self.currentPlot.figure.canvas.restore_region(self.currentPlot.axbkg[index])
    #     axis.draw_artist(spectrum)
    #     self.currentPlot.figure.canvas.blit(axis.bbox)
    #     self.currentPlot.figure.canvas.flush_events()

        
    def setEnlargedSpectrum(self, index, name):
        self.logger.info('setEnlargedSpectrum')
        self.wTab.zoomPlotInfo[self.wTab.currentIndex()] = None 
        if index is not None and name is not None: 
            self.wTab.zoomPlotInfo[self.wTab.currentIndex()] = [index, name]


    def getEnlargedSpectrum(self):
        # self.logger.info('getEnlargedSpectrum')
        result = None
        if self.wTab.zoomPlotInfo[self.wTab.currentIndex()] :
            result = self.wTab.zoomPlotInfo[self.wTab.currentIndex()]
        return result


    #get histo name, type and parameters from REST
    def getSpectrumInfoFromReST(self):
        self.logger.info('getSpectrumInfoFromReST')
        outDict = {}
        inpDict = self.rest.listSpectrum()
        bindList = self.rest.listsbind("*")
        # Dictionary of bindings, keyed by spectrum name
        bindings = {}
        for d in bindList:
            bindings[d["name"]] = d["binding"]
        for el in inpDict:
            if el["name"] in bindings:
                outDict[el["name"]] = {
                    "parameters": el["parameters"],
                    "type": el["type"],
                    "binding": bindings[el["name"]],
                }
            else:
                pass
        self.logger.info('getSpectrumInfoFromReST - return: %s', outDict)
        return outDict


    # update spectrum list for GUI, the widget histo_list is set only here
    def updateSpectrumList(self, init=False):
        self.logger.info('updateSpectrumList')
        self.wConf.histo_list.clear()
        self.wConf.histo_list.setEditText("")
        #Sort because otherwise when ReST update the modified spectrum is append at the end of the list
        for name in sorted(self.getSpectrumInfoRESTDict()):
            if self.wConf.histo_list.findText(name) == -1:
                self.wConf.histo_list.addItem(name)

        if init:
            self.wConf.histo_list.setEditable(True)
            self.wConf.histo_list.setInsertPolicy(QComboBox.NoInsert)
            self.wConf.histo_list.completer().setCompletionMode(QCompleter.PopupCompletion)
            self.wConf.histo_list.completer().setFilterMode(QtCore.Qt.MatchContains)



    #About the gates: unlike spectrum, there is no internal gate dictionnary 
    #which means everytime one gets/sets gate info, one uses the ReST interface, like for the name here:
    #return the gate name applied to a spectrum (identified by index or name)
    def getAppliedGateName(self, **identifier):
        # self.logger.info('getAppliedGateName')
        spectrumName = None
        if "index" in identifier:
            spectrumName = self.nameFromIndex(identifier["index"])
        elif "name" in identifier:
            spectrumName = identifier["name"]
        else:
            self.logger.debug('getAppliedGateName - wrong identifier - expects name=histo_name or index=histo_index')
            return
        gate = self.rest.applylistgate(spectrumName)
        if gate is None or len(gate) == 0:
            self.logger.debug('getAppliedGateName - gate is None')
            return 
        #gate is a list with one dictionary [{'spectrum': 'spectrumName', 'gate': 'gateName'}]
        gateName = gate[0]["gate"]
        if gateName == "-TRUE-" or gateName == "-Ungated-":
            return None 
        else :
            return gateName


    #give spectrum index and line2D of summing region to fill sumRegionDict
    def setSumRegion(self, index, line):
        self.logger.info('setSumRegion')
        if index is None : 
            self.logger.debug('setSumRegion - index: %s', index)
            return
        labelSplit = line.get_label().split("_-_")
        if len(labelSplit) == 3 and labelSplit[0] == "sumReg":
            spectrumName = self.nameFromIndex(index)
            if spectrumName not in self.sumRegionDict:
                self.sumRegionDict[spectrumName] = [line]
            else :
                self.sumRegionDict[spectrumName].append(line)


    #return list with summing regions drawn on the spectrum at index
    def getSumRegion(self, index):
        self.logger.info('getSumRegion - index: %s',index)
        result = None
        if index is None : 
            return
        spectrumName = self.nameFromIndex(index)
        if spectrumName not in self.sumRegionDict:
            pass
        else :
            result = self.sumRegionDict[spectrumName]
        self.logger.info('getSumRegion - spectrumName, result: %s, %s', spectrumName, result)
        return result


    #delete summing region line2D in sumRegionDict, identified with label
    def deleteSumRegionDict(self, labelSumRegion):
        self.logger.info('deleteSumRegionDict - labelSumRegion: %s', labelSumRegion)
        labelSplit = labelSumRegion.split("_-_")
        if len(labelSplit) == 3 and labelSplit[0] == "sumReg":
            for spectrumName, regionList in self.sumRegionDict.items():
                for line in regionList:
                    if line.get_label() == labelSumRegion:
                        #remove all line2D with labelSumRegion in the entire currentPlot 
                        for ax in self.currentPlot.figure.axes:
                            toRemove = [child for child in ax.get_children() if isinstance(child, matplotlib.lines.Line2D) and child.get_label() == labelSumRegion]
                            for linetoRemove in toRemove:
                                linetoRemove.remove()
                        #remove line from dict
                        regionList.remove(line)
    

    #check if needs to delete a spectrumName key in sumRegionDict and all sumRegion associated
    def refreshSpectrumSumRegionDict(self):
        self.logger.info('refreshSpectrumSumRegionDict')
        histoList = [self.wConf.histo_list.itemText(i) for i in range(self.wConf.histo_list.count())]
        keyToDelete = []
        for spectrumNameSumReg in self.sumRegionDict.keys():
            if spectrumNameSumReg not in histoList:
                #in principle there should not be line2D drawn so no need to delete those
                keyToDelete.append(spectrumNameSumReg)
        if len(keyToDelete) > 0:
            for key in keyToDelete:
                del self.sumRegionDict[key]



    ##########################################
    # 6) Accessing the ShMem
    ##########################################


    def connectShMem(self):
        self.logger.info('connectShMem')
        # trying to access the shared memory through SpecTcl Mirror Client
        try:
            # update host name and port, mirror port, and user name from GUI
            hostname = str(self.connectConfig.server.text())
            port = str(self.connectConfig.rest.text())
            user = str(self.connectConfig.user.text())
            mirror = str(self.connectConfig.mirror.text())
            self.logger.debug('connectShMem - host: %s -- user: %s -- RESTPort: %s -- MirrorPort: %s', hostname, user, port, mirror)

            # configuration of the REST plugin
            self.rest = PyREST(self.logger,hostname,port)
            self.wConf.connectButton.setStyleSheet("background-color:rgb(252, 48, 3);")
            self.wConf.connectButton.setText("Disconnected")
            # check if thread already exists, if yes reset, and set traces
            self.stopRestThread.set()
            self.endThread(self.threadRest)
            self.threadRest = threading.Thread(target=self.restThread, args=(6,))
            self.threadRest.start()

            # way to stop connectShMem, url checks are done on trace thread but if find issue it wont be communicated here
            if self.rest.checkSpecTclREST() == False:
                self.logger.warning('connectShMem - invalid URL for SpecTclREST')
                return
            else:
                self.logger.debug("connectShMem - could make REST request of server")
                
            timer1 = QElapsedTimer()
            timer1.start()

            self.logger.debug("connectShMem - attempting update from CPYConverter.")
            s = cpy.CPyConverter().getShMem(bytes(hostname, encoding='utf-8'), bytes(port, encoding='utf-8'), bytes(mirror, encoding='utf-8'), bytes(user, encoding='utf-8'), True)
            self.logger.debug("connectShMem CPyConverter updated without failure")
            # Connected to REST and MIRROR, clear previous definitions.
            spectrumDict = list(self.getSpectrumInfoRESTDict().items())
            for spectrumName, infoDict in spectrumDict:
                    self.removeSpectrum(name=spectrumName, mode="definitive")
                    self.updateSpectrumList()
                    self.refreshSpectrumSumRegionDict()    
       

            # creates a dataframe for spectrum info
            # use the spectrum name to merge both sources (REST and shared memory) of spectrum info
            # info = {"id":[],"names":[],"dim":[],"binx":[],"minx":[],"maxx":[],"biny":[],"miny":[],"maxy":[],"data":[],"parameters":[],"type":[]}

            otherInfo = self.getSpectrumInfoFromReST()
            self.logger.debug("connectShMem Got spectrum information from REST")
            for i, name in enumerate(s[1]):
                self.logger.debug("Looking at: %s", name)
                if name in otherInfo:
                    self.logger.debug("It's in otherinfo.")
                    if s[2][i] == 2:
                        self.logger.debug("s[2][i] == 2")
                        data = s[9][i][ 1:-1, 1:-1]
                        # -- begin -- for auto x-axis definition summary spec
                        if "s" in otherInfo[name]["type"]:
                            minx = s[4][i]
                            maxx = s[5][i] + 1
                        # # get x axis from parameters list
                        # binx = 0
                        # minx = 9e+6
                        # maxx = 0
                        # for par in otherInfo[name]["parameters"]:
                        #     ipar = self.getLastDigitParam(par)
                        #     if ipar < minx :
                        #         minx = ipar
                        #     if ipar > maxx :
                        #         maxx = ipar
                        # maxx += 1
                        # # Assuming 1 parameter per bin
                        # binx = maxx - minx
                        # -- end -- for auto x-axis definition summary spec
                    else:
                        self.logger.debug("s[2][i] != 2 ")
                        data = s[9][i][0:-1]
                        data[0] = 0




                    # print("Simon connectShMem ", name, s[2][i], s[3][i], s[4][i], s[5][i], s[6][i], s[7][i], s[8][i],otherInfo[name]["type"])
                    self.logger.debug("Setting spectrum info")
                    # -- begin -- for auto x-axis definition summary spec
                    if "s" in otherInfo[name]["type"]:
                        self.setSpectrumInfoREST(name, dim=s[2][i], binx=s[3][i]-2, minx=minx, maxx=maxx, biny=s[6][i]-2, miny=s[7][i], maxy=s[8][i], data=data, parameters=otherInfo[name]["parameters"], type=otherInfo[name]["type"])
                    else:
                        self.setSpectrumInfoREST(name, dim=s[2][i], binx=s[3][i]-2, minx=s[4][i], maxx=s[5][i], biny=s[6][i]-2, miny=s[7][i], maxy=s[8][i], data=data, parameters=otherInfo[name]["parameters"], type=otherInfo[name]["type"])
                    # -- end -- for auto x-axis definition summary spec
                    # self.setSpectrumInfoREST(name, dim=s[2][i], binx=s[3][i]-2, minx=s[4][i], maxx=s[5][i], biny=s[6][i]-2, miny=s[7][i], maxy=s[8][i], data=data, parameters=otherInfo[name]["parameters"], type=otherInfo[name]["type"])
                    self.logger.debug('-------------------')

            # update and create parameter, spectrum, and gate lists
            # updateSpectrumList has True flag/arg only here, to define searchable list
            self.logger.debug("connectShMem Updating spectrumlist")
            self.updateSpectrumList(True)

            '''
            # update Modify menu
            gate_list = [self.wConf.listGate.itemText(i) for i in range(self.wConf.listGate.count())]
            print("gate_list", gate_list)
            for gate in gate_list:
                self.wConf.submenuD.addAction(gate, lambda:self.wConf.drag(gate))
                self.wConf.submenuE.addAction(gate, lambda:self.wConf.edit(gate))
            '''
            # If connection established, save recovery information into a file
            recoInfo = self.connectConfig.server.text() + '::' + self.connectConfig.user.text() + '::' \
                    + self.connectConfig.rest.text() + '::' + self.connectConfig.mirror.text() + '::' \
                    + self.connectConfig.spectclPID.currentText()
            self.setRecoVariable('CUTIEPIE_CONNECT', recoInfo)
            self.logger.debug("connectShMem exiting")
        except Exception as e:
            self.logger.exception('connectShMem - Exception')
            raise
        # except:
        #     QMessageBox.about(self, "Warning", "The rest interface for SpecTcl was not started or hostname/port/mirror are not configured!")


    ##########################################
    # 7) Load/save geometry window
    ##########################################


    def findWholeWord(self, w):
        return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

    def findNumbers(self, w):
        return [int(s) for s in re.findall(r'\b\d+\b', w)]

    def findHistoName(self, w):
        return re.findall('"([^"]*)"', w)

    # definition for both legacy and not window defs
    def openGeo(self, filename):
        self.logger.info('openGeo')
        cntr = 0
        coords = []
        spec_dict = {}
        info_scale = {}
        info_range = {}
        x_range = {}
        y_range = {}
        properties = {}


        if (os.stat(filename).st_size == 0):
            self.logger.warning('openGeo - empty geometry file: %s', filename)
            return None
        #new (qtpy) geometry file format
        elif (len(open(filename).readlines()) == 1):
            return eval(open(filename,"r").read())
        #old format
        else:
            # not supported, see comment below about spectrum name 
            return None
            # with open(filename) as f:
            #     for line in f:
            #         if (self.findWholeWord("Geometry")(line)):
            #             # find geo x,y in line
            #             coords = self.findNumbers(line)
            #         elif (self.findWholeWord("Window")(line)):
            #             spectrum = self.findHistoName(line)
            #             spec_dict[cntr] = spectrum[0]
            #             cntr+=1
            #         elif (self.findWholeWord("COUNTSAXIS")(line)):
            #             info_scale[cntr-1] = True
            #         elif (self.findWholeWord("Expanded")(line)):
            #             tmp = self.findNumbers(line)
            #             info_range[cntr-1] = tmp

            # for index, name in spec_dict.items():
            #     scale = False
            #     #pb here because with the old format the spectrum name is in capital letters
            #     #while the REST info are case sensitive.
            #     dim = self.getSpectrumInfoREST("dim", name=name)
            #     x_range = [self.minX, self.maxX]
            #     if dim == 2:
            #         y_range = [self.minY, self.maxY]

            #     print("Simon - openGeo - index, name, dim: ",index, name, dim)

            #     if index in info_scale:
            #         scale = info_scale[index]
            #         if y_range[0] == 0:
            #             y_range[0] = 0.001
            #     if index in info_range:
            #         x_range = info_range[index][0:2]
            #         y_range = info_range[index][2:4]

            #     print("Simon - openGeo - x_range, y_range: ",x_range, y_range)

            #     properties[index] = {"name": name, "x": x_range, "y": y_range, "scale": scale}
            #     self.logger.debug('openGeo - index, properties: %s, %s', index, properties[index])

            # return {'row': coords[0], 'col': coords[1], 'geo': properties}


    def saveGeo(self):
        fileName = self.saveFileDialog()
        self.logger.info('saveGeo - fileName: %s',fileName)
        try:
            f = open(fileName,"w")
            properties = {}
            geo = self.getGeo()
            for index in range(len(geo)):
                try :
                    h_name = geo[index]
                    x_range, y_range = self.getAxisProperties(index)
                    scale = True if self.getSpectrumInfo("log", index=index) else False
                    properties[index] = {"name": h_name, "x": x_range, "y": y_range, "scale": scale}
                except:
                    properties[index] = {"name": '', "x": None, "y": None, "scale": None}
                    pass
            tmp = {"row": int(self.wConf.histo_geo_row.currentText()), "col": int(self.wConf.histo_geo_col.currentText()), "geo": properties}
            QMessageBox.about(self, "Saving...", "Window configuration saved!")
            f.write(str(tmp))
            f.close()
        except :
            self.logger.debug('saveGeo - exception', exc_info=True)
            pass


    def loadGeo(self):
        fileName = self.openFileNameDialog()
        self.logger.info('loadGeo - fileName: %s', fileName)
        try:
            infoGeo = self.openGeo(fileName)
            if infoGeo is None:
                return
            row = infoGeo["row"]
            col = infoGeo["col"]
            # change index in combobox to the actual loaded values
            index_row = self.wConf.histo_geo_row.findText(str(row), QtCore.Qt.MatchFixedString)
            index_col = self.wConf.histo_geo_col.findText(str(col), QtCore.Qt.MatchFixedString)
            notFound = []
            if index_row >= 0 and index_col >= 0:
                self.wConf.histo_geo_row.setCurrentIndex(index_row)
                self.wConf.histo_geo_col.setCurrentIndex(index_col)
                self.setCanvasLayout()
                for index, val_dict in infoGeo["geo"].items():
                    if self.getSpectrumInfoREST("dim", name=val_dict["name"]) is None:
                        notFound.append(val_dict["name"])
                        continue 

                    self.setGeo(index, val_dict["name"])
                    self.setSpectrumInfo(log=val_dict["scale"], index=index)
                    self.setSpectrumInfo(minx=val_dict["x"][0], index=index)
                    self.setSpectrumInfo(maxx=val_dict["x"][1], index=index)
                    self.setSpectrumInfo(miny=val_dict["y"][0], index=index)
                    self.setSpectrumInfo(maxy=val_dict["y"][1], index=index)

                if len(notFound) > 0:
                    self.logger.warning('loadGeo - definition not found for: %s', notFound)

                self.currentPlot.isLoaded = True
                self.wTab.selected_plot_index_bak[self.wTab.currentIndex()] = None
                self.currentPlot.selected_plot_index = None
                self.currentPlot.next_plot_index = -1

            self.addPlot()
            self.updatePlot()
            self.currentPlot.isLoaded = False
        except TypeError:
            self.logger.debug('loadGeo - TypeError exception', exc_info=True)
            pass


    def openFileNameDialog(self):
        self.logger.info('openFileNameDialog')
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Open file...", "","Window Files (*.win);;All Files (*)", options=options)
        if fileName:
            return fileName


    def saveFileDialog(self):
        self.logger.info('saveFileDialog')
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"Save file...","","Window Files (*.win);;All Files (*)", options=options)
        if fileName:
            return fileName


    #############################
    # 8) Zoom/Scaling operations
    #############################

    # can sets x, y scales for 1d and x, y, z scales for 2d depending on the scale identifier and if axisIsLog
    # basically do all the scaling operations
    def setAxisScale(self, ax, index, *scale):
        self.logger.info('setAxisScale - index: %s', index)

        wPlot = self.currentPlot
        axisIsLog = self.getSpectrumInfo("log", index=index)
        axisIsAutoScale = wPlot.histo_autoscale.isChecked()

        # priority to autoscale value, then if not to user defined value (ex: by zoom), and finally to default value
        # log is set last
        # update spectrumInfo if autoscale and/or log if value <=0

        if (self.getSpectrumInfoREST("dim", index=index) == 1) :
            #x limits need to be known for y autoscale in x range
            xmin = self.getSpectrumInfo("minx", index=index)
            xmax = self.getSpectrumInfo("maxx", index=index)
            if "x" in scale and xmin and xmax:
                ax.set_xlim(xmin,xmax) 
            if "y" or "log" in scale:
                ymin = self.getSpectrumInfo("miny", index=index)
                ymax = self.getSpectrumInfo("maxy", index=index)
                if (not ymin or ymin is None or ymin == 0) and (not ymax or ymax is None or ymax == 0):
                    ymin = self.minY
                    ymax = self.maxY               
                if axisIsAutoScale:
                    #search in the current view
                    xmin, xmax = ax.get_xlim()
                    #getMinMaxInRange returns only max for 1d 
                    ymax = self.getMinMaxInRange(index, xmin=xmin, xmax=xmax)
                if axisIsLog:
                    if ymin <= 0:
                        ymin = 0.001
                    if ymax <= 0:
                        self.logger.warning('setAxisScale - all value <0 for : %s - cannot log scale', self.nameFromIndex(index))
                    else:
                        ax.set_ylim(ymin,ymax)
                        ax.set_yscale("log")
                else:
                    ax.set_ylim(ymin,ymax)
                    ax.set_yscale("linear")
                self.setSpectrumInfo(miny=ymin, index=index)
                self.setSpectrumInfo(maxy=ymax, index=index)
        else:
            #x and y limits need to be known for z autoscale in x,y ranges
            xmin = self.getSpectrumInfo("minx", index=index)
            xmax = self.getSpectrumInfo("maxx", index=index)
            ymin = self.getSpectrumInfo("miny", index=index)
            ymax = self.getSpectrumInfo("maxy", index=index)

            if "x" in scale and xmin and xmax:
                ax.set_xlim(xmin,xmax) 
            if "y" in scale and ymin and ymax:
                ax.set_ylim(ymin,ymax)
            if "z" or "log" in scale:
                zmin = self.getSpectrumInfo("minz", index=index)
                zmax = self.getSpectrumInfo("maxz", index=index)
                spectrum = self.getSpectrumInfo("spectrum", index=index)
                if spectrum is None :
                    return
                if (not zmin or zmin is None or zmin==0) and (not zmax or zmax is None or zmax==0):
                    zmin = self.minZ
                    zmax = self.maxZ
                if axisIsAutoScale:
                    #search in the current view
                    xmin, xmax = ax.get_xlim()
                    ymin, ymax = ax.get_ylim()
                    #getMinMaxInRange returns min and max for 2d 
                    zmin, zmax = self.getMinMaxInRange(index, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
                    self.setSpectrumInfo(maxz=zmax, index=index)
                    self.setSpectrumInfo(minz=zmin, index=index)
                spectrum.set_clim(vmin=zmin, vmax=zmax)
                if axisIsLog:
                    self.setCmapNorm("log", index)
                else :
                    #linearCentered not used so far but could be user choice 
                    #while testing linearCentered noticed that it is not very compatible with cutoff
                    #self.setCmapNorm("linearCentered", index)
                    self.setCmapNorm("linear", index)
                self.setSpectrumInfo(spectrum=spectrum, index=index)


    # Where is defined the color bar
    def setCmapNorm(self, scale, index):
        self.logger.info('setCmapNorm')
        validScales = ["linear", "log", "linearCentered"]
        if scale not in validScales or index is None:
            self.logger.debug('setCmapNorm - scale not in validScales or index is None')
            return
        spectrum = self.getSpectrumInfo("spectrum", index=index)
        zmin, zmax = spectrum.get_clim()

        if scale is validScales[0]:
            if zmin > zmax: 
                self.logger.warning('setCmapNorm - zmin > zmax')
                spectrum.set_norm(colors.Normalize(vmin=self.minZ, vmax=self.maxZ))
            else :
                spectrum.set_norm(colors.Normalize(vmin=zmin, vmax=zmax))
        elif scale is validScales[1]:
            if zmin and zmin <= 0 :
                zmin = 0.001
                self.logger.warning('setCmapNorm - LogNorm with zmin<=0, may want to use CenteredNorm')
            spectrum.set_norm(colors.LogNorm(vmin=zmin, vmax=zmax))
            if zmin > zmax: 
                self.logger.warning('setCmapNorm - zmin > zmax')
                spectrum.set_norm(colors.LogNorm(vmin=self.minZ, vmax=self.maxZ))
        elif scale is validScales[2]:
            palette = copy(plt.cm.jet)
            palette.set_bad(color='white')
            data = self.getSpectrumInfo("data", index=index)
            spectrum.set_cmap(palette)
            # set vcenter to variable set by user
            spectrum.set_norm(centeredNorm(data,50000))
        if self.getEnlargedSpectrum() is None:
            ax = spectrum.axes
            self.removeCb(ax)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            # label used in on_press to avoid interaction with it
            label = "colorbar_"+str(index)
            cax.set_label(label)
            self.currentPlot.figure.colorbar(spectrum, cax=cax, orientation='vertical')


    #Callback for plusButton/minusButton
    def zoomInOut(self, arg):
        self.logger.info('zoomInOut - arg: %s', arg)
        # Simon - added following lines to avoid None plot index
        index = self.autoIndex()
        ax = None
        spectrum = self.getSpectrumInfo("spectrum", index=index)
        if spectrum is None : return
        ax = spectrum.axes
        dim = self.getSpectrumInfoREST("dim", index=index)
        if dim == 1 :
            #step if 0.5
            ymin, ymax = ax.get_ylim()
            if arg == "in" :
                ymax = ymax*0.5
            elif arg == "out" :
                ymax = ymax*2
            ax.set_ylim(ymin, ymax)
            self.setSpectrumInfo(miny=ymin, index=index)
            self.setSpectrumInfo(maxy=ymax, index=index)
            self.setSpectrumInfo(spectrum=spectrum, index=index)
        elif dim == 2 :
            zmin, zmax = spectrum.get_clim()
            if arg == "in" :
                zmax = zmax*0.5
            elif arg == "out" :
                zmax = zmax*2
            spectrum.set_clim(zmin, zmax)
            self.setSpectrumInfo(minz=zmin, index=index)
            self.setSpectrumInfo(maxz=zmax, index=index)
            self.setSpectrumInfo(spectrum=spectrum, index=index)
        self.drawGate(index)
        self.currentPlot.canvas.draw()


    # Callback for histo_autoscale, calls setAxisScale
    def autoScaleAxisBox(self, forIndex):
        self.logger.info('autoScaleAxisBox - forIndex: %s', forIndex)
        try:
            ax = None
            if self.currentPlot.isEnlarged:
                ax = self.getSpectrumInfo("axis", index=0)
                dim = self.getSpectrumInfoREST("dim", index=0)
                if ax is None :
                    self.logger.debug('autoScaleAxisBox - isEnlarged TRUE - ax is None ')
                    return
                #Set y for 1D and z for 2D.
                #dont need to specify if log scale, it is checked inside setAxisScale, if 2D histo in log its z axis is set too.
                if dim == 1:
                    self.setAxisScale(ax, 0, "y")
                elif dim == 2:
                    self.setAxisScale(ax, 0, "z")
                #draw gate if there is one
                self.drawGate(0)
            # implemented this condition to do autoscale in addPlot
            elif forIndex is not None:
                ax = self.getSpectrumInfo("axis", index=forIndex)
                dim = self.getSpectrumInfoREST("dim", index=forIndex)
                if ax is None :
                    self.logger.debug('autoScaleAxisBox - forIndex - ax is None ')
                    return
                #Set y for 1D and z for 2D.
                #dont need to specify if log scale, it is checked inside setAxisScale, if 2D histo in log its z axis is set too.
                if dim == 1:
                    self.setAxisScale(ax, forIndex, "y")
                elif dim == 2:
                    self.setAxisScale(ax, forIndex, "z")
            else:
                for index, name in self.getGeo().items():
                    if name:
                        ax = self.getSpectrumInfo("axis", index=index)
                        dim = self.getSpectrumInfoREST("dim", index=index)
                        if ax is None :
                            self.logger.debug('autoScaleAxisBox - isEnlarged FALSE - ax is None ')
                            return
                        #Set y for 1D and z for 2D.
                        #dont need to specify if log scale, it is checked inside setAxisScale, if 2D histo in log its z axis is set too.
                        if dim == 1:
                            self.setAxisScale(ax, index, "y")
                        elif dim == 2:
                            self.setAxisScale(ax, index, "z")
                        #draw gate if there is one
                        self.drawGate(index)
            self.currentPlot.canvas.draw()
        except:
            pass


    # get data max within user defined range
    # For 2D have to give two ranges (x,y), for 1D range x.
    def getMinMaxInRange(self, index, **limits):
        self.logger.info('getMinMaxInRange - limits: %s', limits)
        result = None
        if not limits :
            self.logger.warning('getMinMaxInRange - limits identifier not valid - expect xmin=val, xmax=val etc. for y with 2D')
            return
        if "xmin" and "xmax" in limits:
            xmin = limits["xmin"]
            xmax = limits["xmax"]
        if "ymin" and "ymax" in limits:
            ymin = limits["ymin"]
            ymax = limits["ymax"]

        dim = self.getSpectrumInfoREST("dim", index=index)
        minx = self.getSpectrumInfoREST("minx", index=index)
        maxx = self.getSpectrumInfoREST("maxx", index=index)
        binx = self.getSpectrumInfoREST("binx", index=index)
        # data = self.getSpectrumInfoREST("data", index=index)
        data = self.getSpectrumInfo("data", index=index)
        stepx = (float(maxx)-float(minx))/float(binx)
        binminx = int((xmin-minx)/stepx)
        binmaxx = int((xmax-minx)/stepx)
        if dim == 1:
            try:
                # get max in x range
                #increase by 10% to get axis view a little bigger than max
                result = data[binminx+1:binmaxx+2].max()*1.1
            except :
                self.logger.debug('getMinMaxInRange - dim == 1 - exception occured', exc_info=True)
                return self.maxY
        elif dim == 2:
            try:
                #get max in x,y ranges
                miny = self.getSpectrumInfoREST("miny", index=index)
                maxy = self.getSpectrumInfoREST("maxy", index=index)
                biny = self.getSpectrumInfoREST("biny", index=index)
                stepy = (float(maxy)-float(miny))/float(biny)
                binminy = int((ymin-miny)/stepy)
                binmaxy = int((ymax-miny)/stepy)
                #Dont increase max by 10% here...
                #truncData = data[binminy:binmaxy+1, binminx:binmaxx+1]
                #Following two lines work for "small" array, replaced by custom function
                #maximum = truncData.max()
                #minimum = np.min(truncData[np.nonzero(truncData)])
                # minimum, maximum = self.customMinMax(data, binminy, binmaxy, binminx, binmaxx)
                minimum, maximum = self.customMinMax(data[binminy:binmaxy+1, binminx:binmaxx+1])
                result = minimum, maximum
            except :
                self.logger.debug('getMinMaxInRange - dim == 2 - exception occured', exc_info=True)
                return self.minZ, self.maxZ
        return result


    # Have seen malloc error if data array too large
    # Divide data array in sub-arrays with sub-(min, max) and then find the global-(min, max)
    def customMinMax(self, data):
        self.logger.info('customMinMax')
        minimum = None
        maximum = None
        nbCol = data.shape[1] 
        nbRow = data.shape[0] 

        #len(data) <= 0
        if not data.any():
            return minimum, maximum
        #Arbitrarily choose number max of col and row of 200...
        if nbCol < 200 and nbRow < 200:
            maximum = data.max()
            minimum = np.min(data[np.nonzero(data)])
            return minimum, maximum
        else :
            stepX = nbCol if nbCol < 200 else 200
            stepY = nbRow if nbRow < 200 else 200
            rangeX = list(range(0, data.shape[1], stepX))
            rangeY = list(range(0, data.shape[0], stepY))
            subMax = []
            subMin = []
            yprev = data.shape[0]+1
            xprev = data.shape[1]+1
            for x in rangeX[::-1]:
                for y in rangeY[::-1]:
                    subData = data[y:yprev,x:xprev]
                    nonZeroIndices = np.where(subData > 0)
                    filteredSubData = subData[nonZeroIndices]
                    if filteredSubData is not None and filteredSubData.size > 0:
                        subMax.append(filteredSubData.max())
                        subMin.append(filteredSubData.min())
                    yprev = y
                xprev = x
            if len(subMin) == 0:
                minimum = self.minZ
            if len(subMax) == 0:
                minimum = self.maxZ
            elif len(subMin)>0 and len(subMax)>0 :
                minimum = min(subMin)
                maximum = max(subMax)

            return minimum, maximum


    #return the axis limits in a certain format [[xmin, xmax], [ymin, ymax]]
    def getAxisProperties(self, index):
        self.logger.info('getAxisProperties')
        try:
            ax = self.getSpectrumInfo("axis", index=index)
            if ax is None :
                return None
            else :
                return list(ax.get_xlim()), list(ax.get_ylim())
        except:
            self.logger.debug('getAxisProperties - exception occured', exc_info=True)
            pass
            

    def zoomCallback(self, event):
        self.logger.info('zoomCallback')
        self.currentPlot.zoomPress = True

    #Used by customZoom button, trigger toolbar zoom action
    def customZoomButtonCallback(self):
        self.logger.info('customZoomButtonCallback')
        # Abord zoom action if one is ongoing
        if self.currentPlot.zoomPress:
            self.currentPlot.canvas.toolbar.actions()[1].triggered.emit()
            self.currentPlot.canvas.toolbar.actions()[1].setChecked(False)
            self.currentPlot.customZoomButton.setDown(False)
            self.currentPlot.zoomPress = False
        else:
            self.currentPlot.canvas.toolbar.actions()[1].triggered.emit()
            self.currentPlot.canvas.toolbar.actions()[1].setChecked(True)
            self.currentPlot.customZoomButton.setDown(True)


    #Used by customHome button, reset the axis limits to ReST definitions, for the specified plot at index or for all plots if index not provided
    def customHomeButtonCallback(self, index=None):
        self.logger.info('customHomeButtonCallback - index: %s', index)

        index_list = [idx for idx, name in self.getGeo().items() if index is None]
        if index is not None:
            index_list = [index]
        for idx in index_list:
            ax = None
            spectrum = self.getSpectrumInfo("spectrum", index=idx)
            if spectrum is None : return
            ax = spectrum.axes
            dim = self.getSpectrumInfoREST("dim", index=idx)
            xmin = self.getSpectrumInfoREST("minx", index=idx)
            xmax = self.getSpectrumInfoREST("maxx", index=idx)
            ymin = self.getSpectrumInfoREST("miny", index=idx)
            ymax = self.getSpectrumInfoREST("maxy", index=idx)

            ax.set_xlim(xmin, xmax)
            if dim == 1:
                #Similar to autoscale, in principle ymin and ymax are not defined in ReST for 1D so set to ymin=0 and autoscale for ymax
                #getMinMaxInRange gives only min for 1d
                ymax = self.getMinMaxInRange(idx, xmin=xmin, xmax=xmax)
                ax.set_ylim(ymin, ymax)
                if self.getSpectrumInfo("log", index=idx) :
                    ax.set_yscale("linear")
            # y limits should be known at this point for both cases 1D/2D
            if dim == 2:
                ax.set_ylim(ymin, ymax)  
                #getMinMaxInRange gives min and max for 2d
                zmin, zmax = self.getMinMaxInRange(idx, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
                spectrum.set_clim(vmin=zmin, vmax=zmax)
                self.setCmapNorm("linear", idx)
                self.setSpectrumInfo(maxz=zmax, index=idx)
                self.setSpectrumInfo(minz=zmin, index=idx)
            self.drawGate(idx)
                
            self.setSpectrumInfo(log=None, index=idx)
            self.setSpectrumInfo(minx=xmin, index=idx)
            self.setSpectrumInfo(maxx=xmax, index=idx)
            self.setSpectrumInfo(miny=ymin, index=idx)
            self.setSpectrumInfo(maxy=ymax, index=idx)
            self.setSpectrumInfo(spectrum=spectrum, index=idx)
        self.currentPlot.canvas.draw()


    #Used by logButton, defines the log scale, for the specified plot at index or for all plots if logAll/unlogAll, calls setAxisScale
    def logButtonCallback(self, *arg):
        self.logger.info('logButtonCallback - arg: %s', arg)

        index = None
        logAllPlot = False
        unlogAllPlot = False
        if "logAll" in arg:
            logAllPlot = True
        elif "unlogAll" in arg:
            unlogAllPlot = True
        else :
            index = arg[0]

        wPlot = self.currentPlot
        index_list = [idx for idx, name in self.getGeo().items() if logAllPlot or unlogAllPlot]

        if index is not None:
            index_list = [index]
        for idx in index_list:
            ax = None
            # spectrum = self.getSpectrum(idx)
            spectrum = self.getSpectrumInfo("spectrum", index=idx)
            if spectrum is None : continue 
            ax = spectrum.axes
            # only place where the log spectrum info is set 
            # so if log now it needs to switch to linear, and vice et versa
            if logAllPlot :
                self.setSpectrumInfo(index=idx, log=True)
            elif unlogAllPlot :
                self.setSpectrumInfo(index=idx, log=False)
            elif self.getSpectrumInfo("log", index=idx) and not logAllPlot and not unlogAllPlot:
                self.setSpectrumInfo(index=idx, log=False)
            elif not self.getSpectrumInfo("log", index=idx) and not logAllPlot and not unlogAllPlot:
                self.setSpectrumInfo(index=idx, log=True)

            self.setAxisScale(ax, idx, "log")
        wPlot.canvas.draw()


    #callback when right click on customZoomButton
    def zoom_handle_right_click(self):
        self.logger.info('zoom_handle_right_click')
        menu = QMenu()
        item1 = menu.addAction("Set range") 
        #Empty agrument for customHomeButtonCallback means it will reset all spectra
        item1.triggered.connect(self.customZoomSetRange)
        plotgui = self.currentPlot
        menuPosX = plotgui.mapToGlobal(QtCore.QPoint(0,0)).x() + plotgui.customZoomButton.geometry().topLeft().x()
        menuPosY = plotgui.mapToGlobal(QtCore.QPoint(0,0)).y() + plotgui.customZoomButton.geometry().topLeft().y()
        menuPos = QtCore.QPoint(menuPosX, menuPosY)
        # Shows menu at button position, need to calibrate with 0,0 position
        menu.exec_(menuPos) 


    def customZoomSetRange(self):
        self.logger.info('customZoomSetRange')
        # Set fields to current spectrum range, as starting points 
        index = self.currentPlot.selected_plot_index
        if index is None : 
            return QMessageBox.about(self,"Warning!", "Please add/select a spectrum")
        ax = self.getSpectrumInfo("axis", index=index)
        if ax is None :
            self.logger.warning('customZoomSetRange - axis is None')
            return
        name = self.getSpectrumInfo("name", index=index)
        dim = self.getSpectrumInfo("dim", index=index)
        self.zoomSetRangePopup.setWindowTitle(f"Set zoom for: {name}")
        self.zoomSetRangePopup.indexSpectrum = index
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        self.zoomSetRangePopup.lineeditXmin.setText(f"{xmin:.2f}")
        self.zoomSetRangePopup.lineeditXmax.setText(f"{xmax:.2f}")
        self.zoomSetRangePopup.lineeditYmin.setText(f"{ymin:.2f}")
        self.zoomSetRangePopup.lineeditYmax.setText(f"{ymax:.2f}")
        # Adapt dialogue layout with dim
        if dim == 1 :
            self.zoomSetRangePopup.layout1d()
        elif dim == 2 :
            self.zoomSetRangePopup.layout2d()
        self.zoomSetRangePopup.show()


    def cancelZoomSetRange(self):
        self.logger.info('cancelZoomSetRange')
        self.zoomSetRangePopup.lineeditXmin.clear()
        self.zoomSetRangePopup.lineeditXmax.clear()
        self.zoomSetRangePopup.lineeditYmin.clear()
        self.zoomSetRangePopup.lineeditYmax.clear()
        self.zoomSetRangePopup.close()
    

    def okZoomSetRange(self):
        self.logger.info('okZoomSetRange')
        index = self.zoomSetRangePopup.indexSpectrum
        ax = self.getSpectrumInfo("axis", index=index)
        if ax is None :
            self.logger.debug('okZoomSetRange - ax is None')
            return
        dim = self.getSpectrumInfoREST("dim", index=index)
        rangeXmin = self.zoomSetRangePopup.lineeditXmin.text()
        rangeXmax = self.zoomSetRangePopup.lineeditXmax.text()
        rangeYmin = self.zoomSetRangePopup.lineeditYmin.text()
        rangeYmax = self.zoomSetRangePopup.lineeditYmax.text()

        # Convert and check expected format
        try:
            rangeXmin = float(rangeXmin)
            rangeXmax = float(rangeXmax)
            rangeYmin = float(rangeYmin)
            rangeYmax = float(rangeYmax)
        except ValueError:
            self.logger.warning("okZoomSetRange - Invalid input format for zoom range(s). Please enter valid numbers.")
            return
        
        #Order X min/max may be inverted
        if rangeXmin is not None and rangeXmax is not None and rangeXmax < rangeXmin:
            buff = rangeXmin
            rangeXmin = rangeXmax
            rangeXmax = buff
            self.logger.warning('okZoomSetRange - new range X values swapped because min > max')
        #Order Y min/max may be inverted
        if rangeYmin is not None and rangeYmax is not None and rangeYmax < rangeYmin:
            buff = rangeYmin
            rangeYmin = rangeYmax
            rangeYmax = buff
            self.logger.warning('okZoomSetRange - new range Y values swapped because min > max')

        try:
            #Set new axis limits and save new range in spectrum dict 
            spectrum = self.getSpectrumInfo("spectrum", index=index)
            if dim == 1 :
                ax.set_xlim(float(rangeXmin), float(rangeXmax))
                self.setSpectrumInfo(minx=rangeXmin, index=index)
                self.setSpectrumInfo(maxx=rangeXmax, index=index)
            elif dim == 2 :
                ax.set_xlim(rangeXmin, rangeXmax)
                ax.set_ylim(rangeYmin, rangeYmax)
                self.setSpectrumInfo(minx=rangeXmin, index=index)
                self.setSpectrumInfo(maxx=rangeXmax, index=index)
                self.setSpectrumInfo(miny=rangeYmin, index=index)
                self.setSpectrumInfo(maxy=rangeYmax, index=index)
            self.setSpectrumInfo(spectrum=spectrum, index=index)
            self.drawGate(index)
            self.currentPlot.canvas.draw()
        except NameError as err:
            self.logger.debug('okZoomSetRange - NameError', exc_info=True)
            pass

        self.zoomSetRangePopup.close()



    #callback when right click on customHomeButton
    def handle_right_click(self):
        self.logger.info('handle_right_click')
        menu = QMenu()
        item1 = menu.addAction("Reset all") 
        #Empty agrument for customHomeButtonCallback means it will reset all spectra
        item1.triggered.connect(lambda: self.customHomeButtonCallback())
        plotgui = self.currentPlot
        menuPosX = plotgui.mapToGlobal(QtCore.QPoint(0,0)).x() + plotgui.customHomeButton.geometry().topLeft().x()
        menuPosY = plotgui.mapToGlobal(QtCore.QPoint(0,0)).y() + plotgui.customHomeButton.geometry().topLeft().y()
        menuPos = QtCore.QPoint(menuPosX, menuPosY)
        # Shows menu at button position, need to calibrate with 0,0 position
        menu.exec_(menuPos)     


    #callback when right click on logButton
    def log_handle_right_click(self):
        self.logger.info('log_handle_right_click')
        menu = QMenu()
        item1 = menu.addAction("Log all")
        item2 = menu.addAction("unLog all")
        #Empty agrument for logButtonCallback means it will set log for all spectra
        item1.triggered.connect(lambda: self.logButtonCallback("logAll"))
        item2.triggered.connect(lambda: self.logButtonCallback("unlogAll"))
        plotgui = self.currentPlot
        menuPosX = plotgui.mapToGlobal(QtCore.QPoint(0,0)).x() + plotgui.logButton.geometry().topLeft().x()
        menuPosY = plotgui.mapToGlobal(QtCore.QPoint(0,0)).y() + plotgui.logButton.geometry().topLeft().y()
        menuPos = QtCore.QPoint(menuPosX, menuPosY)
        # Shows menu at button position, need to calibrate with 0,0 position
        menu.exec_(menuPos)


    def cutoff_handle_right_click(self):
        pass 


    #button of the cutoff window, sets the cutoff values in the spectrum dict
    def okCutoff(self):
        self.logger.info('okCutoff')
        index = self.currentPlot.selected_plot_index
        if index is None : 
            self.logger.debug('okCutoff - index is None')
            return 

        cutoffVal = [None, None]
        cutoffMin = self.cutoffp.lineeditMin.text()
        cutoffMax = self.cutoffp.lineeditMax.text()
        #check expected format and save cutoff in spectrum dict 
        if self.cutoffp.lineeditMin.text() != "" and self.cutoffp.lineeditMin.text().isdigit():
            cutoffVal[0] = float(cutoffMin)
            self.setSpectrumInfo(cutoff=cutoffVal, index=index)
        if self.cutoffp.lineeditMax.text() != "" and self.cutoffp.lineeditMax.text().isdigit():
            cutoffVal[1] = float(cutoffMax)
            self.setSpectrumInfo(cutoff=cutoffVal, index=index)
        #Order may be inverted
        if cutoffVal[0] is not None and cutoffVal[1] is not None and cutoffVal[1] < cutoffVal[0]:
            cutoffVal = [cutoffVal[1], cutoffVal[0]]
            self.setSpectrumInfo(cutoff=cutoffVal, index=index)
            self.logger.warning('okCutoff - cutoff values swapped because min > max')
        self.updatePlot()
        self.cutoffp.close()


    def cancelCutoff(self):
        self.cutoffp.close()


    def resetCutoff(self, doUpdate):
        self.logger.info('resetCutoff - doUpdate: %s', doUpdate)
        index = self.currentPlot.selected_plot_index
        if index is None : return 
        cutoffVal = [None, None]
        self.setSpectrumInfo(cutoff=cutoffVal, index=index)
        if doUpdate:
            self.updatePlot()
        self.cutoffp.close()



    #called by cutoffButton, sets the information in the cutoff window
    def cutoffButtonCallback(self, *arg):
        self.logger.info('cutoffButtonCallback')
        index = self.currentPlot.selected_plot_index
        name = self.nameFromIndex(index)
        if name is not None : 
            self.cutoffp.setWindowTitle("Set cutoff for: " + name)
        else :
            self.cutoffp.setWindowTitle("Set cutoff for: ???" )
        self.cutoffp.setGeometry(300,100,300,100)
        if self.cutoffp.isVisible():
            self.cutoffp.close()
        self.cutoffp.show()

        if self.getSpectrumInfo("cutoff", index=index) is not None and len(self.getSpectrumInfo("cutoff", index=index)) > 0:
            cutoffMin, cutoffMax = self.getSpectrumInfo("cutoff", index=index)
            self.cutoffp.lineeditMin.setText(f"{cutoffMin}")
            self.cutoffp.lineeditMax.setText(f"{cutoffMax}")

        #try to deal with all cases to inform user
        if index is None:
            self.cutoffp.lineeditMin.setText("Select spectrum")
            self.cutoffp.lineeditMax.setText("Select spectrum") 
            return
        if self.getSpectrumInfo("cutoff", index=index) is None:
            self.cutoffp.lineeditMin.setText("None")
            self.cutoffp.lineeditMax.setText("None")
        elif self.getSpectrumInfo("cutoff", index=index) is not None and len(self.getSpectrumInfo("cutoff", index=index)) == 0:
            self.cutoffp.lineeditMin.setText("None")
            self.cutoffp.lineeditMax.setText("None")
        elif self.getSpectrumInfo("cutoff", index=index) is not None and len(self.getSpectrumInfo("cutoff", index=index)) > 0:
            cutoffMin, cutoffMax = self.getSpectrumInfo("cutoff", index=index)
            self.cutoffp.lineeditMin.setText(f"{cutoffMin}")
            self.cutoffp.lineeditMax.setText(f"{cutoffMax}")
        else :
            self.logger.warning('cutoffButtonCallback - only one cutoff value in spectrum dict: %s', self.getSpectrumInfo("cutoff", index=index))
            # print("cutoffButtonCallback - warning - only one cutoff value in spectrum dict: ", self.getSpectrumInfo("cutoff", index=index))


    #Used in zoomCallBack to save the new axis limits
    #sleepTime is a small delay to ensure this function is executed after on_release
    #seems necessary to get the updated axis limits (zoom toolbar action ends on_release)
    def updatePlotLimits(self, sleepTime=0):
        self.logger.info('updatePlotLimits - sleepTime: %s', sleepTime)
        # update currentPlot limits with what's on the actual plot
        ax = None
        index = self.currentPlot.selected_plot_index
        ax = self.getSpectrumInfo("axis", index=index)
        if ax is None : 
            self.logger.debug('updatePlotLimits - ax is None')
            return
        # im = ax.images

        time.sleep(sleepTime)

        try:
            x_range, y_range = self.getAxisProperties(index)
            self.setSpectrumInfo(minx=x_range[0], index=index)
            self.setSpectrumInfo(maxx=x_range[1], index=index)
            self.setSpectrumInfo(miny=y_range[0], index=index)
            self.setSpectrumInfo(maxy=y_range[1], index=index)

            #Set axis limits try with spectrum 
            spectrum = self.getSpectrumInfo("spectrum", index=index)
            ax.set_xlim(x_range[0], x_range[1])
            ax.set_ylim(y_range[0], y_range[1])

            # spectrum = self.getSpectrumInfo("spectrum", index=index)
            # spectrum.axes.set_xlim(x_range[0], x_range[1])
            # spectrum.axes.set_ylim(y_range[0], y_range[1])
            self.setSpectrumInfo(spectrum=spectrum, index=index)
            self.drawGate(index)
        except NameError as err:
            self.logger.debug('updatePlotLimits - NameError', exc_info=True)
            print(err)
            pass


    ##################################
    ## 9) Histogram operations
    ##################################

    # remove colorbar
    def removeCb(self, axis):
        im = axis.images
        if im is not None and len(im) > 0:
            try:
                cb = im[-1].colorbar
                cb.remove()
            except :
                self.logger.debug('removeCb - IndexError exception', exc_info=True)
                pass


    # select axes based on indexing, used only in add()
    def select_plot(self, index):
        self.logger.info('select_plot - index: %s', index)
        for i, axis in enumerate(self.currentPlot.figure.axes):
            # retrieve the subplot from the click
            if (i == index and axis is not None):
                return axis


    # returns position in grid based on indexing
    def plotPosition(self, index):
        self.logger.info('plotPosition - index: %s', index)
        cntr = 0
        # convert index to position in geometry
        canvasLayout = self.wTab.layout[self.wTab.currentIndex()]
        for i in range(canvasLayout[0]):
            for j in range(canvasLayout[1]):
                if index == cntr:
                    return i, j
                else:
                    cntr += 1


    # setup histogram limits according to the ReST info
    # called in add(), when the plot is first added
    def setupPlot(self, axis, index):
        self.logger.info('setupPlot - index: %s', index)
        if self.nameFromIndex(index):

            dim = self.getSpectrumInfoREST("dim", index=index)
            minx = self.getSpectrumInfoREST("minx", index=index)
            maxx = self.getSpectrumInfoREST("maxx", index=index)
            binx = self.getSpectrumInfoREST("binx", index=index)

            w = self.getSpectrumInfoREST("data", index=index)

            if self.getSpectrumInfo("cutoff", index=index) is not None:
                if len(self.getSpectrumInfo("cutoff", index=index))>0:
                    #if min/maxCutoff mask data bellow/above the cutoff values
                    minCutoff = self.getSpectrumInfo("cutoff", index=index)[0]
                    maxCutoff = self.getSpectrumInfo("cutoff", index=index)[1]
                    if minCutoff is not None:
                        if dim == 1:
                            w = np.ma.masked_where(w < minCutoff, w)
                        if dim == 2:
                            w = np.ma.masked_where(w < minCutoff, w)
                    if maxCutoff is not None:
                        if dim == 1:
                            w = np.ma.masked_where(w < maxCutoff, w)
                        if dim == 2:
                            w = np.ma.masked_where(w < maxCutoff, w)
            #used in getMinMaxInRange to take into account also the cutoff if there is one
            self.setSpectrumInfo(data=w, index=index)

            # update axis
            if dim == 1:
                axis.set_xlim(minx,maxx)
                axis.set_ylim(self.minY,self.maxY)
                # create histogram
                line, = axis.plot([], [], drawstyle='steps')


                # axisTest = self.getSpectrumInfo("axis", index=index)
                # self.axesChildsTest(axisTest)
                # print("Simon - in setupPlot before spectrum=line- ")

                self.setSpectrumInfo(spectrum=line, index=index)
                if len(w) > 0:
                    X = np.array(self.createRange(binx, minx, maxx))
                    line.set_data(X, w)
                    self.setSpectrumInfo(spectrum=line, index=index)
                
                # self.axesChilds()
                # print("Simon - in setupPlot after spectrum=line- ")
            else:
                miny = self.getSpectrumInfoREST("miny", index=index)
                maxy = self.getSpectrumInfoREST("maxy", index=index)
                biny = self.getSpectrumInfoREST("biny", index=index)
                # empty data for initialization
                if w is None:
                    w = np.zeros((int(binx), int(biny))) 

                self.palette = copy(plt.cm.plasma)
                w = np.ma.masked_where(w == 0, w)
                self.palette.set_bad(color='white')

                #check if enlarged mode, dont want to modify spectrum dict in enlarged mode
                self.setSpectrumInfo(spectrum=axis.imshow(w,
                                                            interpolation='none',
                                                            extent=[float(minx),float(maxx),float(miny),float(maxy)],
                                                            aspect='auto',
                                                            origin='lower',
                                                            vmin=float(self.minZ), vmax=float(self.maxZ),
                                                            cmap=self.palette), index=index)

                if w is not None :
                    spectrum = self.getSpectrumInfo("spectrum", index=index)
                    spectrum.set_data(w)
                    self.setSpectrumInfo(spectrum=spectrum, index=index)

                # setup colorbar only for 2D
                if self.getEnlargedSpectrum() is None:
                    divider = make_axes_locatable(axis)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    # label used in on_press to avoid interaction with it
                    label = "colorbar_"+str(index)
                    cax.set_label(label)
                    self.currentPlot.figure.colorbar(spectrum, cax=cax, orientation='vertical')


    # geometrically add plots to the right place and calls plotting
    # should be called only by addPlot and on_dblclick when entering/exiting enlarged mode
    def add(self, index):
        self.logger.info('add - index: %s',index)
        a = None
        if self.currentPlot.isEnlarged:
            #following line to work with multiple tabs, otherwise the default current axes are in the latest tab
            a = self.currentPlot.figure.get_axes()[0]
        else:
            a = self.select_plot(index)
        #clear plot and if 2D remove color bar
        try:
            self.removeCb(a)
        except:
            pass
        a.clear()
        #set lines 1D or 2D properties and plot limits with spectrum limits
        self.setupPlot(a, index)


    # Callback for histo_geo_add button
    # also called in loadGeo
    # geometrically add plots to the right place
    # plot axis as defined in the ReST interface.
    def addPlot(self):
        self.logger.info('addPlot')

        if self.wConf.histo_list.count() == 0 : 
            QMessageBox.about(self, "Warning", 'Please click on "Connection" and fill in the information')

        try:
            # if we load the geometry from file
            if self.currentPlot.isLoaded:
                self.logger.debug('addPlot - isLoaded TRUE - getGeo: %s',self.getGeo())
                # self.getGeo()
                # for key, value in self.getGeo().items():
                #     index = self.wConf.histo_list.findText(value, QtCore.Qt.MatchFixedString)
                #     if self.getSpectrumInfoREST("dim", name=value) is None: 
                #         self.logger.debug('addPlot - isLoaded TRUE - self.getSpectrumInfoREST("dim", name=value) is None - key, value, getGeo: %s, %s, %s', key, value, self.getGeo())
                #         return
                #     # changing the index to the correct histogram to load
                #     self.wConf.histo_list.setCurrentIndex(index)
                #it turns out the autoscale is wanted in addPlot
                self.currentPlot.histo_autoscale.setChecked(True)
                for key, value in self.getGeo().items():
                    if self.getSpectrumInfoREST("dim", name=value) is None: 
                        continue
                    self.add(key)
                    self.autoScaleAxisBox(key)
            # self adding
            else:
                index = self.nextIndex()
                #Set the plot according to the selected name in the spectrum list widget
                name = str(self.wConf.histo_list.currentText())
                self.logger.debug('addPlot - isLoaded FALSE - index, name: %s, %s',index,name)

                if self.getSpectrumInfoREST("dim", name=name) is None: 
                    self.logger.debug('addPlot - isLoaded FALSE - dim is None')
                    return

                #Reset cutoff
                self.setSpectrumInfo(cutoff=None, index=index)

                self.setGeo(index, name)
                self.add(index)

                #it turns out the autoscale is wanted in addPlot
                self.currentPlot.histo_autoscale.setChecked(True)
                self.autoScaleAxisBox(index)

                #When add with button "add" the default state is unlog
                self.currentPlot.logButton.setDown(False)
                #Reset log
                self.setSpectrumInfo(log=False, index=index)

                #draw gates
                self.drawGate(index)

                #draw dashed red rectangle to indicate where the next plot would be added, based on next_plot_index, selected_plot_index is unchanged.
                #recDashed added only here
                self.removeRectangle()
                self.currentPlot.recDashed = self.createDashedRectangle(self.currentPlot.figure.axes[self.currentPlot.next_plot_index])

                try:
                    self.currentPlot.figure.tight_layout()
                except ValueError:
                    self.logger.debug('addPlot - ValueError exception', exc_info=True)
                    pass

                # print("Simon - in drawPlot - before canvas draw")
                # for testAx in self.currentPlot.figure.axes:
                #     axisLines = testAx.get_lines()
                #     sumRegionLines = [line for line in axisLines if "sumReg_-_" in line.get_label()]
                #     print("Simon - testAx sumRegionLines ", sumRegionLines)
                #     for line in sumRegionLines:
                #         line.draw_artist()
                # self.axesChilds()

                self.currentPlot.canvas.draw()
                self.currentPlot.isSelected = False
        except NameError:
            raise

        # (Re)start auto update once there is a spectrum
        if self.stopAutoUpdateThread.is_set():
            self.autoUpdateStart()

        # When create a new tab, signals are enabled once a spectrum is added 
        if not  self.wTab.countClickTab[self.wTab.currentIndex()]:
            self.bindDynamicSignal()


    #why not using np.linspace(vmin, vmax, bins)
    def createRange(self, bins, vmin, vmax):
        self.logger.info('createRange')
        x = []
        step = (float(vmax)-float(vmin))/float(bins)
        for i in np.arange(float(vmin), float(vmax), step):
            x.append(i + step)
        x.insert(0, float(vmin))
        return x


    # fill spectrum with new data
    # called in addPlot and updatePlot
    # dont actually draw the plot in this function
    def plotPlot(self, index):
        self.logger.info('plotPlot - index: %s', index)
        currentPlot = self.currentPlot

        # Use spectrumInfoREST dont want to change the resolution etc.
        dim = self.getSpectrumInfoREST("dim", index=index)
        minx = self.getSpectrumInfoREST("minx", index=index)
        maxx = self.getSpectrumInfoREST("maxx", index=index)
        binx = self.getSpectrumInfoREST("binx", index=index)
        spectrum = self.getSpectrumInfo("spectrum", index=index)
        w = self.getSpectrumInfoREST("data", index=index)

        if self.getSpectrumInfo("cutoff", index=index) is not None:
            if len(self.getSpectrumInfo("cutoff", index=index))>0:
                #if min/maxCutoff mask data bellow/above the cutoff values
                minCutoff = self.getSpectrumInfo("cutoff", index=index)[0]
                maxCutoff = self.getSpectrumInfo("cutoff", index=index)[1]
                if minCutoff is not None:
                    if dim == 1:
                        w = np.ma.masked_where(w < minCutoff, w)
                    if dim == 2:
                        w = np.ma.masked_where(w < minCutoff, w)
                if maxCutoff is not None:
                    if dim == 1:
                        w = np.ma.masked_where(w > maxCutoff, w)
                    if dim == 2:
                        w = np.ma.masked_where(w > maxCutoff, w)
        if w is None or len(w) <= 0 :
            self.logger.debug('plotPlot - w is None or len(w) <= 0')
            return
        #used in getMinMaxInRange to take into account also the cutoff if there is one
        self.setSpectrumInfo(data=w, index=index)

        if dim == 1:
            X = np.array(self.createRange(binx, minx, maxx))
            spectrum.set_data(X, w)
        else:
            # color modes for later...
            # if (self.wConf.button2D_option.currentText() == 'Light'):
            w = np.ma.masked_where(w == 0, w)
            spectrum.set_data(w)

        self.setSpectrumInfo(spectrum=spectrum, index=index)
        self.currentPlot = currentPlot


    #Callback for histo_geo_update button
    #also used in various functions
    #redraw plot spectrum, update axis scales, redraw gates
    def updatePlot(self):
        self.logger.info('updatePlot')


        # name = str(self.wConf.histo_list.currentText())
        # index = self.autoIndex()
        # name = self.nameFromIndex(index)
        # print("Simon - updatePlot - index, name:", index,name)
        # if index is None or self.getSpectrumInfoREST("dim", name=name) is None: 
        #     self.logger.debug('updatePlot - index is None or self.getSpectrumInfoREST("dim", name=name) is None')
        #     return
            
        # if gatePopup or sumRegionPopup exit with [X]
        self.cleanPopupExit(False)

        try:
            #x_range, y_range = self.getAxisProperties(index)
            if self.currentPlot.isEnlarged:
                index = self.autoIndex()
                name = self.nameFromIndex(index)
                if index is None or self.getSpectrumInfoREST("dim", name=name) is None: 
                    self.logger.debug('updatePlot - index is None or self.getSpectrumInfoREST("dim", name=name) is None')
                    return
                self.logger.debug('updatePlot - self.currentPlot.isEnlarged TRUE')
                ax = self.getSpectrumInfo("axis", index=0)
                if ax is None :
                    self.logger.debug('updatePlot - ax is None')
                    if len(self.getSpectrumInfoDict()) == 0:
                        return QMessageBox.about(self,"Warning!", "Configuration file has probably changed, please reset the window geometry (add plots or load geo file)")
                    return
                self.plotPlot(index)
                #reset the axis limits as it was before enlarge
                #dont need to specify if log scale, it is checked inside setAxisScale, if 2D histo in log its z axis is set too.
                dim = self.getSpectrumInfoREST("dim", index=0)
                if dim == 1:
                    self.setAxisScale(ax, 0, "x", "y")
                elif dim == 2:
                    self.setAxisScale(ax, 0, "x", "y", "z")
                try:
                    self.removeCb(ax)
                except:
                    pass
                #draw gate if there is one
                self.drawGate(0)
            else:
                self.logger.debug('updatePlot - self.currentPlot.isEnlarged FALSE')
                for index, value in self.getGeo().items():
                    ax = self.getSpectrumInfo("axis", index=index)
                    if ax is None :
                        self.logger.debug('updatePlot - ax is None')
                        if len(self.getSpectrumInfoDict()) == 0:
                            return QMessageBox.about(self,"Warning!", "Configuration file has probably changed, please reset the window geometry (add plots or load geo file)")
                        continue
                    self.plotPlot(index)
                    #reset the axis limits as it was before enlarge
                    #dont need to specify if log scale, it is checked inside setAxisScale, if 2D histo in log its z axis is set too.
                    dim = self.getSpectrumInfoREST("dim", index=index)
                    if dim == 1:
                        self.setAxisScale(ax, index, "x", "y")
                    elif dim == 2:
                        self.setAxisScale(ax, index, "x", "y", "z")
                    else :
                        continue
                    #draw gate if there is one
                    self.drawGate(index)

            self.currentPlot.figure.tight_layout()
            self.currentPlot.canvas.draw()
        except NameError:
            self.logger.debug('updatePlot - NameError exception', exc_info=True)
            pass
            #raise


    # looking for first available index to add an histogram
    def check_index(self):
        self.logger.info('check_index')
        keys=list(self.currentPlot.h_dict.keys())
        values = []
        try:
            values = [value["name"] for value in self.currentPlot.h_dict.values()]
        except TypeError as err:
            self.logger.warning('check_index - TypeError exception')
            # print(err)
            return
        if "empty" in values:
            self.currentPlot.index = keys[values.index("empty")]
        else:
            self.currentPlot.index = keys[-1]
            self.currentPlot.isFull = True
        return self.currentPlot.index


    #To avoid None plot index
    def autoIndex(self):
        self.logger.info('autoIndex')
        if self.currentPlot.isSelected == False or self.currentPlot.selected_plot_index is None:
            if self.wTab.selected_plot_index_bak[self.wTab.currentIndex()] is not None:
                self.currentPlot.index = self.wTab.selected_plot_index_bak[self.wTab.currentIndex()]
            else :
                self.currentPlot.index = self.check_index()
        else:
            self.currentPlot.index = self.currentPlot.selected_plot_index
            self.wTab.selected_plot_index_bak[self.wTab.currentIndex()]= self.currentPlot.selected_plot_index

        return self.currentPlot.index


    #go to next index, used in addPlot, so that one can add spectrum without selecting everytime the pad where to draw
    def nextIndex(self):
        self.logger.info('nextIndex')
        tabIndex = self.wTab.currentIndex()

        #Try to deal with all cases... not elegant
        #first case when ex: coming back from zoom mode or if no plot selected
        if self.currentPlot.selected_plot_index is None:
            if self.wTab.selected_plot_index_bak[tabIndex] is not None:
                self.forNextIndex(tabIndex,self.currentPlot.next_plot_index)
            else :
                self.currentPlot.index = self.check_index()
                self.wTab.selected_plot_index_bak[tabIndex]= self.currentPlot.index
                self.currentPlot.next_plot_index = self.setIndex(self.currentPlot.next_plot_index)

        #second case when select a plot before clicking "Add"
        elif self.currentPlot.selected_plot_index == self.currentPlot.next_plot_index:
            self.currentPlot.index = self.currentPlot.selected_plot_index
            self.wTab.selected_plot_index_bak[tabIndex]= self.currentPlot.selected_plot_index
            self.currentPlot.next_plot_index = self.setIndex(self.currentPlot.next_plot_index)

        #third case when click "Add" without selecting a plot, will draw in the next frame
        elif self.currentPlot.selected_plot_index != self.currentPlot.next_plot_index and self.currentPlot.next_plot_index>=0:
            self.forNextIndex(tabIndex,self.currentPlot.next_plot_index)

        return self.currentPlot.index


    #called in nextIndex
    def forNextIndex(self, tabIndex, index):
        self.logger.info('forNextIndex')
        self.currentPlot.index = index
        self.currentPlot.selected_plot_index = index
        self.wTab.selected_plot_index_bak[tabIndex]= self.currentPlot.selected_plot_index
        self.currentPlot.next_plot_index = self.setIndex(index)

    #called in nextIndex and forNextIndex
    def setIndex(self, indexToChange):
        self.logger.info('setIndex')
        row = int(self.wConf.histo_geo_row.currentText())
        col = int(self.wConf.histo_geo_col.currentText())
        try:
            #Once in the nextIndex first case change to next_plot_index to 0 (then +1)
            if indexToChange == -1:
                indexToChange = 0
            if indexToChange <= row*col-1:
                if indexToChange == row*col-1:
                    indexToChange = 0
                else:
                    indexToChange += 1
        except IndexError as e:
            self.logger.warning('setIndex - IndexError occured', exc_info=True)
            # print(f"An IndexError occured: {e}")
        return indexToChange


    #callback for copyAttr.okAttr
    def okCopy(self):
        self.logger.info('okCopy')
        self.applyCopy()
        self.closeCopy()

    #callback for copyAttr.applyAttr
    def applyCopy(self):
        self.logger.info('applyCopy')
        try:
            flags = []
            discard = ["Ok", "Cancel", "Apply", "Select all", "Deselect all"]

            for instance in self.copyAttr.findChildren(QCheckBox):
                if instance.isChecked():
                    flags.append(True)
                else:
                    flags.append(False)

            self.logger.debug('applyCopy - flags: %s', flags)

            dim = self.getSpectrumInfoREST("dim", index=self.currentPlot.selected_plot_index)
            indexes = []
            xlim_src = []
            ylim_src = []
            zlim_src = []
            scale_src = None

            # creating list of target histograms
            for instance in self.copyAttr.findChildren(QPushButton):
                if instance.text() not in discard and instance.isChecked():
                    labelPos = self.copyAttr.copy_log.labelForField(instance)
                    #following gives [row, col]
                    geoPositionSpectrum = [int(i) for i in labelPos.text().split() if i.isdigit()]
                    indexSpectrum = int(self.wConf.histo_geo_col.currentText())*geoPositionSpectrum[0]+geoPositionSpectrum[1]
                    indexes.append(indexSpectrum)

            self.logger.debug('applyCopy - indexes : %s', indexes)


            # src values to copy to destination
            xlim_src = ast.literal_eval(self.copyAttr.axisLimLabelX.text())
            ylim_src = ast.literal_eval(self.copyAttr.axisLimLabelY.text())
            scale_src = self.copyAttr.axisSLabel.text()
            scale_src_bool = True if scale_src == "Log" else False
            zlim_src = [float(self.copyAttr.histoScaleValueminZ.text()), float(self.copyAttr.histoScaleValuemaxZ.text())]

            self.logger.debug('applyCopy - xlim_src, ylim_src, scale_src, zlim_src : %s, %s, %s, %s', xlim_src, ylim_src, scale_src, zlim_src)

            # copy to destination
            for index in indexes:
                # set the limits for x,y
                if flags[0]:
                    self.setSpectrumInfo(minx=xlim_src[0], index=index)
                    self.setSpectrumInfo(maxx=xlim_src[1], index=index)
                if flags[1]:
                    self.setSpectrumInfo(miny=ylim_src[0], index=index)
                    self.setSpectrumInfo(maxy=ylim_src[1], index=index)
                # set log/lin scale
                if flags[2]:
                    self.setSpectrumInfo(log=scale_src_bool, index=index)
                # set minZ/maxZ
                if dim == 2 and (flags[3] or flags[4]):
                    self.setSpectrumInfo(minz=zlim_src[0], index=index)
                    self.setSpectrumInfo(maxz=zlim_src[1], index=index)
            self.updatePlot()
        except:
            self.logger.debug('applyCopy - exception occured', exc_info=True)
            pass


    #callback for copyAttr.cancelAttr
    def closeCopy(self):
        self.logger.info('closeCopy')
        discard = ["Ok", "Cancel", "Apply", "Select all", "Deselect all"]
        for instance in self.copyAttr.findChildren(QPushButton):
            if instance.text() not in discard:
                instance.deleteLater()

        self.copyAttr.close()


    #open copy properties popup
    def copyPopup(self):
        self.logger.info('copyPopup - self.currentPlot.selected_plot_index: %s', self.currentPlot.selected_plot_index)
        if self.copyAttr.isVisible():
            self.copyAttr.close()
        index = self.currentPlot.selected_plot_index
        name = self.nameFromIndex(index)
        dim = self.getSpectrumInfoREST("dim", index=index)

        if dim is None : 
            self.logger.debug('copyPopup - dim is None', exc_info=True)
            return

        # setting up info for source histogram
        self.copyAttr.histoLabel.setText(name)
        # hdim = 2 if self.wConf.button2D.isChecked() else 1
        if dim == 2 :
            spectrum = self.getSpectrumInfo("spectrum", index=index)
            zmin, zmax = spectrum.get_clim()
            self.copyAttr.histoScaleValueminZ.setText(f"{zmin}")
            self.copyAttr.histoScaleValuemaxZ.setText(f"{zmax}")
        self.copyAttr.axisSLabel.setText("Log" if self.getSpectrumInfo("log", index=index) else "Linear")
        xmin = self.getSpectrumInfo("minx", index=index)
        xmax = self.getSpectrumInfo("maxx", index=index)
        ymin = self.getSpectrumInfo("miny", index=index)
        ymax = self.getSpectrumInfo("maxy", index=index)
        self.copyAttr.axisLimLabelX.setText(f"[{xmin:.1f},{xmax:.1f}]")
        self.copyAttr.axisLimLabelY.setText(f"[{ymin:.1f},{ymax:.1f}]")

        #reset QFormLayout
        rowCount = self.copyAttr.copy_log.rowCount()
        for i in range(rowCount) :
            self.copyAttr.copy_log.removeRow(0)

        try:
            for idx, nameTarget in self.getGeo().items():
                if dim == self.getSpectrumInfoREST("dim", index=idx) and idx is not index:
                    instance = QPushButton(nameTarget, self)
                    instance.setCheckable(True)
                    instance.setStyleSheet('QPushButton {color: red;}')
                    row, col = self.plotPosition(idx)
                    self.copyAttr.copy_log.addRow("row: "+str(row)+" col: "+str(col), instance)
                    instance.clicked.connect(lambda state, instance=instance: self.connectCopy(instance))
        except KeyError as e:
            self.logger.warning('copyPopup - KeyError occured', exc_info=True)
            # print(f"KeyError occured: {e}")
        self.copyAttr.show()


    #callback to change color when press spectrum button name, 
    def connectCopy(self, instance):
        self.logger.info('connectCopy')
        if (instance.palette().color(QPalette.Text).name() == "#008000"):
            instance.setStyleSheet('QPushButton {color: red;}')
        else:
            instance.setStyleSheet('QPushButton {color: green;}')


    def selectAll(self):
        self.logger.info('selectAll')
        flag = False
        basic = ["Ok", "Cancel", "Apply"]
        discard = ["Ok", "Cancel", "Apply", "Select all", "Deselect all"]
        for instance in self.copyAttr.findChildren(QPushButton):
            if instance.text() not in discard:
                instance.setChecked(True)
                instance.setStyleSheet('QPushButton {color: green;}')
            else:
                if instance.text() not in basic:
                    if instance.text() == "Select all":
                        instance.setText("Deselect all")
                    else:
                        instance.setText("Select all")
                        flag = True

        if flag == True:
            for instance in self.copyAttr.findChildren(QPushButton):
                if instance.text() not in discard:
                    instance.setChecked(False)
                    instance.setStyleSheet('QPushButton {color: red;}')
                    flag = False


    def histAllAttr(self, b):
        self.logger.info('histAllAttr - b.text(): %s',  b.text())
        if b.text() == "Select all properties":
            if b.isChecked() == True:
                self.copyAttr.axisLimitX.setChecked(True)
                self.copyAttr.axisLimitY.setChecked(True)
                self.copyAttr.axisScale.setChecked(True)
                self.copyAttr.histoScaleminZ.setChecked(True)
                self.copyAttr.histoScalemaxZ.setChecked(True)
            else:
                self.copyAttr.axisLimitX.setChecked(False)
                self.copyAttr.axisLimitY.setChecked(False)
                self.copyAttr.axisScale.setChecked(False)
                self.copyAttr.histoScaleminZ.setChecked(False)
                self.copyAttr.histoScalemaxZ.setChecked(False)

        dim = self.getSpectrumInfoREST("dim", index=self.currentPlot.selected_plot_index)

        if dim == 1:
            self.copyAttr.histoScaleminZ.setEnabled(False)
            self.copyAttr.histoScaleValueminZ.setEnabled(False)
            self.copyAttr.histoScalemaxZ.setEnabled(False)
            self.copyAttr.histoScaleValuemaxZ.setEnabled(False)
        else:
            self.copyAttr.histoScaleminZ.setEnabled(True)
            self.copyAttr.histoScaleValueminZ.setEnabled(True)
            self.copyAttr.histoScalemaxZ.setEnabled(True)
            self.copyAttr.histoScaleValuemaxZ.setEnabled(True)


    ##############
    # 10) Gates
    ##############


    #add to axes the gate/summing region applied to a spectrum (identified by index)
    def drawGate(self, index):
        self.logger.info('drawGate - index: %s',index)
        spectrumName = self.nameFromIndex(index)
        spectrumType = self.getSpectrumInfoREST("type", name=spectrumName)
        dim = self.getSpectrumInfoREST("dim", name=spectrumName)
        parameters = self.getSpectrumInfoREST("parameters", name=spectrumName)
        # parameters are used to identify the gates drawable for each spectrum, for gd the parameters list in gate and spectrum definitions are not directly comparable
        if spectrumType == "gd" :
            parametersFormat = []
            for item in parameters:
                pars = item.split(' ')
                if len(pars) == 2:
                    parametersFormat.append(pars[0])
                    parametersFormat.append(pars[1])
            parameters = parametersFormat
        self.logger.debug('drawGate - spectrumName, spectrumType, dim, paramters: %s, %s, %s, %s', spectrumName, spectrumType, dim, parameters)
        ax = self.getSpectrumInfo("axis", index=index)
        if ax is None:
            self.logger.debug('drawGate - ax is None')
            return

        drawableTypes = {"b": ["s"], "1": ["s"], "g1": ["gs"], "2": ["c", "b"], "g2": ["gc", "gb"], "gd": ["gc", "gb"], "m2": ["NotDefinedYet"], "s": ["NotDefinedYet"]}

        #get all gates that share the same parameters than spectrum 
        #gateList is a list of gate which is a dictionary e.g.
        #For 1D -> [{'name': 'gateName', 'type': 's', 'parameters': ['parameterName'], 'low': 0.0, 'high': 1.0}, {'name': 'gateName2'...}]
        #For 2D -> [{'name': 'gateName', 'type': 'c', 'parameters': ['parameterNameA', 'parameterNameB'], 'points': [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}, {'x': 5, 'y': 6}] }, {'name': 'gateName2'...}]
        gateList = [dict for dict in self.rest.listGate() if "type" in dict and "parameters" in dict and dict["type"] in drawableTypes[spectrumType] and dict["parameters"] == parameters]


        # self.axesChilds()
        for gate in gateList:
            if dim == 1:
                xlim = [gate["low"], gate["high"]]
                ylim = ax.get_ybound()
                for iLine in range(2):
                    #define a label to find the line among axes childs and update its properties (limits, visiblility, annotate...)
                    lineLabel = "gate_-_" + gate["name"] + "_-_" + str(iLine)

                    #Remove gate if exists then redraw it
                    toRemove = [gateLine for gateLine in ax.get_children() if type(gateLine) == matplotlib.lines.Line2D and gateLine.get_label() == lineLabel]
                    for lineToRemove in toRemove :
                        lineToRemove.remove()

                    if self.extraPopup.options.gateHide.isChecked():
                        continue

                    line = mlines.Line2D([xlim[iLine],xlim[iLine]],[ylim[0],ylim[1]], picker=5, color='red', label=lineLabel)
                    ax.add_artist(line)

            elif dim ==2:
                #define a label to find the line among axes childs and update its properties (limits, visiblility, annotate...)
                lineLabel = "gate_-_" + gate["name"] + "_-_"

                #Remove gate if exists then redraw it
                toRemove = [gateLine for gateLine in ax.get_children() if type(gateLine) == matplotlib.lines.Line2D and lineLabel in gateLine.get_label()]
                for lineToRemove in toRemove :
                    lineToRemove.remove()

                if self.extraPopup.options.gateHide.isChecked():
                    continue

                if spectrumType not in ["s"]:
                    xPoints = [pointDict["x"] for pointDict in gate["points"]]
                    yPoints = [pointDict["y"] for pointDict in gate["points"]]
                    # lineLabelList = [ lineLabel + "_-_" + str(iline)  for iline in range(len(xPoints)) ]
                    # append the first point to close the line2D, exept for b gate type
                    if gate["type"] not in ["b", "gb"]:
                        xPoints.append(gate["points"][0]["x"])
                        yPoints.append(gate["points"][0]["y"])

                    line = mlines.Line2D(xPoints,yPoints, picker=5, color='red', label=lineLabel)
                    ax.add_artist(line)

            if self.extraPopup.options.gateAnnotation.isChecked() and not self.extraPopup.options.gateHide.isChecked():
                self.setGateAnnotation(index,True)
            else :
                self.setGateAnnotation(index,False)

        #draw summing region 
        lineListSumReg = self.getSumRegion(index)
        if lineListSumReg is None :
            return
        for sumRegionLine in lineListSumReg:
            if dim == 1 or dim == 2 :
                lineLabel = sumRegionLine.get_label()

                #Remove sumRegion if exists then redraw it
                toRemove = [line for line in ax.get_children() if type(line) == matplotlib.lines.Line2D and line.get_label() == lineLabel]
                for lineToRemove in toRemove :
                    lineToRemove.remove()

                # if self.extraPopup.options.sumRegionHide.isChecked():
                #     continue

                xlim = sumRegionLine.get_xdata()
                ylim = sumRegionLine.get_ydata()
                if dim == 1:
                    ylim = ax.get_ylim()

                line = mlines.Line2D(xlim, ylim, picker=5, color='blue', label=lineLabel)
                ax.add_artist(line)


    #callback for gate annotation, calls setGateAnnotation
    def gateAnnotationCallBack(self):
        self.logger.info('gateAnnotationCallBack - isChecked: %s', self.extraPopup.options.gateAnnotation.isChecked())
        if self.extraPopup.options.gateAnnotation.isChecked():
            doAnnotate = True 
        else :
            doAnnotate = False
            
        if self.currentPlot.isEnlarged:
            self.setGateAnnotation(0,doAnnotate)
        else:
            for index, name in self.getGeo().items():
                if name:
                    self.setGateAnnotation(index,doAnnotate)
        self.currentPlot.canvas.draw()


    # set and unset gate annotation of a spectrum according to doAnnotate flag
    def setGateAnnotation(self, index, doAnnotate):
        self.logger.info('setGateAnnotation - index, doAnnotate: %s, %s', index, doAnnotate)
        ax = self.getSpectrumInfo("axis", index=index)
        if ax is None:
            self.logger.debug('setGateAnnotation - ax is None')
            return
        # spectrum = self.getSpectrumInfo("spectrum", index=index)
        # if spectrum is None :
        #     return 
        # ax = spectrum.axes
        dim = self.getSpectrumInfoREST("dim", index=index)

        for child in ax.get_children():
            if type(child) == matplotlib.lines.Line2D :
                label = child.get_label()
                labelSplit = label.split("_-_")
                if len(labelSplit) == 3 and labelSplit[0] == "gate":
                    gateName = labelSplit[1]
                    gateSegmentNum = labelSplit[2]
                    labelBuff = None

                    # set more explicit labels for user
                    # and a color is chosen for the gate lines
                    if dim == 1:
                        if gateSegmentNum == "0":
                            color = self.getGateColor(gateName)
                            labelBuff = gateName + "_low"
                        elif gateSegmentNum == "1":
                            labelBuff = gateName + "_high"

                        #Remove annotation if exists then will redraw it at updated coordinate if want annotation
                        toRemove = [an for an in ax.get_children() if type(an) == matplotlib.text.Annotation and an.get_text() == labelBuff]
                        if len(toRemove) == 1:
                            toRemove[0].remove()
                        
                        positionX = child.get_xdata()[0]
                        positionY = 0.95
                        offsetX = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.002

                        if doAnnotate:
                            xy = self.getXYAnnotation(self.getSpectrumInfo("name", index=index), gateName, (positionX + offsetX, positionY))
                            # fist xy coordinate in data scale and second coordinate in axes fraction
                            #Need to improve the allocation of annotation position to avoid overlay of annotations
                            annotation = ax.annotate(labelBuff, xy=xy, xycoords=("data", "axes fraction"), color=color, fontsize=8, clip_on=True)
                            # renderer = self.currentPlot.canvas.get_renderer()
                            # bbox = ax.get_tightbbox(renderer)
                            # print("Simon ANNOTATION ", annotation, xy)
                            # renderer = self.currentPlot.canvas.get_renderer()
                            # annotationBbox = annotation.get_tightbbox(renderer)
                            # # ax.annotate(labelBuff, xy=(positionX + 0.05, 0.95), xycoords=("data", "axes fraction"), color=color, fontsize=8)
                            child.set_color(color)
                            # # check if overlay with other annotations, if does then shift bellow the lowest overlaying annotation (intersec in matplotlib > 3.6 would be better than contains)
                            # overlayAnnotationBboxes = [an.get_tightbbox(renderer) for an in ax.get_children() if type(an) == matplotlib.text.Annotation and annotationBbox.contains(an.get_tightbbox(renderer).get_center())]
                            # if len(overlayAnnotationBboxes) > 0:
                            #     yLowestBbox = annotationBbox.get_y()
                            #     for bbox in overlayAnnotationBboxes:
                            #         if bbox.get_y() < yLowestBbox:
                            #             yLowestBbox = bbox.get_y()
                            #     annotation.remove()
                            #     annotation = ax.annotate(labelBuff, xy=(positionX + offsetX, yLowestBbox - annotationBbox.get_height()), xycoords=("data", "data"), color=color, fontsize=8)
                        else :
                            child.set_color('red')
                        # spectrum.axes = ax
                        # self.setSpectrumInfo(spectrum=spectrum, index=index)
                    elif dim == 2:
                        #Skip annotation if exists
                        toRemove = [an for an in ax.get_children() if type(an) == matplotlib.text.Annotation and an.get_text() == gateName]
                        if len(toRemove) == 1:
                            toRemove[0].remove()
                            
                        if doAnnotate:
                            color = self.getGateColor(gateName)
                            positionX = child.get_xdata()[0]
                            positionY = child.get_ydata()[0]
                            offsetX = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.003
                            offsetY = (ax.get_ylim()[1] - ax.get_ylim()[0])*0.003
                            # fist xy coordinate in data scale and second coordinate in axes fraction
                            #Need to improve the allocation of annotation position to avoid overlay of annotations
                            annotation = ax.annotate(gateName, xy=(positionX+offsetX, positionY+offsetY), xycoords="data", color=color, fontsize=8, clip_on=True)
                            # renderer = self.currentPlot.canvas.get_renderer()
                            # bbox = ax.get_tightbbox(renderer)
                            # print("Simon 2D ANNOTATION ", annotation, positionX,positionY)
                            # renderer = self.currentPlot.canvas.get_renderer()
                            # annotationBbox = annotation.get_tightbbox(renderer)
                            # # ax.annotate(labelBuff, xy=(positionX + 0.05, 0.95), xycoords=("data", "axes fraction"), color=color, fontsize=8)
                            child.set_color(color)
                            # # check if overlay with other annotations, if does then shift bellow the lowest overlaying annotation (intersec in matplotlib > 3.6 would be better than contains)
                            # overlayAnnotationBboxes = [an.get_tightbbox(renderer) for an in ax.get_children() if type(an) == matplotlib.text.Annotation and annotationBbox.contains(an.get_tightbbox(renderer).get_center())]
                            # if len(overlayAnnotationBboxes) > 0:
                            #     yLowestBbox = annotationBbox.get_y()
                            #     for bbox in overlayAnnotationBboxes:
                            #         if bbox.get_y() < yLowestBbox:
                            #             yLowestBbox = bbox.get_y()
                            #     annotation.remove()
                            #     annotation = ax.annotate(labelBuff, xy=(positionX + offsetX, yLowestBbox - annotationBbox.get_height()), xycoords=("data", "data"), color=color, fontsize=8)
                        else :
                            child.set_color('red')
                        # spectrum.axes = ax
                        # self.setSpectrumInfo(spectrum=spectrum, index=index)


    #goes with the annotation feature, associate a color to a gateName  
    def getGateColor(self, gateName):
        self.logger.info('getGateColor - gateName : %s', gateName)
        # 8 colors maybe enough (?) (blue and red ignored)
        colorList = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        # if more gates than colors then restart defining color from colorList[0] etc.
        nextColor = colorList[len(self.gateColor) % len(colorList)]
        if gateName not in self.gateColor:
            self.gateColor[gateName] = nextColor
            return nextColor
        else :
            return self.gateColor[gateName]


    # Need better way to assign position
    def getXYAnnotation(self, spectrumName, gateName, xy):
        self.logger.info('getXYAnnotation - spectrumName, gateName, xy : %s, %s, %s', spectrumName, gateName, xy)
        if spectrumName not in self.gateAnnotation:
            self.gateAnnotation[spectrumName] = {gateName: xy}
            return xy
        else :
            for gateDict in self.gateAnnotation[spectrumName] :
                if gateName in gateDict:
                    return xy 
                else :
                    return (xy[0], xy[1]-0.05)




    # add copy of listRegionLine to axis before deleting the later one (considered a temporary line)
    # Identify this region line with label
    def saveSumRegion(self, index):
        self.logger.info('saveSumRegion - index: %s', index)
        spectrumName = self.nameFromIndex(index)
        spectrum = self.getSpectrumInfo("spectrum", index=index)
        ax = spectrum.axes
        dim = self.getSpectrumInfoREST("dim", name=spectrumName)
        if ax is None :
            self.logger.debug('saveSumRegion - ax is None')
            return
        regionName = self.sumRegionPopup.sumRegionNameList.currentText()
        if regionName is None or regionName == "None" :
            self.logger.debug('saveSumRegion - regionName is None or regionName == "None"')
            return
        self.logger.debug('saveSumRegion - spectrumName, dim, regionName: %s, %s, %s', spectrumName, dim, regionName)

        # self.axesChilds()
        if dim == 1:
            ylim = ax.get_ybound()
            #two lines for 1d 
            if len(self.sumRegionPopup.listRegionLine) != 2:
                return
            for iLine in range(2):
                xlim = self.sumRegionPopup.listRegionLine[iLine].get_xdata()
                #define a label to find the line among axes childs and update its properties (limits, visiblility, annotate...)
                lineLabel = "sumReg_-_" + regionName + "_-_" + str(iLine)

                line = mlines.Line2D([xlim[0],xlim[0]], [ylim[0],ylim[1]], picker=5, color='blue', label=lineLabel)
                self.setSumRegion(index, line)
                ax.add_artist(line)

        elif dim ==2:
            #define a label to find the line among axes childs and update its properties (limits, visiblility, annotate...)
            lineLabel = "sumReg_-_" + regionName + "_-_"

            xPoints = []
            yPoints = []
            for iLine, line in enumerate(self.sumRegionPopup.listRegionLine):
                if iLine == 0:
                    for iPoint in range(2):
                        xPoints.append(self.sumRegionPopup.listRegionLine[iLine].get_xdata()[iPoint])
                        yPoints.append(self.sumRegionPopup.listRegionLine[iLine].get_ydata()[iPoint])
                else :
                    xPoints.append(self.sumRegionPopup.listRegionLine[iLine].get_xdata()[1])
                    yPoints.append(self.sumRegionPopup.listRegionLine[iLine].get_ydata()[1])

            line = mlines.Line2D(xPoints,yPoints, picker=5, color='blue', label=lineLabel)
            self.setSumRegion(index, line)
            ax.add_artist(line)



    #callback for createSumRegionButton, show popup to define name
    def createSumRegion(self):
        self.logger.info('createSumRegion CallBack')
        # Pause auto update when edit sum region
        self.skipAutoUpdateThread.set()
        self.sumRegionPopup.sumRegionNameList.setEditable(True)
        self.sumRegionPopup.sumRegionNameList.setInsertPolicy(QComboBox.NoInsert)

        if self.currentPlot.selected_plot_index is None:
            return QMessageBox.about(self,"Warning!", "Please add/select a spectrum")
        else:
            self.sumRegionPopup.clearInfo()

            spectrumName = self.nameFromIndex(self.currentPlot.selected_plot_index)
            dim = self.getSpectrumInfoREST("dim", name=spectrumName)
            ax = self.getSpectrumInfo("axis", index=self.currentPlot.selected_plot_index)
            if ax is None:
                self.logger.debug('createSumRegion - ax is None')
                return

            #refresh sumRegionDict (remove key if spectrum has been deleted in treegui)
            self.refreshSpectrumSumRegionDict()

            #set summing region name list 
            sumRegionLabels = [child.get_label() for child in ax.get_children() if isinstance(child, matplotlib.lines.Line2D) and "_-_" in child.get_label()]
            for label in sumRegionLabels :
                if dim == 1:
                    label = label.split("_-_")
                    #two lines per region for 1d so just take one label
                    if label[0] == "sumReg" and label[2] == '0':
                        self.sumRegionPopup.sumRegionNameList.addItem(label[1])
                elif dim == 2:
                    label = label.split("_-_")
                    if label[0] == "sumReg":
                        self.sumRegionPopup.sumRegionNameList.addItem(label[1])
            self.sumRegionPopup.sumRegionNameList.setCurrentText("None")
            self.sumRegionPopup.sumRegionNameList.completer().setCompletionMode(QCompleter.PopupCompletion)
            self.sumRegionPopup.sumRegionNameList.completer().setFilterMode(QtCore.Qt.MatchContains)
            self.sumRegionPopup.sumRegionNameListSaved = [self.sumRegionPopup.sumRegionNameList.itemText(i) for i in range(self.sumRegionPopup.sumRegionNameList.count()) if self.sumRegionPopup.sumRegionNameList.itemText(i) != "None"]

            self.currentPlot.toCreateSumRegion = True
            self.sumRegionPopup.sumRegionSpectrumIndex = self.currentPlot.selected_plot_index
        self.sumRegionPopup.show()


    #button of the cutoff window, sets the cutoff values in the spectrum dict
    def okSumRegion(self):
        self.logger.info('okSumRegion')
        # if sumRegionName already exists issue warning
        sumRegionName = self.sumRegionPopup.sumRegionNameList.currentText()
        
        if sumRegionName in self.sumRegionPopup.sumRegionNameListSaved:
            self.logger.debug('okSumRegion - sumRegionName: %s already exist', sumRegionName)
            msgBox = QMessageBox(self)
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setWindowFlag(Qt.WindowStaysOnTopHint, True)
            msgBox.setText("Summing region name already exists.")
            msgBox.setInformativeText('Do you want to overwrite "' + sumRegionName + '" summing region definition?')
            msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
            msgBox.setDefaultButton(QMessageBox.Cancel)
            ret = msgBox.exec()
            if ret == QMessageBox.Yes:
                pass
            elif ret == QMessageBox.Cancel:
                return
        #maybe better solution, the separator in region line labels are _-_ so dont want to find that in the region name
        elif "_-_" in sumRegionName :
            self.logger.debug('okSumRegion - sumRegionName has _-_ in its name')
            msgBox = QMessageBox(self)
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setWindowFlag(Qt.WindowStaysOnTopHint, True)
            msgBox.setText('Region name must not include "_-_"')
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.setDefaultButton(QMessageBox.Ok)
            ret = msgBox.exec()
            if ret == QMessageBox.Ok:
                pass
        #"save"/draw a new version of the region line in ax before deleting listRegionLine in closeEvent of the popup
        self.saveSumRegion(self.sumRegionPopup.sumRegionSpectrumIndex)
        self.cancelSumRegion()


    #close sumRegionPopup
    def cancelSumRegion(self, doClose=True):
        self.logger.info('cancelSumRegion')
        self.currentPlot.toCreateSumRegion = False
        if doClose :
            self.sumRegionPopup.close()
        self.updatePlot()
        # self.autoUpdsateStart()


    # Close gatePopup/sumRegionPopup and set flags properly
    def cleanPopupExit(self, doClose=True):
        # if gatePopup exit with [X]
        if (self.currentPlot.toCreateGate or self.currentPlot.toEditGate) and not self.gatePopup.isVisible():
            self.cancelGate(doClose)
        # if sumRegionPopup exit with [X]
        if self.currentPlot.toCreateSumRegion and not self.sumRegionPopup.isVisible():
            self.cancelSumRegion(doClose)


    #delete summing region according to self.sumRegionPopup.sumRegionNameList.currentText()
    def deleteSumRegion(self):
        self.logger.info('deleteSumRegion')
        spectrumName = self.nameFromIndex(self.currentPlot.selected_plot_index)
        dim = self.getSpectrumInfoREST("dim", name=spectrumName)
        ax = self.getSpectrumInfo("axis", index=self.currentPlot.selected_plot_index)
        if ax is None:
            return
        sumRegionName = self.sumRegionPopup.sumRegionNameList.currentText()
        
        #if sumRegionName not saved nothing to delete yet
        if sumRegionName not in self.sumRegionPopup.sumRegionNameListSaved:
            self.logger.debug('deleteSumRegion - sumRegionName: %s doesnt exists', sumRegionName)
            msgBox = QMessageBox(self)
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setWindowFlag(Qt.WindowStaysOnTopHint, True)
            msgBox.setText("Cannot delete summing region " + sumRegionName + " not found")
            msgBox.setStandardButtons(QMessageBox.Ok)
            ret = msgBox.exec()
            if ret == QMessageBox.Ok:
                return
        #delete summing region from sumRegionDict and line2D drawn in currentPlot
        else:
            if dim == 1 :
                for iLine in range(2):
                    label = "sumReg_-_" + sumRegionName + "_-_" + str(iLine)
                    self.deleteSumRegionDict(label)
            elif dim == 2:
                label = "sumReg_-_" + sumRegionName + "_-_"
                self.deleteSumRegionDict(label)

        # reset the current list region name to None
        self.sumRegionPopup.sumRegionNameList.setCurrentText("None")
        self.currentPlot.figure.tight_layout()
        self.currentPlot.canvas.draw()


    # callback for gatePopup.ok button, do some checks and push gate to REST
    def okGate(self):
        self.logger.info('okGate')
        # check if gateName already exists if yes issue warning
        gateName = self.gatePopup.gateNameList.currentText()
        gateNameList = [gate["name"] for gate in self.rest.listGate()]
        # print("Simon - in okGate ", gateNameList, gateName, self.currentPlot.toEditGate)
        if not self.currentPlot.toEditGate:
            if gateName in gateNameList:
                self.logger.debug('okGate - gateName: %s already exists', gateName)
                msgBox = QMessageBox(self)
                msgBox.setIcon(QMessageBox.Warning)
                msgBox.setWindowFlag(Qt.WindowStaysOnTopHint, True)
                msgBox.setText("Gate name already exists.")
                msgBox.setInformativeText('Do you want to overwrite "' + gateName + '" gate definition?')
                msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
                msgBox.setDefaultButton(QMessageBox.Cancel)
                ret = msgBox.exec()
                if ret == QMessageBox.Yes:
                    pass
                elif ret == QMessageBox.Cancel:
                    return
            #maybe better solution, the separator in gate line labels are _-_ so dont want to find that in the gate name
            elif "_-_" in gateName :
                self.logger.debug('okGate - gateName has _-_ in its name')
                msgBox = QMessageBox(self)
                msgBox.setIcon(QMessageBox.Warning)
                msgBox.setWindowFlag(Qt.WindowStaysOnTopHint, True)
                msgBox.setText('Gate name must not include "_-_"')
                msgBox.setStandardButtons(QMessageBox.Ok)
                msgBox.setDefaultButton(QMessageBox.Ok)
                ret = msgBox.exec()
                if ret == QMessageBox.Ok:
                    pass
        self.pushGateToREST(gateName,self.gatePopup.listGateType.currentText())
        #in the following lines removed in listRegionLine to avoid double drawing, since the gate has been pushed to ReST it will be drawn from ReST info
        self.gatePopup.clearInfo()
        self.cancelGate()
        # self.currentPlot.toCreateGate = False
        # self.currentPlot.toEditGate = False
        # self.gatePopup.close()
        # self.updatePlot()


    #callback for gatePopup.cancel button
    def cancelGate(self, doClose=True):
        self.logger.info('cancelGate')
        self.currentPlot.toCreateGate = False
        self.currentPlot.toEditGate = False
        self.disconnectGateSignals()
        if doClose :
            self.gatePopup.close()
        self.updatePlot()

    
    def disconnectGateSignals(self):
        self.logger.info('disconnectGateSignals')
        try:
            if hasattr(self, 'gateReleaser') :
                self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_disconnect(self.gateReleaser)
        except TypeError:
            pass
        try:
            if hasattr(self, 'gateFollower') :
                self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_disconnect(self.gateFollower)
        except TypeError:
            pass
        try:
            if hasattr(self, 'sid') :
                self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_disconnect(self.sid)
        except TypeError:
            pass
        try:
            if hasattr(self, 'sidGateNameListChanged') :
                self.gatePopup.gateNameList.currentTextChanged.disconnect(self.sidGateNameListChanged)
        except TypeError:
            pass
        try:
            if hasattr(self, 'sidGateTypeListChanged') :
                self.gatePopup.listGateType.currentIndexChanged.disconnect(self.sidGateTypeListChanged)
        except TypeError:
            pass
        try:
            if hasattr(self, 'gatePopupPreview') :
                self.gatePopup.preview.clicked.disconnect(self.gatePopupPreview)
        except TypeError:
            pass
        #Had issue with these two signals when multiple tab(?), use shortcutInsertRegionPoint now
        # try:
        #     if hasattr(self, 'sidOnKeyPressEditGate') :
        #         self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_disconnect(self.sidOnKeyPressEditGate)
        # except TypeError:
        #     pass
        # try:
        #     if hasattr(self, 'sidOnKeyReleaseEditGate') :
        #         self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_disconnect(self.sidOnKeyReleaseEditGate)
        # except TypeError:
        #     pass
        try:
            if hasattr(self, 'shortcutInsertRegionPoint') :
                self.shortcutInsertRegionPoint.setEnabled(False)
        except TypeError:
            pass
        try:
            if hasattr(self, 'sidTableIntegrateCopy') :
                self.integratePopup.resultsText.itemSelectionChanged.disconnect(self.sidTableIntegrateCopy)
        except TypeError:
            pass
            


    #Open dialog window to specify name/type and draw gate while this window is openned
    #The flag self.currentPlot.toCreateGate determines if by clicking one sets the gate points (in on_singleclick)
    def createGate(self):
        self.logger.info('createGate')
        # Pause auto update when edit gate
        self.skipAutoUpdateThread.set()
        #Default gate actions is gate creation
        self.gatePopup.gateActionCreate.setChecked(True)
        self.gatePopup.preview.setEnabled(False)
        #in otherOptions.py define a checkbox to have the possibility to disable gate edition
        if self.extraPopup.options.gateEditDisable.isChecked():
            self.gatePopup.gateActionEdit.setChecked(False)
            self.gatePopup.gateActionEdit.setEnabled(False)
        else:
            self.gatePopup.gateActionEdit.setEnabled(True)

        try:
            if hasattr(self, 'sidGateNameListChanged') :
                self.gatePopup.gateNameList.currentTextChanged.disconnect(self.sidGateNameListChanged)
                self.logger.debug('createGate - disconnected sidGateNameListChanged')
        except TypeError:
            pass

        self.gatePopup.gateNameList.setEditable(True)
        self.gatePopup.gateNameList.setInsertPolicy(QComboBox.NoInsert)
        self.gatePopup.gateNameList.setCurrentText("None")


        if self.currentPlot.selected_plot_index is None:
            return QMessageBox.about(self,"Warning!", "Please add at least one spectrum")
        else:
            gateTypesDict = {"b": ["NotDefinedYet"], "1": ["s"], "g1": ["gs"], "2": ["c", "b"], "g2": ["gc", "gb"], "gd": ["gc", "gb"], "m2": ["c", "b"], "s": ["NotDefinedYet"]}
            spectrumType = self.getSpectrumInfoREST("type", index=self.currentPlot.selected_plot_index)
            if spectrumType is None :
                return

            self.gatePopup.clearInfo()
            self.disconnectGateSignals()

            gateTypesList = gateTypesDict[spectrumType]
            for type in gateTypesList:
                if type == "NotDefinedYet":
                    self.logger.debug('createGate - gate type NotDefinedYet')
                    msgBox = QMessageBox(self)
                    msgBox.setIcon(QMessageBox.Warning)
                    msgBox.setWindowFlag(Qt.WindowStaysOnTopHint, True)
                    msgBox.setText('No gate type available for "' + spectrumType + '" spectrum')
                    #msgBox.setInformativeText('Check latest SpecTcl version')
                    msgBox.setStandardButtons(QMessageBox.Ok)
                    msgBox.setDefaultButton(QMessageBox.Ok)
                    ret = msgBox.exec()
                    return
                self.gatePopup.listGateType.addItem(type)

            self.currentPlot.toCreateGate = True
            self.currentPlot.toEditGate = False

            self.sidGateTypeListChanged = self.gatePopup.listGateType.currentIndexChanged.connect(self.gateTypeListChanged)

            self.gatePopup.gateSpectrumIndex = self.currentPlot.selected_plot_index
            self.gatePopup.show()


    #Draw a preview of the gate in case modified by editing the text gatePopup.regionPoint
    def onGatePopupPreview(self):
        self.logger.info('onGatePopupPreview')
        ax = self.getSpectrumInfo("axis", index=self.gatePopup.gateSpectrumIndex)
        if ax is None :
            self.logger.debug('onGatePopupPreview - ax is None')
            return 
        dim = self.getSpectrumInfoREST("dim", index=self.gatePopup.gateSpectrumIndex)
        #points for 1d: [x0, x1] and for 2d:  [[x0, y0], [x1, y1], ...]
        points = self.formatGatePopupPointText(dim)
        #point is not None when all lines have the good format, see formatGatePopupPointText
        if points is None :
            self.logger.debug('onGatePopupPreview - points is None')
            return
        # if no line selected, set editThisGateLine using gatePopup.gateNameList 
        if not hasattr(self, 'editThisGateLine') or self.editThisGateLine is None:
            try:
                gateIdentifier = "gate_-_" + self.gatePopup.gateNameList.currentText() + "_-_"
                self.editThisGateLine = [child for child in ax.get_children() if type(child) == matplotlib.lines.Line2D and gateIdentifier in child.get_label()][0]
            except :
                self.logger.debug('onGatePopupPreview - exception ', exc_info=True) 
                return
        #For 1d there is multiple lines (2), find those with label
        if dim == 1:
            label = self.editThisGateLine.get_label()
            labelSplit = label.split("_-_")
            gateIdentifier = "gate_-_" + labelSplit[1] + "_-_"
            lines = [child for child in ax.get_children() if type(child) == matplotlib.lines.Line2D and gateIdentifier in child.get_label()]
            if len(lines) == len(points):
                for iline, line in enumerate(lines):
                    line.set_xdata([points[iline], points[iline]])
        elif dim == 2:
            lineX = []
            lineY = []
            for point in points:
                lineX.append(point[0])
                lineY.append(point[1])
            specialTypes = ["c", "gc"]
            if self.gatePopup.listGateType.currentText() in specialTypes: 
                #For c and gc the last point need to match the first one to close the contour
                lineX.append(points[0][0])
                lineY.append(points[0][1])
            self.editThisGateLine.set_data(lineX, lineY)
        self.currentPlot.canvas.draw()


    def checkConnections(self):
        # Check if there are any connections to the canvas
        # if self.wTab.wPlot[self.wTab.currentIndex()].canvas.figure.callbacks.callbacks:
        if self.wTab.wPlot[self.wTab.currentIndex()].canvas.callbacks.callbacks:
            for event_name, callbacks_dict in self.wTab.wPlot[self.wTab.currentIndex()].canvas.callbacks.callbacks.items():
                print(f"Event: {event_name}, Callbacks: {callbacks_dict}")


    def find_callbacks(self):
        callbacks_for_main_window = []

        for event_name, callbacks_dict in self.wTab.wPlot[self.wTab.currentIndex()].canvas.callbacks.callbacks.items():
            for callback_id, callback_func in callbacks_dict.items():
                # Check if the callback function is associated with the MainWindow instance
                if hasattr(callback_func, '__self__') and callback_func.__self__ is self:
                    callbacks_for_main_window.append((event_name, callback_func))
        return callbacks_for_main_window


    #Open dialog window to edit an existing gate while this window is openned (can check and change points coordinates)
    def editGate(self):
        self.logger.info('editGate')

        self.gatePopup.gateActionCreate.setChecked(False)
        self.gatePopup.preview.setEnabled(True)

        try:
            if hasattr(self, 'sidGateTypeListChanged') :
                self.gatePopup.listGateType.currentIndexChanged.disconnect(self.sidGateTypeListChanged)
                self.logger.debug('editGate - disconnected sidGateTypeListChanged')
        except TypeError:
            pass

        self.sid = self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_connect('pick_event', self.clickOnGateLine)
        self.gatePopupPreview = self.gatePopup.preview.clicked.connect(self.onGatePopupPreview)

        # Hotkeys, insert and delete gate point (2d)
        # self.sidOnKeyPressEditGate = self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_connect('key_press_event', self.onKeyPressEditGate)
        # self.sidOnKeyReleaseEditGate = self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_connect('key_release_event', self.onKeyReleaseEditGate)
        #two previous lines dont work with multitab, dont understand why because do the same with pick_event and no problem with this one.
        #So use qt shortcut, no release action so reset altPressed flag in on_singleclick_gate_edit
        self.shortcutInsertRegionPoint = QShortcut(QKeySequence("Alt+E"), self)
        self.shortcutInsertRegionPoint.activated.connect(self.onKeyActivateEditGate)


        if self.gatePopup.gateSpectrumIndex is None:
            self.logger.debug('editGate - gateSpectrumIndex is None')
            return QMessageBox.about(self,"Warning!", "Please add at least one spectrum")
        else:
            spectrumName = self.nameFromIndex(self.gatePopup.gateSpectrumIndex)
            dim = self.getSpectrumInfoREST("dim", name=spectrumName)
            ax = self.getSpectrumInfo("axis", index=self.gatePopup.gateSpectrumIndex)
            if ax is None:
                self.logger.debug('editGate - ax is None')
                return
            
            self.gatePopup.clearInfo()

            #set gate name list 
            gateLabels = [child.get_label() for child in ax.get_children() if isinstance(child, matplotlib.lines.Line2D) and "_-_" in child.get_label()]
            for label in gateLabels :
                if dim == 1:
                    label = label.split("_-_")
                    #two lines per gate for 1d so just take one label
                    if label[0] == 'gate' and label[2] == '0':
                        self.gatePopup.gateNameList.addItem(label[1])
                elif dim == 2:
                    label = label.split("_-_")
                    if label[0] == 'gate' :
                        self.gatePopup.gateNameList.addItem(label[1])
            self.gatePopup.gateNameList.setCurrentText("-- select a gate --")
            self.gatePopup.gateNameList.completer().setCompletionMode(QCompleter.PopupCompletion)
            self.gatePopup.gateNameList.completer().setFilterMode(QtCore.Qt.MatchContains)
            self.sidGateNameListChanged = self.gatePopup.gateNameList.currentTextChanged.connect(self.gateNameListChanged)

            self.currentPlot.toEditGate = True
            self.altPressed = False
            self.currentPlot.toCreateGate = False
            self.gatePopup.regionPoint.setReadOnly(False)


    #if change gate type while creating gate, reset the new gate info (name and point)
    def gateTypeListChanged(self):
        self.logger.info('gateTypeListChanged')
        self.gatePopup.gateNameList.clear()
        self.gatePopup.gateNameList.setCurrentText("None")
        for line in self.gatePopup.listRegionLine:
            line.remove()
        self.gatePopup.listRegionLine.clear()
        self.gatePopup.prevPoint.clear()
        self.gatePopup.regionPoint.clear()
        self.gatePopup.gateSpectrumIndex = 0
        self.gatePopup.gateEditOption = None
        

    #change the gate points text and gate type (and line color when edit) according to gate name in the completable combo box
    def gateNameListChanged(self):
        self.logger.info('gateNameListChanged')
        ax = self.getSpectrumInfo("axis", index=self.gatePopup.gateSpectrumIndex)
        if ax is None:
            self.logger.debug('gateNameListChanged - ax is None')
            return

        #Check that the text corresponds to an actual gate name
        gateName = self.gatePopup.gateNameList.currentText()
        gateFoundAtIdx = self.gatePopup.gateNameList.findText(gateName)
        if gateFoundAtIdx == -1 :
            self.gatePopup.regionPoint.clear()
            self.gatePopup.listGateType.clear()
            self.logger.debug('gateNameListChanged - gateFoundAtIdx == -1')
            return
        
        #update points text
        gateIdentifier = "gate_-_" + gateName + "_-_"
        lines = [child for child in ax.get_children() if type(child) == matplotlib.lines.Line2D and gateIdentifier in child.get_label()]
        otherLines = [child for child in ax.get_children() if type(child) == matplotlib.lines.Line2D and gateIdentifier not in child.get_label()]

        #set listGateType
        gate = [dict for dict in self.rest.listGate() if dict["name"] == gateName]
        self.gatePopup.gateNameList.setCurrentText(gateName)
        self.gatePopup.listGateType.addItem(gate[0]["type"])

        #update text points corresponding to selected gate
        self.updateTextGatePopup(lines)

        #set lines style of the selected gate and rest of the lines
        for line in lines :
            line.set_marker(marker='o')
            line.set_color("green")
        for line in otherLines :
            line.set_marker(marker=None)
            line.set_color("red")

        self.currentPlot.canvas.draw()


    # called in on_singleclick_gate to add a gate line to the axis
    def addLine(self, posx, posy, index, label=None):
        self.logger.info('addLine - posx, posy, index, label: %s, %s, %s, %s', posx, posy, index, label)
        spectrum = self.getSpectrumInfo("spectrum", index=index)
        dim = self.getSpectrumInfo("dim", index=index)
        if spectrum is None :
            self.logger.debug('addLine - spectrum is None')
            return
        ax = spectrum.axes

        if dim == 1:
            ymin, ymax = ax.get_ybound()
            l = mlines.Line2D([posx,posx], [ymin,ymax], picker=5, label=label)
            ax.add_line(l)
        elif dim == 2:
            if self.currentPlot.toCreateGate :
                xyPrev = self.gatePopup.prevPoint
                self.gatePopup.prevPoint = [posx, posy]
                # If we are adding the closing segment we want to save the n-1 last point (prev prev) 
                # because closing segment will be removed before drawing from click
                if label == "closing_segment":
                    self.gatePopup.prevPoint = xyPrev
            elif self.currentPlot.toCreateSumRegion :
                xyPrev = self.sumRegionPopup.prevPoint
                self.sumRegionPopup.prevPoint = [posx, posy]
                if label == "closing_segment":
                    self.sumRegionPopup.prevPoint = xyPrev
            if xyPrev is None or len(xyPrev) == 0 :
               return
            else :
                l = mlines.Line2D([xyPrev[0],posx], [xyPrev[1],posy], picker=5, label=label)
                ax.add_line(l)
        if l is None :
            self.logger.debug('addLine - l is None')
            return 
        elif self.currentPlot.toCreateSumRegion :
            l.set_color('b')
        else:
            l.set_color('r')
        return l


    #called in on_singleclick_gate to remove previous point by right click while making a new gate
    def removePrevLine(self):
        self.logger.info('removePrevLine')
        popup = None
        if self.currentPlot.toCreateGate :
            popup = self.gatePopup
        elif self.currentPlot.toCreateSumRegion :
            popup = self.sumRegionPopup
        if popup is not None:
            l = popup.listRegionLine[0]
            l.remove()
            popup.listRegionLine.pop(0)
        

    #reset signals to continue adding/modifying gate point
    def releaseonclick(self, event):
        self.logger.info('releaseonclick')
        try:
            if hasattr(self, 'gateReleaser') :
                self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_disconnect(self.gateReleaser)
        except TypeError:
            pass
        try:
            if hasattr(self, 'gateFollower') :
                self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_disconnect(self.gateFollower)
        except TypeError:
            pass
        self.gatePopup.gateEditOption = None
        self.xyRef = None
        self.movingMarker = []


    # #Disable hotkey to edit a gate
    # def onKeyReleaseEditGate(self, event):
    #     #set to false as soon as the key is released
    #     self.altPressed = False


    # # Callback for hotkey to edit a gate
    # def onKeyPressEditGate(self, event):
    #     if not event.inaxes or not self.currentPlot.toEditGate:
    #         return
    #     #Used in on_singleclick_gate_edit
    #     self.altPressed = True
    #     if event.key == 'alt':
    #         self.altPressed = True

    # Callback for shortcut to edit a gate
    def onKeyActivateEditGate(self):
        self.logger.info('onKeyActivateEditGate - self.currentPlot.toEditGate: %s',self.currentPlot.toEditGate)
        if not self.currentPlot.toEditGate:
            return
        #Used in on_singleclick_gate_edit
        self.altPressed = True


    #insert a point to the edited (selected) contour (only for 2d gate), at the mouse location close to the contour
    def insertPointGate(self, event):
        self.logger.info('insertPointGate')
        dim = self.getSpectrumInfoREST("dim", index=self.gatePopup.gateSpectrumIndex)
        ax = self.getSpectrumInfo("axis", index=self.gatePopup.gateSpectrumIndex)
        if ax is None or dim !=2 :
            self.logger.debug('insertPointGate - ax is None or dim!=2: %s',dim)
            return 
        
        lineX = self.editThisGateLine.get_xdata()
        lineY = self.editThisGateLine.get_ydata()
        # coordinates in pixels for comparison with event position
        lineXY = self.editThisGateLine.get_transform().transform(self.editThisGateLine.get_xydata())

        p = event.x, event.y
        insertAt = None
        for i in range(len(lineXY) - 1 ):
            s0 = lineXY[i]
            s1 = lineXY[i+1]
            d = self.dist_point_to_segment(p, s0, s1)
            if d <= self.epsilon:
                insertAt = i
                break

        if insertAt is not None:
            lineX.insert(insertAt+1, ax.transData.inverted().transform((event.x, event.y))[0])
            lineY.insert(insertAt+1, ax.transData.inverted().transform((event.x, event.y))[1])
            self.editThisGateLine.set_data(lineX, lineY)
            self.currentPlot.canvas.draw_idle()


    #delete a point to the edited (selected) contour (only for 2d gate), at the mouse location close to the contour
    def deletePointGate(self, event):
        self.logger.info('deletePointGate')
        dim = self.getSpectrumInfoREST("dim", index=self.gatePopup.gateSpectrumIndex)
        ax = self.getSpectrumInfo("axis", index=self.gatePopup.gateSpectrumIndex)
        if ax is None or dim !=2 :
            self.logger.debug('deletePointGate - ax is None or dim!=2: %s',dim)
            return 

        lineX = self.editThisGateLine.get_xdata()
        lineY = self.editThisGateLine.get_ydata()
        markerPos = np.array([lineX, lineY])

        p = ax.transData.inverted().transform((event.x, event.y))

        # calculate the distance between each marker and the event.
        distances = np.linalg.norm(markerPos - p.reshape(2, -1), axis=0)
        dataRadius = abs(ax.transData.inverted().transform((self.epsilon, 0))[0])
        # print("Simon - deletePointGate ", markerPos, dataRadius, p, distances)

        specialGateTypes = ["c", "gc"]

        # get the marker index if marker is close to the click position.
        markerIdx = np.where(distances <= dataRadius)[0]
        if markerIdx.size == 0:
            pass
        else:
            # for contour, if remove first point, set last point to second point to keep contour closed
            if markerIdx[0] == 0 and self.gatePopup.listGateType.currentText() in specialGateTypes:
                lineX[-1] = lineX[1]
                lineY[-1] = lineY[1]
            lineX.pop(markerIdx[0])
            lineY.pop(markerIdx[0])
            self.editThisGateLine.set_data(lineX, lineY)
            dumList = [self.editThisGateLine]
            self.updateTextGatePopup(dumList)
            self.currentPlot.canvas.draw_idle()



    #change "in live" the point/line/gate position according to the mouse position
    def followmouse(self, event):
        if self.gatePopup.gateEditOption == "1d_move_line" :
            self.editThisGateLine.set_color("green")
            self.editThisGateLine.set_xdata([event.xdata, event.xdata])
        elif self.gatePopup.gateEditOption == "2d_move_all" :
            lineX = self.editThisGateLine.get_xdata()
            lineY = self.editThisGateLine.get_ydata()
            shiftX = event.xdata - lineX[0] 
            shiftY = event.ydata - lineY[0]
            #not sure why the += changes the type of lineX lineY from list to ndarray? Want to keep it a list
            # lineX += shiftX
            # lineY += shiftY
            lineX = [item + shiftX for item in lineX]
            lineY = [item + shiftY for item in lineY]
            self.editThisGateLine.set_data(lineX, lineY )
        elif self.gatePopup.gateEditOption == "2d_move_point" :
            lineX = self.editThisGateLine.get_xdata()
            lineY = self.editThisGateLine.get_ydata()
            if hasattr(self, 'movingMarker') and len(self.movingMarker) > 0:
                for mIdx in self.movingMarker:
                    lineX[mIdx] = event.xdata
                    lineY[mIdx] = event.ydata
                self.editThisGateLine.set_data(lineX, lineY )
            else :
                markerPos = np.array([lineX, lineY])
                try :
                    ax = self.getSpectrumInfo("axis", index=self.gatePopup.gateSpectrumIndex)
                    if ax is None :
                        return
                    # calculate the distance between each marker and the event.
                    distances = np.linalg.norm(markerPos - self.xyRef.reshape(2, -1), axis=0)
                    # dataRadius = abs(ax.transData.inverted().transform([[self.epsilon, 0]])[0, 0])
                    # dataRadius = self.epsilon

                    #not fan of plt.gcf()...
                    xlims = ax.get_xlim()
                    ylims = ax.get_ylim()
                    figTest = plt.gcf()
                    plottingAreaWidth, plottingAreaHeight = figTest.get_size_inches() * figTest.dpi
                    radX = self.pixel_to_data_distance(self.epsilon, xlims, plottingAreaWidth)
                    radY = self.pixel_to_data_distance(self.epsilon, ylims, plottingAreaHeight)
                    dataRadius = math.sqrt(radX*radX + radY*radY)

                    # get the marker index if marker is close to the click position.
                    markerIdx = np.where(distances <= dataRadius)[0]
                    # print("Simon - followmouse -", markerIdx, markerPos, distances, dataRadius, self.xyRef, radX, radY)
                    if markerIdx.size == 0:
                        pass
                    else:
                        #If select the first point then one should move the last point too, 2 Idx 
                        self.movingMarker = []
                        for mIdx in markerIdx:
                            lineX[mIdx] = event.xdata
                            lineY[mIdx] = event.ydata
                            self.movingMarker.append(mIdx)
                        self.editThisGateLine.set_data(lineX, lineY )
                        self.gateReleaser = self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_connect("button_press_event", self.releaseonclick)
                except NameError:
                    raise
        else :
            pass
        self.currentPlot.canvas.draw_idle()


    # Function to transform pixel distance to data coordinate distance
    def pixel_to_data_distance(self, pixel_distance, axis_limits, plotting_area_size):
        axis_range = axis_limits[1] - axis_limits[0]
        pixel_to_data_ratio = axis_range / plotting_area_size
        return pixel_distance * pixel_to_data_ratio


    #goes with hotkey to edit gate, depending if left or right click, add or delete gate point (2d)
    def on_singleclick_gate_edit(self, event):
        self.logger.info('on_singleclick_gate_edit')
        dim = self.getSpectrumInfoREST("dim", index=self.gatePopup.gateSpectrumIndex)
        ax = self.getSpectrumInfo("axis", index=self.gatePopup.gateSpectrumIndex)
        if ax is None:
            self.logger.debug('on_singleclick_gate_edit - ax is None')
            return
        if dim == 2:
            self.xyRef = np.array([event.xdata,event.ydata])
        #left click
        if event.button == 1 :
            if  hasattr(self, 'altPressed') and self.altPressed :
                self.insertPointGate(event)
        #right click
        if event.button == 3 :
            if hasattr(self, 'altPressed') and self.altPressed :
                self.deletePointGate(event)
            else :
                pass
            self.currentPlot.canvas.draw()
        self.altPressed = False


    #move a gate line if double click on it 
    def on_dblclick_gate_edit(self, event, index):
        self.logger.info('on_dblclick_gate_edit')
        dim = self.getSpectrumInfoREST("dim", index=self.gatePopup.gateSpectrumIndex)
        if dim == 2:
            self.gatePopup.gateEditOption = "2d_move_all"
            # self.xyRef = [event.xdata,event.ydata]
            # self.gateFollower = self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_connect("motion_notify_event", self.followmouse)
            self.gateReleaser = self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_connect("button_press_event", self.releaseonclick)



    #udpate the gatePopup.regionPoint text according to the gateList [line2D]
    def updateTextGatePopup(self, gateList):
        # Add text in gatePopup
        dim = self.getSpectrumInfoREST("dim", index=self.gatePopup.gateSpectrumIndex)
        if dim ==1 :
            lines = gateList
            lineText = ""
            for nbLine in range(len(lines)):
                if nbLine == 0:
                    lineText = lineText + (f"{nbLine}: X= {gateList[nbLine].get_xdata()[0]:.5f}") 
                else :
                    lineText = lineText + (f"\n{nbLine}: X= {gateList[nbLine].get_xdata()[0]:.5f}") 
        elif dim ==2 :
            lines = gateList[0].get_xydata()
            lineText = ""
            specialTypes = ["c", "gc"]
            nbPoints = len(lines)
            if self.gatePopup.listGateType.currentText() in specialTypes:
                #for c or gc range(lineNb-1) because point0 = pointLast
                nbPoints = len(lines)-1
            for nbLine in range(nbPoints):
                if nbLine == 0:
                    lineText = lineText + (f"{nbLine}: X= {lines[nbLine][0]:.5f}   Y= {lines[nbLine][1]:.5f}") 
                else :
                    lineText = lineText + (f"\n{nbLine}: X= {lines[nbLine][0]:.5f}   Y= {lines[nbLine][1]:.5f}") 

        self.gatePopup.regionPoint.clear()
        self.gatePopup.regionPoint.insertPlainText(lineText)


    # For gate editing when pick_event signal
    def clickOnGateLine(self, event):
        self.logger.info('clickOnGateLine')
        #left click
        if event.mouseevent.button != 1 :
            return 
        self.editThisGateLine = None
        # self.xyRef = [event.mouseevent.x,event.mouseevent.y]
        dim = self.getSpectrumInfoREST("dim", index=self.gatePopup.gateSpectrumIndex)
        ax = self.getSpectrumInfo("axis", index=self.gatePopup.gateSpectrumIndex)
        if ax is None :
            self.logger.debug('clickOnGateLine - ax is None')
            return
        
        #Get the line(s) from axis child
        lineCandidate = [child for child in ax.get_children() if type(child) == matplotlib.lines.Line2D and child == event.artist] 
        if len(lineCandidate) > 0:
            self.editThisGateLine = lineCandidate[0]
            if dim == 2 :
                #lineCandidate only one artist from drawGate even for 2d gates, need to decompose into segments and points
                self.editThisGateLine = lineCandidate[0]
        else :
            self.logger.debug('clickOnGateLine - len(lineCandidate) <= 0')
            return
        #Following block sets the name and type in gatePopup and get the segment number to update self.gatePopup.listRegionLine
        labelSplit = lineCandidate[0].get_label().split("_-_")
        if len(labelSplit) != 3 or labelSplit[0] != "gate":
            self.logger.debug('clickOnGateLine - lineLabel has not the expected format')
            return
        gateName = labelSplit[1]
        gate = [dict for dict in self.rest.listGate() if dict["name"] == gateName]

        self.gatePopup.gateNameList.setCurrentText(gateName)
        self.gatePopup.listGateType.clear()
        self.gatePopup.listGateType.addItem(gate[0]["type"])

        #Fill listRegionLine with the axis child, in order to use pushGateToRest later in okGate
        gateIdentifier = "gate_-_" + gateName + "_-_"
        gateLines = [child for child in ax.get_children() if type(child) == matplotlib.lines.Line2D and gateIdentifier in child.get_label()]

        self.updateTextGatePopup(gateLines)
        
        #Start mouse follow signal but only effective once self.gatePopup.gateEditOption is set
        self.gateFollower = self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_connect("motion_notify_event", self.followmouse)
        if dim == 1:
            self.gatePopup.gateEditOption = "1d_move_line"
            self.gateReleaser = self.wTab.wPlot[self.wTab.currentIndex()].canvas.mpl_connect("button_press_event", self.releaseonclick)
        elif dim ==2:
            #Note that to drag the entire gate the self.gatePopup.gateEditOption is set on_dblclick_gate_edit
            #But still it is necessary to use clickOnGateLine such that one nows which gate to drag
            self.editThisGateLine.set_marker(marker='o')
            self.editThisGateLine.set_color("green")
            self.currentPlot.canvas.draw()
            self.gatePopup.gateEditOption = "2d_move_point"


    def dist(self, x, y):
        # Return the distance between two points.
        d = x - y
        return np.sqrt(np.dot(d, d))

    def dist_point_to_segment(self, p, s0, s1):
        #Get the distance of a point to a segment.
        #  *p*, *s0*, *s1* are *xy* sequences
        #This algorithm from
        #http://www.geomalgorithms.com/algorithms.html
        v = s1 - s0
        w = p - s0
        c1 = np.dot(w, v)
        if c1 <= 0:
            return self.dist(p, s0)
        c2 = np.dot(v, v)
        if c2 <= c1:
            return self.dist(p, s1)
        b = c1 / c2
        pb = s0 + b * v
        return self.dist(p, pb)


    ##############################
    # 11) 1D/2D region integration
    ##############################

    def integrate(self):
        self.logger.info('integrate')
        ax = self.getSpectrumInfo("axis", index=self.currentPlot.selected_plot_index)
        if ax is None or self.currentPlot.selected_plot_index is None:
            self.logger.debug('integrate - ax is None or self.currentPlot.selected_plot_index is None')
            return QMessageBox.about(self,"Warning!", "Please add/select one spectrum")
        else:
            index = self.currentPlot.selected_plot_index

            #reset info 
            self.integratePopup.clearInfo()
            self.disconnectGateSignals()

            #Set column headers of the results table
            colHeader = ['Spectrum', 'Region', 'Counts', 'Centroid X', 'Centroid Y', 'FWHM X', 'FWHM Y']
            for col, header in enumerate(colHeader):
                headerItem = QTableWidgetItem(header)
                font = self.integratePopup.resultsText.font()
                font.setBold(True)
                headerItem.setFont(font)
                self.integratePopup.resultsText.setHorizontalHeaderItem(col, headerItem)

            resultsCombined = {}
            results = {}

            #find lineList of summing regions 
            sumRegionLines = self.getSumRegion(index)
            if sumRegionLines is None or len(sumRegionLines) == 0 :
                self.logger.debug('integrate - sumRegionLines is None or len(sumRegionLines) == 0')
                pass
            else:
                #get a results dict for summing region
                results = self.integrateGateLocal(index, sumRegionLines)

            #now get a results dict for gates
            gateIdentifier = "gate_-_" 
            gateLines = [child for child in ax.get_children() if type(child) == matplotlib.lines.Line2D and gateIdentifier in child.get_label()]
            resultsGate = self.integrateGateLocal(index, gateLines)

            #combining integration results 
            if results is not None and len(results) > 0 and resultsGate is not None and len(resultsGate) > 0:
                resultsCombined = results
                for specName, listGateResults in resultsGate.items():
                    #if there is already specName in resultsCombined then append the existing list.
                    if specName in resultsCombined.keys():
                        for listItem in listGateResults:
                            resultsCombined[specName].append(listItem)
                    else :
                        resultsCombined[specName] = listGateResults
            elif results is not None and len(results) > 0:
                resultsCombined = results                
            else :
                resultsCombined = resultsGate

            #if no result set default message
            if resultsCombined is None or len(resultsCombined) == 0:
                self.integratePopup.resultsText.insertRow(0)
                default = "Nothing to integrate"
                newItem = QTableWidgetItem(default)
                self.integratePopup.resultsText.setItem(0, 0, newItem)
                self.integratePopup.show()
                return
            else :
                self.formatResultsIntegrate(resultsCombined)
                #for copy to clipboard
                self.sidTableIntegrateCopy = self.integratePopup.resultsText.itemSelectionChanged.connect(self.copySelectionIntegrateTable)
                self.integratePopup.show()


    def okIntegrate(self):
        self.logger.info('okIntegrate')
        self.disconnectGateSignals()
        self.integratePopup.close()


    def copySelectionIntegrateTable(self):
        self.logger.info('copySelectionIntegrateTable')
        resultTable = self.integratePopup.resultsText
        selectedItems = resultTable.selectedItems()
        if not selectedItems:
            return
        allValues = []
        for irow in range(resultTable.rowCount()):
            rowValues = []
            for icol in range(resultTable.columnCount()):
                item = resultTable.item(irow, icol)
                if item:
                    rowValues.append(item.text())
                else:
                    rowValues.append("")
            #separate rowValues with tab
            if len(rowValues) > 0:
                allValues.append("\t".join(rowValues))
        #separate row with new line
        formattedText = "\n".join(allValues)
        # Set the clipboard text
        QApplication.clipboard().setText(formattedText)


    def formatResultsIntegrate(self, results):
        self.logger.info('formatResultsIntegrate')
        # {'rawSet.0.other': [{'centroid': 0.0, 'fwhm': 0.0, 'counts': 37091.0, 'regionName': 'aaa'}, 
        # [{'centroid': 0.0, 'fwhm': 0.0, 'counts': 37091.0, 'regionName': 'slice_002'}]]}
        dum = []
        irow = 0
        for spectrumName, resultList in results.items():
            #keep following commented lines- anticipate feature to show integration results of all spectrum in current tab
            # foundName = False
            # for row in range(self.integratePopup.resultsText.rowCount()):
            #     item = self.integratePopup.resultsText.item(row, 0)
            #     # print("Simon - in founName loop ", foundName, item, spectrumName, self.integratePopup.resultsText.item(row, 1), self.integratePopup.resultsText.item(row, 2), row )
            #     if item == spectrumName:
            #         foundName = True
            #         break
            # # print("Simon - after founName ", foundName, spectrumName )
            # if foundName :
            #     continue      

            #result is a dict in resultList
            for result in resultList:
                if len(result) == 0:
                    continue
                elif "centroid" in result.keys():
                    result = self.setPrecisionIntegrationResult(result)
                    #a centroid list should mean it is dim == 2
                    if type(dum) == type(result["centroid"]):
                        row = [spectrumName, result["regionName"], str(result["counts"]), str(result["centroid"][0]), str(result["centroid"][1]), str(result["fwhm"][0]), str(result["fwhm"][1])]
                    else:
                        row = [spectrumName, result["regionName"], str(result["counts"]), str(result["centroid"]), '',  str(result["fwhm"]), '']
                
                #have now a row to insert
                self.integratePopup.resultsText.insertRow(irow)
                #set data of the new row
                for icol, cell in enumerate(row):
                    #spectrumName is inserted only the first time
                    if icol == 0 and self.integratePopup.resultsText.findItems(cell, QtCore.Qt.MatchFixedString):
                        continue
                    newItem = QTableWidgetItem(cell)
                    self.integratePopup.resultsText.setItem(irow, icol, newItem)
                irow += 1

        self.integratePopup.show()


    # round numbers in dict toRound at a proper order according to fwhm
    def setPrecisionIntegrationResult(self, toRound):
        self.logger.info('setPrecisionIntegrationResult - toRound: %s', toRound)

        #round sci format to 3 decimals
        order = 3
        sciFormat = "{:." + str(order) + "E}"
        #dim == 2
        if type(['dumList']) == type(toRound["fwhm"]):
            for i, val in enumerate(toRound["fwhm"]):
                if val is None:
                    continue
                toRound["centroid"][i] = sciFormat.format(toRound["centroid"][i])
                toRound["fwhm"][i] = sciFormat.format(toRound["fwhm"][i])
        #dim == 1
        else :
            if toRound["centroid"] is not None:
                toRound["centroid"] = sciFormat.format(toRound["centroid"])
            if toRound["fwhm"] is not None:
                toRound["fwhm"] = sciFormat.format(toRound["fwhm"])

        # #wanted to get the order for rounding from fwhm
        # if "fwhm" in toRound:
        #     if type(['dumList']) == type(toRound["fwhm"]):
        #         for i, val in enumerate(toRound["fwhm"]):
        #             fwhm = val
        #             #format in sci and get the exponent 
        #             # order = int('{:.0E}'.format(fwhm).split('E')[1])
        #             sciFormat = "{:." + str(order) + "E}"
        #             toRound["centroid"][i] = sciFormat.format(toRound["centroid"][i])
        #             toRound["fwhm"][i] = sciFormat.format(toRound["fwhm"][i])

        #make sure counts does not have decimal
        toRound["counts"] = int(toRound["counts"])
        return toRound


    #perform gate/summing region integration with REST functions integrate1D and integrate2D
    def integrateGateLocal(self, index, gateLines):
        self.logger.info('integrateGateLocal - index: %s', index)
        resultsList = []
        resultsDict = {}
        index = self.currentPlot.selected_plot_index

        lineList = gateLines
        if lineList is None or len(lineList) == 0 :
            self.logger.debug('integrateGateLocal - lineList is None or len(lineList) == 0')
            return

        dim = self.getSpectrumInfoREST("dim", index=index)
        spectrumName = self.nameFromIndex(index) 

        step = 1
        if dim == 1:
            #assuming there are two consecutive lines per gate for 1d
            step = 2

        for iLine in range(0, len(lineList), step):
            boundaries = []
            if dim == 1:
                gateName = lineList[iLine].get_label().split("_-_")[1]
                # gate 1Dgate_xamine s {aris.db1.ppac0.uc {1392.232056 1665.277466}}
                boundaries = [lineList[iLine].get_xdata()[0], lineList[iLine+1].get_xdata()[0]]
                # sort such that lowest first
                if boundaries[0] > boundaries[1]:
                    boundaries.sort()
                results = self.rest.integrate1D(spectrumName, boundaries[0], boundaries[1])
            elif dim == 2:
                #{'2Dgate_xamine': {'name': '2Dgate_xamine', 'type': 'c',
                # 'parameters': ['aris.tof.tdc.db3scin_to_db5scin', 'aris.db5.pin.dE'],
                # 'points': [{'x': 126.876877, 'y': 29.429428}, {'x': 125.625626, 'y': 25.825825},
                #            {'x': 126.626625, 'y': 22.522522}, {'x': 129.879883, 'y': 22.522522}, {'x': 130.63063, 'y': 26.126125}, {'x': 129.629623, 'y': 29.129128}]
                #}}
                gateName = lineList[iLine].get_label().split("_-_")[1]
                points = lineList[iLine].get_xydata()
                # dont want band type (not closed contour)
                if points[0][0] != points[-1][0] or points[0][1] != points[-1][1]:
                    continue
                results = self.rest.integrate2D(spectrumName, points)

            try:
                defaultResult = {'centroid': None, 'fwhm': None, 'counts': 0}
                if dim == 2:
                    defaultResult = {'centroid': [None, None], 'fwhm': [None, None], 'counts': 0}
                #if problem with integration rest.integrateGate returns status string
                if type(results) == type('dumString'):
                    results = defaultResult
                #Add gateName to result dict
                results['regionName'] = gateName
                resultsList.append(results)
            except :
                self.logger.debug('integrateGateLocal - exception ', exc_info=True)
                continue
        resultsDict[spectrumName] = resultsList
        return resultsDict


    ############################
    # 12)  Fitting
    ############################

    #callback to open extra function popup
    def spfunPopup(self):
        self.logger.info('spfunPopup callBack')
        if self.extraPopup.isVisible():
            self.extraPopup.close()

        self.extraPopup.show()

    #set axis limit text of the fit in extraPopup and returns these limits
    def axisLimitsForFit(self, ax):
        left, right = ax.get_xlim()
        self.logger.info('axisLimitsForFit - left, right: %s, %s', left, right)
        if self.extraPopup.fit_range_min.text():
            try:
                left = float(self.extraPopup.fit_range_min.text())
            except ValueError:
                self.logger.warning('axisLimitsForFit - Invalid input for Min X. Please enter a valid number.')
        else:
            left = ax.get_xlim()[0]
        if self.extraPopup.fit_range_max.text():
            try:
                right = float(self.extraPopup.fit_range_max.text())
            except ValueError:
                self.logger.warning('axisLimitsForFit - Invalid input for Max X. Please enter a valid number.')
        else:
            right = ax.get_xlim()[1]
        # Make sure xmin is always smaller than xmax.
        if left > right:
            left, right = right, left
            QMessageBox.about(self, "Warning", "xmin > xmax, the provided limits will be swapped for the fit")
        return left, right

    #Main fit function, reads user defined intial parameters and calls appropriate fit model
    def fit(self):
        self.logger.info('fit')
        fit_funct = self.extraPopup.fit_list.currentText()
        index = self.autoIndex()
        spectrumName = self.nameFromIndex(index)
        ax = self.getSpectrumInfo("axis", index=index)
        self.logger.debug('fit - spectrumName, fit_funct, index: %s, %s, %s', spectrumName, fit_funct, index)


        dim = self.getSpectrumInfoREST("dim", index=index)
        binx = self.getSpectrumInfoREST("binx", index=index)
        minxREST = self.getSpectrumInfoREST("minx", index=index)
        maxxREST = self.getSpectrumInfoREST("maxx", index=index)

        config = self.fit_factory._configs.get(fit_funct)
        self.logger.debug('fit - config: %s',config)

        fit = self.fit_factory.create(fit_funct, **config)

        try:
            if spectrumName != "":
                if dim == 1:
                    x = []
                    y = []
                    xtmp = self.createRange(binx, minxREST, maxxREST)

                    if not self.extraPopup.fit_p0.text():
                        self.extraPopup.fit_p0.setText("0")
                    if not self.extraPopup.fit_p1.text():
                        self.extraPopup.fit_p1.setText("0")
                    if not self.extraPopup.fit_p2.text():
                        self.extraPopup.fit_p2.setText("0")
                    if not self.extraPopup.fit_p3.text():
                        self.extraPopup.fit_p3.setText("0")
                    if not self.extraPopup.fit_p4.text():
                        self.extraPopup.fit_p4.setText("0")
                    if not self.extraPopup.fit_p5.text():
                        self.extraPopup.fit_p5.setText("0")
                    if not self.extraPopup.fit_p6.text():
                        self.extraPopup.fit_p6.setText("0")
                    if not self.extraPopup.fit_p7.text():
                        self.extraPopup.fit_p7.setText("0")

                    fitpar = [float(self.extraPopup.fit_p0.text()), float(self.extraPopup.fit_p1.text()),
                              float(self.extraPopup.fit_p2.text()), float(self.extraPopup.fit_p3.text()),
                              float(self.extraPopup.fit_p4.text()), float(self.extraPopup.fit_p5.text()),
                              float(self.extraPopup.fit_p6.text()), float(self.extraPopup.fit_p7.text())]
                    
                    ytmp = (self.getSpectrumInfoREST("data", index=index)).tolist()
                    xmin, xmax = self.axisLimitsForFit(ax)

                    # create new tmp list with subrange for fitting
                    for i in range(len(xtmp)):
                        if (xtmp[i]>xmin and xtmp[i]<=xmax):
                            # xtmp and ytmp offset by one bin compared to data because of ytmp obtained with tolist(), correct x:
                            x.append(xtmp[i-1]+(xtmp[i]-xtmp[i-1])/2)
                            y.append(ytmp[i])
                    x = np.array(x)
                    y = np.array(y)

                    fitResultsText = QTextEdit()
                    if fit_funct == "Skeleton":
                        fitln = fit.start(x, y, xmin, xmax, fitpar, ax, fitResultsText)
                        self.setFitLineLabel(ax, fitln, fitResultsText, spectrumName)
                    else:
                        fitln = fit.start(x, y, xmin, xmax, fitpar, ax, fitResultsText)
                        self.setFitLineLabel(ax, fitln, fitResultsText, spectrumName)
                else:
                    QMessageBox.about(self, "Warning", "Sorry 2D fitting is not implemented yet")
            else:
                QMessageBox.about(self, "Warning", "Histogram not existing. Please load an histogram...")

            self.currentPlot.canvas.draw()

        except NameError as err:
            print(err)
            pass


    # Set fit outputs title based on fit line label, and set fit_results text
    def setFitResultsLineLabel(self, fitLineLabelIdx, resultsText, spectrumName):
        self.logger.info('setFitResultsLineLabel - fitLineLabelIdx: %d',fitLineLabelIdx)
        title = 'Fit '+str(fitLineLabelIdx)+' ('+ spectrumName +') :'
        self.extraPopup.fit_results.append(title)
        self.extraPopup.fit_results.append(resultsText.toPlainText())
        self.extraPopup.fit_results.append(' ')


    # Set fit line label, proper to an axis
    # Increment a fit line index used as line label
    def setFitLineLabel(self, ax, line, resultsText, spectrumName):
        self.logger.info('setFitLineLabel')
        fitLineLabel = "fit-_-"
        fitLabels = self.listFitLineLabels(ax)
        fitIdxs = [int(label) for label in fitLabels]
        fitIdx = 0

        if line is None:
            self.logger.debug('setFitLineLabel - line is None')
            return

        if not fitIdxs:
            fitLineLabel += "0"
        else:
            i = 0
            while i in fitIdxs:
                i += 1
            fitIdx = i
            fitLineLabel += str(i)

        line.set_label(fitLineLabel)
        self.setFitResultsLineLabel(fitIdx, resultsText, spectrumName)
        self.logger.debug('setFitLineLabel - line label: %s',line.get_label())


    # Delete fit function (delete_button callback), reads user defined fit indexes list and delete corresponding lines
    def deleteFit(self):
        self.logger.info('deleteFit')
        index = self.autoIndex()
        ax = self.getSpectrumInfo("axis", index=index)
        userFitIdxs = self.extraPopup.delete_fitIdx_list.text().split()
        availableFitIdxs = self.listFitLineLabels(ax)
        notAvailableFitIdxs = [fitIdx for fitIdx in userFitIdxs if fitIdx not in availableFitIdxs]
        if notAvailableFitIdxs:
            self.logger.warning('deleteFit - fit line(s) index(es): %s cannot be deleted', notAvailableFitIdxs)
        else:
            for fitIdx in userFitIdxs:
                fitLineIdentifier = "fit-_-" + fitIdx
                for fitLine in ax.get_children():
                    if type(fitLine) == matplotlib.lines.Line2D and fitLineIdentifier in fitLine.get_label():
                        fitLine.remove()
                        self.logger.debug('deleteFit - removed fit line: %s', fitLineIdentifier)


    # Find all fit lines from ax and extract their labels
    def listFitLineLabels(self, ax):
        self.logger.info('listFitLineLabels')
        fitLabels = []
        fitLineIdentifier = "fit-_-"
        fitLineLabels = [fitLine.get_label() for fitLine in ax.get_children() if type(fitLine) == matplotlib.lines.Line2D and fitLineIdentifier in fitLine.get_label()]
        for label in fitLineLabels :
            labelSplit = label.split("-_-")
            fitLabels.append(labelSplit[1])
        return fitLabels


    # all_fitIdx_button callback, print in extraPopup.delete_fitIdx_list the indexes of all fit lines in axis
    def printFitLineLabels(self):
        self.logger.info('printFitLineLabels')
        index = self.autoIndex()
        ax = self.getSpectrumInfo("axis", index=index)
        fitLabels = self.listFitLineLabels(ax)
        textLabels = ''
        for label in fitLabels :
            textLabels += label + " "
        self.extraPopup.delete_fitIdx_list.setText(textLabels)


    ############################
    # 13) Peak Finding
    ############################

    def peakState(self, state):
        self.logger.info('peakState')
        for i, btn in enumerate(self.extraPopup.peak.peak_cbox):
            if btn.isChecked() == False:
                try:
                    self.removePeak(i)
                    self.isChecked[i] = False
                except:
                    pass
            else:
                if self.isChecked[i] == False:
                    self.drawSinglePeaks(self.peaks, self.properties, self.datay, i)
                    self.isChecked[i] = True

        self.currentPlot.canvas.draw()

    def create_peak_signals(self, peaks):
        self.logger.info('create_peak_signals')
        try:
            for i in range(len(peaks)):
                self.isChecked[i] = False
                self.extraPopup.peak.peak_cbox[i].stateChanged.connect(self.peakState)
                self.extraPopup.peak.peak_cbox[i].setChecked(True)
        except:
            pass

    def peakAnalClear(self):
        self.logger.info('peakAnalClear')
        self.extraPopup.peak.peak_results.clear()
        self.removeAllPeaks()
        self.resetPeakDict()

    def removePeak(self, i):
        self.logger.info('removePeak')
        self.peak_pos[i][0].remove()
        del self.peak_pos[i]
        self.peak_vl[i].remove()
        del self.peak_vl[i]
        self.peak_hl[i].remove()
        del self.peak_hl[i]
        self.peak_txt[i].remove()
        del self.peak_txt[i]

    def resetPeakDict(self):
        self.logger.info('resetPeakDict')
        self.peak_pos = {}
        self.peak_vl = {}
        self.peak_hl = {}
        self.peak_txt = {}

    def removeAllPeaks(self):
        self.logger.info('removeAllPeaks')
        try:
            for i in range(len(self.peaks)):
                self.extraPopup.peak.peak_cbox[i].setChecked(False)
                self.isChecked[i] = False
        except:
            pass

        self.currentPlot.canvas.draw()


    def drawSinglePeaks(self, peaks, properties, data, index):
        self.logger.info('drawSinglePeaks - index, properties: %s, %s', index, properties)
        ax = self.getSpectrumInfo("axis", index=self.currentPlot.selected_plot_index)
        x = self.datax.tolist()
        self.peak_pos[index] = ax.plot(x[peaks[index]], int(data[peaks[index]]), "v", color="red")
        self.peak_vl[index] = ax.vlines(x=x[peaks[index]], ymin=data[peaks[index]] - properties["prominences"][index], ymax = data[peaks[index]], color = "red")
        self.peak_hl[index] = ax.hlines(y=properties["width_heights"][index], xmin=properties["left_ips"][index], xmax=properties["right_ips"][index], color = "red")
        self.peak_txt[index] = ax.text(x[peaks[index]], int(data[peaks[index]]*1.1), str(int(x[peaks[index]])))


    def update_peak_output(self, peaks, properties):
        self.logger.info('update_peak_output - len(peaks), properties: %s, %s', len(peaks), properties)
        x = self.datax.tolist()
        for i in range(len(peaks)):
            s = "Peak"+str(i+1)+"\n\tpeak @ " + str(int(x[peaks[i]]))+", FWHM="+str(int(properties['widths'][i]))
            self.extraPopup.peak.peak_results.append(s)

    def analyzePeak(self):
        self.logger.info('analyzePeak')
        try:
            index = self.currentPlot.selected_plot_index
            ax = self.getSpectrumInfo("axis", index=index)
            x = []
            y = []
            # input points for peak finding
            width = int(self.extraPopup.peak.peak_width.text())
            dim = self.getSpectrumInfoREST("dim", index=index)
            binx = self.getSpectrumInfoREST("binx", index=index)
            minxREST = self.getSpectrumInfoREST("minx", index=index)
            maxxREST = self.getSpectrumInfoREST("maxx", index=index)

            xtmp = self.createRange(binx, minxREST, maxxREST)
            ytmp = (self.getSpectrumInfoREST("data", index=index)).tolist()

            xmin, xmax = ax.get_xlim()
            self.logger.debug('analyzePeak - xmin, xmax: %s, %s', xmin, xmax)

            # create new tmp list with subrange for fitting
            for i in range(len(xtmp)):
                if (xtmp[i]>=xmin and xtmp[i]<xmax):
                    x.append(xtmp[i])
                    y.append(ytmp[i])
            self.datax = np.array(x)
            self.datay = np.array(y)
            # if (DEBUG):
            #     print(self.datax)
            #     print(self.datay)
            #     print("xtmp", type(self.datax), "with len", len(self.datax.tolist()), "ytmp", type(self.datay), "with len", len(self.datay.tolist()))
            self.peaks, self.properties = find_peaks(self.datay, prominence=1, width=width)

            # if (DEBUG):
            #     print("peak list with indices", self.peaks)
            #     print("peak properties list", self.properties)
            self.update_peak_output(self.peaks, self.properties)
            self.create_peak_signals(self.peaks)

        except:
            pass


    ############################
    # 15) Overlaying pic
    ############################


    def openFigureDialog(self):
        self.logger.info('openFigureDialog')
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Open file...", "","Image Files (*.png *.jpg);;All Files (*)", options=options)
        if fileName:
            return fileName

    def loadFigure(self):
        self.logger.info('loadFigure')
        fileName = self.openFigureDialog()
        self.extraPopup.imaging.loadLISE_name.setText(fileName)
        if (DEBUG):
            print(fileName)
        try:
            if os.path.isfile(fileName):
                self.LISEpic = cv2.imread(fileName, 0)
                cv2.resize(self.LISEpic, (200, 100))
        except:
            pass

    def fineUpMove(self):
        self.imgplot.remove()
        self.ystart += 0.002
        self.drawFigure()

    def fineDownMove(self):
        self.imgplot.remove()
        self.ystart -= 0.002
        self.drawFigure()

    def fineLeftMove(self):
        self.imgplot.remove()
        self.xstart -= 0.002
        self.drawFigure()

    def fineRightMove(self):
        self.imgplot.remove()
        self.xstart += 0.002
        self.drawFigure()

    def moveFigure(self):
        self.logger.info('moveFigure')
        # if (DEBUG):
        #     print(self.extraPopup.imaging.joystick.direction, self.extraPopup.imaging.joystick.distance)
        try:
            self.imgplot.remove()
            if self.extraPopup.imaging.joystick.direction == "up":
                self.ystart += self.extraPopup.imaging.joystick.distance*0.03
            elif self.extraPopup.imaging.joystick.direction == "down":
                self.ystart -= self.extraPopup.imaging.joystick.distance*0.03
            elif self.extraPopup.imaging.joystick.direction == "left":
                self.xstart -= self.extraPopup.imaging.joystick.distance*0.03
            else:
                self.xstart += self.extraPopup.imaging.joystick.distance*0.03
            self.drawFigure()
        except:
            pass

    def indexToStartPosition(self, index):
        self.logger.info('indexToStartPosition')
        row = int(self.wConf.histo_geo_row.currentText())
        col = int(self.wConf.histo_geo_col.currentText())
        # if (DEBUG):
        #     print("row, col",row, col)
        xoffs = float(1/(2*col))
        yoffs = float(1/(2*row))
        i, j = self.plotPosition(index)
        # if (DEBUG):
        #     print("plot position in geometry", i, j)
        xstart = xoffs*(2*j+1)-0.1
        ystart = yoffs*(2*i+1)+0.1

        self.xstart = xstart
        self.ystart = 1-ystart
        # if (DEBUG):
        #     print("self.xstart", self.xstart, "self.ystart", self.ystart)

    def drawFigure(self):
        self.logger.info('drawFigure')
        self.alpha = self.extraPopup.imaging.alpha_slider.value()/10
        self.zoomX = self.extraPopup.imaging.zoomX_slider.value()/10
        self.zoomY = self.extraPopup.imaging.zoomY_slider.value()/10

        ax = plt.axes([self.xstart, self.ystart, self.zoomX, self.zoomY], frameon=True)
        ax.axis('off')
        self.imgplot = ax.imshow(self.LISEpic,
                                 aspect='auto',
                                 alpha=self.alpha)

        self.currentPlot.canvas.draw()

    def deleteFigure(self):
        self.logger.info('deleteFigure')
        self.imgplot.remove()
        self.onFigure = False
        self.currentPlot.canvas.draw()

    def transFigure(self):
        self.logger.info('transFigure')
        self.extraPopup.imaging.alpha_label.setText("Transparency Level ({} %)".format(self.extraPopup.imaging.alpha_slider.value()*10))
        try:
            self.deleteFigure()
            self.drawFigure()
        except:
            pass

    def zoomFigureX(self):
        self.logger.info('zoomFigureX')
        self.extraPopup.imaging.zoomX_label.setText("Zoom X Level ({} %)".format(self.extraPopup.imaging.zoomX_slider.value()*10))
        try:
            self.deleteFigure()
            self.drawFigure()
        except:
            pass

    def zoomFigureY(self):
        self.logger.info('zoomFigureY')
        self.extraPopup.imaging.zoomY_label.setText("Zoom Y Level ({} %)".format(self.extraPopup.imaging.zoomY_slider.value()*10))
        try:
            self.deleteFigure()
            self.drawFigure()
        except:
            pass

    def addFigure(self):
        self.logger.info('addFigure')
        try:
            self.indexToStartPosition(self.currentPlot.selected_plot_index)
            if self.onFigure == False:
                self.drawFigure()
                self.onFigure = True
        except NameError:
            raise
            #QMessageBox.about(self, "Warning", "Please select one histogram...")

    ############################
    # 16) Jupyter Notebook
    ############################

    #Create dataframe for Jupyter and web
    def createDf(self):
        self.logger.info('createDf')
        try:
            spectrumDict = self.getSpectrumInfoRESTDict()
            #reformat spectrumDict which is {spectrumName: {info1: , info2: ,...}} to {spectrumName: [],info1: [], info2: [],...}
            formatedDict = {'name': [], 'dim': [], 'binx': [], 'minx': [], 'maxx': [], 'biny': [], 'miny': [], 'maxy': [], 'data': [], 'parameters': [], 'type': []}
            for spectrumName, infoDict in spectrumDict.items():
                formatedDict["name"].append(spectrumName)
                for keyInfo, valInfo in infoDict.items():
                    #ndarray with data will be parsed to list (1d) or list of list (2d)
                    #this makes the data parsing easier from csv file.
                    #might want the same for the other valInfo...
                    if isinstance(valInfo, np.ndarray):
                        toList = []
                        if len(valInfo.shape) == 1:
                            toList = valInfo.tolist()
                        if len(valInfo.shape) == 2:
                            toList = [[item for item in row] for row in valInfo]
                        valInfo = toList 
                    formatedDict[keyInfo].append(valInfo)
            df = pd.DataFrame.from_dict(formatedDict)
            df.to_csv(self.extraPopup.peak.jup_df_filename.text(), index=False, compression='gzip')
        except:
            pass

    def jupyterStop(self):
        self.logger.info('jupyterStop')
        # stop the notebook process
        log("Sending interrupt signal to jupyter-notebook")
        self.extraPopup.peak.jup_start.setEnabled(True)
        self.extraPopup.peak.jup_stop.setEnabled(False)
        self.extraPopup.peak.jup_start.setStyleSheet("background-color:#3CB371;")
        self.extraPopup.peak.jup_stop.setStyleSheet("")
        stopnotebook()

    def jupyterStart(self):
        self.logger.info('jupyterStart')
        # dump df to gzip
        self.createDf()
        #starting jupyter server
        s = QSettings()
        execname = s.value(SETTING_EXECUTABLE, "jupyter-notebook")
        if not testnotebook(execname):
            while True:
                QMessageBox.information(None, "Error", "It appears that Jupyter Notebook isn't where it usually is. " +
                                        "Ensure you've installed Jupyter correctly and then press Ok to " +
                                        "find the executable 'jupyter-notebook'", QMessageBox.Ok)
                if testnotebook(execname):
                    break
                execname = QFileDialog.getOpenFileName(None, "Find jupyter-notebook executable", QDir.homePath())
                if not execname:
                    # user hit cancel
                    sys.exit(0)
                else:
                    execname = execname[0]
                    if testnotebook(execname):
                        log("Jupyter found at %s" % execname)
                        #save setting
                        s.setValue(SETTING_EXECUTABLE, execname)
                        break

        # setup logging
        # try to write to a log file, or redirect to stdout if debugging
        logname = "JupyterQtPy-"+time.strftime("%Y%m%d-%H%M%S")+".log"
        logfile = os.path.join(str(QDir.currentPath()), ".JupyterQtPy", logname)
        if not os.path.isdir(os.path.dirname(logfile)):
            os.mkdir(os.path.dirname(logfile))
            try:
                if DEBUG:
                    raise IOError()  # force logging to console
                setup_logging(logfile)
            except IOError:
                # no writable directory, log to console
                setup_logging(None)

        # workdir
        directory = s.value(SETTING_BASEDIR, QDir.currentPath())

        # setting window
        view = WebWindow(None, None)
        view.setWindowTitle("Jupyter CutiePie: %s" % directory)
        # logging on docked console
        qtlogger = QtLogger(view)
        qtlogger.newlog.connect(view.loggerdock.log)
        set_logger(lambda message: qtlogger.newlog.emit(message))

        log("Setting home directory --> "+str(directory))

        # start the notebook process
        webaddr = startnotebook(execname, directory=directory)
        view.loadmain(webaddr)

        # resume regular logging
        setup_logging(logfile)

        self.extraPopup.peak.jup_start.setEnabled(False)
        self.extraPopup.peak.jup_stop.setEnabled(True)
        self.extraPopup.peak.jup_start.setStyleSheet("")
        self.extraPopup.peak.jup_stop.setStyleSheet("background-color:#DC143C;")

    ##############################
    # 17) Misc tools
    ##############################
        
    # Override close method of main window
    def closeEvent(self, event):
        self.logger.info('closeEvent - MainWindow')
        # end threads (auto update, poll rest interface)
        self.stopAutoUpdateThread.set()
        self.endThread(self.threadAutoUpdate)
        self.stopRestThread.set()
        self.endThread(self.threadRest)
        # CutiePie closed properly, no need to save recovery info for connection
        self.setRecoVariable('CUTIEPIE_CONNECT', '')
        event.accept()

    
    # End openned threads, probably useless if threads close events are set correctly
    def endThread(self, thread):
        self.logger.info('endThread')
        if thread:
            try:
                if thread.is_alive():
                    thread.join()
            except:
                self.logger.debug('endThread - ', exc_info=True)


    # Gate and summing region popup close is connected to that, resume auto update
    def autoUpdateResume(self):
        self.logger.info('autoUpdateResume')
        self.skipAutoUpdateThread.clear()


    # Triggered when auto update slider value change 
    def autoUpdateStart(self):
        self.logger.info('autoUpdateStart')
        # Get interval from QSlider and update label
        updateInterval = self.extraPopup.options.autoUpdateIntervals[self.extraPopup.options.autoUpdate.value()]
        updateIntervalUser = self.extraPopup.options.autoUpdateIntervalsUser[self.extraPopup.options.autoUpdate.value()]
        self.extraPopup.options.autoUpdateLabel.setText("Update every: {}".format(updateIntervalUser))
        try:
            # If already a thread end it, and start a new 
            if self.threadAutoUpdate:
                if self.threadAutoUpdate.is_alive():
                    self.stopAutoUpdateThread.set()
            self.stopAutoUpdateThread.clear()
            self.skipAutoUpdateThread.clear()
            
            self.threadAutoUpdate = threading.Thread(target=self.autoUpdateThread, args=(updateInterval,))
            self.threadAutoUpdate.start()
        except ValueError:
            self.logger.debug('autoUpdateStart - ValueError exception', exc_info=True)


    #Spectrum update periodically, run on separated thread.
    def autoUpdateThread(self, interval):
        self.logger.info('autoUpdateThread')
        # Loop while stop event is false
        while not self.stopAutoUpdateThread.is_set():
            # Keep loop running but skip update if skip event 
            if self.skipAutoUpdateThread.is_set():
                continue
            # Wait while stop event is false, if event true break loop
            if self.stopAutoUpdateThread.wait(interval):
                break
            try:
                self.updatePlot()
            except ValueError:
                self.logger.debug('autoUpdateThread - ValueError exception', exc_info=True)


    def createRectangle(self, plot):
        self.logger.info('createRectangle')
        rec = matplotlib.patches.Rectangle((0, 0), 1, 1, ls="-", lw=2, ec="red", fc="none", transform=plot.transAxes)
        rec = plot.add_patch(rec)
        rec.set_clip_on(False)
        return rec


    def createDashedRectangle(self, plot):
        self.logger.info('createDashedRectangle')
        rec = matplotlib.patches.Rectangle((0, 0), 1, 1, ls=":", lw=2, ec="red", fc="none", transform=plot.transAxes)
        rec = plot.add_patch(rec)
        rec.set_clip_on(False)
        return rec


    def removeRectangle(self):
        self.logger.info('removeRectangle')
        try:                       
            for ax in self.currentPlot.figure.axes:
                for child in ax.get_children():
                    if type(child) == matplotlib.patches.Rectangle :
                        if child.get_ls() == ":" and child.get_lw() == 2:
                            if self.currentPlot.recDashed is not None :
                                self.currentPlot.recDashed.remove()
                                self.currentPlot.recDashed = None
                        elif child.get_ls() == "-" and child.get_lw() == 2:
                            if self.currentPlot.rec is not None :
                                self.currentPlot.rec.remove()
                                self.currentPlot.rec = None
        except NameError:
            raise

    #for debug
    def axesChilds(self):
        try:      
            for ax in self.currentPlot.figure.axes:
                print("Simon - axes ---------------------------- ",ax)
                for child in ax.get_children():
                    print("Simon - axesChilds - ",child)
                    if type(child) == matplotlib.lines.Line2D :
                        print("Simon - axesChild get_c", child.get_c())
                        print("Simon - axesChild get_lw", child.get_lw())
                        print("Simon - axesChild get_ls", child.get_ls())
                        print("Simon - axesChild get_xdata:",child.get_xdata())
                        print("Simon - axesChild get_ydata:",child.get_ydata())
        except NameError:
            raise


    #for debug
    def axesChildsTest(self, axis=None):
        try:    
            dumTypeList = []

            if type(axis) == type(dumTypeList):
                return
            print("Simon - axes ---------------------------- ",axis)
            for child in axis.get_children():
                print("Simon - axesChilds - ",child)
                if type(child) == matplotlib.lines.Line2D :
                    print("Simon - axesChild get_c", child.get_c())
                    print("Simon - axesChild get_lw", child.get_lw())
                    print("Simon - axesChild get_ls", child.get_ls())
                    print("Simon - axesChild get_xdata:",child.get_xdata())
                    print("Simon - axesChild get_ydata:",child.get_ydata())
        except NameError:
            raise


    def debugModeCallBack(self):
        if self.extraPopup.options.debugMode.isChecked():
            # allows to add only one instance of file handler
            if len(self.logger.handlers) > 0:
                for handler in self.logger.handlers:
                    # print("Simon - filehandlers for loop add", handler)
                    # add the handlers to the logger
                    # makes sure no duplicate handlers are added
                    if not isinstance(handler, logging.handlers.TimedRotatingFileHandler):
                        self.logger.addHandler(self.fileHandler)
            else:
                self.logger.addHandler(self.fileHandler)
        else :
            # close the file handler
            if len(self.logger.handlers) > 0:
                for handler in self.logger.handlers:
                    # print("Simon - filehandlers for loop delete ", handler)
                    # makes sure fileHandler exists
                    if isinstance(handler, logging.handlers.TimedRotatingFileHandler):
                        self.logger.removeHandler(self.fileHandler)


    def rgbString(self, r, g, b):
        return f"\033[38;2;{r};{g};{b}m"

    def colorString(self, string, rgbList):
        if len(rgbList) == 3:
            color = self.rgbString(rgbList[0], rgbList[1], rgbList[2])
            reset = "\033[0m" # Important!
            return str(f"{color}" + string + f"{reset}")
        else :
            return string

    # For summary spectrum, extract the last number from the parameter name
    def getLastDigitParam(self, parameterName):
        self.logger.info('getLastDigitParam - parameterName: %s', parameterName)
        parts = parameterName.split(".")
        if not any(part.isdigit() for part in parts):
            return None
        try:
            return int(parts[-1])
        except ValueError:
            self.logger.debug('getLastDigitParam - ValueError exception', exc_info=True)
            return None
        

    # In home directory, write a file to save some info e.g. useful for a recovery
    def setRecoVariable(self, varName, value):
        self.logger.info('setRecoVariable - varName %s, value %s', varName, value)
        failsafeFile = os.path.expanduser('~/.failsafe_cutiepie')
        try:
            with open(failsafeFile, "w") as failfile:
                failfile.write(f"{varName}={value}\n")
        except FileNotFoundError:
            os.makedirs(os.path.dirname(failsafeFile), exist_ok=True)
            with open(failsafeFile, "w") as failfile:
                failfile.write(f"{varName}={value}\n")




# redirect logging
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

        self.lineeditMin = QLineEdit(self)
        self.lineeditMax = QLineEdit(self)
        self.labelMin = QLabel(self)
        self.labelMin.setText("Min")
        self.labelMax = QLabel(self)
        self.labelMax.setText("Max")
        self.okButton = QPushButton("Ok", self)
        self.cancelButton = QPushButton("Cancel", self)
        self.resetButton = QPushButton("Reset", self)

        mainLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()
        fieldsLayout = QHBoxLayout()
        fieldsLayout.addWidget(self.labelMin)
        fieldsLayout.addWidget(self.lineeditMin)
        fieldsLayout.addWidget(self.labelMax)
        fieldsLayout.addWidget(self.lineeditMax)
        buttonsLayout.addWidget(self.okButton)
        buttonsLayout.addWidget(self.resetButton)
        buttonsLayout.addWidget(self.cancelButton)

        mainLayout.addLayout(fieldsLayout, 1, 0, 1, 0)
        mainLayout.addLayout(buttonsLayout, 2, 0, 1, 0)
        self.setLayout(mainLayout)


class zoomPopup(QDialog):

    # Hold the index of selected spectrum when openned zoomPopup 
    indexSpectrum = None

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setModal(True)

        self.lineeditXmin = QLineEdit(self)
        self.lineeditXmax = QLineEdit(self)
        self.lineeditYmin = QLineEdit(self)
        self.lineeditYmax = QLineEdit(self)
        self.labelXmin = QLabel(self)
        self.labelXmin.setText("Xmin")
        self.labelXmax = QLabel(self)
        self.labelXmax.setText("Xmax")
        self.labelYmin = QLabel(self)
        self.labelYmin.setText("Ymin")
        self.labelYmax = QLabel(self)
        self.labelYmax.setText("Ymax")
        self.okButton = QPushButton("Ok", self)
        self.cancelButton = QPushButton("Cancel", self)
        self.mainLayout = QGridLayout()

    def setVisibleFields(self, value=True):
        self.labelYmin.setVisible(value)
        self.labelYmax.setVisible(value)
        self.lineeditYmin.setVisible(value)
        self.lineeditYmax.setVisible(value)

    def layout1d(self):
        # while self.mainLayout.count() > 0:
        #     item = self.mainLayout.takeAt(0)
        #     if item.widget():
        #         item.widget().deleteLater()
        self.setVisibleFields(False)
        buttonsLayout = QHBoxLayout()
        fields1Layout = QHBoxLayout()
        fields1Layout.addWidget(self.labelXmin)
        fields1Layout.addWidget(self.lineeditXmin)
        fields1Layout.addWidget(self.labelXmax)
        fields1Layout.addWidget(self.lineeditXmax)
        buttonsLayout.addWidget(self.okButton)
        buttonsLayout.addWidget(self.cancelButton)

        self.mainLayout.addLayout(fields1Layout, 1, 0, 1, 0)
        self.mainLayout.addLayout(buttonsLayout, 2, 0, 1, 0)

        self.setLayout(self.mainLayout)

    def layout2d(self):
        # while self.mainLayout.count() > 0:
        #     item = self.mainLayout.takeAt(0)
        #     if item.widget():
        #         item.widget().deleteLater()
        self.setVisibleFields(True)
        buttonsLayout = QHBoxLayout()
        fields1Layout = QHBoxLayout()
        fields1Layout.addWidget(self.labelXmin)
        fields1Layout.addWidget(self.lineeditXmin)
        fields1Layout.addWidget(self.labelXmax)
        fields1Layout.addWidget(self.lineeditXmax)
        fields2Layout = QHBoxLayout()
        fields2Layout.addWidget(self.labelYmin)
        fields2Layout.addWidget(self.lineeditYmin)
        fields2Layout.addWidget(self.labelYmax)
        fields2Layout.addWidget(self.lineeditYmax)
        buttonsLayout.addWidget(self.okButton)
        buttonsLayout.addWidget(self.cancelButton)

        self.mainLayout.addLayout(fields1Layout, 1, 0, 1, 0)
        self.mainLayout.addLayout(fields2Layout, 2, 0, 1, 0)
        self.mainLayout.addLayout(buttonsLayout, 3, 0, 1, 0)
        self.setLayout(self.mainLayout)



class centeredNorm(colors.Normalize):
    def __init__(self, data, vcenter=0, halfrange=None, clip=False):
        if halfrange is None:
            halfrange = np.max(np.abs(data - vcenter))
        super().__init__(vmin=vcenter - halfrange, vmax=vcenter + halfrange, clip=clip)

