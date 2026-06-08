#!/usr/bin/env python3
# import modules and packages

import sys, os, ast
import copy, cv2
import logging, logging.handlers
import threading, time, math, re
from copy import copy, deepcopy
import pandas as pd
import numpy as np
import CPyConverter as cpy

import signal, ctypes
import csv, json
from types import SimpleNamespace
from functools import partial



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
# ensure GUI.py's own directory (Script/ when installed) is importable so
# sub-packages like services/ are found regardless of launch CWD
if fileDir not in sys.path:
    sys.path.insert(0, fileDir)
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

from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QCompleter, QDialog,
    QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QPushButton,
    QShortcut, QSlider, QTabBar, QTableWidget, QTableWidgetItem,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)
from PyQt5.QtGui import QCursor, QKeySequence, QMouseEvent, QPalette
from PyQt5.QtCore import (
    pyqtSignal, pyqtSlot, Qt, QObject, QThread, QTimer, QElapsedTimer,
    QEventLoop, QSettings, QDir, QEvent, QPoint,
)


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
from MenuAndConfigGUI import Configuration
from SpecialFunctionsGUI import SpecialFunctions # all the extra functions we defined
# from OutputGUI import OutputPopup # popup output window
from PlotGUI import Plot # area defined for the histograms
from PlotGUI import Tabs # area defined for the Tabs
from PyREST import PyREST # class interface for SpecTcl REST plugin
from services.spectrum_store import SpectrumStore
from services.thread_workers import RestWorker, AutoUpdateWorker
from services.fit_manager import FitManager
from services.gate_manager import GateManager
from services.sum_region_manager import SumRegionManager
from services.connection_manager import ConnectionManager
from services.plot_controller import PlotController
from CopyPropertiesGUI import CopyProperties
from connectConfigGUI import ConnectConfiguration #class for the connection configuration popup
from MenuGate import MenuGate #class for the gate creation/edition popup
from MenuSumRegion import MenuSumRegion #class for the gate creation/edition popup
from OutputIntegrate import OutputIntegratePopup #popup for gate/summing region integrate outputs
# from OutputIntegrate import TableModel #table model for popup for gate/summing region

from logger import log, setup_logging, set_logger
from notebook_process import testnotebook, startnotebook, stopnotebook
from WebWindow import WebWindow

## Bashir added for alpha filter dialog
from alpha_filter_dialog import AlphaChainIsoFilterDialog


#from collapseMenu import Spoiler

SETTING_BASEDIR = "workdir"
SETTING_EXECUTABLE = "exec"
DEBUG = False
DEBOUNCE_DUR = 0.25
t = None
FIT_PREFIX = "fit-_-"  # if you keep it as class attr, reference with self.FIT_PREFIX

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


        self._abort_fit = False # Bashir added for aborting fit

        self.setWindowTitle("CutiePie - (QtPy) - It's not a bug, it's a feature (cit.) Qt5 and PyQty5 used under open source terms.")
        #### Bashir added for lightgrey visualization #######
        # self.setStyleSheet("background-color: lightgrey;")
        # self.setStyleSheet("QWidget { background-color: #dcdcdc; }")  # light grey everywhere

        self.setMouseTracking(True)


        self.stopAutoUpdateThread = threading.Event()
        self.skipAutoUpdateThread = threading.Event()

        # Bashir, make the first `addPlot()` call your starter
        self.stopAutoUpdateThread.set()

        self.stopRestThread = threading.Event()

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
        self.connectConfig = ConnectConfiguration()

        # gate window
        self.gatePopup = MenuGate()

        # copy attributes windows
        self.copyAttr = CopyProperties()

        # initialize factory from algo_creator
        #self.factory.initialize(self.extraPopup.imaging.clusterAlgo)
        # initialize factory from fit_creator
        self.fit_factory.initialize(self.extraPopup.fit_list)

        # global variables
        #spectra (SpectrumStore): canonical REST registry {name -> {dim,binx,minx,maxx,biny,miny,maxy,data,parameters,type}}
        self.spectra = SpectrumStore()

        self.fit_manager = FitManager(
            window=self,
            fit_factory=self.fit_factory,
            spectra=self.spectra,
            extra_popup=self.extraPopup,
            logger=self.logger,
        )

        self.gate_manager = GateManager(
            window=self,
            spectra=self.spectra,
            gate_popup=self.gatePopup,
            logger=self.logger,
        )
        self.sum_region_manager = SumRegionManager(
            window=self,
            sum_popup=self.sumRegionPopup,
            logger=self.logger,
        )
        self.connection_manager = ConnectionManager(
            window=self,
            wConf=self.wConf,
            connect_config=self.connectConfig,
            stop_rest=self.stopRestThread,
            stop_auto=self.stopAutoUpdateThread,
            skip_auto=self.skipAutoUpdateThread,
            logger=self.logger,
        )
        self.plot_controller = PlotController(
            window=self,
            wTab=self.wTab,
            wConf=self.wConf,
            logger=self.logger,
        )

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


        # Bool set by extra options -> differentiate gates 
        # self.annotateGate = False

        ### Bashir added for enlarged view
        self._enlargeBusy = False
        self._enlarged_cax = None   # (optional) track colorbar made in enlarged view


        #################
        # 2) Signals
        #################

        # top menu signals
        self.wConf.connectButton.clicked.connect(self.connection_manager.connectPopup)
        self.connectConfig.ok.clicked.connect(self.connection_manager.okConnect)
        self.connectConfig.cancel.clicked.connect(self.connection_manager.closeConnect)

        ### Bashir added to auto select connect button if ports are default
        rest_text   = self.connectConfig.rest.text().strip()
        mirror_text = self.connectConfig.mirror.text().strip()
        if rest_text.isdigit() and mirror_text.isdigit():
            # visually show "connected"
            self.wConf.connectButton.setChecked(True)

            # actually perform the connection once the event loop is ready
            QTimer.singleShot(0, self.okConnect)

        #### Bashir chenged ###################
        menu = QMenu(self.wConf.geometryButton)     # parent the menu to the button

        # Geometry menu (same gold)
        menu = QMenu(self.wConf.geometryButton)
        menu.setStyleSheet("""
            QMenu { background-color: #ffd700; color: black; }
            QMenu::item { background-color: #ffd700; }
            QMenu::item:selected { background-color: #e6c200; }
        """)
        menu.setFixedWidth(120)

        actSave = menu.addAction("Save Geometry")
        actSave.triggered.connect(self.saveGeo)

        actLoad = menu.addAction("Load Geometry")
        actLoad.triggered.connect(self.loadGeo)

        self.wConf.geometryButton.setMenu(menu)
        # self.wConf.saveButton.clicked.connect(self.saveGeo)
        # self.wConf.loadButton.clicked.connect(self.loadGeo)
        ######################################################

        self.wConf.exitButton.clicked.connect(self.closeAll)

        # new tab creation
        self.wTab.tabBarClicked.connect(self.clickedTab)
        self.wTab.tabBar().tabMoved.connect(self.movedTab)

        # config menu signals
        self.wConf.histo_geo_add.clicked.connect(self.addPlot)
        self.wConf.histo_geo_update.clicked.connect(lambda: self.updatePlot())
        self.wConf.extraButton.clicked.connect(self.spfunPopup)

        ### Bashir added for auto update #########################
        
        self.autoUpdateIntervalsUser = ["1 sec", "5 secs", "10 secs", "30 secs", "1 min", "3 mins", "5 mins", "10 mins", "Inf."]
        self.autoUpdateIntervals = [1, 5, 10, 30, 60, 180, 300, 600, 9e9]

        """
        val_auto = self.wConf.autoUpdate2.value()
        self.autoUpdateInterval = self.autoUpdateIntervals[val_auto]
        self.autoUpdateIntervalUser = self.autoUpdateIntervalsUser[val_auto]
        self.wConf.autoUpdateLabel2.setText(f"Update every: {self.autoUpdateIntervalsUser[val_auto]}")

        # update label as slider moves
        self.wConf.autoUpdate2.valueChanged.connect(
            lambda i: self.wConf.autoUpdateLabel2.setText(f"Update every: {self.autoUpdateIntervalsUser[i]}")
        )

        # (only if you want to start/restart the auto-update thread on change)
        self.wConf.autoUpdate2.valueChanged.connect(lambda _: self.autoUpdateStart())
        """
        # populate combo and set default (index 0 → 1 sec)
        self.wConf.autoUpdate2.clear()
        self.wConf.autoUpdate2.addItems(self.autoUpdateIntervalsUser)
        self.wConf.autoUpdate2.setCurrentIndex(8)

        # make existing code that calls .value()/.setValue() still work
        self.wConf.autoUpdate2.value = self.wConf.autoUpdate2.currentIndex
        self.wConf.autoUpdate2.setValue = self.wConf.autoUpdate2.setCurrentIndex

        # initial label
        # i0 = self.wConf.autoUpdate2.currentIndex()
        # self.wConf.autoUpdateLabel2.setText(f"Update every:")

        # keep label updated and restart your existing thread logic
        # self.wConf.autoUpdate2.currentIndexChanged.connect(
        #     lambda: self.wConf.autoUpdateLabel2.setText(f"Update every:")
        # )
        self.wConf.autoUpdate2.currentIndexChanged.connect(lambda _: self.autoUpdateStart())

        ##########################################################

        
        #### Bashir added
        self.wConf.cmapSelector.currentTextChanged.connect(self.onColormapChange)
        self.old_cmap = None  # to store the previous colormap
        self.geometry_applied = False
        # self.wConf.darkModeButton.clicked.connect(self.toggleDarkMode)


        ##### Bashir commented out to examine apply button
        # self.wConf.histo_geo_row.activated.connect( self.setCanvasLayout )
        # self.wConf.histo_geo_col.activated.connect( self.setCanvasLayout )
        self.wConf.histo_geo_apply_btn.clicked.connect(self.setCanvasLayout)
        # self.setCanvasLayout()
        # QShortcut(QKeySequence("Return"), self, activated=self.wConf.histo_geo_apply_btn.click)
        # QShortcut(QKeySequence("Enter"), self, activated=self.wConf.histo_geo_apply_btn.click)
        ####################################################################


        self.wConf.createGate.clicked.connect(self.gate_manager.createGate)
        self.wConf.createGate.setEnabled(False)
        self.gatePopup.ok.clicked.connect(self.gate_manager.okGate)
        self.gatePopup.cancel.clicked.connect(self.gate_manager.cancelGate)
        self.gatePopup.gateActionCreate.clicked.connect(self.gate_manager.createGate)
        self.gatePopup.gateActionEdit.clicked.connect(self.gate_manager.editGate)
        self.gatePopup.clearInfoSignal.connect(self.gatePopup.clearInfo)
        self.gatePopup.clearInfoSignal.connect(self.autoUpdateResume) 

        # summing region
        self.wConf.createSumRegionButton.clicked.connect(self.sum_region_manager.createSumRegion)
        self.sumRegionPopup.ok.clicked.connect(self.sum_region_manager.okSumRegion)
        self.sumRegionPopup.cancel.clicked.connect(self.sum_region_manager.cancelSumRegion)
        self.sumRegionPopup.delete.clicked.connect(self.sum_region_manager.deleteSumRegion)
        self.sumRegionPopup.clearInfoSignal.connect(self.sumRegionPopup.clearInfo)
        self.sumRegionPopup.clearInfoSignal.connect(self.autoUpdateResume)


        # self.wConf.editGate.setToolTip("Key bindings for Modify->Edit:\n"
        #                               "'i' insert vertex\n"
        #                               "'d' delete vertex\n")

        #integrate gate and summing region
        self.wConf.integrateGateAndRegion.clicked.connect(self.sum_region_manager.integrate)
        self.integratePopup.ok.clicked.connect(self.sum_region_manager.okIntegrate)

        self.tabp.okButton.clicked.connect(self.okTab)
        self.tabp.cancelButton.clicked.connect(self.cancelTab)

        self.cutoffp.okButton.clicked.connect(self.okCutoff)
        self.cutoffp.cancelButton.clicked.connect(self.cancelCutoff)
        self.cutoffp.resetButton.clicked.connect(lambda: self.resetCutoff(True))

        # zoom callback
        self.wTab.wPlot[self.wTab.currentIndex()].zoom_action.triggered.connect(self.zoomCallback)
        
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
        #### Bashir added for zooming hotkeys ####
        QShortcut(QKeySequence("+"), self.wTab.wPlot[self.wTab.currentIndex()]).activated.connect(lambda: self.zoomInOut("in"))
        QShortcut(QKeySequence("-"), self.wTab.wPlot[self.wTab.currentIndex()]).activated.connect(lambda: self.zoomInOut("out"))
        ###############################################

        # cutoff button
        self.wTab.wPlot[self.wTab.currentIndex()].cutoffButton.clicked.connect(self.cutoffButtonCallback)
        self.wTab.wPlot[self.wTab.currentIndex()].cutoffButton.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
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

        # extra popup — wired directly to fit_manager
        self.extraPopup.fit_button.clicked.connect(self.fit_manager.fit)
        self.extraPopup.plot_csv_button.clicked.connect(self.fit_manager.on_plot_csv_clicked)
        self.extraPopup.fit_csv_button.clicked.connect(self.fit_manager.on_fit_csv_clicked)
        self.extraPopup.abort_button.clicked.connect(self.fit_manager.on_abort_clicked)
        self.extraPopup.all_fitIdx_button.clicked.connect(self.fit_manager.printFitLineLabels)
        self.extraPopup.delete_button.clicked.connect(self.fit_manager.deleteFit)

        self.extraPopup.peak.peak_analysis.clicked.connect(self.analyzePeak)
        self.extraPopup.peak.peak_analysis_clear.clicked.connect(self.peakAnalClear)

        self.extraPopup.peak.jup_start.clicked.connect(self.jupyterStart)
        self.extraPopup.peak.jup_stop.clicked.connect(self.jupyterStop)

        self.extraPopup.options.gateAnnotation.clicked.connect(self.gate_manager.gateAnnotationCallBack)
        self.extraPopup.options.gateHide.clicked.connect(self.updatePlot)
        self.extraPopup.options.debugMode.clicked.connect(self.debugModeCallBack)
        ##### Bashir commented out the auto update in the main gate
        # self.extraPopup.options.autoUpdate.valueChanged.connect(self.autoUpdateStart)
        ################################################

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

        self.wTab.wPlot[self.wTab.currentIndex()].zoom_action.triggered.connect(self.zoomCallback)
        self.wTab.wPlot[self.wTab.currentIndex()].histo_autoscale.clicked.connect(lambda: self.autoScaleAxisBox(None))
        self.wTab.wPlot[self.wTab.currentIndex()].customZoomButton.clicked.connect(self.customZoomButtonCallback)
        self.wTab.wPlot[self.wTab.currentIndex()].customZoomButton.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.wTab.wPlot[self.wTab.currentIndex()].customZoomButton.customContextMenuRequested.connect(self.zoom_handle_right_click)
        self.wTab.wPlot[self.wTab.currentIndex()].plusButton.clicked.connect(lambda: self.zoomInOut("in"))
        self.wTab.wPlot[self.wTab.currentIndex()].minusButton.clicked.connect(lambda: self.zoomInOut("out"))
        self.wTab.wPlot[self.wTab.currentIndex()].cutoffButton.clicked.connect(self.cutoffButtonCallback)
        self.wTab.wPlot[self.wTab.currentIndex()].cutoffButton.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
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


    #Event filter for search in histo_list widget, and restrict position of moved tab
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
            #### Bashir added for mouse hovering ####
            if event.xdata is not None:
                self.mouse_x = event.xdata
                # self.mouse_y = event.ydata
            #########################################

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
        # print("Simon - on_release - self.currentPlot.zoomPress",self.currentPlot.zoomPress,self.currentPlot.zoom_action.isChecked())
        if self.currentPlot.zoomPress:
            self.currentPlot.zoom_action.triggered.emit()
            self.currentPlot.zoom_action.setChecked(False)
            self.currentPlot.customZoomButton.setDown(False)
            QTimer.singleShot(100, self.updatePlotLimits)
            self.currentPlot.zoomPress = False


    # when mouse pressed in main window
    # Introduced for endding zoom action (toolbar) see on_release and on_press too
    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.logger.info('mousePressEvent - self.currentPlot.zoomPress: %s',self.currentPlot.zoomPress)
        # print("Simon - mousePressEvent - ",self.currentPlot.zoomPress,self.currentPlot.zoom_action.isChecked())
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
            self.currentPlot.zoom_action.triggered.emit()
            self.currentPlot.zoom_action.setChecked(False)
            self.currentPlot.customZoomButton.setDown(False)
            self.currentPlot.zoomPress = False


    #callback for button_press_event
    def on_press(self, event):
        self.logger.info('on_press')
        # print("Simon - on_press - ",self.currentPlot.zoomPress,self.currentPlot.zoom_action.isChecked())
        #if initate zoom (magnifying glass) but dont press in axes, reset the action
        if self.currentPlot.zoomPress and not event.inaxes: 
            self.currentPlot.zoom_action.triggered.emit()
            self.currentPlot.zoom_action.setChecked(False)
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

        ##### Bashir added for energy calibration ####################
        # --- Calibration capture (Ctrl + Left-click while the calibration dialog is open) ---
        cal = getattr(self.fit_manager, "_cal", None)
        if cal and getattr(cal, "active", False) and (event.inaxes is cal.ax):
            ge = getattr(event, "guiEvent", None)
            ctrl = bool(ge and (ge.modifiers() & Qt.ControlModifier))
            if ctrl and event.button == 1 and (event.xdata is not None):
                cal.add_point(event.xdata)
                return  # swallow so it doesn’t trigger other actions


##################################################################################################


        if event.dblclick:
            if self.currentPlot.toCreateGate or self.currentPlot.toCreateSumRegion :
                #dont need to do anything special here, the 2d line is closed when click okGate
                # self.on_dblclick_gate(event, index)
                pass
            elif self.currentPlot.toEditGate:
                self.gate_manager.on_dblclick_gate_edit(event, index)
            else :
                self.on_dblclick(index)
        else :
            if self.currentPlot.toCreateGate :
                #1: left mouse button, 3: right mouse button
                if event.button == 1:
                    self.gate_manager.on_singleclick_gate(event, index)
                if event.button == 3:
                    self.gate_manager.on_singleclick_gate_right(index)
            elif self.currentPlot.toEditGate:
                self.gate_manager.on_singleclick_gate_edit(event)
            elif self.currentPlot.toCreateSumRegion :
                #1: left mouse button, 3: right mouse button
                if event.button == 1:
                    self.sum_region_manager.on_singleclick_sumRegion(event, index)
                if event.button == 3:
                    self.sum_region_manager.on_singleclick_sumRegion_right(index)
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
            # wPlot.cutoffButton.setDown(True)
            pass
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


    """
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

            print("Entering expanded spectrum view...")
            
            #important that zoomPlotInfo is set only while in zoom mode (not None only here)
            self.setEnlargedSpectrum(idx, name)
            self.currentPlot.next_plot_index = self.currentPlot.selected_plot_index
            print("self.currentPlot.next_plot_index",self.currentPlot.next_plot_index)
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
            print("self.wTab.selected_plot_index_bak[self.wTab.currentIndex()]", self.wTab.selected_plot_index_bak[self.wTab.currentIndex()])
            
            #setup single pad canvas
            self.currentPlot.InitializeCanvas(1,1,False)

            self.add(idx)
            autosclae_status = self.currentPlot.histo_autoscale.isChecked()
            self.updatePlot(autosclae_status)
            # self.updatePlot()
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

            t1 = time.time()
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
            t2 = time.time()
            print("on_dblclick: time={:.2f}".format(t2-t1))
    """
    
            
    #### Bashir's changes to avoid re-initialization of Canvas
    #called by on_press when not in ceate/edit gate mode
    def on_dblclick(self, idx):
        self.logger.info('on_dblclick - idx, self.wTab.currentIndex(): %s, %s' ,idx, self.wTab.currentIndex())

        #### Bashir added to gaurd against quick
        if self._enlargeBusy:
            return
        self._enlargeBusy = True
        self.skipAutoUpdateThread.set()

        try:
            name = self.nameFromIndex(idx)
            index = self.wConf.histo_list.findText(name)
            self.wConf.histo_list.setCurrentIndex(index)

            if self.currentPlot.isEnlarged == False: # entering enlarged mode
                self.logger.debug('on_dblclick - isEnlarged TRUE')
                if name == "empty" or index == -1:
                    self.logger.warning('on_dblclick - empty axes cannot enlarge')
                    return
                self.removeRectangle()

                print("Entering expanded spectrum view...")
                
                #important that zoomPlotInfo is set only while in zoom mode (not None only here)
                self.setEnlargedSpectrum(idx, name)
                self.currentPlot.next_plot_index = self.currentPlot.selected_plot_index
                print("self.currentPlot.next_plot_index",self.currentPlot.next_plot_index)
                self.currentPlot.isEnlarged = True
                # disabling adding histograms
                self.wConf.histo_geo_add.setEnabled(False)
                # disabling changing canvas layout
                self.wConf.histo_geo_row.setEnabled(False)
                self.wConf.histo_geo_col.setEnabled(False)
                self.wConf.histo_geo_apply_btn.setEnabled(False)
                self.wConf.geometryButton.setEnabled(False)

                # enabling gate creation
                self.wConf.createGate.setEnabled(True)
                # plot corresponding histogram
                self.wTab.selected_plot_index_bak[self.wTab.currentIndex()]= deepcopy(idx)
                print("self.wTab.selected_plot_index_bak[self.wTab.currentIndex()]", self.wTab.selected_plot_index_bak[self.wTab.currentIndex()])
                
                # t1 = time.time()
                ############### Bashir ##################################################
                # Save the current axes objects
                self.currentPlot._saved_axes = self.currentPlot.figure.axes.copy()

                # Hide all current axes
                for ax in self.currentPlot._saved_axes:
                    ax.set_visible(False)
                
                dim = self.getSpectrumInfoREST("dim", index=idx)
                if dim == 2:
                    spectrum_old = self.getSpectrumInfo("spectrum", index=idx)
                    self.old_cmap = spectrum_old.get_cmap()

                elif dim == 1:
                    # Save current y-limits of the target axes to restore later (when autoscale is OFF)
                    ax0 = self.getSpectrumInfo("axis", index=idx)
                    if ax0 is not None:
                        if not hasattr(self.currentPlot, "_saved_ylims"):
                            self.currentPlot._saved_ylims = {}
                        self.currentPlot._saved_ylims[idx] = ax0.get_ylim()
                else:
                    pass

                ###########################################################################
                #setup single pad canvas
                self.currentPlot.InitializeCanvas(1,1,False)
                autoscale_status = self.currentPlot.histo_autoscale.isChecked()
                self.add(idx)
                self.updatePlot()
                
                ###################################################################

                ax = self.getSpectrumInfo("axis", index=idx)
                if dim == 1:
                    if not autoscale_status and hasattr(self.currentPlot, "_saved_ylims") and idx in self.currentPlot._saved_ylims:
                        ax.set_ylim(*self.currentPlot._saved_ylims[idx])   # <-- restore y only
                # self.updatePlot()
                # t2 = time.time()
                # print("on_dblclick: time={:.2f}".format(t2-t1))

                ########### Bashir: reuse color map
                if dim == 2:
                    spectrum = self.getSpectrumInfo("spectrum", index=idx)

                    if spectrum is not None:
                        # print("Reusing color map for enlarged spectrum...")
                        spectrum.set_cmap(self.old_cmap)
                        if ax:
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes("right", size="5%", pad=0.05)
                            self.currentPlot.figure.colorbar(spectrum, cax=cax, orientation="vertical")

                            self.currentPlot.figure.tight_layout(rect=[0, 0, 0.95, 1])
                            self.currentPlot.canvas.draw_idle()
                #############################################################################################

            else:
                self.logger.debug('on_dblclick - isEnlarged FALSE')
                # enabling adding histograms
                self.wConf.histo_geo_add.setEnabled(True)
                # enabling changing canvas layout
                self.wConf.histo_geo_row.setEnabled(True)
                self.wConf.histo_geo_col.setEnabled(True)
                self.wConf.histo_geo_apply_btn.setEnabled(True)
                self.wConf.geometryButton.setEnabled(True)


                # disabling gate creation
                self.wConf.createGate.setEnabled(False)

                #important that zoomPlotInfo is set only while in zoom mode (None only here)
                #tempIdxEnlargedSpectrum is used to draw back the dashed red rectangle, which pad was enlarged
                tempIdxEnlargedSpectrum = self.getEnlargedSpectrum()[0]
                self.setEnlargedSpectrum(None, None)
                self.currentPlot.isEnlarged = False

                canvasLayout = self.wTab.layout[self.wTab.currentIndex()]
                self.logger.debug('on_dblclick - canvasLayout: %s',canvasLayout)

                ####################### Bashir ###################################################################
                # t1 = time.time()
                #draw back the original canvas
                # self.currentPlot.InitializeCanvas(canvasLayout[0], canvasLayout[1], False)

                # Clear the temporary enlarged plot
                for ax in self.currentPlot.figure.axes:
                    self.currentPlot.figure.delaxes(ax)

                # Restore the saved axes and make them visible again
                for ax in self.currentPlot._saved_axes:
                    self.currentPlot.figure.add_axes(ax)  # This may be redundant but ensures they're in the figure
                    ax.set_visible(True)

                self.currentPlot._saved_axes = None  # Optional: clean up
                # self.currentPlot.canvas.draw()
                ##################################################################################################

                autoscale_status = self.currentPlot.histo_autoscale.isChecked()
                for index, name in self.getGeo().items():

                    # if name is not None and name != "" and name != "empty":
                    if name is not None and name != "" and name != "empty" and index == idx:
                        self.add(index)
                        ax = self.getSpectrumInfo("axis", index=index)
                        
                        #reset the axis limits as it was before enlarge
                        #dont need to specify if log scale, it is checked inside setAxisScale, if 2D histo in log its z axis is set too.
                        dim = self.getSpectrumInfoREST("dim", index=index)
                        if dim == 1:
                            self.plotPlot(index)
                            if not autoscale_status and hasattr(self.currentPlot, "_saved_ylims") and index in self.currentPlot._saved_ylims:
                                ax.set_ylim(*self.currentPlot._saved_ylims[index])   # <-- restore y only
                            else:
                                if autoscale_status:
                                    self.setAxisScale(ax, index, "x", "y")

                        elif dim == 2:
                            self.plotPlot(index, self.old_cmap)
                            self.setSpectrumInfo(cmap=self.old_cmap, index=idx)
                            # if autoscale_status:
                            self.setAxisScale(ax, index, "x", "y", "z")
                        self.drawGate(index)
                #drawing back the dashed red rectangle on the unenlarged spectrum
                self.removeRectangle()
                self.currentPlot.recDashed = self.createDashedRectangle(self.currentPlot.figure.axes[tempIdxEnlargedSpectrum])
                #self.updatePlot() #replaced by the content of updatePlot in the above for loop (avoid looping twice)
                # self.currentPlot.figure.tight_layout()
                # self.drawAllGates()
                self.currentPlot.canvas.draw_idle()

        finally:   
            self.skipAutoUpdateThread.clear()
            self._enlargeBusy = False 

            # t2 = time.time()
            # print("on_dblclick back: time={:.2f}".format(t2-t1))

    #obselete
    # def get_key(self, val):
    #     return next((key for key, value in self.currentPlot.h_dict.items() if any(val == value2 for value2 in value.values())), None)


    #callback for exitButton
    def closeAll(self):
        self.logger.info('closeAll')
        self.close()


     #callback when right click on tab
    """
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
    """

    #### Bashir added to delate/rename when clicking on the tab not the entire window
    def tab_handle_right_click(self):
        self.logger.info('tab_handle_right_click')
        if getattr(self.currentPlot, "zoomPress", False):
            return

        tab_bar = self.wTab.tabBar()

        # Figure out which tab is under the mouse
        global_pos = QCursor.pos()
        local_pos = tab_bar.mapFromGlobal(global_pos)
        index = tab_bar.tabAt(local_pos)
        if index < 0:
            return  # not over any tab → do nothing

        menu = QMenu(self)
        actRename = menu.addAction("Rename")
        actDelete = menu.addAction("Delete")

        chosen = menu.exec_(global_pos)
        if chosen == actRename:
            self.renameTab(index)
        elif chosen == actDelete:
            self.closeTab(index)

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
        # self.setCanvasLayout()
        # print("clickedTab - index: %s",index)
        # End current auto update thread, to avoid thread issue, will start a new thread if/when tab is not empty
        self._stop_auto_thread()

        # For now, if change tab while working on gate, close any ongoing gate action
        if self.currentPlot.toCreateGate or self.currentPlot.toEditGate or self.gatePopup.isVisible():
            self.cancelGate()
            return
        if self.currentPlot.toCreateSumRegion or self.sumRegionPopup.isVisible():
            self.cancelSumRegion()
            return

        # Abord zoom action if click a tab
        if self.currentPlot.zoomPress:
            self.currentPlot.zoom_action.triggered.emit()
            self.currentPlot.zoom_action.setChecked(False)
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
        
    """
    ##### Old click Tab ######
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
            self.currentPlot.zoom_action.triggered.emit()
            self.currentPlot.zoom_action.setChecked(False)
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
    """


    # Helper to set histo_geo widget and enable/disable buttons, when interact with tabs
    def tabGeoWidgetAndFlags(self, index):
        self.currentPlot = self.wTab.wPlot[index]
        nRow = self.wTab.layout[index][0]
        nCol = self.wTab.layout[index][1]
        self.logger.debug('tabGeoWidgetAndFlags - canvas layout: %s, %s',nRow, nCol)

        #nRow-1 because nRow (nCol) is the number of row (col) and the following sets an index starting at 0
        #### Bashir changed to examine apply function to set the row and col
        self.wConf.histo_geo_row.setCurrentIndex(nRow-1)
        self.wConf.histo_geo_col.setCurrentIndex(nCol-1)
        # self.wConf.histo_geo_row.setValue(nRow)
        # self.wConf.histo_geo_col.setValue(nCol)
        ######################################################################
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
        ##### Bashir changed to examine apply function to set the row and col
        nRow = int(self.wConf.histo_geo_row.currentText())
        nCol = int(self.wConf.histo_geo_col.currentText())
        # print("setCanvasLayout - nRow: %s, nCol: %s", nRow, nCol)
        # nRow = self.wConf.histo_geo_row.value()
        # nCol = self.wConf.histo_geo_col.value()
        ######################################################################
        self.wTab.layout[indexTab] = [nRow, nCol]
        self.wTab.wPlot[indexTab].InitializeCanvas(nRow, nCol)
        self.wTab.selected_plot_index_bak[indexTab] = None
        self.currentPlot.selected_plot_index = None
        self.currentPlot.next_plot_index = -1

        """
        if self.wConf.darkModeButton.isChecked():
            self.toggleDarkMode()
        """
        # ✅ mark geometry applied
        self.geometry_applied = True

    
    ####### Bashir added to exit gui once exited from SpecTcl
    def tie_lifetime_to_parent():
        try:
            libc = ctypes.CDLL("libc.so.6")
            PR_SET_PDEATHSIG = 1
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGHUP, 0, 0, 0)
            signal.signal(signal.SIGHUP, lambda *_: os._exit(0))
            if os.getppid() == 1:
                sys.exit(0)
        except Exception:
            pass

    # ---- ensure GUI dies when SpecTcl dies ----
    tie_lifetime_to_parent()
    ########################################################################
    
###############################################
# 5) Connection to REST for gates
###############################################


    ###############################################
    # 5) Connection to REST for gates
    ##
    #############################################


    #Set spectrum info from ReST in self.spectra (identified by histo name and can update multiple info at once)
    #self.spectra is used to keep track of the treegui definition (fixed)
    def setSpectrumInfoREST(self, name, **info):
        self.logger.info('setSpectrumInfoREST - name, info: %s, %s',info, name)
        self.spectra.set(name, **info)


    #Get spectrum info from self.spectra (identified by histo name or index and info name)
    #template of expected arguments e.g.: ("dim", index=5) takes only the first info parameter (here "dim") (one per call)
    #Important that it gets only the info from self.spectra here.
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
        if name is not None:
            return self.spectra.get(name, info[0])


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
            #print("Giordano - in getSpectrumInfo - ",self.wTab.currentIndex(), index, info[0])
            return self.wTab.spectrum_dict[self.wTab.currentIndex()][index][info[0]]


    #Remove spectrum from self.wTab.spectrum_dict:
    #important: only functions that delete item in spectrum_dict and self.spectra
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
        self.spectra.remove(name)
    
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


    #get full spectrum dict from self.spectra:
    def getSpectrumInfoRESTDict(self):
        return self.spectra.as_dict()
 

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
        #Initialize with the same info as in self.spectra.
        for key, value in self.spectra.as_dict()[name].items():
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
            print(index, name)

    def getEnlargedSpectrum(self):
        # self.logger.info('getEnlargedSpectrum')
        result = None
        if self.wTab.zoomPlotInfo[self.wTab.currentIndex()] :
            result = self.wTab.zoomPlotInfo[self.wTab.currentIndex()]
        return result


    #get histo name, type and parameters from REST
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
            ##### Bashir changed to examine the apply button
            tmp = {"row": int(self.wConf.histo_geo_row.currentText()), "col": int(self.wConf.histo_geo_col.currentText()), "geo": properties}
            # tmp = {
            #     "row": self.wConf.histo_geo_row.value(),
            #     "col": self.wConf.histo_geo_col.value(),
            #     "geo": properties
            # }
            #######################################################################

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
            
            ### Bashir added -1
            row = infoGeo["row"] - 1
            col = infoGeo["col"] - 1
            # change index in combobox to the actual loaded values
            #### Bashir changed to examine the apply button
            index_row = self.wConf.histo_geo_row.findText(str(row), QtCore.Qt.MatchFixedString)
            index_col = self.wConf.histo_geo_col.findText(str(col), QtCore.Qt.MatchFixedString)
            # self.wConf.histo_geo_row.setValue(row)
            # self.wConf.histo_geo_col.setValue(col)
            index_row = row
            index_col = col
            #####################################################

# Later usage of index_row / index_col continues to work

            notFound = []
            if index_row >= 0 and index_col >= 0:
                #### Bashir changed to examine the apply button
                self.wConf.histo_geo_row.setCurrentIndex(index_row)
                self.wConf.histo_geo_col.setCurrentIndex(index_col)
                # self.wConf.histo_geo_row.setValue(index_row)
                # self.wConf.histo_geo_col.setValue(index_col)
                #####################################################
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
            if "x" in scale and xmin is not None and xmax is not None:
                ax.set_xlim(xmin,xmax) 
            if "y" in scale or "log" in scale:
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

            if "x" in scale and xmin is not None and xmax is not None:
                ax.set_xlim(xmin,xmax)
            if "y" in scale and ymin is not None and ymax is not None:
                ax.set_ylim(ymin,ymax)
            if "z" in scale or "log" in scale:
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
        #### Bashir added to disable auto scaling when zooming in/out
        self.currentPlot.histo_autoscale.setChecked(False)
        #############################################################

        self.logger.info('zoomInOut - arg: %s', arg)
        # Simon - added following lines to avoid None plot index
        index = self.autoIndex()
        ax = None
        spectrum = self.getSpectrumInfo("spectrum", index=index)
        if spectrum is None : return
        ax = spectrum.axes
        dim = self.getSpectrumInfoREST("dim", index=index)
        if dim == 1 :
            """
            #### Bashir added to zoom in x around mouse location ###
            xmin, xmax = ax.get_xlim()
            center_x = getattr(self, 'mouse_x', (xmin + xmax) / 2)
            xscale = 0.5
            scale = xscale if arg == "in" else 1 / xscale
            half_range_x = (xmax - xmin) * scale / 2

            # xmin_new = max(0, center_x - half_range_x)
            xmin_new = 0.0
            xmax_new = center_x + half_range_x

            ax.set_xlim(xmin_new, xmax_new)
            self.setSpectrumInfo(minx=xmin_new, index=index)
            self.setSpectrumInfo(maxx=xmax_new, index=index)
            #########################################################
            """
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
            """
            #### Bashir added to zoom in x around mouse location ###
            xmin, xmax = ax.get_xlim()
            center_x = getattr(self, 'mouse_x', (xmin + xmax) / 2)
            xscale = 0.3
            scale = xscale if arg == "in" else 1 / xscale
            half_range_x = (xmax - xmin) * scale / 2

            # xmin_new = max(0, center_x - half_range_x)
            xmin_new = 0.0
            xmax_new = center_x + half_range_x

            ax.set_xlim(xmin_new, xmax_new)
            self.setSpectrumInfo(minx=xmin_new, index=index)
            self.setSpectrumInfo(maxx=xmax_new, index=index)
            #########################################################
            """
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
        
        #### Bashir added to disable autoscale while zooming ####
        self.currentPlot.histo_autoscale.setChecked(False)
        #########################################################

        # Abord zoom action if one is ongoing
        if self.currentPlot.zoomPress:
            self.currentPlot.zoom_action.triggered.emit()
            self.currentPlot.zoom_action.setChecked(False)
            self.currentPlot.customZoomButton.setDown(False)
            self.currentPlot.zoomPress = False
        else:
            self.currentPlot.zoom_action.triggered.emit()
            self.currentPlot.zoom_action.setChecked(True)
            self.currentPlot.customZoomButton.setDown(True)


    #Used by customHome button, reset the axis limits to ReST definitions, for the specified plot at index or for all plots if index not provided
    def customHomeButtonCallback(self, index=None):
        #### Bashir added to enable autoscale at home ####
        # self.currentPlot.histo_autoscale.setChecked(True)
        #########################################################

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
        item1 = menu.addAction("Set Zoom Range Manually") 
        #Empty agrument for customHomeButtonCallback means it will reset all spectra
        # Set fields to current spectrum range, as starting points
        index = self.currentPlot.selected_plot_index
        if index is None :
            return QMessageBox.about(self,"Warning!", "Please add/select a spectrum")
        else:        
            # Set fields to current spectrum range, as starting points
            item1.triggered.connect(self.cutoffButtonCallback)
            plotgui = self.currentPlot
            menuPosX = plotgui.mapToGlobal(QtCore.QPoint(0,0)).x() + plotgui.customZoomButton.geometry().topLeft().x()
            menuPosY = plotgui.mapToGlobal(QtCore.QPoint(0,0)).y() + plotgui.customZoomButton.geometry().topLeft().y()
            menuPos = QtCore.QPoint(menuPosX, menuPosY)
            # Shows menu at button position, need to calibrate with 0,0 position
            menu.exec_(menuPos) 

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


    #button of the cutoff window, sets the cutoff values in the spectrum dict
    def okCutoff(self):
        self.logger.info('okCutoff')
        index = self.currentPlot.selected_plot_index
        if index is None : 
            self.logger.debug('okCutoff - index is None')
            return
        spectrum = self.getSpectrumInfo("spectrum", index=index)
        if spectrum is None : return
        ax = self.getSpectrumInfo("axis", index=index)
        if ax is None :
            self.logger.debug('okCutoff - ax is None')
            return

        dim = self.getSpectrumInfoREST("dim", index=index)        
        rangeXmin = self.cutoffp.lineeditXMin.text()
        rangeXmax = self.cutoffp.lineeditXMax.text()
        rangeYmin = self.cutoffp.lineeditYMin.text()
        rangeYmax = self.cutoffp.lineeditYMax.text()                

        cutoffVal = [None, None]
        cutoffMin = self.cutoffp.lineeditZMin.text()
        cutoffMax = self.cutoffp.lineeditZMax.text()

        # Convert and check expected format
        try:
            rangeXmin = float(rangeXmin)
            rangeXmax = float(rangeXmax)
            rangeYmin = float(rangeYmin)
            rangeYmax = float(rangeYmax)
        except ValueError:
            self.logger.warning("okCutoff Range - Invalid input format for zoom range(s). Please enter valid numbers.")
            return

        #Order X min/max may be inverted
        if rangeXmin is not None and rangeXmax is not None and rangeXmax < rangeXmin:
            buff = rangeXmin
            rangeXmin = rangeXmax
            rangeXmax = buff
            self.logger.warning('okCutoff Range - new range X values swapped because min > max')
        #Order Y min/max may be inverted
        if rangeYmin is not None and rangeYmax is not None and rangeYmax < rangeYmin:
            buff = rangeYmin
            rangeYmin = rangeYmax
            rangeYmax = buff
            self.logger.warning('okCutoff Range - new range Y values swapped because min > max')
        #check expected format and save cutoff in spectrum dict 
        if dim == 2:
            if self.cutoffp.lineeditZMin.text() != "" and self.cutoffp.lineeditZMin.text().isdigit():
                cutoffVal[0] = float(cutoffMin)
                self.setSpectrumInfo(cutoff=cutoffVal, index=index)
            if self.cutoffp.lineeditZMax.text() != "" and self.cutoffp.lineeditZMax.text().isdigit():
                cutoffVal[1] = float(cutoffMax)
                self.setSpectrumInfo(cutoff=cutoffVal, index=index)              
            #Order may be inverted
            if cutoffVal[0] is not None and cutoffVal[1] is not None and cutoffVal[1] < cutoffVal[0]:
                cutoffVal = [cutoffVal[1], cutoffVal[0]]
                self.setSpectrumInfo(cutoff=cutoffVal, index=index)
        try:
            #Set new axis limits and save new range in spectrum dict
            spectrum = self.getSpectrumInfo("spectrum", index=index)
            ax.set_xlim(float(rangeXmin), float(rangeXmax))
            ax.set_ylim(float(rangeYmin), float(rangeYmax))
            self.setSpectrumInfo(minx=rangeXmin, index=index)
            self.setSpectrumInfo(maxx=rangeXmax, index=index)
            self.setSpectrumInfo(miny=rangeYmin, index=index)
            self.setSpectrumInfo(maxy=rangeYmax, index=index)                
            if dim == 2 :
                spectrum.set_clim(cutoffVal[0], cutoffVal[1])
                self.setSpectrumInfo(minz=cutoffVal[0], index=index)
                self.setSpectrumInfo(maxz=cutoffVal[1], index=index)
                self.setSpectrumInfo(spectrum=spectrum, index=index)
            self.setSpectrumInfo(spectrum=spectrum, index=index)
            self.drawGate(index)
            self.currentPlot.canvas.draw()
            #self.updatePlot()
        except NameError as err:
            self.logger.debug('okZoomSetRange - NameError', exc_info=True)
            pass

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

        #### Bashir added to disable autoscale while zooming ###
        self.currentPlot.histo_autoscale.setChecked(False)
        ########################################################

        index = self.currentPlot.selected_plot_index
        if index is None :
            return QMessageBox.about(self,"Warning!", "Please Add/Select a Spectrum")
        name = self.nameFromIndex(index)
        if name is not None : 
            self.cutoffp.setWindowTitle("Set zoom range for: " + name)
        else :
            self.cutoffp.setWindowTitle("Set zoom range for: ???" )
        self.cutoffp.setGeometry(300,100,300,100)
        if self.cutoffp.isVisible():
            self.cutoffp.close()
        if self.getSpectrumInfo("cutoff", index=index) is not None and len(self.getSpectrumInfo("cutoff", index=index)) > 0:
            ax = self.getSpectrumInfo("axis", index=index)
            dim = self.getSpectrumInfoREST("dim", index=index) 
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            self.cutoffp.lineeditXMin.setText(f"{xmin:.1f}")
            self.cutoffp.lineeditXMax.setText(f"{xmax:.1f}")
            self.cutoffp.lineeditYMin.setText(f"{ymin:.1f}")
            self.cutoffp.lineeditYMax.setText(f"{ymax:.1f}")
            if dim == 2:
                spectrum = self.getSpectrumInfo("spectrum", index=index)
                zmin, zmax = spectrum.get_clim()
                self.cutoffp.lineeditZMin.setText(f"{zmin:.1f}")
                self.cutoffp.lineeditZMax.setText(f"{zmax:.1f}")

            if dim == 1 :
                self.cutoffp.layout1d()
            elif dim == 2 :
                self.cutoffp.layout2d()
            self.cutoffp.show()
             
        else :
            QMessageBox.about(self,"Warning!", "Please Add/Select a Spectrum")            
            self.logger.warning('cutoffButtonCallback - you broke something really bad - spectrum dict: %s', self.getSpectrumInfo("cutoff", index=index))

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
            minx = self.getSpectrumInfo("minx", index=index)
            maxx = self.getSpectrumInfo("maxx", index=index)
            binx = self.getSpectrumInfo("binx", index=index)

            biny = self.getSpectrumInfo("biny", index=index)            
            
            w = self.getSpectrumInfoREST("data", index=index)

            if self.getSpectrumInfo("cutoff", index=index) is not None:
                if len(self.getSpectrumInfo("cutoff", index=index))>0:
                    #if min/maxCutoff mask data bellow/above the cutoff values
                    minCutoff = self.getSpectrumInfo("cutoff", index=index)[0]
                    maxCutoff = self.getSpectrumInfo("cutoff", index=index)[1]
                    # print("BS-setupPlot, maxCutoff, index: ", maxCutoff, index)
                    if minCutoff is not None:
                        if dim == 1:
                            w = np.ma.masked_where(w < minCutoff, w)
                            # print("BS-setupPlot, minCutoff, index: ", minCutoff, index)
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
                # axis.set_ylim(self.minY,self.maxY)
                # create histogram
                line, = axis.plot([], [], drawstyle='steps')
                name = self.getSpectrumInfo("name", index=index)
                axis.set_title("{}".format(name))

                self.setSpectrumInfo(spectrum=line, index=index)
                if len(w) > 0:
                    X = np.array(self.createRange(binx, minx, maxx))
                    line.set_data(X, w)
                    self.setSpectrumInfo(spectrum=line, index=index)
                
            else:
                #### Bashir added to get the original min/max values from ReST ####
                minxREST = self.getSpectrumInfoREST("minx", index=index)
                maxxREST = self.getSpectrumInfoREST("maxx", index=index)

                minyREST = self.getSpectrumInfoREST("miny", index=index)
                maxyREST = self.getSpectrumInfoREST("maxy", index=index)
                ####################################################################

                # empty data for initialization
                if w is None:
                    w = np.zeros((int(binx), int(biny))) 

                self.palette = copy(plt.cm.plasma)
                w = np.ma.masked_where(w == 0, w)
                self.palette.set_bad(color='white')

                #check if enlarged mode, dont want to modify spectrum dict in enlarged mode
                spectrum = axis.imshow(w,
                                        interpolation='none',
                                        extent=[float(minxREST), float(maxxREST), float(minyREST), float(maxyREST)],
                                        # extent=[float(minx), float(maxx), float(miny), float(maxy)],
                                        aspect='auto',
                                        origin='lower',
                                        vmin=float(self.minZ), vmax=float(self.maxZ),
                                        cmap=self.palette)
                self.setSpectrumInfo(spectrum=spectrum, index=index)
                
                name = self.getSpectrumInfo("name", index=index)
                # axis.set_title("{}, [binx, biny] = [{}, {}] points".format(name, self.len2binratiox, self.len2binratioy))
                axis.set_title("{}".format(name))

                
                if w is not None :
                    # spectrum = self.getSpectrumInfo("spectrum", index=index)
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
        if not self.geometry_applied:
            print("addPlot: Apply Geometry first!, return")
            return

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
                    # print("addPlot: key={}, value={}".format(key, value))
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
                # t1 = time.time()
                self.add(index)

                #it turns out the autoscale is wanted in addPlot
                ### Bashir removed hard autoscale while adding plot 
                self.currentPlot.histo_autoscale.setChecked(True)
                self.autoScaleAxisBox(index)

                ##### AutoScale Bashir ###################################
                #search in the current view
                dim = self.getSpectrumInfoREST("dim", index=index)
                ax = self.getSpectrumInfo("axis", index=index)
                xmin, xmax = ax.get_xlim()
                #getMinMaxInRange returns only max for 1d
                
                if dim == 1:
                    ymin = self.getSpectrumInfo("miny", index=index)
                    ymax = self.getMinMaxInRange(index, xmin=xmin, xmax=xmax)
                    ax.set_ylim(ymin,ymax)
                else:
                    pass
                    ymin, ymax = ax.get_xlim()
                    zmin, zmax = self.getMinMaxInRange(index, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
                    # print("addPlot: zmax={}".format(zmax))
                    self.setSpectrumInfo(maxz=zmax, index=index)
                    self.setSpectrumInfo(minz=zmin, index=index)
                    spectrum = self.getSpectrumInfo("spectrum", index=index)
                    spectrum.set_clim(vmin=zmin, vmax=zmax)
                    self.setSpectrumInfo(spectrum=spectrum, index=index)
                ##############################################################
                # t2 = time.time()
                # print("addPlot: time={:.2f}".format(t2-t1))
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
        
                self.currentPlot.canvas.draw_idle()
                self.currentPlot.isSelected = False
        except NameError:
            raise

        # (Re)start auto update once there is a spectrum
        if self.stopAutoUpdateThread.is_set():
            self.autoUpdateStart()

        # When create a new tab, signals are enabled once a spectrum is added
        if not self.wTab.countClickTab[self.wTab.currentIndex()]:
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
    def plotPlot(self, index, cmap=None):
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
            if cmap is not None:
                spectrum.set_cmap(cmap)
            elif self.old_cmap is not None:
                spectrum.set_cmap(self.old_cmap)
            elif hasattr(self, "palette") and self.palette is not None:
                spectrum.set_cmap(self.palette)
            else:
                spectrum.set_cmap(plt.get_cmap("viridis"))  # final fallback


        self.setSpectrumInfo(spectrum=spectrum, index=index)
        self.currentPlot = currentPlot


    ### Bashir added for auto-update signal
    @pyqtSlot()
    def _updatePlotOnGui(self):
        self.updatePlot()
    #Callback for histo_geo_update button
    #also used in various functions
    #redraw plot spectrum, update axis scales, redraw gates
    def updatePlot(self):
        #### Bashir added to automatically enable autoscale when updating the plot
        auto_scale_status = self.currentPlot.histo_autoscale.isChecked()
        self.currentPlot.histo_autoscale.setChecked(auto_scale_status)
        ########################################
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
                if auto_scale_status:
                    if dim == 1:
                        self.setAxisScale(ax, 0, "x", "y")
                if dim == 2:
                    self.setAxisScale(ax, 0, "x", "y", "z")
    
                #######################################################
                    spectrum = self.getSpectrumInfo("spectrum", index=index)

                    if spectrum is not None and dim == 2:
                        # spectrum.set_cmap(self.old_cmap)
                        if ax:
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes("right", size="5%", pad=0.05)
                            self.currentPlot.figure.colorbar(spectrum, cax=cax, orientation="vertical")

                            self.currentPlot.figure.tight_layout(rect=[0, 0, 0.95, 1])
                            self.currentPlot.canvas.draw_idle()
                ########################################################
                    
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
                    # print("updatePlot: old_cmap1: ", self.old_cmap)
                    self.plotPlot(index)
                    # print("updatePlot: old_cmap2: ", self.old_cmap)
                    #reset the axis limits as it was before enlarge
                    #dont need to specify if log scale, it is checked inside setAxisScale, if 2D histo in log its z axis is set too.
                    dim = self.getSpectrumInfoREST("dim", index=index)
                    #### Bashir commented out to avoid autoscale while updating
                    if auto_scale_status:
                        if dim == 1:
                            self.setAxisScale(ax, index, "x", "y")
                        elif dim == 2:
                            self.setAxisScale(ax, index, "x", "y", "z")
                        else:
                            pass
                    
                    #draw gate if there is one
                    self.drawGate(index)

            self.currentPlot.figure.tight_layout()
            self.currentPlot.canvas.draw()
        except NameError:
            self.logger.debug('updatePlot - NameError exception', exc_info=True)
            pass
            #raise


    ####### Bashir added for color map #######################
    def onColormapChange(self, cmap_name: str):
        """Callback when the colormap selector changes."""
        self.logger.info("onColormapChange - cmap: %s", cmap_name)

        if not self.getGeo():
            self.logger.debug("onColormapChange - no active plots")
            return

        try:
            if cmap_name.lower() == "custom":
                # Ask user for colormap definition file
                QMessageBox.information(
                    self,
                    "Custom Colormap Format from a .txt File",
                    "Each line should be:\n<low> <high> <red> <green> <blue>\n"
                    "Example:\n0.0 0.5 0 1 0\n0.5 0.7 1 0 0\n0.7 1.0 0 0 1\n\n"
                    "<low> and <high> represent count percentiles.\n"
                    "The <low> value of the first line must be 0.0, and the <low> of each line must match the <high> of the previous line.\n\n"
                    "<red>, <green>, and <blue> are RGB values between 0 and 1."
                )

                filename, _ = QFileDialog.getOpenFileName(
                    self, "Open Custom Colormap", "", "Text Files (*.txt)"
                )
                if not filename:
                    self.logger.debug("No file selected for custom cmap")
                    return

                bounds = []
                color_list = []

                with open(filename) as f:
                    for line in f:
                        parts = line.split()
                        if not parts or len(parts) < 5:
                            continue
                        lo, hi = float(parts[0]), float(parts[1])
                        r, g, b = map(float, parts[2:])
                        # take lower bound and its color
                        bounds.append(lo)
                        color_list.append((r, g, b))
                    # also add the very last high bound with last color
                    bounds.append(hi)
                    color_list.append((r, g, b))

                # Build custom colormap
                self.palette = colors.LinearSegmentedColormap.from_list(
                    "custom_cmap", list(zip(bounds, color_list)), N=256
                )
                self.palette.set_bad(color="white")
            else:
                # Normal built-in colormap
                self.palette = copy(plt.get_cmap(cmap_name))
                self.palette.set_bad(color="white")

            # Apply to all 2D plots
            for index, _ in self.getGeo().items():
                if self.getSpectrumInfoREST("dim", index=index) != 2:
                    continue

                spectrum = self.getSpectrumInfo("spectrum", index=index)
                if spectrum is None:
                    continue

                spectrum.set_cmap(self.palette)
                self.setSpectrumInfo(spectrum=spectrum, index=index)

                # persist cmap if we’re in enlarged mode
                if self.currentPlot.isEnlarged:
                    self.old_cmap = spectrum.get_cmap()

                ax = self.getSpectrumInfo("axis", index=index)
                if ax is not None:
                    try:
                        self.removeCb(ax)
                    except Exception:
                        pass
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    self.currentPlot.figure.colorbar(spectrum, cax=cax, orientation="vertical")

            self.currentPlot.canvas.draw_idle()

        except Exception as e:
            self.logger.error("onColormapChange - exception: %s", str(e), exc_info=True)


    ####### Bashir added for dark mode ##########################

    """
    def toggleDarkMode(self):
        if self.wConf.darkModeButton.isChecked():
            # Dark mode
            self.wConf.darkModeButton.setText("Light Mode")
            dark_bg = "#1e1e1e"    # pleasant dark, not pure black like VS Code

            for plot in self.wTab.wPlot.values():
                plot.figure.set_facecolor(dark_bg)
                for ax in plot.figure.axes:
                    ax.set_facecolor(dark_bg)
                    ax.tick_params(colors="white")
                    ax.xaxis.label.set_color("white")
                    ax.yaxis.label.set_color("white")
                    ax.title.set_color("white")
                plot.canvas.draw_idle()

        else:
            # Light mode
            self.wConf.darkModeButton.setText("Dark Mode")
            light_bg = "white"

            for plot in self.wTab.wPlot.values():
                plot.figure.set_facecolor(light_bg)
                for ax in plot.figure.axes:
                    ax.set_facecolor(light_bg)
                    ax.tick_params(colors="black")
                    ax.xaxis.label.set_color("black")
                    ax.yaxis.label.set_color("black")
                    ax.title.set_color("black")
                plot.canvas.draw_idle()
    """
    ############################################################################
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
        #### Bashir changed to examine the apply button##
        row = int(self.wConf.histo_geo_row.currentText())
        col = int(self.wConf.histo_geo_col.currentText())
        # row = self.wConf.histo_geo_row.value()
        # col = self.wConf.histo_geo_col.value()
        #################################################

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



    ############################
    # 12)  Fitting
    ############################

    #callback to open extra function popup
    def spfunPopup(self):
        self.logger.info('spfunPopup callBack')
        if self.extraPopup.isVisible():
            self.extraPopup.close()

        self.extraPopup.show()

    # ------------------------------------------------------------------
    # Fit methods — delegated to FitManager (see gui/services/fit_manager.py)
    # ------------------------------------------------------------------
    def fit(self):                       return self.fit_manager.fit()
    def deleteFit(self):                 return self.fit_manager.deleteFit()
    def listFitLineLabels(self, ax):     return self.fit_manager.listFitLineLabels(ax)
    def printFitLineLabels(self):        return self.fit_manager.printFitLineLabels()
    def setFitLineLabel(self, ax, line, resultsText, spectrumName):
        return self.fit_manager.setFitLineLabel(ax, line, resultsText, spectrumName)
    def setFitResultsLineLabel(self, fitLineLabelIdx, resultsText, spectrumName):
        return self.fit_manager.setFitResultsLineLabel(fitLineLabelIdx, resultsText, spectrumName)
    def axisLimitsForFit(self, ax):      return self.fit_manager.axisLimitsForFit(ax)

    # ------------------------------------------------------------------
    # Gate methods — delegated to GateManager (see gui/services/gate_manager.py)
    # ------------------------------------------------------------------
    def drawGate(self, index):                   return self.gate_manager.drawGate(index)
    def cancelGate(self, doClose=True):          return self.gate_manager.cancelGate(doClose)
    def disconnectGateSignals(self):             return self.gate_manager.disconnectGateSignals()
    def addLine(self, *a, **kw):                 return self.gate_manager.addLine(*a, **kw)
    def removePrevLine(self):                    return self.gate_manager.removePrevLine()
    def clickOnGateLine(self, event):            return self.gate_manager.clickOnGateLine(event)
    def dist(self, x, y):                        return self.gate_manager.dist(x, y)
    def find_callbacks(self, *a):                return self.gate_manager.find_callbacks(*a)

    # ------------------------------------------------------------------
    # SumRegion methods — delegated to SumRegionManager (see gui/services/sum_region_manager.py)
    # ------------------------------------------------------------------
    def setSumRegion(self, index, line):         return self.sum_region_manager.setSumRegion(index, line)
    def getSumRegion(self, index):               return self.sum_region_manager.getSumRegion(index)
    def deleteSumRegionDict(self, label):        return self.sum_region_manager.deleteSumRegionDict(label)
    def refreshSpectrumSumRegionDict(self):      return self.sum_region_manager.refreshSpectrumSumRegionDict()
    def saveSumRegion(self, index):              return self.sum_region_manager.saveSumRegion(index)
    def createSumRegion(self):                   return self.sum_region_manager.createSumRegion()
    def okSumRegion(self):                       return self.sum_region_manager.okSumRegion()
    def cancelSumRegion(self, doClose=True):     return self.sum_region_manager.cancelSumRegion(doClose)
    def cleanPopupExit(self, doClose=True):      return self.sum_region_manager.cleanPopupExit(doClose)
    def deleteSumRegion(self):                   return self.sum_region_manager.deleteSumRegion()
    def integrate(self):                         return self.sum_region_manager.integrate()
    def okIntegrate(self):                       return self.sum_region_manager.okIntegrate()
    def copySelectionIntegrateTable(self):       return self.sum_region_manager.copySelectionIntegrateTable()
    def formatResultsIntegrate(self, results):   return self.sum_region_manager.formatResultsIntegrate(results)
    def setPrecisionIntegrationResult(self, d):  return self.sum_region_manager.setPrecisionIntegrationResult(d)
    def integrateGateLocal(self, idx, lines):    return self.sum_region_manager.integrateGateLocal(idx, lines)

    # -- ConnectionManager shims --
    def connectShMem(self):              return self.connection_manager.connectShMem()
    def connectPopup(self):              return self.connection_manager.connectPopup()
    def okConnect(self):                 return self.connection_manager.okConnect()
    def closeConnect(self):              return self.connection_manager.closeConnect()
    def autoUpdateStart(self):           return self.connection_manager.autoUpdateStart()
    def autoUpdateResume(self):          return self.connection_manager.autoUpdateResume()
    def updateSpectrumList(self, init=False): return self.connection_manager.updateSpectrumList(init)
    def updateFromTraces(self, tracesDetails): return self.connection_manager.updateFromTraces(tracesDetails)
    def _stop_auto_thread(self):         return self.connection_manager._stop_auto_thread()
    def _stop_rest_thread(self):         return self.connection_manager._stop_rest_thread()



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
        #### Bashir changed to examine the apply button
        row = int(self.wConf.histo_geo_row.currentText())
        col = int(self.wConf.histo_geo_col.currentText())
        # row = self.wConf.histo_geo_row.value()
        # col = self.wConf.histo_geo_col.value()
        ######################################################

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
        self.connection_manager._stop_auto_thread()
        self.connection_manager._stop_rest_thread()
        event.accept()

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

class centeredNorm(colors.Normalize):
    def __init__(self, data, vcenter=0, halfrange=None, clip=False):
        if halfrange is None:
            halfrange = np.max(np.abs(data - vcenter))
        super().__init__(vmin=vcenter - halfrange, vmax=vcenter + halfrange, clip=clip)


# Bashir --- ADD: simple dialog to collect fit points and plot ---
class FitParamsDialog(QDialog):
    """
    Holds rows of {mu, A, sigma, tau, model, name}, with:
      - Name: editable text for current point
      - Append / Load CSV / Plot
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Collected Fit Parameters")
        self.resize(800, 520)

        self.data = []        # list of dicts
        self.pending = None   # dict to be appended when user clicks "Append"

        layout = QVBoxLayout(self)

        self.hint = QLabel("Last fit: (nothing yet)")
        layout.addWidget(self.hint)

        # --- Name (label) editor for the pending point ---
        row = QHBoxLayout()
        row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("enter a label for this point (e.g., peak window / ROI)")
        row.addWidget(self.name_edit)
        layout.addLayout(row)

        # --- Table ---
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["mu", "A", "sigma", "tau1", "tau2", "model", "name"])
        self.table.setSortingEnabled(True)
        layout.addWidget(self.table)

        # --- Buttons ---
        btns = QHBoxLayout()
        self.btn_append = QPushButton("Append current")
        self.btn_load   = QPushButton("Load CSV…")
        self.btn_save   = QPushButton("Save CSV…")       
        self.btn_plot   = QPushButton("Plot")
        btns.addWidget(self.btn_append)
        btns.addWidget(self.btn_load)
        btns.addWidget(self.btn_save)                    
        btns.addWidget(self.btn_plot)

        self.btn_append.clicked.connect(self.on_append)
        self.btn_load.clicked.connect(self.on_load)
        self.btn_save.clicked.connect(self.on_save)      
        self.btn_plot.clicked.connect(self.on_plot)
        layout.addLayout(btns)


    def set_pending(self, point_dict, suggested_name=None):
        """Supply the most recent fit’s point (not automatically stored)."""
        self.pending = point_dict or None
        if self.pending:
            self.hint.setText(
                f"Last fit → mu={self.pending.get('mu'):.6g}, "
                f"A={self.pending.get('A'):.6g}, "
                f"sigma={self.pending.get('sigma'):.6g}, "
                f"tau={self.pending.get('tau'):.6g} "
                f"({self.pending.get('model')} · {self.pending.get('spectrum','')})"
            )
            # prefer caller’s suggestion; fall back to pending.name/spectrum
            default_name = (suggested_name
                            or self.pending.get('name')
                            or self.pending.get('spectrum', ''))
            self.name_edit.setText(default_name)
        else:
            self.hint.setText("Last fit: (nothing yet)")
            self.name_edit.clear()

    def _append_row(self, p):
        """
        Append one row to the table from a point-dict `p`.
        Expected keys: 'mu','A','sigma',('tau1' or 'tau'),('tau2' or 'tau'),'model','name'
        Falls back to single-tail 'tau' if tau1/tau2 are missing.
        """
        import math

        # Ensure the table has the 7 expected columns (mu, A, sigma, tau1, tau2, model, name)
        if self.table.columnCount() < 7:
            self.table.setColumnCount(7)
            self.table.setHorizontalHeaderLabels(
                ["mu", "A", "sigma", "tau1", "tau2", "model", "name"]
            )

        was_sorting = self.table.isSortingEnabled()
        if was_sorting:
            self.table.setSortingEnabled(False)

        row = self.table.rowCount()
        self.table.insertRow(row)

        # ---- helpers ----
        def _coerce_float(v):
            try:
                f = float(v)
                return f if math.isfinite(f) else None
            except Exception:
                return None

        def _fmt_num(v):
            f = _coerce_float(v)
            return f"{f:.6g}" if f is not None else ""

        def _add_cell(col, text, editable=False, align_right=True):
            it = QTableWidgetItem(text)
            # Right-align numeric columns for readability
            if align_right:
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if not editable:
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, col, it)
            return it

        # ---- normalize taus (support single-tail dicts that only carry 'tau') ----
        t1 = p.get("tau1", p.get("tau", None))
        t2 = p.get("tau2", p.get("tau", None))

        # ---- fill cells ----
        _add_cell(0, _fmt_num(p.get("mu", None)))
        _add_cell(1, _fmt_num(p.get("A", None)))
        _add_cell(2, _fmt_num(p.get("sigma", None)))
        _add_cell(3, _fmt_num(t1))
        _add_cell(4, _fmt_num(t2))
        _add_cell(5, str(p.get("model", "")), align_right=False)

        name_text = p.get("name", "")
        name_item = _add_cell(6, name_text, editable=True, align_right=False)

        # Optionally stash raw numeric values for future use (e.g., export)
        # in case the display text was formatted:
        # (Qt's sort uses display text; for true numeric sort, you'd subclass QTableWidgetItem.)
        self.table.item(row, 0).setData(Qt.UserRole, _coerce_float(p.get("mu", None)))
        self.table.item(row, 1).setData(Qt.UserRole, _coerce_float(p.get("A", None)))
        self.table.item(row, 2).setData(Qt.UserRole, _coerce_float(p.get("sigma", None)))
        self.table.item(row, 3).setData(Qt.UserRole, _coerce_float(t1))
        self.table.item(row, 4).setData(Qt.UserRole, _coerce_float(t2))

        if was_sorting:
            self.table.setSortingEnabled(True)

    def on_append(self):
        if not self.pending:
            QMessageBox.information(self, "Append", "No current fit to append.")
            return
        p = self.pending.copy()
        label = self.name_edit.text().strip()
        if not label:
            # fallback if user left it blank
            label = p.get('name') or p.get('spectrum', '') or "untitled"
        p['name'] = label
        self.data.append(p)
        self._append_row(p)
        self.pending = None
        self.hint.setText("Last fit: (appended)")
        self.name_edit.clear()

    def on_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load CSV", "", "CSV files (*.csv)")
        if not path: return
        try:
            import csv
            was_sorting = self.table.isSortingEnabled()
            if was_sorting:
                self.table.setSortingEnabled(False)

            self.table.setRowCount(0)
            self.data = []

            with open(path, newline='') as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    name = r.get('name', r.get('spectrum', ''))
                    p = {
                        'mu': float(r.get('mu', 'nan')),
                        'A': float(r.get('A', 'nan')),
                        'sigma': float(r.get('sigma', 'nan')),
                        'tau1': float(r.get('tau1', 'nan')),
                        'tau2': float(r.get('tau2', 'nan')),
                        'model': r.get('model', 'AlphaEMG12'),
                        'name': name,
                    }
                    self.data.append(p)
                    self._append_row(p)

            if was_sorting:
                self.table.setSortingEnabled(True)
        except Exception as e:
            QMessageBox.warning(self, "Load CSV", f"Failed to load CSV:\n{e}")


    def on_plot(self):
        if not self.data:
            QMessageBox.information(self, "Plot", "No data to plot. Append or load first.")
            return

        # keep only rows with required fields and finite numbers
        def _finite(p, k):
            try: return np.isfinite(float(p.get(k, np.nan)))
            except Exception: return False
        pts = [p for p in self.data
               if all(k in p for k in ('mu','A','sigma')) and
                  (_finite(p,'mu') and _finite(p,'A') and _finite(p,'sigma'))]
        # tolerate single-tail rows by copying tau → tau1=tau2=tau
        for p in pts:
            if 'tau1' not in p and 'tau' in p: p['tau1'] = p['tau']
            if 'tau2' not in p and 'tau' in p: p['tau2'] = p['tau']
        pts = [p for p in pts if _finite(p,'tau1') and _finite(p,'tau2')]
        pts = sorted(pts, key=lambda d: float(d['mu']))

        mu   = np.array([float(d['mu'])    for d in pts])
        Aval = np.array([float(d['A'])     for d in pts])
        sig  = np.array([float(d['sigma']) for d in pts])
        tau1 = np.array([float(d['tau1'])  for d in pts])
        tau2 = np.array([float(d['tau2'])  for d in pts])

        # Reuse a single window if already open
        if not hasattr(self, "_plot_fig") or self._plot_fig is None:
            self._plot_fig, self._plot_axs = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
            try:
                self._plot_fig.canvas.manager.set_window_title("Fit parameters vs μ (energy)")
            except Exception:
                pass
        fig, axs = self._plot_fig, self._plot_axs
        # Clear and redraw
        for ax in axs: ax.cla()
        axs[0].plot(mu, Aval, marker='o'); axs[0].set_ylabel("A")
        axs[1].plot(mu, sig,  marker='o'); axs[1].set_ylabel("σ")
        axs[2].plot(mu, tau1, marker='o', label="τ₁"); axs[2].set_ylabel("τ")
        axs[2].plot(mu, tau2, marker='o', label="τ₂"); axs[2].legend(loc="best")
        # Optional combined view of τ ratio or Δτ for diagnostics
        axs[3].plot(mu, tau2 - tau1, marker='o'); axs[3].set_ylabel("τ₂ - τ₁")
        axs[3].set_xlabel("μ (energy)")
        fig.tight_layout()
        plt.show(block=False)


    def on_save(self):
        if not self.data:
            QMessageBox.information(self, "Save CSV", "No data to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV files (*.csv)")
        if not path:
            return
        try:
            import csv
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["mu","A","sigma","tau1","tau2","model","name"])
                w.writeheader()
                for d in self.data:
                    w.writerow({
                        "mu": d.get("mu",""),
                        "A": d.get("A",""),
                        "sigma": d.get("sigma",""),
                        "tau1": d.get("tau1",""),
                        "tau2": d.get("tau2",""),
                        "model": d.get("model",""),
                        "name": d.get("name",""),
                    })
            QMessageBox.information(self, "Save CSV", f"Saved to:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Save CSV", f"Could not save:\n{e}")
