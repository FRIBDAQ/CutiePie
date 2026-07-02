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

# Single source of truth for the auto-update combo: index i of the names maps
# to seconds at the same index (ConnectionManager.autoUpdateStart relies on it).
AUTO_UPDATE_INTERVALS      = [1, 5, 10, 30, 60, 180, 300, 600, 9e9]
AUTO_UPDATE_INTERVAL_NAMES = ["1 sec", "5 secs", "10 secs", "30 secs",
                              "1 min", "3 mins", "5 mins", "10 mins", "Inf."]


def tie_lifetime_to_parent():
    """Exit the GUI when the parent (SpecTcl) dies. Linux-only; no-op elsewhere."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGHUP, 0, 0, 0)
        signal.signal(signal.SIGHUP, lambda *_: os._exit(0))
        if os.getppid() == 1:
            sys.exit(0)
    except Exception:
        pass

# 0) Class definition
class MainWindow(QMainWindow):

    # result of a background applylistgate fetch;
    # emitted from a worker thread, delivered on the GUI thread
    _gateNameFetched = pyqtSignal(str, object)

    def __init__(self, factory, fit_factory, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # ensure GUI dies when SpecTcl dies (was an import-time call in the class body)
        tie_lifetime_to_parent()

        # initialize debug logging
        logging.basicConfig(datefmt='%d-%b-%y %H:%M:%S')        
        
        self.logger = logging.getLogger(__name__)
        # WARNING in normal operation so per-tick debug/info calls in the render
        # and hover hot paths don't build LogRecords nobody consumes; flipped to
        # DEBUG by debugModeCallBack while debug mode is on.
        self.logger.setLevel(logging.WARNING)

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
            fit_factory=self.fit_factory,
            spectra=self.spectra,
            extra_popup=self.extraPopup,
            parent_widget=self,
            logger=self.logger,
        )

        self.gate_manager = GateManager(
            spectra=self.spectra,
            name_from_index=self.nameFromIndex,
            get_spectrum_info=self.getSpectrumInfo,
            get_is_enlarged=lambda: self.currentPlot.isEnlarged,
            get_geo=self.getGeo,
            get_sum_region=lambda index, name: self.sum_region_manager.getSumRegion(index, name),
            get_current_canvas=lambda: self.wTab.wPlot[self.wTab.currentIndex()].canvas,
            integrate_popup=self.integratePopup,
            get_integrate_copy=lambda: getattr(self, 'sidTableIntegrateCopy', None),
            gate_hide_cb=self.extraPopup.options.gateHide,
            gate_annotation_cb=self.extraPopup.options.gateAnnotation,
            gate_edit_disable_cb=self.extraPopup.options.gateEditDisable,
            sum_region_popup=self.sumRegionPopup,
            skip_auto=self.skipAutoUpdateThread,
            get_rest=lambda: self.rest,
            gate_popup=self.gatePopup,
            parent_widget=self,
            logger=self.logger,
        )
        self.sum_region_manager = SumRegionManager(
            spectra=self.spectra,
            name_from_index=self.nameFromIndex,
            get_spectrum_info=self.getSpectrumInfo,
            histo_list=self.wConf.histo_list,
            integrate_popup=self.integratePopup,
            skip_auto=self.skipAutoUpdateThread,
            add_line=lambda *a, **kw: self.gate_manager.addLine(*a, mode="sum_region", **kw),
            remove_prev_line=lambda: self.gate_manager.removePrevLine(mode="sum_region"),
            get_rest=lambda: self.rest,
            check_and_cancel_gate=self._check_and_cancel_gate,
            sum_popup=self.sumRegionPopup,
            parent_widget=self,
            logger=self.logger,
        )
        self.connection_manager = ConnectionManager(
            wConf=self.wConf,
            connect_config=self.connectConfig,
            spectra=self.spectra,
            update_intervals=AUTO_UPDATE_INTERVALS,
            update_intervals_user=AUTO_UPDATE_INTERVAL_NAMES,
            stop_rest=self.stopRestThread,
            stop_auto=self.stopAutoUpdateThread,
            skip_auto=self.skipAutoUpdateThread,
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

        self.plot_controller = PlotController(
            wTab=self.wTab,
            wConf=self.wConf,
            spectra=self.spectra,
            get_current_plot=lambda: self.currentPlot,
            get_geo=self.getGeo,
            set_geo=self.setGeo,
            get_spectrum_info=self.getSpectrumInfo,
            set_spectrum_info=self.setSpectrumInfo,
            get_spectrum_info_dict=self.getSpectrumInfoDict,
            name_from_index=self.nameFromIndex,
            get_enlarged_spectrum=self.getEnlargedSpectrum,
            auto_index=self.autoIndex,
            next_index=self.nextIndex,
            bind_dynamic_signal=self.bindDynamicSignal,
            draw_gate=self.drawGate,
            clean_popup_exit=self.cleanPopupExit,
            auto_update_start=self.autoUpdateStart,
            stop_auto_update_thread=self.stopAutoUpdateThread,
            cutoff_popup=self.cutoffp,
            min_y=self.minY,
            max_y=self.maxY,
            min_z=self.minZ,
            max_z=self.maxZ,
            parent_widget=self,
            logger=self.logger,
        )

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

        self._gate_name_cache: dict = {}  # spectrum_name → (gate_or_None, monotonic_ts)
        self._gate_name_inflight: set = set()  # names with a background fetch running
        self._hoveredSpectrumName = None       # spectrum currently under the pointer
        self._gateNameFetched.connect(self._on_gate_name_fetched)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._do_resize)


        #################
        # 2) Signals
        #################

        # top menu signals
        self.wConf.connectButton.clicked.connect(self.connection_manager.connectPopup)
        self.connectConfig.ok.clicked.connect(self.connection_manager.okConnect)
        self.connectConfig.cancel.clicked.connect(self.connection_manager.closeConnect)
        self.connection_manager.connectionEstablished.connect(self.setCanvasLayout)
        self.connection_manager.spectrumRemoved.connect(self._on_spectrum_removed_rest)
        self.connection_manager.spectrumListChanged.connect(self.refreshSpectrumSumRegionDict)
        self.connection_manager.updatePlotRequested.connect(self._updatePlotOnGui)

        ### Bashir added to auto select connect button if ports are default
        rest_text   = self.connectConfig.rest.text().strip()
        mirror_text = self.connectConfig.mirror.text().strip()
        if rest_text.isdigit() and mirror_text.isdigit():
            # visually show "connected"
            self.wConf.connectButton.setChecked(True)

            # actually perform the connection once the event loop is ready
            QTimer.singleShot(0, self.okConnect)

        #### Bashir chenged ###################
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
        
        # populate combo from the shared constant; default index 8 → "Inf."
        self.wConf.autoUpdate2.clear()
        self.wConf.autoUpdate2.addItems(AUTO_UPDATE_INTERVAL_NAMES)
        self.wConf.autoUpdate2.setCurrentIndex(8)


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
        # palette, old_cmap, and geometry_applied live on PlotController

        # self.wConf.darkModeButton.clicked.connect(self.toggleDarkMode)


        ##### Bashir commented out to examine apply button
        # self.wConf.histo_geo_row.activated.connect( self.setCanvasLayout )
        # self.wConf.histo_geo_col.activated.connect( self.setCanvasLayout )
        self.wConf.histo_geo_apply_btn.clicked.connect(self.setCanvasLayout)
        # self.setCanvasLayout()
        # QShortcut(QKeySequence("Return"), self, activated=self.wConf.histo_geo_apply_btn.click)
        # QShortcut(QKeySequence("Enter"), self, activated=self.wConf.histo_geo_apply_btn.click)
        ####################################################################


        self.wConf.createGate.clicked.connect(
            lambda: self.gate_manager.createGate(self.currentPlot.selected_plot_index))
        self.wConf.createGate.setEnabled(False)
        self.gatePopup.ok.clicked.connect(self.gate_manager.okGate)
        self.gatePopup.cancel.clicked.connect(self.gate_manager.cancelGate)
        self.gatePopup.gateActionCreate.clicked.connect(
            lambda: self.gate_manager.createGate(self.currentPlot.selected_plot_index))
        self.gatePopup.gateActionEdit.clicked.connect(self.gate_manager.editGate)
        self.gatePopup.clearInfoSignal.connect(self.gatePopup.clearInfo)
        self.gatePopup.clearInfoSignal.connect(self.autoUpdateResume)
        self.gate_manager.canvasDrawRequested.connect(self._on_gm_canvas_draw)
        self.gate_manager.canvasDrawIdleRequested.connect(self._on_gm_canvas_draw_idle)
        self.gate_manager.updatePlotRequested.connect(self.updatePlot)
        self.gate_manager.gateCreationStarted.connect(self._on_gate_creation_started)
        self.gate_manager.gateEditingStarted.connect(self._on_gate_editing_started)
        self.gate_manager.gateEnded.connect(self._on_gate_ended)

        # summing region
        self.wConf.createSumRegionButton.clicked.connect(
            lambda: self.sum_region_manager.createSumRegion(*self._current_plot_ctx()))
        self.sumRegionPopup.ok.clicked.connect(self.sum_region_manager.okSumRegion)
        self.sumRegionPopup.cancel.clicked.connect(self.sum_region_manager.cancelSumRegion)
        self.sumRegionPopup.delete.clicked.connect(
            lambda: self.sum_region_manager.deleteSumRegion(*self._current_plot_ctx()))
        self.sumRegionPopup.clearInfoSignal.connect(self.sumRegionPopup.clearInfo)
        self.sumRegionPopup.clearInfoSignal.connect(self.autoUpdateResume)
        self.sum_region_manager.canvasDrawRequested.connect(self._on_srm_canvas_draw)
        self.sum_region_manager.figureTightLayoutRequested.connect(self._on_srm_tight_layout)
        self.sum_region_manager.updatePlotRequested.connect(self.updatePlot)
        self.sum_region_manager.sumRegionStarted.connect(self._on_sum_region_started)
        self.sum_region_manager.sumRegionEnded.connect(self._on_sum_region_ended)
        self.sum_region_manager.gateSignalsDisconnectRequested.connect(self.disconnectGateSignals)
        self.sum_region_manager.sidTableConnectionUpdated.connect(self._on_side_table_conn_updated)

        # self.wConf.editGate.setToolTip("Key bindings for Modify->Edit:\n"
        #                               "'i' insert vertex\n"
        #                               "'d' delete vertex\n")

        #integrate gate and summing region
        self.wConf.integrateGateAndRegion.clicked.connect(
            lambda: self.sum_region_manager.integrate(*self._current_plot_ctx()))
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
        self.extraPopup.fit_button.clicked.connect(
            lambda: self.fit_manager.fit(*self._current_plot_ctx()))
        self.extraPopup.plot_csv_button.clicked.connect(self.fit_manager.on_plot_csv_clicked)
        self.extraPopup.fit_csv_button.clicked.connect(self.fit_manager.on_fit_csv_clicked)
        self.extraPopup.abort_button.clicked.connect(self.fit_manager.on_abort_clicked)
        self.extraPopup.all_fitIdx_button.clicked.connect(
            lambda: self.fit_manager.printFitLineLabels(*self._current_plot_ctx()))
        self.extraPopup.delete_button.clicked.connect(
            lambda: self.fit_manager.deleteFit(*self._current_plot_ctx()))

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
                self.wTab.wPlot[index].histo_autoscale.disconnect()
                self.wTab.wPlot[index].customZoomButton.disconnect()
                self.wTab.wPlot[index].plusButton.disconnect()
                self.wTab.wPlot[index].minusButton.disconnect()
                self.wTab.wPlot[index].copyButton.disconnect()
                self.wTab.wPlot[index].customHomeButton.disconnect()
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
    @staticmethod
    def _setLabelText(label, text):
        """setText only when the text changed — every set triggers a Qt relayout,
        and histoHover runs on every mouse-motion event."""
        if label.text() != text:
            label.setText(text)

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
            name  = self.nameFromIndex(index)
            self._hoveredSpectrumName = name  # read by _on_gate_name_fetched
            si    = self.spectra.get_record(name) or {}
            dim   = si.get("dim")
            params    = si.get("parameters") or []
            sp_type   = si.get("type", "")
            xTitle    = params[0] if params else ""
            coordinates = self.getPointerInfo(event, "coordinates", index)
            if dim == 1:
                if sp_type == "g1":
                    xTitle = xTitle + ", ..."
                self._setLabelText(self.currentPlot.histoLabel, "Spectrum: " + name + "\nX: " + xTitle)
                self._setLabelText(self.currentPlot.pointerLabel, f"Pointer:\nX: {coordinates[0]:.2f} Y: {coordinates[1]:.0f} Count: {coordinates[2]:.0f}")
            elif dim == 2:
                yTitle = params[1] if len(params) > 1 else ""
                if sp_type in ("g2", "m2", "gd"):
                    xTitle = xTitle + ", ..."
                    yTitle = yTitle + ", ..."
                self._setLabelText(self.currentPlot.histoLabel, "Spectrum: " + name + "\nX: " + xTitle + " Y: " + yTitle)
                self._setLabelText(self.currentPlot.pointerLabel, f"Pointer:\nX: {coordinates[0]:.2f} Y: {coordinates[1]:.2f}  Count: {coordinates[2]:.0f}")
                if sp_type == "s":
                    xTitle = xTitle + ", ..."
                    self._setLabelText(self.currentPlot.histoLabel, "Spectrum: " + name + "\nX: " + xTitle)
            gateName = self.getAppliedGateName(index=index)
            if gateName is not None:
                self._setLabelText(self.currentPlot.gateLabel, "Gate applied: "+gateName+"\n")
            else :
                self._setLabelText(self.currentPlot.gateLabel, "Gate applied: \n")
        except Exception:
            # self.logger.debug('histoHover - exception', exc_info=True)
            self._hoveredSpectrumName = None
            self._setLabelText(self.currentPlot.histoLabel, "Spectrum: \nX: Y:")
            self._setLabelText(self.currentPlot.pointerLabel, "Pointer:\nX: Y: Count: ")
            self._setLabelText(self.currentPlot.gateLabel, "Gate applied: \n")


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
        self._resize_timer.start(150)

    def _do_resize(self):
        self.logger.debug('_do_resize')
        self.currentPlot.figure.tight_layout()
        self.currentPlot.canvas.draw_idle()


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
                _srm_name = self.nameFromIndex(index)
                if event.button == 1:
                    self.sum_region_manager.on_singleclick_sumRegion(event, index, _srm_name)
                if event.button == 3:
                    self.sum_region_manager.on_singleclick_sumRegion_right(index, _srm_name)
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
            self.currentPlot.canvas.draw_idle()


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

                self.logger.debug('on_dblclick - entering expanded spectrum view')

                #important that zoomPlotInfo is set only while in zoom mode (not None only here)
                self.setEnlargedSpectrum(idx, name)
                self.currentPlot.next_plot_index = self.currentPlot.selected_plot_index
                self.logger.debug('on_dblclick - next_plot_index: %s', self.currentPlot.next_plot_index)
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
                self.logger.debug('on_dblclick - selected_plot_index_bak: %s', self.wTab.selected_plot_index_bak[self.wTab.currentIndex()])
                
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
                    self.plot_controller.old_cmap = spectrum_old.get_cmap()

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
                        spectrum.set_cmap(self.plot_controller.old_cmap)
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
                            self.plotPlot(index, self.plot_controller.old_cmap)
                            self.setSpectrumInfo(cmap=self.plot_controller.old_cmap, index=idx)
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
            except Exception:
                self.logger.debug('clickedTab - exception occured', exc_info=True)
                pass
        


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
    def setCanvasLayout(self):                   return self.plot_controller.setCanvasLayout()

    
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
        # log keys only — info can carry the full counts array (P5)
        self.logger.info('setSpectrumInfoREST - name: %s, keys: %s', name, list(info))
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
        self.logger.debug('setSpectrumInfo - info: %s',info)
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
        self.logger.debug('getSpectrumInfo - info, identifier: %s, %s', info, identifier)
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
    @property
    def rest(self):
        """REST client owned by ConnectionManager; exposed here for backward compat."""
        cm = getattr(self, 'connection_manager', None)
        return cm._rest if cm is not None else None

    def _check_and_cancel_gate(self, doClose):
        """Called by SumRegionManager.cleanPopupExit to handle gate-side cleanup."""
        if (self.currentPlot.toCreateGate or self.currentPlot.toEditGate) \
                and not self.gatePopup.isVisible():
            self.cancelGate(doClose)

    @pyqtSlot()
    def _on_srm_canvas_draw(self):
        self.currentPlot.canvas.draw()

    @pyqtSlot()
    def _on_srm_tight_layout(self):
        self.currentPlot.figure.tight_layout()
        self.currentPlot.canvas.draw()

    @pyqtSlot(int)
    def _on_sum_region_started(self, index):
        self.currentPlot.toCreateSumRegion = True

    @pyqtSlot()
    def _on_sum_region_ended(self):
        self.currentPlot.toCreateSumRegion = False

    @pyqtSlot(object)
    def _on_side_table_conn_updated(self, conn):
        self.sidTableIntegrateCopy = conn

    @pyqtSlot()
    def _on_gm_canvas_draw(self):
        self.currentPlot.canvas.draw()

    @pyqtSlot()
    def _on_gm_canvas_draw_idle(self):
        self.currentPlot.canvas.draw_idle()

    @pyqtSlot(int)
    def _on_gate_creation_started(self, index):
        self.currentPlot.toCreateGate = True
        self.currentPlot.toEditGate   = False

    @pyqtSlot()
    def _on_gate_editing_started(self):
        self.currentPlot.toEditGate   = True
        self.currentPlot.toCreateGate = False

    @pyqtSlot()
    def _on_gate_ended(self):
        self.currentPlot.toCreateGate = False
        self.currentPlot.toEditGate   = False

    @pyqtSlot(str)
    def _on_spectrum_removed_rest(self, name):
        """Display-side cleanup when ConnectionManager removes a spectrum from REST binding."""
        for tabIdx, plotVal in self.wTab.wPlot.items():
            to_delete = [key for key, value in plotVal.h_dict_geo.items() if name in value]
            for key in to_delete:
                if key in self.wTab.spectrum_dict[tabIdx]:
                    spectrum = self.wTab.spectrum_dict[tabIdx][key]["spectrum"]
                    if hasattr(spectrum, 'axes'):
                        ax = spectrum.axes
                        self.removeCb(ax)
                        ax.clear()
                    plotVal.h_dict_geo[key] = "empty"
                    del spectrum

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
        #"data" is intentionally NOT copied: the canonical array lives solely in the
        #SpectrumStore and is derived (with cutoff) on demand by the plot controller,
        #so it is never duplicated into the per-tab display tier. The empty "data"
        #placeholder from the template above is kept for dict-shape consistency.
        record = self.spectra.get_record(name)
        if record is None:
            self.logger.warning('setGeo - %s not in SpectrumStore; slot left with name only', name)
            return
        for key, value in record.items():
            if key == "data":
                continue
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


    _GATE_NAME_TTL = 2.0  # seconds — max staleness of gate-name label during mouse hover

    def getAppliedGateName(self, **identifier):
        """Return the gate name applied to a spectrum — cache only, never blocking.

        Stale-while-revalidate: a fresh cache entry is
        returned as-is; a cold/expired one returns the stale value (or None)
        immediately and triggers a background REST fetch. The hover label
        corrects itself when the result lands (_on_gate_name_fetched), so the
        GUI thread never waits on HTTP mid-hover."""
        spectrumName = None
        if "index" in identifier:
            spectrumName = self.nameFromIndex(identifier["index"])
        elif "name" in identifier:
            spectrumName = identifier["name"]
        else:
            self.logger.debug('getAppliedGateName - wrong identifier')
            return None
        now = time.monotonic()
        cached = self._gate_name_cache.get(spectrumName)
        if cached is not None and (now - cached[1]) < self._GATE_NAME_TTL:
            return cached[0]
        self._refreshGateNameAsync(spectrumName)
        return cached[0] if cached is not None else None

    def _refreshGateNameAsync(self, spectrumName):
        """Fetch applylistgate on a worker thread; result lands via _gateNameFetched."""
        if spectrumName in self._gate_name_inflight:
            return
        self._gate_name_inflight.add(spectrumName)

        def fetch():
            gate = self.connection_manager.applylistgate(spectrumName)
            try:
                self._gateNameFetched.emit(spectrumName, gate)
            except RuntimeError:
                pass  # window destroyed during shutdown

        threading.Thread(target=fetch, daemon=True,
                         name=f"gate-name-fetch-{spectrumName}").start()

    @pyqtSlot(str, object)
    def _on_gate_name_fetched(self, spectrumName, gate):
        self._gate_name_inflight.discard(spectrumName)
        try:
            if gate is None or len(gate) == 0:
                result = None
            else:
                gn = gate[0]["gate"]
                result = None if gn in ("-TRUE-", "-Ungated-") else gn
        except Exception:
            self.logger.debug('_on_gate_name_fetched - malformed reply for %s',
                              spectrumName, exc_info=True)
            result = None
        self._gate_name_cache[spectrumName] = (result, time.monotonic())
        # correct the hover label if the pointer is still on this spectrum
        if self._hoveredSpectrumName == spectrumName and self.currentPlot is not None:
            text = "Gate applied: " + result + "\n" if result is not None else "Gate applied: \n"
            self._setLabelText(self.currentPlot.gateLabel, text)




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
        if os.stat(filename).st_size == 0:
            self.logger.warning('openGeo - empty geometry file: %s', filename)
            return None

        # Sniff the format from the first meaningful (non-blank, non-comment) line:
        # legacy Xamine/dispwind ".win" files begin with a "Geometry R,C" line, while
        # native (qtpy) files are a single-line Python dict literal beginning with "{".
        firstMeaningful = ""
        with open(filename) as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    firstMeaningful = stripped
                    break

        if firstMeaningful.lower().startswith("geometry"):
            return self.parseOldGeo(filename)
        if firstMeaningful.startswith("{"):
            return eval(open(filename, "r").read())
        self.logger.warning('openGeo - unrecognized geometry file format: %s', filename)
        return None

    def parseOldGeo(self, filename):
        """Parse a legacy Xamine/dispwind ``.win`` geometry file into the same
        structure :meth:`openGeo` returns for the native format::

            {"row": <nrows>, "col": <ncols>,
             "geo": {flatIndex: {"name": str,
                                 "x": [min, max] | None,
                                 "y": [min, max] | None,
                                 "scale": bool}}}

        Windows get sequential flat indices in file order (matching the original
        loader). ``COUNTSAXIS`` maps to log scale; ``Expanded`` supplies the x/y view
        range when present, otherwise the spectrum keeps its natural range. ``SCALE``,
        ``Refresh``, ``MAPPED`` and other per-window settings are ignored.
        """
        self.logger.info('parseOldGeo - filename: %s', filename)
        nrow = ncol = None
        properties = {}
        index  = 0
        name   = None
        scale  = False
        xRange = None
        yRange = None

        def _numbers(text):
            return [float(n) for n in re.findall(r'-?\d+(?:\.\d+)?', text)]

        try:
            with open(filename) as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith('#'):
                        continue
                    keyword = line.split()[0].lower()
                    if keyword == 'geometry':
                        nums = _numbers(line)
                        if len(nums) >= 2:
                            nrow, ncol = int(nums[0]), int(nums[1])
                    elif keyword == 'window':
                        name, scale, xRange, yRange = None, False, None, None
                        match = re.search(r'"([^"]*)"', line)
                        if match:
                            name = match.group(1)
                    elif keyword == 'countsaxis':
                        scale = True
                    elif keyword == 'expanded':
                        nums = _numbers(line)
                        if len(nums) >= 2:
                            xRange = [nums[0], nums[1]]
                        if len(nums) >= 4:
                            yRange = [nums[2], nums[3]]
                    elif keyword == 'endwindow':
                        if name:
                            properties[index] = {"name": name, "x": xRange,
                                                 "y": yRange, "scale": scale}
                        index += 1
                        name, scale, xRange, yRange = None, False, None, None
        except OSError:
            self.logger.warning('parseOldGeo - could not read %s', filename, exc_info=True)
            return None

        if nrow is None or ncol is None:
            self.logger.warning('parseOldGeo - no "Geometry" line found in %s', filename)
            return None
        return {"row": nrow, "col": ncol, "geo": properties}


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
                except Exception:
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


    def _resolveSpectrumName(self, name):
        """Return a spectrum name present in the store that matches `name`, tolerating
        case differences (legacy .win files often store names upper-cased). Returns the
        exact name if it exists, a unique case-insensitive match otherwise, or None."""
        if self.getSpectrumInfoREST("dim", name=name) is not None:
            return name
        lowered = name.lower()
        matches = [n for n in self.spectra.all_names() if n.lower() == lowered]
        return matches[0] if len(matches) == 1 else None

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
                    if not val_dict["name"]:
                        continue
                    resolved = self._resolveSpectrumName(val_dict["name"])
                    if resolved is None:
                        notFound.append(val_dict["name"])
                        continue

                    self.setGeo(index, resolved)
                    self.setSpectrumInfo(log=val_dict["scale"], index=index)
                    # Old .win files may omit the view range (no "Expanded"); when it
                    # is absent the spectrum keeps its natural full range from the store.
                    if val_dict.get("x") is not None:
                        self.setSpectrumInfo(minx=val_dict["x"][0], index=index)
                        self.setSpectrumInfo(maxx=val_dict["x"][1], index=index)
                    if val_dict.get("y") is not None:
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
    def setAxisScale(self, ax, index, *scale):   return self.plot_controller.setAxisScale(ax, index, *scale)



    # Where is defined the color bar
    def setCmapNorm(self, scale, index):         return self.plot_controller.setCmapNorm(scale, index)


    #Callback for plusButton/minusButton
    def zoomInOut(self, arg):                    return self.plot_controller.zoomInOut(arg)


    # Callback for histo_autoscale, calls setAxisScale
    def autoScaleAxisBox(self, forIndex):        return self.plot_controller.autoScaleAxisBox(forIndex)


    # get data max within user defined range
    # For 2D have to give two ranges (x,y), for 1D range x.
    def getMinMaxInRange(self, index, **limits):  return self.plot_controller.getMinMaxInRange(index, **limits)


    # Have seen malloc error if data array too large
    # Divide data array in sub-arrays with sub-(min, max) and then find the global-(min, max)
    def customMinMax(self, data):                return self.plot_controller.customMinMax(data)


    #return the axis limits in a certain format [[xmin, xmax], [ymin, ymax]]
    def getAxisProperties(self, index):          return self.plot_controller.getAxisProperties(index)
            

    def zoomCallback(self, event):               return self.plot_controller.zoomCallback(event)

    #Used by customZoom button, trigger toolbar zoom action
    def customZoomButtonCallback(self):          return self.plot_controller.customZoomButtonCallback()


    #Used by customHome button, reset the axis limits to ReST definitions, for the specified plot at index or for all plots if index not provided
    def customHomeButtonCallback(self, index=None): return self.plot_controller.customHomeButtonCallback(index)


    #Used by logButton, defines the log scale, for the specified plot at index or for all plots if logAll/unlogAll, calls setAxisScale
    def logButtonCallback(self, *arg):           return self.plot_controller.logButtonCallback(*arg)


    #callback when right click on customZoomButton
    def zoom_handle_right_click(self):           return self.plot_controller.zoom_handle_right_click()

    #callback when right click on customHomeButton
    def handle_right_click(self):                return self.plot_controller.handle_right_click()


    #callback when right click on logButton
    def log_handle_right_click(self):            return self.plot_controller.log_handle_right_click()


    #button of the cutoff window, sets the cutoff values in the spectrum dict
    def okCutoff(self):                          return self.plot_controller.okCutoff()
        
    def cancelCutoff(self):                      return self.plot_controller.cancelCutoff()

    def resetCutoff(self, doUpdate):             return self.plot_controller.resetCutoff(doUpdate)

    #called by cutoffButton, sets the information in the cutoff window
    def cutoffButtonCallback(self, *arg):        return self.plot_controller.cutoffButtonCallback(*arg)

    #Used in zoomCallBack to save the new axis limits
    def updatePlotLimits(self):                  return self.plot_controller.updatePlotLimits()


    ##################################
    ## 9) Histogram operations
    ##################################

    # remove colorbar
    def removeCb(self, axis):                    return self.plot_controller.removeCb(axis)


    # select axes based on indexing, used only in add()
    def select_plot(self, index):                return self.plot_controller.select_plot(index)


    # returns position in grid based on indexing
    def plotPosition(self, index):               return self.plot_controller.plotPosition(index)


    # setup histogram limits according to the ReST info
    # called in add(), when the plot is first added
    def setupPlot(self, axis, index):            return self.plot_controller.setupPlot(axis, index)


    # geometrically add plots to the right place and calls plotting
    # should be called only by addPlot and on_dblclick when entering/exiting enlarged mode
    def add(self, index):                        return self.plot_controller.add(index)


    # Callback for histo_geo_add button
    # also called in loadGeo
    # geometrically add plots to the right place
    # plot axis as defined in the ReST interface.
    def addPlot(self):                           return self.plot_controller.addPlot()


    #why not using np.linspace(vmin, vmax, bins)
    def createRange(self, bins, vmin, vmax):     return self.plot_controller.createRange(bins, vmin, vmax)


    # fill spectrum with new data
    # called in addPlot and updatePlot
    # dont actually draw the plot in this function
    def plotPlot(self, index, cmap=None):        return self.plot_controller.plotPlot(index, cmap)


    ### Bashir added for auto-update signal
    @pyqtSlot()
    def _updatePlotOnGui(self):                  return self.plot_controller._updatePlotOnGui()
    #Callback for histo_geo_update button
    #also used in various functions
    #redraw plot spectrum, update axis scales, redraw gates
    def updatePlot(self):                        return self.plot_controller.updatePlot()


    ####### Bashir added for color map #######################
    def onColormapChange(self, cmap_name: str):  return self.plot_controller.onColormapChange(cmap_name)


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

    def _current_plot_ctx(self):
        """Return (index, name, ax) for the currently selected plot slot.
        Used by fit_manager signal lambdas to pass resolved context without a bridge."""
        idx = self.autoIndex()
        return idx, self.nameFromIndex(idx), self.getSpectrumInfo("axis", index=idx)

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
        except Exception:
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
    def fit(self):               return self.fit_manager.fit(*self._current_plot_ctx())
    def deleteFit(self):         return self.fit_manager.deleteFit(*self._current_plot_ctx())
    def listFitLineLabels(self, ax):     return self.fit_manager.listFitLineLabels(ax)
    def printFitLineLabels(self): return self.fit_manager.printFitLineLabels(*self._current_plot_ctx())
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

    # ------------------------------------------------------------------
    # SumRegion methods — delegated to SumRegionManager (see gui/services/sum_region_manager.py)
    # ------------------------------------------------------------------
    def setSumRegion(self, index, line):
        name = self.nameFromIndex(index)
        return self.sum_region_manager.setSumRegion(index, line, name)
    def getSumRegion(self, index):
        name = self.nameFromIndex(index)
        return self.sum_region_manager.getSumRegion(index, name)
    def deleteSumRegionDict(self, label):
        return self.sum_region_manager.deleteSumRegionDict(label, self.currentPlot.figure.axes)
    def refreshSpectrumSumRegionDict(self):      return self.sum_region_manager.refreshSpectrumSumRegionDict()
    def saveSumRegion(self, index):
        name = self.nameFromIndex(index)
        return self.sum_region_manager.saveSumRegion(index, name)
    def createSumRegion(self):                   return self.sum_region_manager.createSumRegion(*self._current_plot_ctx())
    def okSumRegion(self):                       return self.sum_region_manager.okSumRegion()
    def cancelSumRegion(self, doClose=True):     return self.sum_region_manager.cancelSumRegion(doClose)
    def cleanPopupExit(self, doClose=True):      return self.sum_region_manager.cleanPopupExit(doClose)
    def deleteSumRegion(self):                   return self.sum_region_manager.deleteSumRegion(*self._current_plot_ctx())
    def integrate(self):                         return self.sum_region_manager.integrate(*self._current_plot_ctx())
    def okIntegrate(self):                       return self.sum_region_manager.okIntegrate()
    def copySelectionIntegrateTable(self):       return self.sum_region_manager.copySelectionIntegrateTable()
    def formatResultsIntegrate(self, results):   return self.sum_region_manager.formatResultsIntegrate(results)
    def setPrecisionIntegrationResult(self, d):  return self.sum_region_manager.setPrecisionIntegrationResult(d)
    def integrateGateLocal(self, idx, lines):
        name = self.nameFromIndex(idx)
        return self.sum_region_manager.integrateGateLocal(idx, name, lines)

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
                except Exception:
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
        except Exception:
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
        except Exception:
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

        except Exception:
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
        except Exception:
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
        except Exception:
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
        except Exception:
            pass

    def zoomFigureX(self):
        self.logger.info('zoomFigureX')
        self.extraPopup.imaging.zoomX_label.setText("Zoom X Level ({} %)".format(self.extraPopup.imaging.zoomX_slider.value()*10))
        try:
            self.deleteFigure()
            self.drawFigure()
        except Exception:
            pass

    def zoomFigureY(self):
        self.logger.info('zoomFigureY')
        self.extraPopup.imaging.zoomY_label.setText("Zoom Y Level ({} %)".format(self.extraPopup.imaging.zoomY_slider.value()*10))
        try:
            self.deleteFigure()
            self.drawFigure()
        except Exception:
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
        except Exception:
            pass

    def jupyterStop(self):
        self.logger.info('jupyterStop')
        # stop the notebook process
        log("Sending interrupt signal to jupyter-notebook")
        self.extraPopup.peak.jup_start.setEnabled(True)
        self.extraPopup.peak.jup_stop.setEnabled(False)
        self.extraPopup.peak.jup_start.setStyleSheet("background-color:#3CB371;")
        self.extraPopup.peak.jup_stop.setStyleSheet("")
        if getattr(self, "jupyterView", None) is not None:
            self.jupyterView.close()
            self.jupyterView = None
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

        # setting window — anchored on self: a parentless local QMainWindow
        # is finalized by the cyclic GC after this method returns (the
        # WebWindow<->CustomWebView reference cycle is its only holder) and
        # the window silently disappears mid-session
        self.jupyterView = view = WebWindow(None, None)
        view.setWindowTitle("Jupyter CutiePie: %s" % directory)
        # logging on docked console
        qtlogger = QtLogger(view)
        qtlogger.newlog.connect(view.loggerdock.log)
        set_logger(lambda message: qtlogger.newlog.emit(message))

        log("Setting home directory --> "+str(directory))

        # start the notebook process
        try:
            webaddr = startnotebook(execname, directory=directory)
        except (RuntimeError, ValueError) as e:
            self.logger.error('jupyterStart - notebook failed to start: %s', e)
            view.close()
            self.jupyterView = None
            setup_logging(logfile)
            QMessageBox.warning(self, "Jupyter",
                                "The Jupyter notebook server failed to start:\n%s" % e)
            return
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
        stopnotebook()   # no-op when not running; otherwise avoid an orphan server
        event.accept()

    def createRectangle(self, plot):             return self.plot_controller.createRectangle(plot)


    def createDashedRectangle(self, plot):       return self.plot_controller.createDashedRectangle(plot)


    def removeRectangle(self):                  return self.plot_controller.removeRectangle()

    #for debug
    def axesChilds(self):                       return self.plot_controller.axesChilds()


    #for debug
    def axesChildsTest(self, axis=None):        return self.plot_controller.axesChildsTest(axis)


    def debugModeCallBack(self):
        if self.extraPopup.options.debugMode.isChecked():
            # record creation only while the file handler can consume it (P5)
            self.logger.setLevel(logging.DEBUG)
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
            self.logger.setLevel(logging.WARNING)


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
