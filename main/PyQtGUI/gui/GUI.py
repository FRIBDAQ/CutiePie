#!/usr/bin/env python3
# import modules and packages

import sys, os
import logging, logging.handlers
import threading, re
from copy import deepcopy

import signal, ctypes



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
    QApplication, QFileDialog, QGridLayout, QMainWindow, QMenu,
    QShortcut, QTabBar, QWidget,
)
from PyQt5.QtGui import QCursor, QKeySequence, QMouseEvent
from PyQt5.QtCore import (
    pyqtSignal, pyqtSlot, Qt, QTimer,
)


import matplotlib
matplotlib.use("Qt5Agg")
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
from PlotGUI import Tabs # area defined for the Tabs
from services.spectrum_store import SpectrumStore
from services.log_throttle import LogThrottle
from services.fit_manager import FitManager
from services.gate_manager import GateManager
from services.sum_region_manager import SumRegionManager
from services.connection_manager import ConnectionManager
from services.plot_controller import PlotController
from controllers.copy_properties_controller import CopyPropertiesController
from controllers.geometry_controller import GeometryController
from controllers.overlay_controller import OverlayController
from controllers.jupyter_controller import JupyterController
from controllers.peak_scan_controller import PeakScanController
from dialogs import QtLogger, TabPopup, cutoffPopup
from view_state import ViewState
from controllers.peak_fit2_controller import PeakFit2Controller
from adapters.connection_adapter import ConnectionAdapter
from adapters.fit_adapter import FitAdapter
from adapters.gate_adapter import GateAdapter
from adapters.sum_region_adapter import SumRegionAdapter
from CopyPropertiesGUI import CopyProperties
from connectConfigGUI import ConnectConfiguration #class for the connection configuration popup
from MenuGate import MenuGate #class for the gate creation/edition popup
from MenuSumRegion import MenuSumRegion #class for the gate creation/edition popup
from OutputIntegrate import OutputIntegratePopup #popup for gate/summing region integrate outputs
# from OutputIntegrate import TableModel #table model for popup for gate/summing region

from notebook_process import stopnotebook


#from collapseMenu import Spoiler

SETTING_BASEDIR = "workdir"
SETTING_EXECUTABLE = "exec"
DEBUG = False


# Fallback for an installed tree, where configure.ac is not shipped. Keep this
# in sync with AC_INIT in configure.ac, which stays the source of truth.
_VERSION_FALLBACK = "v1.6-002"


def cutiepie_version():
    """CutiePie version for the window title. In a source checkout, read it
    live from configure.ac's AC_INIT line (configure.ac sits two levels up
    from this gui/ folder) so it always tracks the source of truth."""
    try:
        cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "..", "configure.ac")
        with open(cfg, encoding="utf-8") as f:
            m = re.search(r"AC_INIT\(\s*\[?\s*CutiePie\s*\]?\s*,\s*\[?\s*([^,\]\)\s]+)",
                          f.read())
        if m:
            return m.group(1)
    except Exception:
        pass
    return _VERSION_FALLBACK



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

        # __init__ is a sequence of setup phases plus the activation below;
        # each helper is a contiguous slice of what used to be one 550-line
        # body, called in the original order. Order is load-bearing: services
        # capture widgets and threading events built before them, and the
        # signal wiring captures the services.
        self._setup_logging()
        self._build_widgets(factory, fit_factory)
        self._build_services()
        self._init_runtime_state()
        self._wire_signals()

        self.currentPlot = self.wTab.plot(self.wTab.currentIndex()) # definition of current plot

        # per-tab button/canvas wiring — single source of truth:
        # the same routine that rebinds on tab switch does the initial bind
        self.bindDynamicSignal()

    def _setup_logging(self):
        """Root-logger configuration and the two handlers."""
        # Single source of truth for root-logger config. Runs before any
        # setup_logging() call, so THIS is the config that takes effect:
        # default stderr handler at WARNING.
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
            filename="debugCutiePie.log", when='m', interval=10, backupCount=1,
            delay=True)
        self.fileHandler.setLevel(logging.DEBUG)
        self.fileHandler.setFormatter(formatterFileHandler)

        #add only stream handler here, the fileHandler is added/removed by user action
        self.logger.addHandler(self.streamHandler)
        #following line to avoid main logger printing log in addition to its handlers
        self.logger.propagate = False

    def _build_widgets(self, factory, fit_factory):
        """Window shell, layouts, the toolbar/tab widgets and the nine popups.

        factory/fit_factory are the two __init__ arguments this phase stores
        and initializes; they are passed in rather than read off self so the
        statements stay in their original order."""
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)

        self.factory = factory
        self.fit_factory = fit_factory


        self._abort_fit = False # Bashir added for aborting fit

        _version = cutiepie_version()
        _version_tag = (" " + _version) if _version else ""
        self.setWindowTitle("CutiePie" + _version_tag + " - (QtPy) - It's not a bug, it's a feature (cit.) Qt5 and PyQty5 used under open source terms.")
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

    def _build_services(self):
        """The spectrum store and the five application services."""
        # global variables
        #spectra (SpectrumStore): canonical REST registry {name -> {dim,binx,minx,maxx,biny,miny,maxy,data,parameters,type}}
        self.spectra = SpectrumStore()

        # FIRST, before any service: every one of them is constructed against
        # these accessors, and the two late-bound seams below (currentPlot,
        # connection_manager) are resolved per call, so neither has to exist yet.
        self.view_state = ViewState(
            spectra=self.spectra,
            tabs=self.wTab,
            get_current_plot=lambda: self.currentPlot,
            applylistgate=lambda name: self.connection_manager.applylistgate(name),
            gate_name_fetched=self._gateNameFetched,
            logger=self.logger,
        )


        self.fit_manager = FitManager(
            fit_factory=self.fit_factory,
            spectra=self.spectra,
            parent_widget=self,
            logger=self.logger,
        )

        self.gate_manager = GateManager(
            spectra=self.spectra,
            name_from_index=self.view_state.nameFromIndex,
            get_spectrum_info=self.view_state.getSpectrumViewInfo,
            get_is_enlarged=lambda: self.currentPlot.isEnlarged,
            get_geo=self.view_state.getGeo,
            get_sum_region=lambda index, name: self.sum_region_manager.getSumRegion(index, name),
            get_current_canvas=lambda: self.wTab.plot(self.wTab.currentIndex()).canvas,
            integrate_popup=self.integratePopup,
            get_integrate_copy=lambda: getattr(self._sum_region_adapter, 'sid_table_integrate_copy', None),
            get_hide=lambda: self.extraPopup.options.gateHide.isChecked(),
            get_annotate=lambda: self.extraPopup.options.gateAnnotation.isChecked(),
            get_edit_disable=lambda: self.extraPopup.options.gateEditDisable.isChecked(),
            get_readout=lambda: self.gatePopup.regionPoint.toPlainText(),
            get_gate_type=lambda: self.gatePopup.listGateType.currentText(),
            get_gate_name=lambda: self.gatePopup.gateNameList.currentText(),
            sum_region_popup=self.sumRegionPopup,
            skip_auto=self.skipAutoUpdateThread,
            get_rest=lambda: self.rest,
            gate_popup=self.gatePopup,
            parent_widget=self,
            logger=self.logger,
        )
        self.sum_region_manager = SumRegionManager(
            spectra=self.spectra,
            name_from_index=self.view_state.nameFromIndex,
            get_spectrum_info=self.view_state.getSpectrumViewInfo,
            get_histo_names=lambda: [self.wConf.histo_list.itemText(i)
                                     for i in range(self.wConf.histo_list.count())],
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
            spectra=self.spectra,
            get_current_plot=lambda: self.currentPlot,
            get_geo=self.view_state.getGeo,
            set_geo=self.view_state.setGeo,
            get_spectrum_info=self.view_state.getSpectrumViewInfo,
            set_spectrum_info=self.view_state.setSpectrumViewInfo,
            get_spectrum_info_dict=self.view_state.getSpectrumViewDict,
            name_from_index=self.view_state.nameFromIndex,
            get_enlarged_spectrum=self.view_state.getEnlargedSpectrum,
            auto_index=self.autoIndex,
            next_index=self.nextIndex,
            bind_dynamic_signal=self.bindDynamicSignal,
            draw_gate=self.gate_manager.drawGate,
            clean_popup_exit=self.cleanPopupExit,
            auto_update_start=self.autoUpdateStart,
            stop_auto_update_thread=self.stopAutoUpdateThread,
            min_y=self.minY,
            max_y=self.maxY,
            min_z=self.minZ,
            max_z=self.maxZ,
            parent_widget=self,
            logger=self.logger,
        )

        self.overlay_controller = OverlayController(
            imaging=self.extraPopup.imaging,
            get_current_plot=lambda: self.currentPlot,
            get_selected_index=lambda: self.currentPlot.selected_plot_index,
            get_grid=lambda: (int(self.wConf.histo_geo_row.currentText()),
                              int(self.wConf.histo_geo_col.currentText())),
            plot_position=self.plotPosition,
            open_image_dialog=lambda: self.openFigureDialog(),
            parent_widget=self,
            logger=self.logger,
        )

        self.jupyter_controller = JupyterController(
            peak_tab=self.extraPopup.peak,
            get_store_dict=self.view_state.getSpectrumStoreDict,
            get_statistics=self.connection_manager.getSpectrumStatistics,
            qt_logger_factory=QtLogger,
            parent_widget=self,
            logger=self.logger,
        )

        self.geometry_controller = GeometryController(
            tabs=self.wTab,
            conf=self.wConf,
            spectra=self.spectra,
            get_current_plot=lambda: self.currentPlot,
            set_current_plot=lambda plot: setattr(self, "currentPlot", plot),
            get_store_info=self.view_state.getSpectrumStoreInfo,
            get_view_info=self.view_state.getSpectrumViewInfo,
            set_view_info=self.view_state.setSpectrumViewInfo,
            get_geo=self.view_state.getGeo,
            set_geo=self.view_state.setGeo,
            plot_controller=self.plot_controller,
            gate_manager=self.gate_manager,
            sum_region_manager=self.sum_region_manager,
            connection_manager=self.connection_manager,
            gate_popup=self.gatePopup,
            sum_region_popup=self.sumRegionPopup,
            set_canvas_layout=self.setCanvasLayout,
            add_plot=self.addPlot,
            auto_update_start=self.autoUpdateStart,
            bind_dynamic_signal=self.bindDynamicSignal,
            tab_geo_widget_and_flags=self.tabGeoWidgetAndFlags,
            # late-bound: the dialogs are attributes that tests and future code
            # may replace, so resolve them per call rather than at construction
            open_file_dialog=lambda: self.openFileNameDialog(),
            save_file_dialog=lambda: self.saveFileDialog(),
            parent_widget=self,
            logger=self.logger,
        )

        self.peak_fit2_controller = PeakFit2Controller(
            peak_tab=self.extraPopup.peak,
            spectra=self.spectra,
            get_store_info=self.view_state.getSpectrumStoreInfo,
            get_view_info=self.view_state.getSpectrumViewInfo,
            plot_controller=self.plot_controller,
            # the fit records stay on MainWindow until 8d moves the last group
            # that writes them; read through the seam, never rebound here
            tabs=self.wTab,
            get_current_plot=lambda: self.currentPlot,
            # late-bound: both popups are attributes that can be replaced, and
            # a torn-down one must read as "not blocking"
            get_gate_popup=lambda: self.gatePopup,
            get_sum_popup=lambda: self.sumRegionPopup,
            name_from_index=self.view_state.nameFromIndex,
            parent_widget=self,
            logger=self.logger,
        )

        self.peak_scan_controller = PeakScanController(
            peak_tab=self.extraPopup.peak,
            get_current_plot=lambda: self.currentPlot,
            get_selected_index=lambda: self.currentPlot.selected_plot_index,
            get_store_info=self.view_state.getSpectrumStoreInfo,
            get_view_info=self.view_state.getSpectrumViewInfo,
            plot_controller=self.plot_controller,
            logger=self.logger,
        )

        self.copy_props = CopyPropertiesController(
            copy_attr=self.copyAttr,
            get_selected_index=lambda: self.currentPlot.selected_plot_index,
            set_autoscale=lambda on: self.currentPlot.histo_autoscale.setChecked(on),
            get_store_info=self.view_state.getSpectrumStoreInfo,
            get_view_info=self.view_state.getSpectrumViewInfo,
            set_view_info=self.view_state.setSpectrumViewInfo,
            get_geo=self.view_state.getGeo,
            name_from_index=self.view_state.nameFromIndex,
            plot_position=self.plotPosition,
            plot_controller=self.plot_controller,
            parent_widget=self,
            logger=self.logger,
        )

    def _init_runtime_state(self):
        """Per-session scratch state: PF2, image overlay, gate-name cache.
        Peak Finder 1's scan results and marker handles moved onto
        PeakScanController with its methods."""
        # Peak Finder 2 state — the fit records, the arming flags, the drag
        # context and the connection registry — lives on PeakFit2Controller.

        # overlay: onFigure says whether one is up; imgplot/overlay_ax are the
        # live artists (None until Add) and LISEpic the image (None until Load).
        # Delete and the four nudge buttons are wired straight to Qt slots, so
        # these have to exist before the first click, not after the first draw.
        self.onFigure = False
        self.LISEpic = None
        self.imgplot = None
        self.overlay_ax = None
        self.xstart = 0.0
        self.ystart = 0.0


        # Bool set by extra options -> differentiate gates 
        # self.annotateGate = False

        ### Bashir added for enlarged view
        self._enlargeBusy = False
        self._enlarged_cax = None   # (optional) track colorbar made in enlarged view

        # the gate-name cache moved onto ViewState with its only reader
        self._hoveredSpectrumName = None       # spectrum currently under the pointer
        # histoHover runs per mouse-motion event, so an unexpected error there
        # gets one WARNING per interval rather than one per pixel
        self._hoverLogThrottle = LogThrottle(interval_secs=30.0)
        self._gateNameFetched.connect(self._on_gate_name_fetched)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._do_resize)

    def _wire_signals(self):
        """Every widget/service connection, and the four adapters."""
        #################
        # 2) Signals
        #################

        # top menu signals
        self.wConf.connectButton.clicked.connect(self.connectPopup)
        self.connectConfig.ok.clicked.connect(self.okConnect)
        self.connectConfig.cancel.clicked.connect(self.closeConnect)
        self.connection_manager.connectionEstablished.connect(self.setCanvasLayout)
        self.connection_manager.spectrumListChanged.connect(self.sum_region_manager.refreshSpectrumSumRegionDict)
        self.connection_manager.updatePlotRequested.connect(self._updatePlotOnGui)
        self._connection_adapter = ConnectionAdapter(
            self.connection_manager, self.wTab, self.wConf.connectButton,
            self.wConf.histo_list, self.plot_controller.removeCb,
            self, self.logger)
        self._fit_adapter = FitAdapter(
            self.fit_manager, self.plot_controller,
            self.extraPopup, self.cutoffp, self.logger)

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
        menu.setFixedWidth(150)

        actSave = menu.addAction("Save Geometry")
        actSave.triggered.connect(self.saveGeo)

        actLoad = menu.addAction("Load Geometry")
        actLoad.triggered.connect(self.loadGeo)

        actSaveAll = menu.addAction("Save All Tabs")
        actSaveAll.triggered.connect(self.saveGeoAll)

        actLoadAll = menu.addAction("Load All Tabs")
        actLoadAll.triggered.connect(self.loadGeoAll)

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
        self.wConf.histo_geo_update.clicked.connect(lambda: self.plot_controller.updatePlot())
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
        self.wConf.cmapSelector.currentTextChanged.connect(self.plot_controller.onColormapChange)
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
        self.gatePopup.ok.clicked.connect(
            lambda: self.gate_manager.okGate(self.gatePopup.gateNameList.currentText()))
        # Same clicked(bool)->doClose quirk as the sum-region Cancel:
        # call with the default so gate Cancel discards + closes the popup.
        self.gatePopup.cancel.clicked.connect(
            lambda: self.gate_manager.cancelGate())
        self.gatePopup.gateActionCreate.clicked.connect(
            lambda: self.gate_manager.createGate(self.currentPlot.selected_plot_index))
        self.gatePopup.clearInfoSignal.connect(self.gatePopup.clearInfo)
        self.gatePopup.clearInfoSignal.connect(self.connection_manager.autoUpdateResume)
        self.gate_manager.updatePlotRequested.connect(self.plot_controller.updatePlot)
        self._gate_adapter = GateAdapter(
            self.gate_manager, self.gatePopup,
            lambda: self.currentPlot, self.logger)

        # summing region
        self.wConf.createSumRegionButton.clicked.connect(
            lambda: self.sum_region_manager.createSumRegion(*self._current_plot_ctx()))
        self.sumRegionPopup.ok.clicked.connect(
            lambda: self.sum_region_manager.okSumRegion(
                self.sumRegionPopup.sumRegionNameList.currentText()))
        # clicked(bool) would pass checked=False into doClose, so the bare
        # connection made Cancel run with doClose=False and never close the
        # popup. Call with the intended default so Cancel discards the
        # in-progress region AND closes (closeEvent -> clearInfo + resume).
        self.sumRegionPopup.cancel.clicked.connect(
            lambda: self.sum_region_manager.cancelSumRegion())
        self.sumRegionPopup.delete.clicked.connect(
            lambda: self.sum_region_manager.deleteSumRegion(
                *self._current_plot_ctx(),
                self.sumRegionPopup.sumRegionNameList.currentText()))
        self.sumRegionPopup.clearInfoSignal.connect(self.sumRegionPopup.clearInfo)
        self.sumRegionPopup.clearInfoSignal.connect(self.connection_manager.autoUpdateResume)
        self.sum_region_manager.updatePlotRequested.connect(self.plot_controller.updatePlot)
        self.sum_region_manager.gateSignalsDisconnectRequested.connect(self.gate_manager.disconnectGateSignals)
        self._sum_region_adapter = SumRegionAdapter(
            self.sum_region_manager, lambda: self.currentPlot,
            self.sumRegionPopup, self.integratePopup,
            self.copySelectionIntegrateTable, self.logger)

        # self.wConf.editGate.setToolTip("Key bindings for Modify->Edit:\n"
        #                               "'i' insert vertex\n"
        #                               "'d' delete vertex\n")

        #integrate gate and summing region
        self.wConf.integrateGateAndRegion.clicked.connect(
            lambda: self.sum_region_manager.integrate(*self._current_plot_ctx()))

        self.tabp.okButton.clicked.connect(self.okTab)
        self.tabp.cancelButton.clicked.connect(self.cancelTab)

        self.cutoffp.okButton.clicked.connect(self.okCutoff)
        self.cutoffp.cancelButton.clicked.connect(self.cancelCutoff)
        self.cutoffp.resetButton.clicked.connect(lambda: self.plot_controller.resetCutoff(True))

        #### Bashir added for zooming hotkeys ####
        QShortcut(QKeySequence("+"), self.wTab.plot(self.wTab.currentIndex())).activated.connect(lambda: self.plot_controller.zoomInOut("in"))
        QShortcut(QKeySequence("-"), self.wTab.plot(self.wTab.currentIndex())).activated.connect(lambda: self.plot_controller.zoomInOut("out"))
        ###############################################

        # copy attributes
        self.copyAttr.histoAll.clicked.connect(lambda: self.histAllAttr(self.copyAttr.histoAll))
        self.copyAttr.okAttr.clicked.connect(self.okCopy)
        self.copyAttr.applyAttr.clicked.connect(self.applyCopy)
        self.copyAttr.cancelAttr.clicked.connect(self.closeCopy)
        self.copyAttr.selectAll.clicked.connect(self.selectAll)

        # extra popup — wired to fit_manager; popup field reads happen HERE
        # (the service takes plain arguments, never widget references)
        self.extraPopup.fit_button.clicked.connect(
            lambda: self.fit_manager.fit(*self._current_plot_ctx(), *self._fit_inputs()))
        self.extraPopup.fit_csv_button.clicked.connect(
            lambda: self.fit_manager.on_fit_csv_clicked(*self._fit_inputs()))
        # lambdas shield the slots from clicked(bool)'s checked arg — a bare
        # connect would pass False as the path/context
        self.extraPopup.save_fit_button.clicked.connect(
            lambda: self.fit_manager.save_fit_curve())
        self.extraPopup.load_fit_button.clicked.connect(
            lambda: self.fit_manager.load_fit_curve(*self._current_plot_ctx()))
        self.extraPopup.all_fitIdx_button.clicked.connect(
            lambda: self.fit_manager.printFitLineLabels(*self._current_plot_ctx()))
        self.extraPopup.delete_button.clicked.connect(
            lambda: self.fit_manager.deleteFit(*self._current_plot_ctx(),
                                               self.extraPopup.delete_fitIdx_list.text()))

        self.extraPopup.peak.peak_analysis.clicked.connect(self.analyzePeak)
        self.extraPopup.peak.peak_analysis_clear.clicked.connect(self.peakAnalClear)
        # Peak Finder 2 (click-to-fit): Start is a checkable toggle; lambda
        # shields Clear from clicked(bool)'s checked arg
        self.extraPopup.peak.peak2_start.toggled.connect(self.peakFit2Toggle)
        self.extraPopup.peak.peak2_fix.toggled.connect(self.peakFit2FixToggle)
        self.extraPopup.peak.peak2_delete.clicked.connect(lambda: self._peak2_delete_selected())
        self.extraPopup.peak.peak2_clear.clicked.connect(lambda: self.peakFit2Clear())
        self.extraPopup.peak.peak2_config.clicked.connect(lambda: self.peakFit2Config())
        self.extraPopup.peak.peak2_table.itemSelectionChanged.connect(self._peak2_row_selected)
        # shape menus: restore the last-used selection, then persist on change.
        # The slot also re-fits the selected fit.
        self._peak2_load_shape_menus()
        self.extraPopup.peak.peak2_signal.currentIndexChanged.connect(self._peak2_shape_changed)
        self.extraPopup.peak.peak2_bg.currentIndexChanged.connect(self._peak2_shape_changed)
        self.extraPopup.peak.peak2_cb_tail.currentIndexChanged.connect(self._peak2_shape_changed)
        # peak-selection list: wired ONCE here (the old per-scan
        # stateChanged.connect on the fixed checkbox grid stacked a duplicate
        # connection on every Scan); lambdas shield from clicked(bool)'s
        # checked arg — All would otherwise receive False
        self.extraPopup.peak.peak_list.itemChanged.connect(self.peakItemChanged)
        self.extraPopup.peak.peak_all.clicked.connect(lambda: self.setAllPeaksChecked(True))
        self.extraPopup.peak.peak_none.clicked.connect(lambda: self.setAllPeaksChecked(False))

        self.extraPopup.peak.jup_start.clicked.connect(self.jupyterStart)
        self.extraPopup.peak.jup_stop.clicked.connect(self.jupyterStop)
        # precedent: lambda shields the slot from clicked(bool)'s checked arg
        self.extraPopup.peak.jup_save.clicked.connect(lambda: self.createDf())

        self.extraPopup.options.gateAnnotation.clicked.connect(self.gate_manager.gateAnnotationCallBack)
        # lambda, not a direct connect: clicked carries a bool that would land on
        # updatePlot's force parameter and turn the redraw into a skippable one
        self.extraPopup.options.gateHide.clicked.connect(lambda: self.plot_controller.updatePlot())
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
        self.wTab.plot(self.wTab.currentIndex()).canvas.setFocusPolicy( QtCore.Qt.ClickFocus )
        self.wTab.plot(self.wTab.currentIndex()).canvas.setFocus()

        # create helpers
        self.wConf.histo_list.installEventFilter(self)
        self.gatePopup.gateNameList.installEventFilter(self)

        # Hotkeys
        # zoom (click-drag)
        self.shortcutZoomDrag = QShortcut(QKeySequence("Alt+Z"), self)
        # self.shortcutZoomDrag.activated.connect(self.zoomKeyCallback)
        self.shortcutZoomDrag.activated.connect(self.plot_controller.customZoomButtonCallback)



    ################################
    # 3) Implementation of Signals
    ################################

    #So that signals work for each tab, called in clickedTab()
    def bindDynamicSignal(self):
        self.logger.info('bindDynamicSignal')
        for index in self.wTab.sessions.indices():
            if self.wTab.isClickBound(index):
                self.wTab.plot(index).logButton.disconnect()
                self.wTab.plot(index).cutoffButton.disconnect()
                self.wTab.plot(index).histo_autoscale.disconnect()
                self.wTab.plot(index).customZoomButton.disconnect()
                self.wTab.plot(index).plusButton.disconnect()
                self.wTab.plot(index).minusButton.disconnect()
                self.wTab.plot(index).copyButton.disconnect()
                self.wTab.plot(index).customHomeButton.disconnect()
                self.wTab.setClickBound(index, False)

        self.wTab.plot(self.wTab.currentIndex()).zoom_action.triggered.connect(self.plot_controller.zoomCallback)
        self.wTab.plot(self.wTab.currentIndex()).histo_autoscale.clicked.connect(lambda: self.plot_controller.autoScaleAxisBox(None))
        self.wTab.plot(self.wTab.currentIndex()).customZoomButton.clicked.connect(self.plot_controller.customZoomButtonCallback)
        self.wTab.plot(self.wTab.currentIndex()).customZoomButton.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.wTab.plot(self.wTab.currentIndex()).customZoomButton.customContextMenuRequested.connect(self.plot_controller.zoom_handle_right_click)
        self.wTab.plot(self.wTab.currentIndex()).plusButton.clicked.connect(lambda: self.plot_controller.zoomInOut("in"))
        self.wTab.plot(self.wTab.currentIndex()).minusButton.clicked.connect(lambda: self.plot_controller.zoomInOut("out"))
        self.wTab.plot(self.wTab.currentIndex()).cutoffButton.clicked.connect(self.plot_controller.cutoffButtonCallback)
        self.wTab.plot(self.wTab.currentIndex()).cutoffButton.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.wTab.plot(self.wTab.currentIndex()).copyButton.clicked.connect(self.copyPopup)
        self.wTab.plot(self.wTab.currentIndex()).customHomeButton.clicked.connect(lambda: self.plot_controller.customHomeButtonCallback(self.currentPlot.selected_plot_index))
        self.wTab.plot(self.wTab.currentIndex()).customHomeButton.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.wTab.plot(self.wTab.currentIndex()).customHomeButton.customContextMenuRequested.connect(self.plot_controller.handle_right_click)
        self.wTab.plot(self.wTab.currentIndex()).logButton.clicked.connect(lambda: self.plot_controller.logButtonCallback(self.currentPlot.selected_plot_index))
        self.wTab.plot(self.wTab.currentIndex()).logButton.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.wTab.plot(self.wTab.currentIndex()).logButton.customContextMenuRequested.connect(self.plot_controller.log_handle_right_click)

        self.resizeID = self.wTab.plot(self.wTab.currentIndex()).canvas.mpl_connect("resize_event", self.on_resize)
        self.pressID = self.wTab.plot(self.wTab.currentIndex()).canvas.mpl_connect("button_press_event", self.on_press)
        self.wTab.plot(self.wTab.currentIndex()).canvas.mpl_connect("button_release_event", self.on_release)

        self.wTab.plot(self.wTab.currentIndex()).canvas.mpl_connect("motion_notify_event", self.histoHover)

        self.wTab.setClickBound(self.wTab.currentIndex(), True)


    def connect(self):
        self.wTab.plot(self.wTab.currentIndex()).canvas.mpl_connect("button_press_event", self.on_press)


    def disconnect(self):
        self.wTab.plot(self.wTab.currentIndex()).canvas.mpl_disconnect(self.pressID)


    #Event filter for search in histo_list widget, and restrict position of moved tab
    def eventFilter(self, obj, event):
        if obj in (self.wConf.histo_list, self.gatePopup.gateNameList) and event.type() == QtCore.QEvent.HoverEnter:
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

    def _blankHoverLabels(self):
        """Clear the three readout labels — the pointer is not over a spectrum."""
        self._hoveredSpectrumName = None
        self._setLabelText(self.currentPlot.histoLabel, "Spectrum: \nX: Y:")
        self._setLabelText(self.currentPlot.pointerLabel, "Pointer:\nX: Y: Count: ")
        self._setLabelText(self.currentPlot.gateLabel, "Gate applied: \n")


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
            name  = self.view_state.nameFromIndex(index)
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
            else:
                # The pad names a spectrum the store does not have — deleted
                # server-side, or a geometry naming something SpecTcl never sent.
                # Without this the two labels keep the previous pad's text and
                # the readout reports a spectrum the pointer is not over.
                self._blankHoverLabels()
                return
            gateName = self.view_state.getAppliedGateName(index=index)
            if gateName is not None:
                self._setLabelText(self.currentPlot.gateLabel, "Gate applied: "+gateName+"\n")
            else :
                self._setLabelText(self.currentPlot.gateLabel, "Gate applied: \n")
        except (IndexError, ValueError, TypeError):
            # Ordinary and frequent: the pointer is over a pad whose axes are not
            # in this figure, over an empty pad, or over one whose spectrum has no
            # counts under the cursor yet, so getPointerInfo hands back ''. Blank
            # the readout and say nothing.
            self._blankHoverLabels()
        except Exception:
            # Anything else is a defect somewhere below — a bad store record, a
            # wrong-tier axis read — and it used to look exactly like the pointer
            # leaving the axes. Report it, but not once per mouse-motion event.
            emit, suppressed = self._hoverLogThrottle.allow()
            if emit:
                self.logger.warning(
                    'histoHover - unexpected error, readout blanked%s',
                    ' (%d similar suppressed)' % suppressed if suppressed else '',
                    exc_info=True)
            self._blankHoverLabels()


    #called in histoHover, return the bin position under mouse pointer
    def getPointerInfo(self, event, info, index):
        result = ['','','']
        if self.view_state.getEnlargedSpectrum():
            index = self.view_state.getEnlargedSpectrum()[0]
        try:
            ax = self.view_state.getSpectrumViewInfo("axis", index=index)
            dim = self.view_state.getSpectrumStoreInfo("dim", index=index)
            minx = self.view_state.getSpectrumStoreInfo("minx", index=index)
            maxx = self.view_state.getSpectrumStoreInfo("maxx", index=index)
            binx = self.view_state.getSpectrumStoreInfo("binx", index=index)
            data = self.view_state.getSpectrumStoreInfo("data", index=index)
            if ax is None or len(data) <= 0:
                return result
            x, y = ax.transData.inverted().transform([event.x, event.y])
            stepx = (float(maxx)-float(minx))/float(binx)
            binminx = int((x-minx)/stepx)
            # clamp: just outside the axis (or a sliver left of minx) must not
            # produce a negative index, which silently wraps to the array end
            binminx = max(0, min(binminx, int(binx) - 1))
            if dim == 1:
                if "coordinates" == info:
                    # +1 is the shm underflow-bin convention: 1D data keeps the
                    # underflow channel at index 0 (connection_manager slices
                    # [0:-1] and zeroes data[0]); 2D data is sliced [1:-1,1:-1]
                    # so it needs no shift.
                    count = data[binminx+1:binminx+2]
                    result = [x,y,count[0]]
                elif "bins" == info:
                    result = [binminx,'','']
            elif dim == 2:
                if "coordinates" == info:
                    miny = self.view_state.getSpectrumStoreInfo("miny", index=index)
                    maxy = self.view_state.getSpectrumStoreInfo("maxy", index=index)
                    biny = self.view_state.getSpectrumStoreInfo("biny", index=index)
                    stepy = (float(maxy)-float(miny))/float(biny)
                    binminy = int((y-miny)/stepy)
                    binminy = max(0, min(binminy, int(biny) - 1))
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
            QTimer.singleShot(100, self.plot_controller.updatePlotLimits)
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

        if not withinLimits or event.button() == Qt.RightButton:
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
            index = self.wTab.selectedPad(self.wTab.currentIndex())
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
                _srm_name = self.view_state.nameFromIndex(index)
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
        axisIsLog = self.view_state.getSpectrumViewInfo("log", index=index)
        wPlot = self.currentPlot
        logBut = wPlot.logButton
        if axisIsLog :
            logBut.setDown(True)
        else :
            logBut.setDown(False) 
        # similar to log button, cutoff button change status according to spectrum info
        cutoffVal = self.view_state.getSpectrumViewInfo("cutoff", index=index)
        if cutoffVal is not None and len(cutoffVal)>1 and (cutoffVal[0] is not None or cutoffVal[1] is not None):
            # wPlot.cutoffButton.setDown(True)
            pass
        else :
            wPlot.cutoffButton.setDown(False)
            self.plot_controller.resetCutoff(False)

        # If we are not zooming on one histogram we can select one histogram
        # and a red rectangle will contour the plot
        if self.currentPlot.isEnlarged == False:
            self.logger.debug('on_singleclick - isEnlarged FALSE')
            self.plot_controller.removeRectangle()
            self.currentPlot.isSelected = True
            self.currentPlot.next_plot_index = self.currentPlot.selected_plot_index
            self.currentPlot.rec = self.plot_controller.createRectangle(self.currentPlot.figure.axes[index])
            #tried to blit here but not successful (?) important delay for canvas with many plots
            self.currentPlot.canvas.draw_idle()


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
            name = self.view_state.nameFromIndex(idx)
            index = self.wConf.histo_list.findText(name)
            self.wConf.histo_list.setCurrentIndex(index)

            if self.currentPlot.isEnlarged == False: # entering enlarged mode
                self.logger.debug('on_dblclick - isEnlarged TRUE')
                if name == "empty" or index == -1:
                    self.logger.warning('on_dblclick - empty axes cannot enlarge')
                    return
                self.plot_controller.removeRectangle()

                self.logger.debug('on_dblclick - entering expanded spectrum view')

                #important that zoomInfo is set only while in zoom mode (not None only here)
                self.view_state.setEnlargedSpectrum(idx, name)
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
                self.wTab.setSelectedPad(self.wTab.currentIndex(), deepcopy(idx))
                self.logger.debug('on_dblclick - selectedPad: %s', self.wTab.selectedPad(self.wTab.currentIndex()))
                
                # t1 = time.time()
                ############### Bashir ##################################################
                # Save the current axes objects
                self.currentPlot._saved_axes = self.currentPlot.figure.axes.copy()

                # Hide all current axes
                for ax in self.currentPlot._saved_axes:
                    ax.set_visible(False)
                
                dim = self.view_state.getSpectrumStoreInfo("dim", index=idx)
                if dim == 2:
                    spectrum_old = self.view_state.getSpectrumViewInfo("spectrum", index=idx)
                    self.plot_controller.old_cmap = spectrum_old.get_cmap()

                elif dim == 1:
                    # Save current y-limits of the target axes to restore later (when autoscale is OFF)
                    ax0 = self.view_state.getSpectrumViewInfo("axis", index=idx)
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
                self.plot_controller.add(idx)
                self.plot_controller.updatePlot()
                
                ###################################################################

                ax = self.view_state.getSpectrumViewInfo("axis", index=idx)
                if dim == 1:
                    if not autoscale_status and hasattr(self.currentPlot, "_saved_ylims") and idx in self.currentPlot._saved_ylims:
                        ax.set_ylim(*self.currentPlot._saved_ylims[idx])   # <-- restore y only
                # self.updatePlot()
                # t2 = time.time()
                # print("on_dblclick: time={:.2f}".format(t2-t1))

                ########### Bashir: reuse color map
                if dim == 2:
                    spectrum = self.view_state.getSpectrumViewInfo("spectrum", index=idx)

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

                #important that zoomInfo is set only while in zoom mode (None only here)
                #tempIdxEnlargedSpectrum is used to draw back the dashed red rectangle, which pad was enlarged
                tempIdxEnlargedSpectrum = self.view_state.getEnlargedSpectrum()[0]
                self.view_state.setEnlargedSpectrum(None, None)
                self.currentPlot.isEnlarged = False

                canvasLayout = self.wTab.tabLayout(self.wTab.currentIndex())
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
                for index, name in self.view_state.getGeo().items():

                    # if name is not None and name != "" and name != "empty":
                    if name is not None and name != "" and name != "empty" and index == idx:
                        self.plot_controller.add(index)
                        ax = self.view_state.getSpectrumViewInfo("axis", index=index)
                        
                        #reset the axis limits as it was before enlarge
                        #dont need to specify if log scale, it is checked inside setAxisScale, if 2D histo in log its z axis is set too.
                        dim = self.view_state.getSpectrumStoreInfo("dim", index=index)
                        if dim == 1:
                            self.plot_controller.plotPlot(index)
                            if not autoscale_status and hasattr(self.currentPlot, "_saved_ylims") and index in self.currentPlot._saved_ylims:
                                ax.set_ylim(*self.currentPlot._saved_ylims[index])   # <-- restore y only
                            else:
                                if autoscale_status:
                                    self.plot_controller.setAxisScale(ax, index, "x", "y")

                        elif dim == 2:
                            self.plot_controller.plotPlot(index, self.plot_controller.old_cmap)
                            self.view_state.setSpectrumViewInfo(cmap=self.plot_controller.old_cmap, index=idx)
                            # if autoscale_status:
                            self.plot_controller.setAxisScale(ax, index, "x", "y", "z")
                        self.gate_manager.drawGate(index)
                #drawing back the dashed red rectangle on the unenlarged spectrum
                self.plot_controller.removeRectangle()
                self.currentPlot.recDashed = self.plot_controller.createDashedRectangle(self.currentPlot.figure.axes[tempIdxEnlargedSpectrum])
                # the re-add above cleared the pad that was enlarged, taking any
                # fit artists with it; redraw them from the stored records
                self.peakFit2RedrawAll()
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
            self.gate_manager.cancelGate()
        if self.currentPlot.toCreateSumRegion or self.sumRegionPopup.isVisible():
            self.sum_region_manager.cancelSumRegion()

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
            self.gate_manager.cancelGate()
        if self.currentPlot.toCreateSumRegion or self.sumRegionPopup.isVisible():
            self.sum_region_manager.cancelSumRegion()

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
        self.connection_manager._stop_auto_thread()

        # For now, if change tab while working on gate, close any ongoing gate action
        if self.currentPlot.toCreateGate or self.currentPlot.toEditGate or self.gatePopup.isVisible():
            self.gate_manager.cancelGate()
            return
        if self.currentPlot.toCreateSumRegion or self.sumRegionPopup.isVisible():
            self.sum_region_manager.cancelSumRegion()
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
            self.currentPlot = self.wTab.plot(index)
            self.tabGeoWidgetAndFlags(index)   
                  
        else:
            try:
                self.tabGeoWidgetAndFlags(index)
                self.plot_controller.removeRectangle()
                self.bindDynamicSignal()

                # If tab not empty, (re)start auto update
                for indexPlot, name in self.view_state.getGeo().items():
                    if name:
                        ax = self.view_state.getSpectrumViewInfo("axis", index=indexPlot)
                        if ax is not None :
                            self.autoUpdateStart()
                            break
            except Exception:
                self.logger.debug('clickedTab - exception occured', exc_info=True)
        


    # Helper to set histo_geo widget and enable/disable buttons, when interact with tabs
    def tabGeoWidgetAndFlags(self, index):
        self.currentPlot = self.wTab.plot(index)
        nRow = self.wTab.tabLayout(index)[0]
        nCol = self.wTab.tabLayout(index)[1]
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
            self.plot_controller.removeRectangle()
            self.bindDynamicSignal()

        # change only default tab text ! might be confusing not sure if useful
        self.wTab.orderDefaultName()


    #set geometry of the canvas
    def setCanvasLayout(self):
        # the geometry combos and tab widget live here; the service only
        # keeps the geometry_applied flag (markGeometryApplied).
        self.logger.info('setCanvasLayout')
        indexTab = self.wTab.currentIndex()
        nRow = int(self.wConf.histo_geo_row.currentText())
        nCol = int(self.wConf.histo_geo_col.currentText())
        self.wTab.setTabLayout(indexTab, [nRow, nCol])
        self.wTab.plot(indexTab).InitializeCanvas(nRow, nCol)
        self.wTab.setSelectedPad(indexTab, None)
        self.currentPlot.selected_plot_index = None
        self.currentPlot.next_plot_index     = -1
        self.plot_controller.markGeometryApplied()

    
###############################################
# 5) Connection to REST for gates
###############################################


    ###############################################
    # 5) Connection to REST for gates
    ##
    #############################################


    #Get spectrum info from self.spectra (identified by histo name or index and info name)
    #template of expected arguments e.g.: ("dim", index=5) takes only the first info parameter (here "dim") (one per call)
    #Important that it gets only the info from self.spectra here.


    #Update spectrum info in tabSlots (identified by index and can update multiple info at once)
    #Important that only the per-tab slots dict is changed here
    #work in normal and enlarged mode


    #Get spectrum info from tabSlots (identified by index and info name)
    #template of expected arguments e.g.: ("dim", index=5) takes only the first info parameter (here "dim") (one per call)
    #Important that it gets only the info from the current tab's slots here.
    #work in normal and enlarged mode


    #Remove spectrum from per-tab slots:
    #important: only functions that delete item in per-tab slots and self.spectra
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
            self.gate_manager.cancelGate(doClose)

    #get full per-tab slots dict for the current tab:


    #get full spectrum dict from self.spectra:
 

    #Find name with geo index:


    #sets h_dict_geo {key=index, value=histoName}
    #Use only this function to set the geometry dict when add plot (against using it elsewhere because it initializes per-tab slots)


    #returns h_dict_geo {key=index, value=histoName}
    #Use only this function to get the geometry dict, to know the name at index there is nameFromIndex


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

        






    @pyqtSlot(str, object)
    def _on_gate_name_fetched(self, spectrumName, gate):
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
        self.view_state.storeGateName(spectrumName, result)
        # correct the hover label if the pointer is still on this spectrum
        if self._hoveredSpectrumName == spectrumName and self.currentPlot is not None:
            text = "Gate applied: " + result + "\n" if result is not None else "Gate applied: \n"
            self._setLabelText(self.currentPlot.gateLabel, text)




    ##########################################
    # 7) Load/save geometry window
    ##########################################


    # Geometry save/load lives on GeometryController. These stay because they
    # are what the File menu actions are connected to, and because other call
    # sites (openGeo, the Jupyter export) use the dialogs.

    def saveGeo(self):
        self.geometry_controller.saveGeo()

    def saveGeoAll(self):
        self.geometry_controller.saveGeoAll()

    def _resolveSpectrumName(self, name):
        return self.geometry_controller._resolveSpectrumName(name)

    def _applyGeometryToCurrentTab(self, infoGeo):
        return self.geometry_controller.applyGeometryToCurrentTab(infoGeo)

    def loadGeo(self):
        self.geometry_controller.loadGeo()

    def loadGeoAll(self):
        self.geometry_controller.loadGeoAll()

    def _applySession(self, tabsInfo):
        self.geometry_controller.applySession(tabsInfo)

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
    # The scaling, zoom and cutoff operations themselves live on PlotController;
    # what stays here reads the popup fields and hands them over as arguments.

    #button of the cutoff window, sets the cutoff values in the spectrum dict
    def okCutoff(self):
        return self.plot_controller.okCutoff(
            self.cutoffp.lineeditXMin.text(), self.cutoffp.lineeditXMax.text(),
            self.cutoffp.lineeditYMin.text(), self.cutoffp.lineeditYMax.text(),
            self.cutoffp.lineeditZMin.text(), self.cutoffp.lineeditZMax.text())
        
    def cancelCutoff(self):
        self.cutoffp.close()


    ##################################
    ## 9) Histogram operations
    ##################################
    # Rendering lives on PlotController. What stays here reads a widget the
    # service no longer holds — the tab layout, the spectrum combo.

    # returns position in grid based on indexing
    def plotPosition(self, index):
        return self.plot_controller.plotPosition(
            index, self.wTab.tabLayout(self.wTab.currentIndex()))


    # Callback for histo_geo_add button
    # also called in loadGeo
    # geometrically add plots to the right place
    # plot axis as defined in the ReST interface.
    def addPlot(self):
        selected = (self.wConf.histo_list.currentText()
                    if self.wConf.histo_list.count() else None)
        return self.plot_controller.addPlot(
            selected, self.wTab.isClickBound(self.wTab.currentIndex()))


    ### Bashir added for auto-update signal
    # Kept as a slot on this side: the auto-update worker emits from its own
    # thread, and the decorator is what makes the delivery a queued one.
    @pyqtSlot()
    def _updatePlotOnGui(self):                  return self.plot_controller._updatePlotOnGui()


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
        return self.currentPlot.index


    #To avoid None plot index
    def autoIndex(self):
        self.logger.info('autoIndex')
        if self.currentPlot.isSelected == False or self.currentPlot.selected_plot_index is None:
            if self.wTab.selectedPad(self.wTab.currentIndex()) is not None:
                self.currentPlot.index = self.wTab.selectedPad(self.wTab.currentIndex())
            else :
                self.currentPlot.index = self.check_index()
        else:
            self.currentPlot.index = self.currentPlot.selected_plot_index
            self.wTab.setSelectedPad(self.wTab.currentIndex(), self.currentPlot.selected_plot_index)

        return self.currentPlot.index

    def _current_plot_ctx(self):
        """Return (index, name, ax) for the currently selected plot slot.
        Used by fit_manager signal lambdas to pass resolved context without a bridge."""
        idx = self.autoIndex()
        return idx, self.view_state.nameFromIndex(idx), self.view_state.getSpectrumViewInfo("axis", index=idx)

    def _fit_inputs(self):
        """Gather the fit popup fields FitManager needs as plain values:
        (fit_funct, [20 parameter texts], range_min_text, range_max_text)."""
        p = self.extraPopup
        texts = [getattr(p, f"fit_p{i}").text() for i in range(20)]
        return (p.fit_list.currentText(), texts,
                p.fit_range_min.text(), p.fit_range_max.text())

    #go to next index, used in addPlot, so that one can add spectrum without selecting everytime the pad where to draw
    def nextIndex(self):
        self.logger.info('nextIndex')
        tabIndex = self.wTab.currentIndex()

        #Try to deal with all cases... not elegant
        #first case when ex: coming back from zoom mode or if no plot selected
        if self.currentPlot.selected_plot_index is None:
            if self.wTab.selectedPad(tabIndex) is not None:
                self.forNextIndex(tabIndex,self.currentPlot.next_plot_index)
            else :
                self.currentPlot.index = self.check_index()
                self.wTab.setSelectedPad(tabIndex, self.currentPlot.index)
                self.currentPlot.next_plot_index = self.setIndex(self.currentPlot.next_plot_index)

        #second case when select a plot before clicking "Add"
        elif self.currentPlot.selected_plot_index == self.currentPlot.next_plot_index:
            self.currentPlot.index = self.currentPlot.selected_plot_index
            self.wTab.setSelectedPad(tabIndex, self.currentPlot.selected_plot_index)
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
        self.wTab.setSelectedPad(tabIndex, self.currentPlot.selected_plot_index)
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


    # Copy Properties lives on CopyPropertiesController. These stay because
    # they are what the popup's buttons are connected to.

    #callback for copyAttr.okAttr
    def okCopy(self):
        self.copy_props.okCopy()

    #callback for copyAttr.applyAttr
    def applyCopy(self):
        self.copy_props.applyCopy()

    #callback for copyAttr.cancelAttr
    def closeCopy(self):
        self.copy_props.closeCopy()

    #open copy properties popup
    def copyPopup(self):
        self.copy_props.copyPopup()

    #callback on press of a target spectrum button
    def connectCopy(self, instance):
        self.copy_props.connectCopy(instance)

    def selectAll(self):
        self.copy_props.selectAll()

    def histAllAttr(self, b):
        self.copy_props.histAllAttr(b)



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
    # Fit methods — FitManager does the work (see gui/services/fit_manager.py).
    # What is left here gathers the popup fields and the current-plot context
    # the service is not allowed to read for itself.
    # ------------------------------------------------------------------
    def fit(self):
        return self.fit_manager.fit(*self._current_plot_ctx(), *self._fit_inputs())
    def deleteFit(self):
        return self.fit_manager.deleteFit(*self._current_plot_ctx(),
                                          self.extraPopup.delete_fitIdx_list.text())
    def printFitLineLabels(self): return self.fit_manager.printFitLineLabels(*self._current_plot_ctx())
    def setFitLineLabel(self, ax, line, resultsText, spectrumName):
        return self.fit_manager.setFitLineLabel(ax, line, resultsText, spectrumName)
    def setFitResultsLineLabel(self, fitLineLabelIdx, resultsText, spectrumName):
        return self.fit_manager.setFitResultsLineLabel(fitLineLabelIdx, resultsText, spectrumName)
    def axisLimitsForFit(self, ax):
        return self.fit_manager.axisLimitsForFit(
            ax, self.extraPopup.fit_range_min.text(), self.extraPopup.fit_range_max.text())

    # Gate methods live entirely on GateManager now (gui/services/gate_manager.py);
    # call it directly. The gate popup is wired to it in _wire_signals.

    # ------------------------------------------------------------------
    # SumRegion methods — SumRegionManager does the work
    # (see gui/services/sum_region_manager.py). What is left here reads the
    # region-name combo and the current-plot context on its behalf.
    # ------------------------------------------------------------------
    def setSumRegion(self, index, line):
        name = self.view_state.nameFromIndex(index)
        return self.sum_region_manager.setSumRegion(index, line, name)
    def getSumRegion(self, index):
        name = self.view_state.nameFromIndex(index)
        return self.sum_region_manager.getSumRegion(index, name)
    def deleteSumRegionDict(self, label):
        return self.sum_region_manager.deleteSumRegionDict(label, self.currentPlot.figure.axes)
    def saveSumRegion(self, index):
        name = self.view_state.nameFromIndex(index)
        return self.sum_region_manager.saveSumRegion(
            index, name, self.sumRegionPopup.sumRegionNameList.currentText())
    def createSumRegion(self):                   return self.sum_region_manager.createSumRegion(*self._current_plot_ctx())
    def okSumRegion(self):
        return self.sum_region_manager.okSumRegion(
            self.sumRegionPopup.sumRegionNameList.currentText())
    def cleanPopupExit(self, doClose=True):
        return self.sum_region_manager.cleanPopupExit(doClose, self.sumRegionPopup.isVisible())
    def deleteSumRegion(self):
        return self.sum_region_manager.deleteSumRegion(
            *self._current_plot_ctx(), self.sumRegionPopup.sumRegionNameList.currentText())
    def integrate(self):                         return self.sum_region_manager.integrate(*self._current_plot_ctx())
    def copySelectionIntegrateTable(self):
        # the integrate table lives here now; this reads it + writes clipboard.
        resultTable = self.integratePopup.resultsText
        if not resultTable.selectedItems():
            return
        allValues = []
        for irow in range(resultTable.rowCount()):
            rowValues = []
            for icol in range(resultTable.columnCount()):
                item = resultTable.item(irow, icol)
                rowValues.append(item.text() if item else "")
            if rowValues:
                allValues.append("\t".join(rowValues))
        QApplication.clipboard().setText("\n".join(allValues))
    def integrateGateLocal(self, idx, lines):
        name = self.view_state.nameFromIndex(idx)
        return self.sum_region_manager.integrateGateLocal(idx, name, lines)

    # -- Connect popup adapters: the popup widget and its field reads live
    # here, the service takes plain arguments. Everything else on
    # ConnectionManager is called directly.
    def connectShMem(self):
        return self.connection_manager.connectShMem(
            str(self.connectConfig.server.text()),
            str(self.connectConfig.rest.text()),
            str(self.connectConfig.user.text()),
            str(self.connectConfig.mirror.text()),
        )
    def connectPopup(self):
        self.logger.info('callback connectPopup')
        self.connectConfig.show()
    def okConnect(self):
        self.logger.info('okConnect')
        self.connectShMem()
        self.closeConnect()
    def closeConnect(self):
        self.logger.info('closeConnect callback')
        self.connectConfig.close()
    def autoUpdateStart(self):           return self.connection_manager.autoUpdateStart(self.wConf.autoUpdate2.currentIndex())



    ############################
    # 13) Peak Finding
    ############################

    # Peak Finder 1 lives on PeakScanController. These stay because they are
    # the targets of .connect() calls in bindDynamicSignal; the list handler
    # also has to keep MainWindow's signature, since Qt hands it the item.

    def analyzePeak(self):
        self.peak_scan_controller.analyzePeak()

    def peakAnalClear(self):
        self.peak_scan_controller.peakAnalClear()

    def peakItemChanged(self, item):
        self.peak_scan_controller.peakItemChanged(item)

    def setAllPeaksChecked(self, checked):
        self.peak_scan_controller.setAllPeaksChecked(checked)


    ############################
    # 14b) Peak Finder 2 — click-to-fit (gaussian + linear background)
    ############################

    # ---- Peak Finder 2, stage 8b ------------------------------------
    # Arming and canvas connections moved to PeakFit2Controller. Four of these
    # are .connect() targets in _wire_signals; the other four are called by
    # the 8c/8d methods still on this class.

    # ---- Peak Finder 2 ----------------------------------------------
    # The whole cluster lives on PeakFit2Controller; what remains here is the
    # set _wire_signals connects, plus peakFit2RedrawAll, kept as the entry
    # point a future enlarge/redraw fix will need.

    # The two label maps below are the controller's dicts by reference, not
    # copies, so the reverse lookup in _peak2_sync_menus_to_spec cannot drift
    # from the forward one.
    _PEAK2_SIGNAL_BY_LABEL = PeakFit2Controller._PEAK2_SIGNAL_BY_LABEL
    _PEAK2_BG_BY_LABEL = PeakFit2Controller._PEAK2_BG_BY_LABEL



    # ---- Peak Finder 2, stage 8c ------------------------------------
    # Press dispatch and drag moved to PeakFit2Controller, and the arming
    # flags and drag context went with them (8b's four seams are gone).
    # These shims remain only because the 8d methods still on this class call
    # them; they come off with that group.







    def peakFit2Toggle(self, checked):
        self.peak_fit2_controller.peakFit2Toggle(checked)

    def peakFit2FixToggle(self, checked):
        self.peak_fit2_controller.peakFit2FixToggle(checked)





    def _peak2_load_shape_menus(self):
        self.peak_fit2_controller._peak2_load_shape_menus()

    def _peak2_shape_changed(self, *args):
        self.peak_fit2_controller._peak2_shape_changed(*args)





    def _peak2_delete_selected(self):
        self.peak_fit2_controller._peak2_delete_selected()

    def _peak2_row_selected(self):
        self.peak_fit2_controller._peak2_row_selected()




    def peakFit2Config(self):
        self.peak_fit2_controller.peakFit2Config()



    def peakFit2RedrawAll(self):
        self.peak_fit2_controller.peakFit2RedrawAll()

    def peakFit2Clear(self):
        self.peak_fit2_controller.peakFit2Clear()


    ############################
    # 15) Overlaying pic
    ############################


    # Image overlay lives on OverlayController. These stay because they are
    # what the Special Functions -> Imaging buttons and sliders are connected
    # to.

    def openFigureDialog(self):
        self.logger.info('openFigureDialog')
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Open file...", "","Image Files (*.png *.jpg);;All Files (*)", options=options)
        if fileName:
            return fileName

    def loadFigure(self):      self.overlay_controller.loadFigure()

    def addFigure(self):       self.overlay_controller.addFigure()

    def deleteFigure(self):    self.overlay_controller.deleteFigure()

    def fineUpMove(self):      self.overlay_controller.fineUpMove()

    def fineDownMove(self):    self.overlay_controller.fineDownMove()

    def fineLeftMove(self):    self.overlay_controller.fineLeftMove()

    def fineRightMove(self):   self.overlay_controller.fineRightMove()

    def moveFigure(self):      self.overlay_controller.moveFigure()

    def transFigure(self):     self.overlay_controller.transFigure()

    def zoomFigureX(self):     self.overlay_controller.zoomFigureX()

    def zoomFigureY(self):     self.overlay_controller.zoomFigureY()

    ############################
    # 16) Jupyter Notebook
    ############################

    # Jupyter lives on JupyterController. These stay because they are the
    # Special Functions button targets, and closeEvent calls jupyterStop on
    # the way out.

    def createDf(self):        self.jupyter_controller.createDf()

    def jupyterStart(self):    self.jupyter_controller.jupyterStart()

    def jupyterStop(self):     self.jupyter_controller.jupyterStop()

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






    def debugModeCallBack(self):
        if self.extraPopup.options.debugMode.isChecked():
            # record creation only while the file handler can consume it
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



# redirect logging





