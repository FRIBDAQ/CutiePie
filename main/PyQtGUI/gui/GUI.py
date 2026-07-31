#!/usr/bin/env python3
# import modules and packages

import sys, os, ast
import cv2
import logging, logging.handlers
import threading, time, re
from copy import deepcopy
import numpy as np

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
    QApplication, QDialog,
    QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QPushButton,
    QListWidgetItem, QShortcut, QTabBar,
    QTableWidgetItem, QVBoxLayout, QWidget,
)
from PyQt5.QtGui import QCursor, QKeySequence, QMouseEvent, QPalette
from PyQt5.QtCore import (
    pyqtSignal, pyqtSlot, Qt, QObject, QTimer,
    QSettings, QDir,
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
from services.display_slot import DisplaySlot, SLOT_KEYS
from services import geometry_io
from services.dataframe_export import export_spectrum_csv
from services.peak_finder import (
    PEAK_ALGORITHMS, find_peaks_in_range, format_peak_labels, format_peak_output,
    autocomponent_refit, find_duplicate_mu, fit_composite, fit_composite_auto,
    fix_peak_window, format_composite_fit_row, nearest_component_index,
    fwhm_to_sigma, nearest_window_edge, sigma_to_fwhm, validate_gauss_edit,
)
from services.figure_overlay import compute_overlay_position, apply_joystick_move, apply_fine_move
from services.log_throttle import LogThrottle
from services.fit_manager import FitManager
from services.gate_manager import GateManager
from services.sum_region_manager import SumRegionManager
from services.connection_manager import ConnectionManager
from services.plot_controller import PlotController
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

from logger import log, setup_logging, set_logger
from notebook_process import testnotebook, startnotebook, stopnotebook
from WebWindow import WebWindow


#from collapseMenu import Spoiler

SETTING_BASEDIR = "workdir"
SETTING_EXECUTABLE = "exec"
DEBUG = False


# Fallback for an installed tree, where configure.ac is not shipped. Keep this
# in sync with AC_INIT in configure.ac, which stays the source of truth.
_VERSION_FALLBACK = "v1.6-002"


def cutiepie_version():
    """CutiePie version for the window title.

    In a source checkout, read it live from configure.ac's AC_INIT line
    (configure.ac sits two levels up from this gui/ folder) so it always tracks
    the source of truth. In an installed tree — where configure.ac is not
    shipped — fall back to the hardcoded default above."""
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


class _NumericItem(QTableWidgetItem):
    """Peak Finder 2 results-table cell that sorts by a stored numeric value
    (Qt.UserRole) rather than its displayed '<value> ± <err>' string."""
    def __lt__(self, other):
        try:
            return float(self.data(QtCore.Qt.UserRole)) < float(other.data(QtCore.Qt.UserRole))
        except (TypeError, ValueError):
            return super().__lt__(other)

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
        # Single source of truth for root-logger config. Runs before
        # any setup_logging() call, so THIS is the config that takes effect:
        # default stderr handler at WARNING. (datefmt is inert here — the
        # default format carries no %(asctime)s.) setup_logging() only swaps the
        # logger.py sink; it no longer reconfigures root.
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

        self.fit_manager = FitManager(
            fit_factory=self.fit_factory,
            spectra=self.spectra,
            parent_widget=self,
            logger=self.logger,
        )

        self.gate_manager = GateManager(
            spectra=self.spectra,
            name_from_index=self.nameFromIndex,
            get_spectrum_info=self.getSpectrumViewInfo,
            get_is_enlarged=lambda: self.currentPlot.isEnlarged,
            get_geo=self.getGeo,
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
            name_from_index=self.nameFromIndex,
            get_spectrum_info=self.getSpectrumViewInfo,
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
            get_geo=self.getGeo,
            set_geo=self.setGeo,
            get_spectrum_info=self.getSpectrumViewInfo,
            set_spectrum_info=self.setSpectrumViewInfo,
            get_spectrum_info_dict=self.getSpectrumViewDict,
            name_from_index=self.nameFromIndex,
            get_enlarged_spectrum=self.getEnlargedSpectrum,
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

    def _init_runtime_state(self):
        """Per-session scratch state: peak finding, PF2, image overlay, gate-name cache."""
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

        # Peak Finder 2 (click-to-fit): armed-mode connection id, per-fit
        # records [{number, index, name, result, artists}, ...] (full curves
        # stored so fits can be redrawn/refit independent of artist survival)
        # and the running peak counter
        # per-canvas press-handler registry {canvas: cid}; the unified handler
        # stays connected wherever fits live so future drag/edit survive Stop
        self.peak2_conns = {}
        self.peak2_armed = False
        self.peak2_fix_armed = False
        self.peak2_drag = None   # active drag-to-refit context, or None
        self.peak2_fits = []
        self.peak2_count = 0

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

        self._gate_name_cache: dict = {}  # spectrum_name → (gate_or_None, monotonic_ts)
        self._gate_name_inflight: set = set()  # names with a background fetch running
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
        # lambdas shield the slots from clicked(bool)'s checked arg (the E17 trap:
        # a bare connect would pass False as the path/context)
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
        # shields Clear from clicked(bool)'s checked arg (the E17 trap)
        self.extraPopup.peak.peak2_start.toggled.connect(self.peakFit2Toggle)
        self.extraPopup.peak.peak2_fix.toggled.connect(self.peakFit2FixToggle)
        self.extraPopup.peak.peak2_delete.clicked.connect(lambda: self._peak2_delete_selected())
        self.extraPopup.peak.peak2_clear.clicked.connect(lambda: self.peakFit2Clear())
        self.extraPopup.peak.peak2_config.clicked.connect(lambda: self.peakFit2Config())
        self.extraPopup.peak.peak2_table.itemSelectionChanged.connect(self._peak2_row_selected)
        # shape menus: restore the last-used selection, then persist on change.
        # E2 only persists here (no self-update yet); E3 extends the slot to
        # re-fit the selected fit.
        self._peak2_load_shape_menus()
        self.extraPopup.peak.peak2_signal.currentIndexChanged.connect(self._peak2_shape_changed)
        self.extraPopup.peak.peak2_bg.currentIndexChanged.connect(self._peak2_shape_changed)
        self.extraPopup.peak.peak2_cb_tail.currentIndexChanged.connect(self._peak2_shape_changed)
        # peak-selection list: wired ONCE here (the old per-scan
        # stateChanged.connect on the fixed checkbox grid stacked a duplicate
        # connection on every Scan); lambdas shield from clicked(bool)'s
        # checked arg (the E17 trap — All would otherwise receive False)
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
            else:
                # The pad names a spectrum the store does not have — deleted
                # server-side, or a geometry naming something SpecTcl never sent.
                # Without this the two labels keep the previous pad's text and
                # the readout reports a spectrum the pointer is not over.
                self._blankHoverLabels()
                return
            gateName = self.getAppliedGateName(index=index)
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
        if self.getEnlargedSpectrum():
            index = self.getEnlargedSpectrum()[0]
        try:
            ax = self.getSpectrumViewInfo("axis", index=index)
            dim = self.getSpectrumStoreInfo("dim", index=index)
            minx = self.getSpectrumStoreInfo("minx", index=index)
            maxx = self.getSpectrumStoreInfo("maxx", index=index)
            binx = self.getSpectrumStoreInfo("binx", index=index)
            data = self.getSpectrumStoreInfo("data", index=index)
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
                    miny = self.getSpectrumStoreInfo("miny", index=index)
                    maxy = self.getSpectrumStoreInfo("maxy", index=index)
                    biny = self.getSpectrumStoreInfo("biny", index=index)
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
        axisIsLog = self.getSpectrumViewInfo("log", index=index)
        wPlot = self.currentPlot
        logBut = wPlot.logButton
        if axisIsLog :
            logBut.setDown(True)
        else :
            logBut.setDown(False) 
        # similar to log button, cutoff button change status according to spectrum info
        cutoffVal = self.getSpectrumViewInfo("cutoff", index=index)
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
            name = self.nameFromIndex(idx)
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
                self.wTab.setSelectedPad(self.wTab.currentIndex(), deepcopy(idx))
                self.logger.debug('on_dblclick - selectedPad: %s', self.wTab.selectedPad(self.wTab.currentIndex()))
                
                # t1 = time.time()
                ############### Bashir ##################################################
                # Save the current axes objects
                self.currentPlot._saved_axes = self.currentPlot.figure.axes.copy()

                # Hide all current axes
                for ax in self.currentPlot._saved_axes:
                    ax.set_visible(False)
                
                dim = self.getSpectrumStoreInfo("dim", index=idx)
                if dim == 2:
                    spectrum_old = self.getSpectrumViewInfo("spectrum", index=idx)
                    self.plot_controller.old_cmap = spectrum_old.get_cmap()

                elif dim == 1:
                    # Save current y-limits of the target axes to restore later (when autoscale is OFF)
                    ax0 = self.getSpectrumViewInfo("axis", index=idx)
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

                ax = self.getSpectrumViewInfo("axis", index=idx)
                if dim == 1:
                    if not autoscale_status and hasattr(self.currentPlot, "_saved_ylims") and idx in self.currentPlot._saved_ylims:
                        ax.set_ylim(*self.currentPlot._saved_ylims[idx])   # <-- restore y only
                # self.updatePlot()
                # t2 = time.time()
                # print("on_dblclick: time={:.2f}".format(t2-t1))

                ########### Bashir: reuse color map
                if dim == 2:
                    spectrum = self.getSpectrumViewInfo("spectrum", index=idx)

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
                tempIdxEnlargedSpectrum = self.getEnlargedSpectrum()[0]
                self.setEnlargedSpectrum(None, None)
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
                for index, name in self.getGeo().items():

                    # if name is not None and name != "" and name != "empty":
                    if name is not None and name != "" and name != "empty" and index == idx:
                        self.plot_controller.add(index)
                        ax = self.getSpectrumViewInfo("axis", index=index)
                        
                        #reset the axis limits as it was before enlarge
                        #dont need to specify if log scale, it is checked inside setAxisScale, if 2D histo in log its z axis is set too.
                        dim = self.getSpectrumStoreInfo("dim", index=index)
                        if dim == 1:
                            self.plot_controller.plotPlot(index)
                            if not autoscale_status and hasattr(self.currentPlot, "_saved_ylims") and index in self.currentPlot._saved_ylims:
                                ax.set_ylim(*self.currentPlot._saved_ylims[index])   # <-- restore y only
                            else:
                                if autoscale_status:
                                    self.plot_controller.setAxisScale(ax, index, "x", "y")

                        elif dim == 2:
                            self.plot_controller.plotPlot(index, self.plot_controller.old_cmap)
                            self.setSpectrumViewInfo(cmap=self.plot_controller.old_cmap, index=idx)
                            # if autoscale_status:
                            self.plot_controller.setAxisScale(ax, index, "x", "y", "z")
                        self.gate_manager.drawGate(index)
                #drawing back the dashed red rectangle on the unenlarged spectrum
                self.plot_controller.removeRectangle()
                self.currentPlot.recDashed = self.plot_controller.createDashedRectangle(self.currentPlot.figure.axes[tempIdxEnlargedSpectrum])
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
                for indexPlot, name in self.getGeo().items():
                    if name:
                        ax = self.getSpectrumViewInfo("axis", index=indexPlot)
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
    def getSpectrumStoreInfo(self, *info, **identifier):
        # self.logger.info('getSpectrumStoreInfo - info, identifier: %s, %s',info, identifier)
        name = None
        if not identifier and self.getEnlargedSpectrum():
            name = self.getEnlargedSpectrum()[1]
        elif "index" in identifier:
            name = self.nameFromIndex(identifier["index"])
        elif "name" in identifier:
            name = identifier["name"]
        else:
            self.logger.debug('getSpectrumStoreInfo - wrong identifier - expects name=histo_name or index=histo_index or shoud be in zoomed mode')
            # print("getSpectrumViewInfo - wrong identifier - expects name=histo_name or index=histo_index or shoud be in zoomed mode")
            return
        if name is not None:
            return self.spectra.get(name, info[0])


    #Update spectrum info in tabSlots (identified by index and can update multiple info at once)
    #Important that only the per-tab slots dict is changed here
    #work in normal and enlarged mode
    def setSpectrumViewInfo(self, **info):
        self.logger.debug('setSpectrumViewInfo - info: %s',info)
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
            self.logger.debug('setSpectrumViewInfo - wrong identifier - expects index=histo_index or shoud be in zoomed mode')
            # print("setSpectrumViewInfo - wrong identifier - expects index=histo_index or shoud be in zoomed mode")
            return
        # print("Simon - setSpectrumViewInfo - ", index,name,info["index"])
        for key, value in info.items():
            if key in SLOT_KEYS and index is not None:
                if index not in self.wTab.tabSlots(self.wTab.currentIndex()):
                    self.logger.debug('setSpectrumViewInfo - %s not in tabSlots', name)
                    return
                slot = self.wTab.tabSlots(self.wTab.currentIndex())[index]
                setattr(slot, key, value)          # typed DisplaySlot field (key is whitelisted above)
                #set axes info at the same time than spectrum
                if key == "spectrum":
                    slot.axis = value.axes


    #Get spectrum info from tabSlots (identified by index and info name)
    #template of expected arguments e.g.: ("dim", index=5) takes only the first info parameter (here "dim") (one per call)
    #Important that it gets only the info from the current tab's slots here.
    #work in normal and enlarged mode
    def getSpectrumViewInfo(self, *info, **identifier):
        self.logger.debug('getSpectrumViewInfo - info, identifier: %s, %s', info, identifier)
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
            self.logger.debug('getSpectrumViewInfo - wrong identifier - expects index=histo_index or shoud be in zoomed mode')
            # print("getSpectrumViewInfo - wrong identifier - expects index=histo_index or shoud be in zoomed mode")
            return
        if index is not None and index in self.wTab.tabSlots(self.wTab.currentIndex()) and info[0] in SLOT_KEYS:
            return getattr(self.wTab.tabSlots(self.wTab.currentIndex())[index], info[0])   # typed access (info[0] whitelisted above)


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
    def getSpectrumViewDict(self):
        return self.wTab.tabSlots(self.wTab.currentIndex())


    #get full spectrum dict from self.spectra:
    def getSpectrumStoreDict(self):
        return self.spectra.as_dict()
 

    #Find name with geo index:
    def nameFromIndex(self, index):
        # self.logger.info('nameFromIndex - index: %s', index)
        #Can call getSpectrumViewInfo and setSpectrumViewInfo with an identifier but still check if in zoom mode,
        #which is important for autoScaleAxis/setAxisScale
        if self.getEnlargedSpectrum():
            return self.getEnlargedSpectrum()[1]
        elif index in self.currentPlot.h_dict_geo:
            return self.currentPlot.h_dict_geo[index]


    #sets h_dict_geo {key=index, value=histoName}
    #Use only this function to set the geometry dict when add plot (against using it elsewhere because it initializes per-tab slots)
    def setGeo(self, index, name):
        self.logger.info('setGeo - index, name: %s, %s', index, name)
        self.currentPlot.h_dict_geo[index] = name
        #Set also here the per-tab slots with only the spectra defined in the geo
        if index not in self.wTab.tabSlots(self.wTab.currentIndex()):
            self.wTab.tabSlots(self.wTab.currentIndex())[index] = DisplaySlot()
        slot = self.wTab.tabSlots(self.wTab.currentIndex())[index]
        slot.name = name
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
            setattr(slot, key, value)          # typed DisplaySlot field


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
        self.wTab.setZoomInfo(self.wTab.currentIndex(), None)
        if index is not None and name is not None:
            self.wTab.setZoomInfo(self.wTab.currentIndex(), [index, name])

    def getEnlargedSpectrum(self):
        # self.logger.info('getEnlargedSpectrum')
        result = None
        if self.wTab.zoomInfo(self.wTab.currentIndex()):
            result = self.wTab.zoomInfo(self.wTab.currentIndex())
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


    def saveGeo(self):
        fileName = self.saveFileDialog()
        self.logger.info('saveGeo - fileName: %s', fileName)
        if not fileName:
            return
        try:
            properties = {}
            geo = self.getGeo()
            for index in range(len(geo)):
                try:
                    h_name = geo[index]
                    x_range, y_range = self.plot_controller.getAxisProperties(index)
                    scale = True if self.getSpectrumViewInfo("log", index=index) else False
                    properties[index] = {"name": h_name, "x": x_range, "y": y_range, "scale": scale}
                except Exception:
                    self.logger.debug('saveGeo - pad %s skipped', index, exc_info=True)
                    properties[index] = {"name": '', "x": None, "y": None, "scale": None}
            ##### Bashir changed to examine the apply button
            tmp_text = geometry_io.serialize_geometry(
                self.wConf.histo_geo_row.currentText(),
                self.wConf.histo_geo_col.currentText(),
                properties)
            #######################################################################
            with open(fileName, "w") as f:
                f.write(tmp_text)
        except Exception:
            # was a bare `except:` that logged at debug and still showed
            # the success dialog (shown before the write, at that)
            self.logger.exception('saveGeo - failed to save %s', fileName)
            QMessageBox.warning(self, "Saving...", "Could not save the window configuration — see the log.")
            return
        QMessageBox.about(self, "Saving...", "Window configuration saved!")


    def saveGeoAll(self):
        """Save EVERY tab's geometry as one v2 session file (design
        2026-07-10). Reads the per-tab VIEW tier (slots), not live axes —
        background tabs' axes aren't reliably current, and the view tier is
        exactly what load consumes. (Single-tab saveGeo keeps its live-axes
        read — deliberate asymmetry.) Enlarged state is transient: never saved."""
        fileName = self.saveFileDialog()
        self.logger.info('saveGeoAll - fileName: %s', fileName)
        if not fileName:
            return
        try:
            tabs = []
            for tabIdx in sorted(self.wTab.sessions.indices()):
                nRow, nCol = self.wTab.tabLayout(tabIdx)
                plotW = self.wTab.plot(tabIdx)
                slots = self.wTab.tabSlots(tabIdx) if tabIdx in self.wTab.sessions else {}
                properties = {}
                for index in range(nRow * nCol):
                    try:
                        h_name = plotW.h_dict_geo.get(index, "")
                        slot = slots.get(index)
                        x_range = y_range = None
                        scale = False
                        if slot is not None:
                            # slot fields default to [] (DisplaySlot); 0.0 is a
                            # legitimate limit, so test emptiness, not truthiness
                            if slot.minx not in ("", [], None) and slot.maxx not in ("", [], None):
                                x_range = [float(slot.minx), float(slot.maxx)]
                            if slot.miny not in ("", [], None) and slot.maxy not in ("", [], None):
                                y_range = [float(slot.miny), float(slot.maxy)]
                            scale = bool(slot.log) if slot.log not in ([], None) else False
                        properties[index] = {"name": h_name if h_name != "empty" else "",
                                             "x": x_range, "y": y_range, "scale": scale}
                    except Exception:
                        self.logger.debug('saveGeoAll - tab %s pad %s skipped',
                                          tabIdx, index, exc_info=True)
                        properties[index] = {"name": '', "x": None, "y": None, "scale": None}
                tabs.append({"name": self.wTab.tabText(tabIdx), "row": nRow,
                             "col": nCol, "geo": properties})
            tmp_text = geometry_io.serialize_session(tabs)
            with open(fileName, "w") as f:
                f.write(tmp_text)
        except Exception:
            self.logger.exception('saveGeoAll - failed to save %s', fileName)
            QMessageBox.warning(self, "Saving...", "Could not save the session — see the log.")
            return
        QMessageBox.about(self, "Saving...", "All tabs saved!")


    def _resolveSpectrumName(self, name):
        """Return a spectrum name present in the store that matches `name`, tolerating
        case differences (legacy .win files often store names upper-cased). Returns the
        exact name if it exists, a unique case-insensitive match otherwise, or None."""
        if self.getSpectrumStoreInfo("dim", name=name) is not None:
            return name
        lowered = name.lower()
        matches = [n for n in self.spectra.all_names() if n.lower() == lowered]
        return matches[0] if len(matches) == 1 else None

    def _applyGeometryToCurrentTab(self, infoGeo):
        """Apply one tab's geometry payload ({"row","col","geo"}) to the
        CURRENT tab. Extracted verbatim from loadGeo so the single-tab load
        and the session load (loadGeoAll) run the same code. Returns the
        list of spectrum names that could not be resolved."""
        ### Bashir added -1
        row = infoGeo["row"] - 1
        col = infoGeo["col"] - 1
        # change index in combobox to the actual loaded values
        #### Bashir changed to examine the apply button
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
                self.setSpectrumViewInfo(log=val_dict["scale"], index=index)
                # Old .win files may omit the view range (no "Expanded"); when it
                # is absent the spectrum keeps its natural full range from the store.
                if val_dict.get("x") is not None:
                    self.setSpectrumViewInfo(minx=val_dict["x"][0], index=index)
                    self.setSpectrumViewInfo(maxx=val_dict["x"][1], index=index)
                if val_dict.get("y") is not None:
                    self.setSpectrumViewInfo(miny=val_dict["y"][0], index=index)
                    self.setSpectrumViewInfo(maxy=val_dict["y"][1], index=index)

            if len(notFound) > 0:
                self.logger.warning('loadGeo - definition not found for: %s', notFound)

            self.currentPlot.isLoaded = True
            self.wTab.setSelectedPad(self.wTab.currentIndex(), None)
            self.currentPlot.selected_plot_index = None
            self.currentPlot.next_plot_index = -1

        self.addPlot()
        self.plot_controller.updatePlot()
        self.currentPlot.isLoaded = False
        return notFound


    def loadGeo(self):
        fileName = self.openFileNameDialog()
        self.logger.info('loadGeo - fileName: %s', fileName)
        if not fileName:
            return
        # Detect the format instead of assuming a single-tab file: a multi-tab
        # session dropped here used to crash with KeyError 'row'.
        tagged = geometry_io.read_geometry_any(fileName, self.logger)
        if tagged is None:
            QMessageBox.warning(self, "Load Geometry",
                                "Not a readable geometry file — nothing was changed.")
            return
        kind, payload = tagged
        if kind == "session":
            nTabs = len(payload.get("tabs") or [])
            reply = QMessageBox.question(
                self, "Load Geometry",
                f"This file is a multi-tab session ({nTabs} tabs). "
                "Load all tabs? This replaces your current tabs.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self._applySession(payload["tabs"])
            return
        self._applyGeometryToCurrentTab(payload)


    def loadGeoAll(self):
        """Load a session file, REPLACING all tabs (user-approved semantics,
        design 2026-07-10). The file is parsed and validated COMPLETELY before
        any tab is touched, so a bad file can never half-destroy the
        workspace. A v1 single-tab file here loads as a one-tab session."""
        fileName = self.openFileNameDialog()
        self.logger.info('loadGeoAll - fileName: %s', fileName)
        if not fileName:
            return
        tagged = geometry_io.read_geometry_any(fileName, self.logger)
        if tagged is None:
            QMessageBox.warning(self, "Load All Tabs",
                                "Not a readable geometry/session file — nothing was changed.")
            return
        kind, payload = tagged
        try:
            if kind == "single":
                tabsInfo = [{"name": "Tab 1", "row": int(payload["row"]),
                             "col": int(payload["col"]), "geo": payload["geo"]}]
            else:
                tabsInfo = payload["tabs"]
        except (KeyError, TypeError, ValueError):
            QMessageBox.warning(self, "Load All Tabs",
                                "Geometry file is missing required fields — nothing was changed.")
            return

        self._applySession(tabsInfo)


    def _applySession(self, tabsInfo):
        """Replace ALL tabs with the parsed session `tabsInfo` (a list of
        {"name","row","col","geo"}). Shared by loadGeoAll and loadGeo's
        session branch. Callers must have parsed + validated the file first —
        this only mutates the workspace, so it can never half-destroy it on a
        bad file."""
        # quiesce: same guards clickedTab uses, then stop the auto-update tick
        if self.currentPlot.toCreateGate or self.currentPlot.toEditGate or self.gatePopup.isVisible():
            self.gate_manager.cancelGate()
        if self.currentPlot.toCreateSumRegion or self.sumRegionPopup.isVisible():
            self.sum_region_manager.cancelSumRegion()
        self.connection_manager._stop_auto_thread()

        # rebuild the tab set with existing primitives only (danger
        # zone: deleteTab reindexes the parallel dicts and plt.closes figures)
        self.wTab.setCurrentIndex(0)
        self.currentPlot = self.wTab.plot(0)
        while len(self.wTab.sessions) > 1:
            self.wTab.deleteTab(len(self.wTab.sessions) - 1)
        for k in range(1, len(tabsInfo)):
            self.wTab.addTab(k)

        notFoundByTab = {}
        for k, tabInfo in enumerate(tabsInfo):
            self.wTab.setCurrentIndex(k)
            self.tabGeoWidgetAndFlags(k)
            infoGeo = {"row": tabInfo["row"], "col": tabInfo["col"], "geo": tabInfo["geo"]}
            try:
                notFound = self._applyGeometryToCurrentTab(infoGeo)
            except TypeError:
                self.logger.debug('_applySession - TypeError applying tab %s', k, exc_info=True)
                notFound = []
            if notFound:
                notFoundByTab[str(tabInfo.get("name") or f"Tab {k+1}")] = notFound
            self.wTab.setTabText(k, str(tabInfo.get("name") or f"Tab {k+1}"))

        self.wTab.setCurrentIndex(0)
        self.tabGeoWidgetAndFlags(0)
        self.bindDynamicSignal()
        if notFoundByTab:
            lines = "\n".join(f"{tab}: {', '.join(names)}"
                              for tab, names in notFoundByTab.items())
            QMessageBox.warning(self, "Load All Tabs",
                                "Some spectra were not found on the connected SpecTcl; "
                                "their pads were left empty:\n\n" + lines)
        self.autoUpdateStart()


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
        return idx, self.nameFromIndex(idx), self.getSpectrumViewInfo("axis", index=idx)

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


    #callback for copyAttr.okAttr
    def okCopy(self):
        self.logger.info('okCopy')
        self.applyCopy()
        self.closeCopy()

    #callback for copyAttr.applyAttr
    def applyCopy(self):
        self.logger.info('applyCopy')
        try:
            # read each property checkbox by name; a positional list built from
            # findChildren() would silently re-point if CopyProperties ever
            # reorders or gains a checkbox. histoAll is the master toggle, not a
            # property, so it is not one of these.
            copy_xlim  = self.copyAttr.axisLimitX.isChecked()
            copy_ylim  = self.copyAttr.axisLimitY.isChecked()
            copy_scale = self.copyAttr.axisScale.isChecked()
            copy_minz  = self.copyAttr.histoScaleminZ.isChecked()
            copy_maxz  = self.copyAttr.histoScalemaxZ.isChecked()

            self.logger.debug('applyCopy - x: %s, y: %s, scale: %s, minz: %s, maxz: %s',
                              copy_xlim, copy_ylim, copy_scale, copy_minz, copy_maxz)

            dim = self.getSpectrumStoreInfo("dim", index=self.currentPlot.selected_plot_index)
            indexes = []
            xlim_src = []
            ylim_src = []
            zlim_src = []
            scale_src = None

            # creating list of target histograms
            discard = ["Ok", "Cancel", "Apply", "Select all", "Deselect all"]
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

            # autoscale off, or the trailing updatePlot recomputes y/z from the
            # data and discards the copied values (same pattern as
            # zoomInOut / cutoffButtonCallback)
            self.currentPlot.histo_autoscale.setChecked(False)

            # copy to destination
            for index in indexes:
                # the target axes are read up front because the y bottom has to
                # be clamped before it is stored, not only before it is drawn: a
                # linear source pad reports a zero or slightly negative bottom,
                # and a log-scaled target rejects that outright (matplotlib warns
                # and keeps its own, so only half the range copies). Clamp to the
                # same floor setAxisScale uses for its log branch. With no axes
                # yet the scale is unknowable, so the raw value is stored and
                # setAxisScale clamps it on read as before.
                ax = self.getSpectrumViewInfo("axis", index=index)
                ymin_dst, ymax_dst = ylim_src[0], ylim_src[1]
                if ax is not None and ax.get_yscale() == "log" and ymin_dst <= 0:
                    ymin_dst = 0.001

                # set the limits for x,y
                if copy_xlim:
                    self.setSpectrumViewInfo(minx=xlim_src[0], index=index)
                    self.setSpectrumViewInfo(maxx=xlim_src[1], index=index)
                if copy_ylim:
                    self.setSpectrumViewInfo(miny=ymin_dst, index=index)
                    self.setSpectrumViewInfo(maxy=ymax_dst, index=index)
                # set log/lin scale
                if copy_scale:
                    self.setSpectrumViewInfo(log=scale_src_bool, index=index)
                # set minZ/maxZ (either box copies both bounds)
                if dim == 2 and (copy_minz or copy_maxz):
                    self.setSpectrumViewInfo(minz=zlim_src[0], index=index)
                    self.setSpectrumViewInfo(maxz=zlim_src[1], index=index)
                # apply to the target axes directly: updatePlot's only
                # limits-application path is autoscale-gated, so view-tier
                # writes alone never reach the screen (okCutoff precedent)
                if ax is None:
                    continue
                if copy_xlim:
                    ax.set_xlim(xlim_src[0], xlim_src[1])
                if copy_ylim:
                    ax.set_ylim(ymin_dst, ymax_dst)
                if dim == 2 and (copy_minz or copy_maxz):
                    spectrum = self.getSpectrumViewInfo("spectrum", index=index)
                    if spectrum is not None:
                        spectrum.set_clim(zlim_src[0], zlim_src[1])
                if copy_scale:
                    self.plot_controller.setAxisScale(ax, index, "log")
            self.plot_controller.updatePlot()
        except Exception:
            self.logger.exception('applyCopy - copy properties failed')


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
        dim = self.getSpectrumStoreInfo("dim", index=index)

        if dim is None : 
            self.logger.debug('copyPopup - dim is None', exc_info=True)
            return

        # setting up info for source histogram
        self.copyAttr.histoLabel.setText(name)
        # hdim = 2 if self.wConf.button2D.isChecked() else 1
        if dim == 2 :
            spectrum = self.getSpectrumViewInfo("spectrum", index=index)
            zmin, zmax = spectrum.get_clim()
            self.copyAttr.histoScaleValueminZ.setText(f"{zmin}")
            self.copyAttr.histoScaleValuemaxZ.setText(f"{zmax}")
        self.copyAttr.axisSLabel.setText("Log" if self.getSpectrumViewInfo("log", index=index) else "Linear")
        xmin = self.getSpectrumViewInfo("minx", index=index)
        xmax = self.getSpectrumViewInfo("maxx", index=index)
        ymin = self.getSpectrumViewInfo("miny", index=index)
        ymax = self.getSpectrumViewInfo("maxy", index=index)
        self.copyAttr.axisLimLabelX.setText(f"[{xmin:.1f},{xmax:.1f}]")
        self.copyAttr.axisLimLabelY.setText(f"[{ymin:.1f},{ymax:.1f}]")

        #reset QFormLayout
        rowCount = self.copyAttr.copy_log.rowCount()
        for i in range(rowCount) :
            self.copyAttr.copy_log.removeRow(0)

        try:
            for idx, nameTarget in self.getGeo().items():
                if dim == self.getSpectrumStoreInfo("dim", index=idx) and idx != index:
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

        dim = self.getSpectrumStoreInfo("dim", index=self.currentPlot.selected_plot_index)

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
        name = self.nameFromIndex(index)
        return self.sum_region_manager.setSumRegion(index, line, name)
    def getSumRegion(self, index):
        name = self.nameFromIndex(index)
        return self.sum_region_manager.getSumRegion(index, name)
    def deleteSumRegionDict(self, label):
        return self.sum_region_manager.deleteSumRegionDict(label, self.currentPlot.figure.axes)
    def saveSumRegion(self, index):
        name = self.nameFromIndex(index)
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
        name = self.nameFromIndex(idx)
        return self.sum_region_manager.integrateGateLocal(idx, name, lines)

    # -- Connect popup adapters: the popup widget and its field reads live
    #    here, the service takes plain arguments. Everything else on
    #    ConnectionManager is called directly. --
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

    def _syncPeakMarker(self, row, checked):
        """Draw or remove one peak's markers to match its list check state."""
        if not checked and self.isChecked.get(row, False):
            try:
                self.removePeak(row)
            except Exception:
                self.logger.debug('_syncPeakMarker - peak artist cleanup failed', exc_info=True)
            self.isChecked[row] = False
        elif checked and not self.isChecked.get(row, False):
            self.drawSinglePeaks(self.peaks, self.properties, self.datay, row)
            self.isChecked[row] = True

    def peakItemChanged(self, item):
        self.logger.info('peakItemChanged')
        row = self.extraPopup.peak.peak_list.row(item)
        self._syncPeakMarker(row, item.checkState() == Qt.Checked)
        self.currentPlot.canvas.draw()

    def setAllPeaksChecked(self, checked):
        self.logger.info('setAllPeaksChecked - checked: %s', checked)
        peak_list = self.extraPopup.peak.peak_list
        state = Qt.Checked if checked else Qt.Unchecked
        # block itemChanged while flipping the states, then sync the markers
        # in one pass with a single canvas redraw
        peak_list.blockSignals(True)
        try:
            for row in range(peak_list.count()):
                peak_list.item(row).setCheckState(state)
        finally:
            peak_list.blockSignals(False)
        for row in range(peak_list.count()):
            self._syncPeakMarker(row, checked)
        self.currentPlot.canvas.draw()

    def populatePeakList(self):
        """Rebuild the checkable peak list, one row per found peak (no cap —
        replaces the fixed 12-checkbox grid), and draw every marker checked."""
        self.logger.info('populatePeakList')
        # drop markers left over from a previous Scan (the old grid redrew
        # over its stale artist handles, leaking them onto the canvas)
        self.setAllPeaksChecked(False)
        peak_list = self.extraPopup.peak.peak_list
        peak_list.blockSignals(True)
        try:
            peak_list.clear()
            labels = format_peak_labels(self.peaks, self.properties, self.datax)
            for i, label in enumerate(labels):
                item = QListWidgetItem(label)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                peak_list.addItem(item)
                self.isChecked[i] = False
        finally:
            peak_list.blockSignals(False)
        for i in range(len(self.peaks)):
            self._syncPeakMarker(i, True)
        self.currentPlot.canvas.draw()

    def peakAnalClear(self):
        self.logger.info('peakAnalClear')
        self.extraPopup.peak.peak_results.clear()
        self.removeAllPeaks()
        peak_list = self.extraPopup.peak.peak_list
        peak_list.blockSignals(True)
        try:
            peak_list.clear()
        finally:
            peak_list.blockSignals(False)
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
        self.setAllPeaksChecked(False)


    def drawSinglePeaks(self, peaks, properties, data, index):
        self.logger.info('drawSinglePeaks - index, properties: %s, %s', index, properties)
        ax = self.getSpectrumViewInfo("axis", index=self.currentPlot.selected_plot_index)
        x = self.datax.tolist()
        self.peak_pos[index] = ax.plot(x[peaks[index]], int(data[peaks[index]]), "v", color="red")
        self.peak_vl[index] = ax.vlines(x=x[peaks[index]], ymin=data[peaks[index]] - properties["prominences"][index], ymax = data[peaks[index]], color = "red")
        self.peak_hl[index] = ax.hlines(y=properties["width_heights"][index], xmin=properties["left_ips"][index], xmax=properties["right_ips"][index], color = "red")
        self.peak_txt[index] = ax.text(x[peaks[index]], int(data[peaks[index]]*1.1), str(int(x[peaks[index]])))


    def update_peak_output(self, peaks, properties):
        self.logger.info('update_peak_output - len(peaks), properties: %s, %s', len(peaks), properties)
        for s in format_peak_output(peaks, properties, self.datax):
            self.extraPopup.peak.peak_results.append(s)

    def analyzePeak(self):
        self.logger.info('analyzePeak')
        try:
            index = self.currentPlot.selected_plot_index
            ax = self.getSpectrumViewInfo("axis", index=index)
            # input points for peak finding
            width = int(self.extraPopup.peak.peak_width.text())
            binx = self.getSpectrumStoreInfo("binx", index=index)
            minxREST = self.getSpectrumStoreInfo("minx", index=index)
            maxxREST = self.getSpectrumStoreInfo("maxx", index=index)

            xtmp = self.plot_controller.createRange(binx, minxREST, maxxREST)
            ytmp = (self.getSpectrumStoreInfo("data", index=index)).tolist()

            xmin, xmax = ax.get_xlim()
            algo_name = self.extraPopup.peak.peak_algo.currentText()
            finder = PEAK_ALGORITHMS.get(algo_name, find_peaks_in_range)
            self.logger.debug('analyzePeak - algo, xmin, xmax: %s, %s, %s',
                              algo_name, xmin, xmax)

            self.datax, self.datay, self.peaks, self.properties = \
                finder(xtmp, ytmp, xmin, xmax, width)

            self.update_peak_output(self.peaks, self.properties)
            self.populatePeakList()

        except Exception:
            # Peak analysis is best-effort: a bad width entry, an empty view,
            # or a find_peaks failure must not crash the GUI — but must not be
            # silent either (the user would see nothing happen with no clue why).
            self.logger.exception('analyzePeak - peak analysis failed')


    ############################
    # 14b) Peak Finder 2 — click-to-fit (gaussian + linear background)
    ############################

    def _peak2_connect(self, canvas):
        """Ensure the unified press handler is connected on `canvas`."""
        if canvas is None or canvas in self.peak2_conns:
            return
        self.peak2_conns[canvas] = canvas.mpl_connect(
            "button_press_event", self.onPeakFit2Press)

    def _peak2_disconnect(self, canvas):
        cid = self.peak2_conns.pop(canvas, None)
        if cid is not None:
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                self.logger.debug('_peak2_disconnect failed', exc_info=True)

    def _peak2_fit_canvases(self):
        """Canvases that currently hold at least one recorded fit's artists."""
        canvases = set()
        for rec in self.peak2_fits:
            for art in rec.get("artists") or ():
                try:
                    canvases.add(art.axes.figure.canvas)
                except Exception:
                    pass
        return canvases

    def _peak2_sync_connections(self):
        """Drop the handler from canvases that are neither armed nor holding
        fits; keep it wherever fits still live (so drag/edit will work after
        Stop)."""
        keep = self._peak2_fit_canvases()
        if self.peak2_armed or self.peak2_fix_armed:
            try:
                keep.add(self.wTab.plot(self.wTab.currentIndex()).canvas)
            except Exception:
                pass
        for canvas in list(self.peak2_conns):
            if canvas not in keep:
                self._peak2_disconnect(canvas)

    def _peak2_spectrum_arrays(self, name):
        """(xc, y) — bin-centre x and counts for the spectrum called `name`, or
        None. Mirrors the fit handler's array setup; used by drag-refit, the
        edit popup and the shape-menu refit.

        Keyed by NAME, never by pad index: a pad index only means anything
        against the tab that is currently showing (nameFromIndex reads
        currentPlot.h_dict_geo, and answers with the enlarged spectrum for any
        index while a pad is enlarged). A fit records the spectrum it was made
        on, and a refit triggered from the popup can happen while a different
        tab is up or after the geometry moved that spectrum, so resolving the
        index again would hand back a different spectrum's counts. The store is
        name-keyed and tab-independent, so this stays correct either way; an
        unknown name yields None and the callers report 'spectrum unavailable'."""
        # Ask the store outright rather than inferring absence from the TypeError
        # a None binx would raise downstream: a removed spectrum is an expected
        # state, not an error, and the explicit test cannot be defeated by a
        # record that survives removal with some fields still readable.
        if not self.spectra.contains(name):
            self.logger.debug('_peak2_spectrum_arrays - %s is no longer in the store', name)
            return None
        try:
            binx = self.getSpectrumStoreInfo("binx", name=name)
            minx = self.getSpectrumStoreInfo("minx", name=name)
            maxx = self.getSpectrumStoreInfo("maxx", name=name)
            xtmp = self.plot_controller.createRange(binx, minx, maxx)
            ytmp = self.getSpectrumStoreInfo("data", name=name)
            xc = np.asarray(xtmp[:-1]) + 0.5 * np.diff(np.asarray(xtmp))
            return xc, np.asarray(ytmp)[1:]
        except Exception:
            self.logger.debug('_peak2_spectrum_arrays failed for %s', name, exc_info=True)
            return None

    @staticmethod
    def _peak2_live_axes(rec):
        """The axes this fit is still drawn on, or None once the pad went away
        under it.

        Two teardowns have to be caught and they leave different wreckage.
        Removing a spectrum clears its pad (`_on_spectrum_removed_rest` calls
        `ax.clear()`), which sets every cleared artist's `.axes` to None.
        Applying a new geometry instead DETACHES the old axes
        (`InitializeCanvas` runs `figure.delaxes`), which leaves both
        `artist.axes` and `axes.figure` pointing at real objects and only drops
        the axes out of `figure.axes`. So a plain None test waves the geometry
        case straight through, and the refit then draws onto a pad that is no
        longer part of the figure — invisible, and reported as success.
        Attachment is the test that catches both."""
        arts = rec.get("artists") or ()
        ax = arts[0].axes if arts else None
        if ax is None or ax.figure is None:
            return None
        return ax if ax in ax.figure.axes else None

    def _peak2_try_grab(self, event):
        """Drag-to-refit: if the left-press landed within the pick radius of a
        fit's end-handle, start dragging that window edge. Returns True when a
        drag starts. Works whether or not Start is armed."""
        if self.peak2_drag is not None:
            return False
        ax = event.inaxes
        tol_px = 8.0
        for rec in self.peak2_fits:
            arts = rec.get("artists") or ()
            if not arts or arts[0].axes is not ax:
                continue
            xx = rec["result"].get("xx")
            if xx is None or len(xx) < 2:
                continue
            try:
                lo_px = ax.transData.transform((xx[0], 0.0))[0]
                hi_px = ax.transData.transform((xx[-1], 0.0))[0]
            except Exception:
                continue
            edge = nearest_window_edge(event.x, lo_px, hi_px, tol_px)
            if edge is None:
                continue
            (guide,) = ax.plot([event.xdata, event.xdata], list(ax.get_ylim()),
                               color="tab:green", lw=1.0, ls="--", zorder=5)
            canvas = ax.figure.canvas
            self.peak2_drag = dict(
                rec=rec, edge=edge, ax=ax, guide=guide,
                cid_move=canvas.mpl_connect("motion_notify_event", self._peak2_on_drag_motion),
                cid_up=canvas.mpl_connect("button_release_event", self._peak2_on_drag_release))
            self._peak2_status(f"[drag] Peak {rec['number']}: drag the {edge} edge, "
                               "release to refit.")
            canvas.draw_idle()
            return True
        return False

    def _peak2_on_drag_motion(self, event):
        """Move the dashed guide line to follow the cursor during a drag."""
        d = self.peak2_drag
        if d is None or event.inaxes is not d["ax"] or event.xdata is None:
            return
        try:
            d["guide"].set_xdata([event.xdata, event.xdata])
            d["ax"].figure.canvas.draw_idle()
        except Exception:
            self.logger.debug('_peak2_on_drag_motion failed', exc_info=True)

    def _peak2_on_drag_release(self, event):
        """Release: refit the fit with the dragged edge moved (other edge + μ
        seed kept). A drag is explicit — its failures are always reported and
        never cap-silenced; on failure the previous fit is kept untouched."""
        d = self.peak2_drag
        if d is None:
            return
        ax = d["ax"]
        canvas = ax.figure.canvas
        # tear down the drag interaction first
        for cid in (d["cid_move"], d["cid_up"]):
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                pass
        try:
            d["guide"].remove()
        except Exception:
            pass
        self.peak2_drag = None

        rec, edge = d["rec"], d["edge"]
        new_x = event.xdata
        if new_x is None:
            self._peak2_status(f"[drag] Peak {rec['number']}: cancelled (released off the pad).")
            canvas.draw_idle()
            return
        prev = rec["result"]
        xx = prev["xx"]
        lo, hi = ((float(new_x), float(xx[-1])) if edge == "lo"
                  else (float(xx[0]), float(new_x)))
        arrays = self._peak2_spectrum_arrays(rec["name"])
        if arrays is None:
            self._peak2_status(f"[drag] Peak {rec['number']}: spectrum unavailable.")
            canvas.draw_idle()
            return
        xc, y = arrays
        # auto-components: the new window may now cover extra peaks (add) or
        # have dropped some (shrink) — refit the component set to match it
        r = autocomponent_refit(xc, y, lo, hi, prev)
        if not r["ok"]:
            self._peak2_status(f"[failed] window edit (Peak {rec['number']}): "
                               f"{r.get('error', 'fit failed')}")
            canvas.draw_idle()
            return
        for art in rec.get("artists") or ():
            try:
                art.remove()
            except Exception:
                pass
        rec["result"] = r
        rec["artists"] = self._peak2_draw(ax, r)
        self._peak2_update_row(rec["number"], r, tag="edited")
        ncomp = len(r["components"])
        extra = f" ({ncomp} components)" if ncomp > 1 else ""
        self._peak2_status(f"Peak {rec['number']} (window edited){extra}: "
                           f"μ = {self._peak2_result_mu(r):.6g}")
        canvas.draw_idle()

    def _peak2_update_row(self, number, r, tag=None):
        """Refresh the table row for fit `number` in place after a refit."""
        row = format_composite_fit_row(number, r, tag=tag)
        t = self.extraPopup.peak.peak2_table
        for ri in range(t.rowCount()):
            it = t.item(ri, 0)
            if it is not None and int(round(float(it.data(QtCore.Qt.UserRole)))) == number:
                for ci, (text, sortval) in enumerate(row["cells"]):
                    cell = t.item(ri, ci)
                    if cell is not None:
                        cell.setText(text)
                        cell.setData(QtCore.Qt.UserRole, float(sortval))
                        cell.setToolTip(row["tooltip"])
                break

    def _peak2_try_edit(self, event):
        """Right-click inside a fit's blue fill opens the edit popup for that
        fit. Returns True when a popup opened."""
        ax = event.inaxes
        for rec in self.peak2_fits:
            arts = rec.get("artists") or ()
            if len(arts) < 3 or arts[0].axes is not ax:
                continue
            try:
                hit, _ = arts[2].contains(event)     # the fill (PolyCollection)
            except Exception:
                hit = False
            if hit:
                self._peak2_open_edit(rec, event.xdata)
                return True
        return False

    def _peak2_open_edit(self, rec, click_x):
        """Modal μ/σ/FWHM editor for one fit. On a multi-component fit the edited
        component is the one whose μ is nearest the right-clicked x (named in the
        dialog title). σ↔FWHM are linked (factor 2.3548); fields the user changed
        become `fixed=` on that component, the rest stay free; Apply refits over
        the same window in place, Cancel does nothing."""
        prev = rec["result"]
        comps = prev["components"]
        if click_x is None:                       # right-click without an x → first
            click_x = comps[0]["mu"]
        ci = nearest_component_index(comps, click_x) or 0
        c = comps[ci]
        mu_txt0 = f"{c['mu']:.6g}"
        sig_txt0 = f"{c['sigma']:.6g}"
        dlg = QDialog(self)
        title = f"Edit Peak {rec['number']}"
        if len(comps) > 1:
            title += f" — component {ci + 1}/{len(comps)} (μ≈{c['mu']:.4g})"
        dlg.setWindowTitle(title)
        mu_edit = QLineEdit(mu_txt0)
        sigma_edit = QLineEdit(sig_txt0)
        fwhm_edit = QLineEdit(f"{c['fwhm']:.6g}")
        form = QFormLayout()
        form.addRow("μ", mu_edit)
        form.addRow("σ", sigma_edit)
        form.addRow("FWHM", fwhm_edit)

        # link σ <-> FWHM live (guard against the re-entrant echo)
        self._peak2_edit_linking = False

        def _from_sigma(_):
            if self._peak2_edit_linking:
                return
            self._peak2_edit_linking = True
            try:
                fwhm_edit.setText(f"{sigma_to_fwhm(sigma_edit.text()):.6g}")
            except (ValueError, TypeError):
                pass
            finally:
                self._peak2_edit_linking = False

        def _from_fwhm(_):
            if self._peak2_edit_linking:
                return
            self._peak2_edit_linking = True
            try:
                sigma_edit.setText(f"{fwhm_to_sigma(fwhm_edit.text()):.6g}")
            except (ValueError, TypeError):
                pass
            finally:
                self._peak2_edit_linking = False

        sigma_edit.textEdited.connect(_from_sigma)
        fwhm_edit.textEdited.connect(_from_fwhm)

        apply_btn = QPushButton("Apply")
        cancel_btn = QPushButton("Cancel")
        apply_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        btns = QHBoxLayout()
        btns.addWidget(apply_btn)
        btns.addWidget(cancel_btn)
        lay = QVBoxLayout()
        lay.addLayout(form)
        lay.addLayout(btns)
        dlg.setLayout(lay)

        if dlg.exec_() != QDialog.Accepted:
            return

        # fields whose text changed become fixed (σ text also changes when the
        # user edits FWHM, via the link — so a width edit either way is caught)
        fixed = {}
        try:
            if mu_edit.text() != mu_txt0:
                fixed["mu"] = float(mu_edit.text())
            if sigma_edit.text() != sig_txt0:
                fixed["sigma"] = float(sigma_edit.text())
        except ValueError:
            self._peak2_status(f"[edit] Peak {rec['number']}: invalid number — unchanged.")
            return
        if not fixed:
            self._peak2_status(f"[edit] Peak {rec['number']}: nothing changed.")
            return
        xx = prev["xx"]
        bad = validate_gauss_edit(fixed, lo=float(xx[0]), hi=float(xx[-1]))
        if bad:
            self._peak2_status(f"[edit] Peak {rec['number']}: {bad} — unchanged.")
            return

        ax = self._peak2_live_axes(rec)
        arrays = None if ax is None else self._peak2_spectrum_arrays(rec["name"])
        if arrays is None:
            self._peak2_status(f"[edit] Peak {rec['number']}: spectrum unavailable.")
            return
        xc, y = arrays
        # validate uses the flat mu/sigma names; the fit core takes the
        # suffixed per-component names (mu{k}/sigma{k} of the edited component),
        # while the other components are seeded on their current centroids so
        # they stay put through the refit
        suffix = ci + 1
        fixed_c = {(f"mu{suffix}" if k == "mu" else f"sigma{suffix}"): v
                   for k, v in fixed.items()}
        seeds = {f"mu{i + 1}": comp["mu"] for i, comp in enumerate(comps)}
        r = fit_composite(xc, y, float(xx[0]), float(xx[-1]), prev["spec"],
                          fixed=fixed_c, seeds=seeds)
        if not r["ok"]:
            self._peak2_status(f"[failed] edit (Peak {rec['number']}): "
                               f"{r.get('error', 'fit failed')}")
            return
        for art in rec.get("artists") or ():
            try:
                art.remove()
            except Exception:
                pass
        rec["result"] = r
        rec["artists"] = self._peak2_draw(ax, r)
        self._peak2_update_row(rec["number"], r, tag="edited")
        self._peak2_status(f"Peak {rec['number']} (edited): "
                           f"μ = {self._peak2_result_mu(r):.6g}")
        ax.figure.canvas.draw_idle()

    def peakFit2Toggle(self, checked):
        """Start/Stop toggle: arm (or disarm) the current tab's canvas so each
        left-click fits a gaussian+linear around the clicked position. The press
        handler stays connected wherever fits remain, so future drag/edit
        interactions survive Stop."""
        self.logger.info('peakFit2Toggle - checked: %s', checked)
        btn = self.extraPopup.peak.peak2_start
        self.peak2_armed = bool(checked)
        if checked:
            # Start and Fix Peak are mutually exclusive arming modes
            self.extraPopup.peak.peak2_fix.setChecked(False)
            self._peak2_connect(self.wTab.plot(self.wTab.currentIndex()).canvas)
            btn.setText("Stop")
            btn.setStyleSheet("background-color:#ff6b6b;")
            self._peak2_status(
                "[armed] Left-click a peak on the pad to fit it. "
                "Click Stop to disarm.")
        else:
            btn.setText("Start")
            btn.setStyleSheet("background-color:#bcee68;")
            self._peak2_sync_connections()

    def peakFit2FixToggle(self, checked):
        """Fix Peak toggle: arm fixed-μ fitting on the current tab's canvas,
        mutually exclusive with Start. While armed, each left-click fits
        gaussian+linear with μ pinned exactly at the clicked x (window centred
        on the click; see `_peak2_max_window_bins` for its size)."""
        self.logger.info('peakFit2FixToggle - checked: %s', checked)
        btn = self.extraPopup.peak.peak2_fix
        self.peak2_fix_armed = bool(checked)
        if checked:
            self.extraPopup.peak.peak2_start.setChecked(False)   # mutual exclusion
            self._peak2_connect(self.wTab.plot(self.wTab.currentIndex()).canvas)
            btn.setStyleSheet("background-color:#ff6b6b;")
            self._peak2_status(
                "[armed: fix μ] Left-click at a peak centre to fit with μ "
                "pinned there. Click Fix Peak again to disarm.")
        else:
            btn.setStyleSheet("background-color:#bcee68;")
            self._peak2_sync_connections()

    def _peak2_status(self, msg):
        """Show the latest Peak Finder 2 status/feedback line (armed/config/
        skip/failed/error)."""
        self.extraPopup.peak.peak2_status.setText(msg)

    _PEAK2_SIGNAL_BY_LABEL = {"Gaussian": "gaussian", "Crystal ball": "crystal_ball"}
    _PEAK2_BG_BY_LABEL = {"Linear": "poly1", "Quadratic": "poly2", "Cubic": "poly3"}

    def _peak2_current_spec(self):
        """The fit spec selected in the shape menus. Single component here; a
        multi-component fit only arises from E5's auto-add-on-drag."""
        p = self.extraPopup.peak
        return {
            "signal": self._PEAK2_SIGNAL_BY_LABEL.get(p.peak2_signal.currentText(),
                                                      "gaussian"),
            "n_components": 1,
            "background": self._PEAK2_BG_BY_LABEL.get(p.peak2_bg.currentText(),
                                                      "poly1"),
            "tail_side": p.peak2_cb_tail.currentText(),
        }

    @staticmethod
    def _peak2_result_mu(r):
        """μ of a composite fit's primary (first) component."""
        return r["components"][0]["mu"]

    def _peak2_load_shape_menus(self):
        """Restore the last-used shape-menu selections from QSettings (signals
        blocked so restoring does not trigger the persist/refit slot)."""
        s = QSettings()
        p = self.extraPopup.peak
        for widget, key in ((p.peak2_signal, "signal_shape"),
                            (p.peak2_bg, "background_shape"),
                            (p.peak2_cb_tail, "cb_tail_side")):
            val = s.value(f"PeakFinder2/{key}", "", type=str)
            if val:
                widget.blockSignals(True)
                widget.setCurrentText(val)
                widget.blockSignals(False)

    def _peak2_shape_changed(self, *_):
        """A shape menu changed (E3 self-update): persist the selection, then —
        if a fit row is selected — re-fit THAT fit in place with the new model.
        With no row selected the menu only sets the default for the next new
        fit. This slot fires only on a genuine user change: the menu-sync on
        row-select (`_peak2_sync_menus_to_spec`) blocks the combo signals, so a
        selection never lands here."""
        s = QSettings()
        p = self.extraPopup.peak
        s.setValue("PeakFinder2/signal_shape", p.peak2_signal.currentText())
        s.setValue("PeakFinder2/background_shape", p.peak2_bg.currentText())
        s.setValue("PeakFinder2/cb_tail_side", p.peak2_cb_tail.currentText())
        num = self._peak2_selected_number()
        if num is None:
            return
        rec = next((rc for rc in self.peak2_fits if rc.get("number") == num), None)
        if rec is not None:
            self._peak2_refit_selected(rec)

    def _peak2_refit_selected(self, rec):
        """Re-fit `rec` in place over its stored window with the current menu
        model (keeping the fit's own component count; μ seeded from the fit so
        it stays on the same peak). Reuses the drag/edit redraw path; on failure
        the previous fit is kept untouched."""
        prev = rec["result"]
        xx = prev["xx"]
        lo, hi = float(xx[0]), float(xx[-1])
        spec = self._peak2_current_spec()
        spec["n_components"] = prev["spec"].get("n_components", 1)
        seeds = {f"mu{i + 1}": c["mu"] for i, c in enumerate(prev["components"])}
        ax = self._peak2_live_axes(rec)
        arrays = None if ax is None else self._peak2_spectrum_arrays(rec["name"])
        if arrays is None:
            self._peak2_status(f"[shape] Peak {rec['number']}: spectrum unavailable.")
            return
        xc, y = arrays
        r = fit_composite(xc, y, lo, hi, spec, seeds=seeds)
        if not r["ok"]:
            self._peak2_status(f"[shape] Peak {rec['number']}: "
                               f"{r.get('error', 'refit failed')} — unchanged.")
            return
        for art in rec.get("artists") or ():
            try:
                art.remove()
            except Exception:
                pass
        rec["result"] = r
        rec["artists"] = self._peak2_draw(ax, r)
        self._peak2_update_row(rec["number"], r, tag="edited")
        self._peak2_status(f"Peak {rec['number']} → {spec['signal']}/"
                           f"{spec['background']}: μ = {self._peak2_result_mu(r):.6g}")
        ax.figure.canvas.draw_idle()

    def _peak2_sync_menus_to_spec(self, spec):
        """Write `spec` back into the three shape menus WITH combo signals
        blocked, so syncing the menus to the selected fit never triggers
        `_peak2_shape_changed`'s re-fit (the E3 re-entrancy guard)."""
        p = self.extraPopup.peak
        sig = next((k for k, v in self._PEAK2_SIGNAL_BY_LABEL.items()
                    if v == spec.get("signal")), "Gaussian")
        bg = next((k for k, v in self._PEAK2_BG_BY_LABEL.items()
                   if v == spec.get("background")), "Linear")
        for widget, text in ((p.peak2_signal, sig), (p.peak2_bg, bg),
                             (p.peak2_cb_tail, spec.get("tail_side", "low"))):
            widget.blockSignals(True)
            widget.setCurrentText(text)
            widget.blockSignals(False)

    def _peak2_add_row(self, peak_no, r, tag=None):
        """Insert one fitted peak as a row in the results table (compact
        columns + numeric sort keys; hover shows the full detail)."""
        row = format_composite_fit_row(peak_no, r, tag=tag)
        t = self.extraPopup.peak.peak2_table
        t.setSortingEnabled(False)     # don't re-sort mid-insert
        ri = t.rowCount()
        t.insertRow(ri)
        for ci, (text, sortval) in enumerate(row["cells"]):
            item = _NumericItem(text)
            # the # cell's sort value (== peak number) doubles as the row->fit
            # lookup key for delete/highlight
            item.setData(QtCore.Qt.UserRole, float(sortval))
            item.setToolTip(row["tooltip"])
            t.setItem(ri, ci, item)
        t.setSortingEnabled(True)

    def _peak2_selected_number(self):
        """Fit number of the currently-selected table row, or None."""
        t = self.extraPopup.peak.peak2_table
        ri = t.currentRow()
        if ri < 0 or t.item(ri, 0) is None:
            return None
        try:
            return int(round(float(t.item(ri, 0).data(QtCore.Qt.UserRole))))
        except (TypeError, ValueError):
            return None

    def _peak2_delete_selected(self):
        """Delete the selected fit: its table row, its record, and its artists
        on the pad."""
        num = self._peak2_selected_number()
        if num is None:
            self._peak2_status("[delete] Select a fit row first.")
            return
        rec = next((rc for rc in self.peak2_fits if rc.get("number") == num), None)
        canvases = set()
        if rec is not None:
            for art in rec.get("artists") or ():
                try:
                    canvases.add(art.axes.figure.canvas)
                    art.remove()
                except Exception:
                    self.logger.debug('_peak2_delete_selected - remove failed', exc_info=True)
            self.peak2_fits.remove(rec)
        t = self.extraPopup.peak.peak2_table
        for ri in range(t.rowCount()):
            it = t.item(ri, 0)
            if it is not None and int(round(float(it.data(QtCore.Qt.UserRole)))) == num:
                t.removeRow(ri)
                break
        try:
            canvases.add(self.currentPlot.canvas)
        except Exception:
            pass
        for c in canvases:
            try:
                c.draw_idle()
            except Exception:
                self.logger.debug('_peak2_delete_selected - redraw failed', exc_info=True)
        # a canvas that just lost its last fit and isn't armed can release the handler
        self._peak2_sync_connections()
        self._peak2_status(f"[delete] Removed Peak {num}.")

    def _peak2_row_selected(self):
        """Highlight the selected fit's curve on the pad (thicker + orange);
        restore the others to the default red. Also syncs the shape menus to the
        selected fit's model (E3) so the menus reflect the fit you'd act on."""
        sel = self._peak2_selected_number()
        sel_rec = None
        canvases = set()
        for rec in self.peak2_fits:
            arts = rec.get("artists") or ()
            if not arts:
                continue
            curve = arts[0]
            try:
                if rec.get("number") == sel:
                    sel_rec = rec
                    curve.set_linewidth(3.2)
                    curve.set_color("tab:orange")
                else:
                    curve.set_linewidth(1.8)
                    curve.set_color("tab:red")
                canvases.add(curve.axes.figure.canvas)
            except Exception:
                self.logger.debug('_peak2_row_selected - restyle failed', exc_info=True)
        # sync the menus to the selected fit's spec (signals blocked inside, so
        # this never triggers _peak2_shape_changed's re-fit)
        if sel_rec is not None:
            self._peak2_sync_menus_to_spec(sel_rec["result"].get("spec", {}))
        for c in canvases:
            try:
                c.draw_idle()
            except Exception:
                self.logger.debug('_peak2_row_selected - redraw failed', exc_info=True)

    def _peak2_other_mode_active(self):
        """True while another pad interaction owns clicks: rubber-band
        zoom, gate create/edit, or summing-region create. An armed Peak
        Finder 2 must not also fit on those presses."""
        cp = self.currentPlot
        if cp.zoomPress or cp.toCreateGate or cp.toEditGate or cp.toCreateSumRegion:
            return True
        try:
            return self.gatePopup.isVisible() or self.sumRegionPopup.isVisible()
        except Exception:
            return False

    def _peak2_draw(self, ax, r):
        """Draw one fit result on `ax` (curve + dashed bg + blue net-area fill +
        square end-handles, plus thin dashed per-component curves when the fit
        has more than one component) and return the artist tuple; the curve is
        always index 0 and the fill index 2 (drag-grab / edit-hit rely on that).
        Factored out so a fit record can be redrawn from its stored curves at
        any time (lifecycle safety)."""
        (curve,) = ax.plot(r["xx"], r["y_fit"], color="tab:red", lw=1.8)
        (bgline,) = ax.plot(r["xx"], r["y_bg"], color="grey", lw=1.0, ls="--")
        fill = ax.fill_between(r["xx"], r["y_bg"], r["y_fit"],
                               where=r["y_fit"] >= r["y_bg"],
                               color="tab:blue", alpha=0.45)
        # square end-handles (grab targets for drag-to-refit)
        (handles,) = ax.plot([r["xx"][0], r["xx"][-1]],
                             [r["y_fit"][0], r["y_fit"][-1]],
                             marker="s", ms=6, ls="None",
                             color="tab:red", mec="black", mew=0.6, zorder=6)
        # per-component overlays (each drawn over the background); only when the
        # fit is a genuine multi-component one — a single component == the curve
        comps = []
        y_comp = r.get("y_comp") or []
        if len(y_comp) > 1:
            for yc in y_comp:
                (ln,) = ax.plot(r["xx"], yc, color="tab:red", lw=0.8, ls=":")
                comps.append(ln)
        for art in (curve, bgline, fill, handles, *comps):
            if hasattr(art, "set_gid"):
                art.set_gid("peakfit2")
        return (curve, bgline, fill, handles, *comps)

    def _peak2_max_window_bins(self):
        """The Config cap (max fit window in bins), or None when unset."""
        try:
            raw = QSettings().value("PeakFinder2/max_window_bins", "", type=str)
            n = int(raw)
            return n if n > 0 else None
        except Exception:
            return None

    def peakFit2Config(self):
        """Config dialog: max fit window in bins (empty = no cap)."""
        self.logger.info('peakFit2Config')
        current = self._peak2_max_window_bins()
        text, ok = QInputDialog.getText(
            self, "Peak Finder — Config",
            "Max fit window (in bins), empty = no cap:\n"
            "With a cap set, clicks that can't be fitted are skipped silently.",
            text="" if current is None else str(current))
        if not ok:
            return
        s = QSettings()
        text = text.strip()
        if text == "":
            s.setValue("PeakFinder2/max_window_bins", "")
            self._peak2_status("[config] Max window: no cap.")
            return
        try:
            n = int(text)
            if n <= 0:
                raise ValueError
        except ValueError:
            self._peak2_status("[config] Max window must be a positive integer or empty — unchanged.")
            return
        s.setValue("PeakFinder2/max_window_bins", str(n))
        self._peak2_status(f"[config] Max window: {n} bins.")

    def onPeakFit2Press(self, event):
        """Unified Peak Finder 2 press handler. Priority: (1) an end-handle grab
        starts a drag; (2) a right-click inside a fit's fill opens the edit
        popup; (3) a left-click while armed (Start or Fix) fits. Zoom / gate /
        summing-region presses are never treated as any of those."""
        if event.inaxes is None or event.xdata is None:
            return
        if self._peak2_other_mode_active():
            return
        if event.button == 1 and not event.dblclick and self._peak2_try_grab(event):
            return
        if event.button == 3:
            self._peak2_try_edit(event)
            return
        if event.button != 1 or event.dblclick:
            return
        if not (self.peak2_armed or self.peak2_fix_armed):
            return
        self._peak2_fit_at_press(event)

    def _peak2_fit_at_press(self, event):
        """Armed-mode fit: gaussian+linear around the click on the clicked pad
        (fix-μ when Fix Peak is armed, automatic window otherwise), draw the
        curve + dashed background + blue net-area fill, and add a results row.
        Guards are owned by the `onPeakFit2Press` dispatcher."""
        try:
            if "colorbar_" in event.inaxes.get_label():
                return
            # resolve the clicked pad (same rule as on_press)
            index = list(self.currentPlot.figure.axes).index(event.inaxes)
            if self.currentPlot.isEnlarged:
                index = self.wTab.selectedPad(self.wTab.currentIndex())

            name = self.nameFromIndex(index)
            if not name:
                self._peak2_status("[skip] Clicked pad holds no spectrum.")
                return
            if self.getSpectrumStoreInfo("dim", index=index) != 1:
                self._peak2_status("[skip] Peak Finder works on 1D spectra only.")
                return

            binx     = self.getSpectrumStoreInfo("binx", index=index)
            minxREST = self.getSpectrumStoreInfo("minx", index=index)
            maxxREST = self.getSpectrumStoreInfo("maxx", index=index)
            xtmp = self.plot_controller.createRange(binx, minxREST, maxxREST)
            ytmp = self.getSpectrumStoreInfo("data", index=index)
            # bin centres to match the counts array; the fit window is chosen
            # automatically from the data around the click (plan A)
            xc = np.asarray(xtmp[:-1]) + 0.5 * np.diff(np.asarray(xtmp))

            # Config cap (bins -> x units); with a cap set, unfittable clicks
            # are skipped silently by design
            cap_bins = self._peak2_max_window_bins()
            bw = float(maxxREST - minxREST) / float(binx)
            max_hw = 0.5 * cap_bins * bw if cap_bins else None

            cx = float(event.xdata)
            spec = self._peak2_current_spec()
            if self.peak2_fix_armed:
                # Fix Peak: pin μ at the clicked x over a window centred on the
                # click (cap sizes it, else 50 bins) — never estimate_fit_window,
                # which would snap onto a bigger neighbour. Failures are always
                # reported here; the cap only sizes the window, it does not
                # silence the click.
                lo, hi = fix_peak_window(cx, bw, cap_bins=cap_bins)
                r = fit_composite(xc, np.asarray(ytmp)[1:], lo, hi, spec,
                                  fixed={"mu1": cx})
                tag = "fixed μ"
                if not r["ok"]:
                    self._peak2_status(f"[failed] {r.get('error', 'fit failed')}")
                    return
            else:
                r = fit_composite_auto(xc, np.asarray(ytmp)[1:], cx, spec,
                                       max_half_window=max_hw)
                tag = None
                if not r["ok"]:
                    if cap_bins:
                        self.logger.debug('_peak2_fit_at_press - capped fit skipped: %s',
                                          r.get('error'))
                        return
                    # failures don't consume a peak number
                    self._peak2_status(f"[failed] {r.get('error', 'fit failed')}")
                    return
                # duplicate suppression (auto mode only): an off-peak flank
                # click re-fits an already-fitted peak on the same spectrum;
                # skip it if the centroid lands within ~1 bin of an existing fit
                same = [rec for rec in self.peak2_fits if rec.get("name") == name]
                new_mu = self._peak2_result_mu(r)
                dup = find_duplicate_mu(
                    new_mu, [self._peak2_result_mu(rec["result"]) for rec in same], bw)
                if dup is not None:
                    self._peak2_status(f"[skip] already fitted near μ = {new_mu:.6g} "
                                       f"(Peak {same[dup]['number']}).")
                    return
            self.peak2_count += 1
            self._peak2_add_row(self.peak2_count, r, tag=tag)

            artists = self._peak2_draw(event.inaxes, r)
            # full record (data included) so the fit can be redrawn or refit
            # later without depending on artist survival
            self.peak2_fits.append(dict(number=self.peak2_count, index=index,
                                        name=name, result=r, artists=artists))
            # keep the handler alive on this canvas even after Stop, so future
            # drag/edit interactions on existing fits keep working
            self._peak2_connect(event.inaxes.figure.canvas)
            self.currentPlot.canvas.draw_idle()
        except Exception:
            # a click must never crash the GUI; report instead
            self.logger.exception('_peak2_fit_at_press - fit failed')
            self._peak2_status("[error] Fit failed — see log.")

    def peakFit2RedrawAll(self):
        """Redraw every recorded fit from its stored curves onto its pad's
        current axis (recovers from axis rebuilds, e.g. enlarge/un-enlarge)."""
        self.logger.info('peakFit2RedrawAll - %d record(s)', len(self.peak2_fits))
        canvases = set()
        for rec in self.peak2_fits:
            for art in rec.get("artists") or ():
                try:
                    art.remove()
                except Exception:
                    self.logger.debug('peakFit2RedrawAll - stale artist', exc_info=True)
            try:
                ax = self.getSpectrumViewInfo("axis", index=rec["index"])
            except Exception:
                ax = None
            if ax is None:
                rec["artists"] = ()
                continue
            rec["artists"] = self._peak2_draw(ax, rec["result"])
            canvases.add(ax.figure.canvas)
        for canvas in canvases:
            try:
                canvas.draw_idle()
            except Exception:
                self.logger.debug('peakFit2RedrawAll - redraw failed', exc_info=True)

    def peakFit2Clear(self):
        """Remove every Peak Finder 2 artist and clear its output box."""
        self.logger.info('peakFit2Clear')
        canvases = set()
        for rec in self.peak2_fits:
            for art in rec.get("artists") or ():
                try:
                    canvases.add(art.axes.figure.canvas)
                except Exception:
                    pass
                try:
                    art.remove()
                except Exception:
                    self.logger.debug('peakFit2Clear - artist remove failed', exc_info=True)
        self.peak2_fits = []
        self.peak2_count = 0
        self.extraPopup.peak.peak2_table.setRowCount(0)
        self._peak2_status("")
        # no fits remain: drop the handler from canvases unless still armed
        self._peak2_sync_connections()
        # redraw every canvas that held a fit, including other tabs'
        try:
            canvases.add(self.currentPlot.canvas)
        except Exception:
            pass
        for canvas in canvases:
            try:
                canvas.draw_idle()
            except Exception:
                self.logger.debug('peakFit2Clear - redraw failed', exc_info=True)


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
        if not fileName:
            return
        self.extraPopup.imaging.loadLISE_name.setText(fileName)
        try:
            if os.path.isfile(fileName):
                self.LISEpic = cv2.imread(fileName, 0)
        except Exception:
            # Best-effort image load — a bad path / unreadable file must not
            # crash the GUI, but must not be silent either.
            self.logger.exception('loadFigure - image load failed')

    def _removeOverlayArtist(self):
        """Detach the overlay image and its axes if one is drawn; True when
        there was something to remove.

        The axes goes too. drawFigure adds a fresh one on every call and only
        the image used to be removed, so a slider drag (valueChanged fires
        continuously) appended one empty axes per tick to figure.axes — the
        same list the pad lookups index (`list(figure.axes).index(inaxes)`), so
        a click on a leaked axes answers with an index past the end of the grid.
        A geometry change or an enlarge detaches the axes behind our back
        (InitializeCanvas delaxes everything), hence the membership test. Both
        artists are removed from whatever figure they were drawn on rather than
        from currentPlot's, which is a different figure once the user has
        switched tabs."""
        if self.imgplot is None:
            return False
        self.imgplot.remove()
        self.imgplot = None
        if self.overlay_ax is not None:
            fig = self.overlay_ax.figure
            if fig is not None and self.overlay_ax in fig.axes:
                fig.delaxes(self.overlay_ax)
            self.overlay_ax = None
        return True

    def _fineMove(self, direction):
        if not self._removeOverlayArtist():
            return
        self.xstart, self.ystart = apply_fine_move(self.xstart, self.ystart, direction)
        self.drawFigure()

    def fineUpMove(self):    self._fineMove("up")

    def fineDownMove(self):  self._fineMove("down")

    def fineLeftMove(self):  self._fineMove("left")

    def fineRightMove(self): self._fineMove("right")

    def _redrawOverlay(self):
        """Slider handlers: redraw the overlay in place, or do nothing when
        none is up. Deliberately leaves onFigure alone — routing these through
        deleteFigure cleared the flag while the image stayed on screen, and the
        next Add then drew a second overlay over the first."""
        if self._removeOverlayArtist():
            self.drawFigure()

    def moveFigure(self):
        self.logger.info('moveFigure')
        try:
            if not self._removeOverlayArtist():
                return
            self.xstart, self.ystart = apply_joystick_move(
                self.xstart, self.ystart,
                self.extraPopup.imaging.joystick.direction,
                self.extraPopup.imaging.joystick.distance)
            self.drawFigure()
        except Exception:
            # Best-effort overlay nudge — a bad joystick read must not crash the
            # GUI, but must not be silent either. (The missing-overlay case is
            # handled above now, not by this clause.)
            self.logger.exception('moveFigure - overlay move failed')

    def indexToStartPosition(self, index):
        self.logger.info('indexToStartPosition')
        row = int(self.wConf.histo_geo_row.currentText())
        col = int(self.wConf.histo_geo_col.currentText())
        i, j = self.plotPosition(index)
        self.xstart, self.ystart = compute_overlay_position(row, col, i, j)

    def drawFigure(self):
        self.logger.info('drawFigure')
        if self.LISEpic is None:
            # Reachable from a redraw when a later Load failed under a live
            # overlay (cv2.imread answers None instead of raising): the image
            # is gone from the canvas, so the flag must not still claim one.
            self.logger.warning('drawFigure - no image loaded')
            self.onFigure = False
            return
        self.alpha = self.extraPopup.imaging.alpha_slider.value()/10
        self.zoomX = self.extraPopup.imaging.zoomX_slider.value()/10
        self.zoomY = self.extraPopup.imaging.zoomY_slider.value()/10

        self.overlay_ax = ax = self.currentPlot.figure.add_axes(
            [self.xstart, self.ystart, self.zoomX, self.zoomY], frameon=True)
        ax.axis('off')
        self.imgplot = ax.imshow(self.LISEpic,
                                 aspect='auto',
                                 alpha=self.alpha)

        self.currentPlot.canvas.draw()

    def deleteFigure(self):
        self.logger.info('deleteFigure')
        if not self._removeOverlayArtist():
            return
        self.onFigure = False
        self.currentPlot.canvas.draw()

    def transFigure(self):
        self.logger.info('transFigure')
        self.extraPopup.imaging.alpha_label.setText("Transparency Level ({} %)".format(self.extraPopup.imaging.alpha_slider.value()*10))
        self._redrawOverlay()

    def zoomFigureX(self):
        self.logger.info('zoomFigureX')
        self.extraPopup.imaging.zoomX_label.setText("Zoom X Level ({} %)".format(self.extraPopup.imaging.zoomX_slider.value()*10))
        self._redrawOverlay()

    def zoomFigureY(self):
        self.logger.info('zoomFigureY')
        self.extraPopup.imaging.zoomY_label.setText("Zoom Y Level ({} %)".format(self.extraPopup.imaging.zoomY_slider.value()*10))
        self._redrawOverlay()

    def addFigure(self):
        self.logger.info('addFigure')
        if self.LISEpic is None:
            QMessageBox.warning(self, "Overlay", "Load an image first.")
            return
        # plotPosition returns None for an unselected pad, so the unpack in
        # indexToStartPosition raises TypeError — this is the case the old
        # `except NameError: raise` was reaching for and never caught.
        if self.currentPlot.selected_plot_index is None:
            QMessageBox.warning(self, "Overlay", "Please select one histogram.")
            return
        if self.onFigure:
            return
        self.indexToStartPosition(self.currentPlot.selected_plot_index)
        self.drawFigure()
        self.onFigure = True

    ############################
    # 16) Jupyter Notebook
    ############################

    #Create dataframe for Jupyter and web
    def createDf(self):
        self.logger.info('createDf')
        try:
            export_spectrum_csv(self.getSpectrumStoreDict(),
                                self.extraPopup.peak.jup_df_filename.text(),
                                statistics_fetcher=self.connection_manager.getSpectrumStatistics().get)
        except Exception:
            # Export is best-effort: a failure here must not crash the GUI or
            # block jupyterStart (the notebook can still open). But it must not
            # be silent either — otherwise the notebook loads stale/missing data
            # with no clue why. Log the traceback instead of swallowing it.
            self.logger.exception('createDf - spectrum export failed')

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
                path, _ = QFileDialog.getOpenFileName(None, "Find jupyter-notebook executable", QDir.homePath())
                if not path:
                    # user cancelled: abort starting Jupyter, keep the GUI alive
                    # (the old tuple-truthiness check made Cancel unreachable,
                    # and the cancel path called sys.exit(0) — killing the GUI)
                    self.logger.warning('jupyterStart - jupyter-notebook not located; start aborted by user')
                    return
                execname = path
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

