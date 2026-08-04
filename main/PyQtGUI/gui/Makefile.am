bin_PROGRAMS = _CutiePie.bin

include_HEADERS = CPyHelper.h

_CutiePie_bin_SOURCES = gui.cpp 
# _CutiePie_bin_CXXFLAGS = @PYTHON_CXXFLAGS@ 
_CutiePie_bin_CXXFLAGS = @PYTHON_CXXFLAGS@ -DINSTALLED_IN='"@prefix@"' 
_CutiePie_bin_LDFLAGS = @PYTHON_LDFLAGS@

install-exec-hook:
	$(mkinstalldirs) @prefix@/Skel @prefix@/VMUSBSkel @prefix@/DDASSkel
	$(mkinstalldirs) @prefix@/Script
	$(INSTALL_PROGRAM) _CutiePie @prefix@/bin
	$(INSTALL_DATA) @srcdir@/*.py @prefix@/Script
	$(mkinstalldirs) @prefix@/Script/services
	$(INSTALL_DATA) @srcdir@/services/*.py @prefix@/Script/services
	$(mkinstalldirs) @prefix@/Script/adapters
	$(INSTALL_DATA) @srcdir@/adapters/*.py @prefix@/Script/adapters
	$(mkinstalldirs) @prefix@/Script/controllers
	$(INSTALL_DATA) @srcdir@/controllers/*.py @prefix@/Script/controllers
	$(INSTALL_DATA) @srcdir@/Main.py @prefix@/Skel 
	$(INSTALL_DATA) @srcdir@/algo_skel_creator.py @prefix@/Skel
	$(INSTALL_DATA) @srcdir@/fit_skel_creator.py @prefix@/Skel
	$(INSTALL_DATA) @srcdir@/jupyterqtpy_notebook_config.py @prefix@/Skel
	$(INSTALL_DATA) @srcdir@/Main.py @prefix@/VMUSBSkel
	$(INSTALL_DATA) @srcdir@/algo_skel_creator.py @prefix@/VMUSBSkel
	$(INSTALL_DATA) @srcdir@/fit_skel_creator.py @prefix@/VMUSBSkel
	$(INSTALL_DATA) @srcdir@/jupyterqtpy_notebook_config.py @prefix@/VMUSBSkel
	$(INSTALL_DATA) @srcdir@/Main.py @prefix@/DDASSkel
	$(INSTALL_DATA) @srcdir@/algo_skel_creator.py @prefix@/DDASSkel
	$(INSTALL_DATA) @srcdir@/fit_skel_creator.py @prefix@/DDASSkel
	$(INSTALL_DATA) @srcdir@/jupyterqtpy_notebook_config.py @prefix@/DDASSkel
	(chmod -R g+w @prefix@)
	(chmod -R o+x @prefix@)	

EXTRA_DIST = Main.py GUI.py connectConfigGUI.py MenuAndConfigGUI.py \
	MenuGate.py MenuSumRegion.py otherOptions.py OutputIntegrate.py OutputGUI.py Functions1DGUI.py Functions2DGUI.py JoystickGUI.py PlotGUI.py\
	shm_parser.py CannyEdgePlot.py ImgSegPlot.py algo_factory.py algo_skel_creator.py fit_factory.py fit_skel_creator.py \
	fit_gaus_creator.py fit_gp1_creator.py fit_gp2_creator.py fit_p1_creator.py fit_p2_creator.py fit_exp_creator.py skel_plot.py kmean_creator.py gmm_creator.py imgseg_creator.py cannye_creator.py \
	WebWindow.py notebook_process.py logger.py jupyterqtpy_notebook_config.py SpecialFunctionsGUI.py PyREST.py CopyPropertiesGUI.py \
	services/__init__.py services/spectrum_store.py services/thread_workers.py \
	services/fit_manager.py services/gate_manager.py services/sum_region_manager.py \
	services/connection_manager.py services/plot_controller.py adapters/__init__.py \
	adapters/fit_adapter.py adapters/connection_adapter.py adapters/gate_adapter.py adapters/sum_region_adapter.py \
	services/dataframe_export.py services/display_slot.py \
	services/figure_overlay.py services/geometry_io.py services/log_throttle.py services/peak_finder.py \
	services/shape_file.py services/tab_session.py \
	controllers/__init__.py controllers/copy_properties_controller.py controllers/geometry_controller.py controllers/overlay_controller.py controllers/jupyter_controller.py controllers/peak_scan_controller.py controllers/peak_fit2_controller.py
