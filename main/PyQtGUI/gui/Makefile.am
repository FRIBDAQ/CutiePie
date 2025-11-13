bin_PROGRAMS = _CutiePie.bin

include_HEADERS = CPyHelper.h

_CutiePie_bin_SOURCES = gui.cpp 
# _CutiePie_bin_CXXFLAGS = @PYTHON_CXXFLAGS@ 
_CutiePie_bin_CXXFLAGS = @PYTHON_CXXFLAGS@ -DINSTALLED_IN='"@prefix@"' 
_CutiePie_bin_LDFLAGS = @PYTHON_LDFLAGS@

install-exec-hook:
	$(mkinstalldirs) @prefix@/share/d3web
	$(mkinstalldirs) @prefix@/Skel @prefix@/VMUSBSkel @prefix@/DDASSkel
	$(mkinstalldirs) @prefix@/Script
	$(INSTALL_DATA) @srcdir@/d3.js/d3.v6.min.js @prefix@/share/d3web
	$(INSTALL_DATA) @srcdir@/d3.js/plotly-1.57.1.js @prefix@/share/d3web
	$(INSTALL_DATA) @srcdir@/d3.js/style.css @prefix@/share/d3web
	$(INSTALL_DATA) @srcdir@/d3.js/TimeAndDate.js @prefix@/share/d3web	
	$(INSTALL_DATA) @srcdir@/d3.js/index.html @prefix@/share/d3web	
	$(INSTALL_DATA) @srcdir@/*.py @prefix@/Script
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
	WebWindow.py notebook_process.py logger.py jupyterqtpy_notebook_config.py d3.js SpecialFunctionsGUI.py PyREST.py CopyPropertiesGUI.py
