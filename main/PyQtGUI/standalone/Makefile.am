bin_PROGRAMS = CutiePie.bin

include_HEADERS = CPyHelper.h

CutiePie_bin_SOURCES = gui.cpp 
CutiePie_bin_CXXFLAGS = @PYTHON_CXXFLAGS@ -DINSTALLED_IN='"@prefix@"' -lpthread \
	-I@top_srcdir@/Core
CutiePie_bin_LDFLAGS = @PYTHON_LDFLAGS@

install-exec-hook:
	$(INSTALL_PROGRAM) CutiePie @prefix@/bin
