bin_PROGRAMS = CutiePie

include_HEADERS = CPyHelper.h

CutiePie_SOURCES = gui.cpp 
CutiePie_CXXFLAGS = @PYTHON_CXXFLAGS@ -DINSTALLED_IN='"@prefix@"' -lpthread \
	-I@top_srcdir@/Core
CutiePie_LDFLAGS = @PYTHON_LDFLAGS@

