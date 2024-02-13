
lib_LTLIBRARIES=libMirrorClient.la



include_HEADERS=MirrorClient.h SpecTclMirrorClient.h
libMirrorClient_la_SOURCES=MirrorClient.cpp SpecTclMirrorClient.cpp \
	CPortManager.cpp CPortManager.h \
	CPortManagerException.h CPortManagerException.cpp \
	MirrorClientInternals.h MirrorClientInternals.cpp \
	CSocket.cpp CTCPBadSocketState.cpp CTCPConnectionFailed.cpp \
	CTCPConnectionLost.cpp CTCPNoSuchHost.cpp CTCPNoSuchService.cpp \
	Exception.cpp io.cpp ErrnoException.cpp ErrnoException.h \
	os.cpp client.cpp \
	CSocket.h CTCPBadSocket.h CTCPConnectionFailed.h \
	CTCPConnectionLost.h CTCPNoSuchHost.h CTCPNoSucnService.h \
	ErrnoException.h Exception.h io.h SpecTclMirrorClient.h xamineDataTypes.h \
	os.h


libMirrorClient_la_CXXFLAGS=-I@top_srcdir@/Core -I@top_srcdir@/Display \
	@RESTCLIENT_CPPFLAGS@   @TCL_CFLAGS@ \
	@JSONCPP_CFLAGS@ -DSPECTCL_BIN="\"@bindir@\""
libMirrorClient_la_LIBADD= @JSONCPP_LIBS@ @TCL_LIBS@

libMirrorClient_la_LDFLAGS=	@RESTCLIENT_LDFLAGS@ 


install-exec-local:
	$(mkinstalldirs)  @bindir@
	$(INSTALL_SCRIPT) @srcdir@/mirrorclientscript.tcl @bindir@





TESTS=unittests

EXTRA_DIST=mirrorclientscript.tcl
