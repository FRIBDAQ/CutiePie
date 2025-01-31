
lib_LTLIBRARIES=libMirrorClient.la
bin_PROGRAMS=mirrorclient

mirrorclient_SOURCES=cmdline.c cmdline.h mirror.cpp \
	CmdInfo.h CmdInfo.cpp \
	MirrorCommand.h MirrorCommand.cpp 

mirrorclient_DEPENDENCIES=libMirrorClient.la

mirrorclient_CXXFLAGS=	-DSPECTCL_BIN="\"@prefix@/bin\"" \
	-DSPECTCL_TCLLIBS="\"@prefix@/TclLibs\"" \
	-I@top_srcdir@/Display @TCL_CFLAGS@ @TCLPLUS_CFLAGS@

mirrorclient_LDADD=@builddir@/libMirrorClient.la \
	@RESTCLIENT_LDFLAGS@ \
	@TCLPLUS_LDFLAGS@ @TCL_LIBS@ @JSONCPP_LIBS@



BUILT_SOURCES = cmdline.c cmdline.h 
noinst_HEADERS=cmdline.h

cmdline.h cmdline.c : @srcdir@/mirroropts.ggo
	@GENGETOPT@ --unamed-opts <@srcdir@/mirroropts.ggo



include_HEADERS=MirrorClient-linux.h MirrorClient-test.h SpecTclMirrorClient.h
libMirrorClient_la_SOURCES=MirrorClient-linux.cpp MirrorClient-test.cpp SpecTclMirrorClient.cpp \
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
	$(mkinstalldirs)  @bindir@ @includedir@
	$(INSTALL_SCRIPT) @srcdir@/mirrorclientscript.tcl @bindir@
	echo "#include <MirrorClient-linux.h>" > _mirrorclient.hdr
	$(INSTALL_DATA) _mirrorclient.hdr @includedir@/MirrorClient.h


EXTRA_DIST=mirrorclientscript.tcl
