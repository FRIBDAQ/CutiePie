# Process this file with autoconf to produce a configure script.

AC_INIT(CutiePie, v1.5-005, daqhelp@frib.msu.edu)

#AC_CONFIG_SRCDIR(SpecTcl/MySpecTclApp.cpp)
AC_CONFIG_AUX_DIR(config)

AM_INIT_AUTOMAKE([foreign])
AM_CONFIG_HEADER(config.h)
AC_CANONICAL_HOST
AC_CONFIG_MACRO_DIR([m4])

AC_EXEEXT
AC_OBJEXT

AX_CXX_COMPILE_STDCXX_11([noext])

# we use libtool, thus we need to include it
LT_INIT

#  This should track the version in
#  AM_INIT_AUTOMAKE it will be used to
#  define the version information in shared
#  images.

SOVERSION="4:0"
#

# Checks for programs.


AC_PROG_AWK
AC_PROG_CXX
AC_PROG_CC
AC_PROG_CPP
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AM_PROG_LEX
AC_PROG_YACC
AC_PROG_GCC_TRADITIONAL
AC_PROG_LIBTOOL

# PYthon C/C++ extension environment build flags:

PKG_CHECK_MODULES(PYTHON, [python3-embed], [PYTHON_CXXFLAGS=`pkg-config --cflags python3-embed`; PYTHON_LDFLAGS=`pkg-config --libs python3-embed`],
			  [PYTHON_CXXFLAGS=`pkg-config python3 --cflags`, PYTHON_LDFLAGS=`pkg-config python3 --libs`])
AC_SUBST(PYTHON_CXXFLAGS)
AC_SUBST(PYTHON_LDFLAGS)

# Tcl compilations/linking:

PKG_CHECK_MODULES(TCL, [tcl8.6], [TCL_CPPFLAGS=`pkg-config --cflags tcl8.6`; TCL_LDFLAGS=`pkg-config --libs tcl8.6`])

# Checks for header files.

AC_HEADER_STDC
AC_HEADER_SYS_WAIT
AC_CHECK_HEADER(time.h)
AC_CHECK_HEADERS([regex.h sstream stddef.h stdexcept stdio.h stdlib.h])
AC_CHECK_HEADERS([string string.h socket.h types.h unistd.h utility vector])
AC_CHECK_HEADERS([shm.h ipc.h])
# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STAT
AC_HEADER_TIME
AC_STRUCT_TM
AC_C_CONST
AC_C_INLINE
AC_TYPE_OFF_T
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_TYPE_SIGNAL

# Checks for library functions.
AC_FUNC_MMAP
AC_REPLACE_FUNCS([atexit dup2 ftruncate getpagesize memset putenv])
AC_REPLACE_FUNCS([strchr strcspn strerror strrchr bzero mkstemp stat]) 
AC_REPLACE_FUNCS([malloc memcmp strftime alloca vfork strtok_r])
AC_CHECK_FUNCS([select socket strspn fork drand48 seed48]) 
AC_CHECK_FUNCS([pipe sbrk shm_open shmget fcntl])
AC_CHECK_FUNCS([floor getcwd gethostname gettimeofday mbrlen])
AC_CHECK_FUNCS(time)
AC_CHECK_LIB(m, pow)
AC_CHECK_LIB(m, sqrt)
AC_CHECK_LIB(m, logb)
AC_CHECK_FUNCS([strdup strstr strtol tzset mktime strtod timezone])
AC_CHECK_LIB(rt, shm_open, [RTLIB=-lrt
AC_DEFINE([HAVE_SHM_OPEN], [1])],[RTLIB=""])
AC_SUBST(RTLIB)

AC_CHECK_FILE([${srcdir}/libtclplus/configure.ac],
	[AC_MSG_NOTICE([Using previously incorporated tcl++ from the source tree at ${srcdir}/libtclplus])],
	[AC_MSG_NOTICE([Need to build local tagged copy of tcl++])
	${srcdir}/tcl++incorp])
(cd ${srcdir}/libtclplus; ./configure --prefix=${prefix}; make clean install)
result=$?
AS_VAR_IF([result], [0], [], AC_MSG_ERROR([Failed to build and install tcl++ with result: ${result}]))
AC_SUBST(TCLPLUS_CFLAGS, [-I${prefix}/include])
AC_SUBST(TCLPLUS_LDFLAGS, ["-L${prefix}/lib -ltclPlus -lException"])


# PAGESIZE in limits.h
AC_MSG_CHECKING([for PAGESIZE in limits.h])
    AC_TRY_COMPILE([
#include <limits.h>
#ifndef PAGESIZE
# error pagesize not defined
#endif
], , my_cv_have_pagesize=yes, my_cv_have_pagesize=no)
    if test $my_cv_have_pagesize = yes ; then
	AC_DEFINE(HAVE_DECL_PAGESIZE, 1, [True if PAGESIZE is defined in limits.h as on HP-UX])
    fi
AC_MSG_RESULT($my_cv_have_pagesize)

	# PADSIZE in limits.h
AC_MSG_CHECKING([for PADSIZE in limits.h])
    AC_TRY_COMPILE([
#include <limits.h>
#ifndef PADSIZE
# error padsize not defined
#endif
], , my_cv_have_padsize=yes, my_cv_have_padsize=no)
    if test $my_cv_have_padsize = yes ; then
	AC_DEFINE(HAVE_DECL_PADSIZE, 1, [True if PADSIZE defined in limits.h])
    fi
AC_MSG_RESULT($my_cv_have_padsize)

	# INT_MAX in limits.h
AC_MSG_CHECKING([for INT_MAX in limits.h])
    AC_TRY_COMPILE([
#include <limits.h>
#ifndef INT_MAX
# error intmax not defined
#endif
], , my_cv_have_int_max=yes, my_cv_have_int_max=no)
    if test $my_cv_have_int_max = yes ; then
	AC_DEFINE(HAVE_DECL_INT_MAX, 1, [True if INT_MAX defined in limits.h])
    fi
AC_MSG_RESULT($my_cv_have_int_max)

	# MAXINT in values.h
AC_MSG_CHECKING([for MAXINT in values.h])
    AC_TRY_COMPILE([
#include <values.h>
#ifndef MAXINT
# error maxint not defined
#endif
], , my_cv_have_maxint=yes, my_cv_have_maxint=no)
    if test $my_cv_have_maxint = yes ; then
	AC_DEFINE(HAVE_DECL_MAXINT, 1, [True if MAXINT defined in values.h])
    fi
AC_MSG_RESULT($my_cv_have_maxint)

	# Check for 64 bit longs
AC_MSG_CHECKING([whether long int is 64 bits])
    AC_TRY_RUN([
int main(int argc, char **argv) {
	if(sizeof(long) == 8)
		return 0;
	else
		return 1;
}  
], [AC_DEFINE(HAVE_TRUE_SIXTY_FOUR, 1, [True if long int is 8 bits]) AC_MSG_RESULT(yes)], AC_MSG_RESULT(no), AC_MSG_RESULT(assuming not on target machine))


# Determine what OS we are running so we know whether or not to 
# declare some things extern "C". Also helpful for some Cygwin
# and Ultrix specific calls. 

LD_EXTRA_FLAGS=""
X11EXTRA_LDFLAGS=""
AC_MSG_CHECKING([for supported OS])
OS="$host_os"                          # Default guess.
case $host_os in
  *cygwin* ) 
	AC_DEFINE(CYGWIN, 1, [Set to 1 if Cygwin environment])
	AC_MSG_RESULT([Cygwin...  xdr.h broken for c++])

	# cygwin we have doesn't seem to get the object extension
	# quite right:

	OBJEXT="o"
	AC_SUBST(OBJEXT)
	AC_DEFINE(BROKEN_XDR_H, 1,
	         [defined if xdr.h unusuable for c++])


	# Windows doesn't allow undefined symbols at link time
	#
	LD_EXTRA_FLAGS="$LD_EXTRA_FLAGS -no-undefined -lrpc" 
	X11EXTRA_LDFLAGS="-lSM -lICE"
	;;
  *linux* )
	AC_DEFINE(LINUX, 1, [Set to 1 if Linux environment])
	AC_MSG_RESULT([Linux])
	;;
  *ultrix* )
	AC_DEFINE(ULTRIX, 1, [Set to 1 if Ultrix environment])
	AC_MSG_RESULT([Ultrix])
	;;
  *osf* )
	AC_DEFINE(OSF, 1, [Set to 1 if OSF environment])
	AC_MSG_RESULT([OSF1])
	WCHAR_T_DEF=-D_WCHAR_T
	OBJEXT=o

	# Missing on OSF1 for some reason
	AC_SUBST(OBJEXT)

	# This does away with "weak symbol" warnings
	LD_EXTRA_FLAGS="$LD_EXTRA_FLAGS -Wl,\"-S\""
	;;
  *darwin* )
	OBJEXT="o"
	AC_SUBST(OBJEXT)
	AC_DEFINE(DARWIN, 1, [Set to 1 if Darwin environment])
	AC_DEFINE(BROKEN_XDR_H, 1,
		  [defined if xdr.h is unusable for c++])
	AC_MSG_RESULT([Darwin .. xdr.h broken for c++])
	OS="darwin"              # It's crapped up with versions otherwise.
	;;
  * )
	AC_MSG_RESULT([ok])
	;;
esac


AM_PATH_CPPUNIT
AM_CONDITIONAL([CPPUNIT], [test -n "${CPPUNIT_LIBS}"] )
AM_CONDITIONAL([CPPUNITDONOTBUILD], [test -n ""])
AM_CONDITIONAL([THIS_IS_NOT_DEFINED], [test -n ""])



#
#   We'd like to compile with -pedantic if supported
#   Intel looks like gcc so we check explicitly for a gcc
#   named compiler
#

AC_MSG_CHECKING([gcc so use -pedantic?])

if `echo $CC | grep -q gcc` 
then 
     CFLAGS="$CFLAGS -pedantic -Wno-long-long"
     CXXFLAGS="$CXXFLAGS -pedantic -Wno-long-long"
     AC_MSG_RESULT(yes)
else  
     AC_MSG_RESULT(no)
fi

##
#  There's a utility that uses gengetopt:

AX_GENGETOPT


######################################  Checks for jsoncpp


PKG_CHECK_MODULES(JSONCPP, [jsoncpp], [], [AC_MSG_ERROR([jsoncpp package must be installed])])
AC_MSG_RESULT([$JSONCPP])

AC_SUBST(JSONCPP_CFLAGS)
AC_SUBST(JSONCPP_LIBS)

###################################### Checks for restclient-cpp
# Note there's no debian package for that.
# FRIB installs it in /usr/opt/restclient which shoulid have:
#  .../lib/librestclient-cpp.so
#  ../include/restclient-cpp/restclient.h
#     configure can overridet this location and will prompt for that
#     if not found using --with-librestclient=topdir.
# The package is on github at: https://github.com/mrtazz/restclient-cpp


AC_MSG_CHECKING([Checking for librestclient-cpp])
AC_MSG_NOTICE([If you need to install it it can be gotten at https://github.com/mrtazz/restclient-cpp])

AC_ARG_WITH([restclient-cpp],
	[AS_HELP_STRING([--with-restclient-cpp=topdir],[ Provide top installation directory of librestclient-cpp])],
	[restclient_topdir=$withval],[restclient_topdir="/usr/opt/restclient"])

#  Make sure we can find this beast:

AC_CHECK_FILE([${restclient_topdir}/lib/librestclient-cpp.so], [libfound="yes"], [libfound="no"])
AC_CHECK_FILE([${restclient_topdir}/include/restclient-cpp/restclient.h], [hdrfound="yes"], [hdrfound="no"])

if test "${libfound}" = "no"
then
   AC_MSG_FAILURE([Unable to find ${restclient_topdir}/lib/librestclient-cpp.so.  Use correct --with-restclient-cpp value.  If necessary, download/build from https://github.com/mrtazz/restclient-cpp])
fi

if test "${hdrfound}" = "no"
then
   AC_MSG_FAILURE([Unable to find ${restclient_topdir}/include/restclient-cpp/restclient.h.  Use correct --with-restclient-cpp value. If necessary, download/build from https://github.com/mrtazz/restclient-cpp])
fi

RESTCLIENT_LDFLAGS="-L${restclient_topdir}/lib -lrestclient-cpp -Wl,-rpath=${restclient_topdir}/lib"
RESTCLIENT_CPPFLAGS="-I${restclient_topdir}/include"

AC_SUBST(RESTCLIENT_LDFLAGS)
AC_SUBST(RESTCLIENT_CPPFLAGS)



AC_SUBST(SOVERSION)



# create output files.
# Removed           Tape/Makefile

AC_OUTPUT(Makefile
	SpecTclRESTClient/Makefile
	  PyQtGUI/Makefile	
	  PyQtGUI/src/Makefile
	  PyQtGUI/sip/Makefile
	  PyQtGUI/gui/Makefile
	  PyQtGUI/test/Makefile
	  PyQtGUI/standalone/Makefile
	  mirrorclient/Makefile
	  )
          
