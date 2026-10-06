#!/bin/bash
#
#  Build and install CutiePie natively on macOS.
#
#  Usage:
#     ./build-macos.sh               compile + install (configures first if needed)
#     ./build-macos.sh --configure   rerun autoreconf/configure, then compile + install
#
#  Rerun with --configure after editing configure.ac or any Makefile.am.
#  Override locations with environment variables:
#     CONDA_ENV   conda env holding Python, PyQt5, numpy and sip  (/opt/miniconda3/envs/cutiepie)
#     BUILD_DIR   out-of-tree build directory                      ($HOME/cutiepie-build)
#     PREFIX      install prefix                                   ($HOME/opt/cutiepie)
#     RESTCLIENT  restclient-cpp install                           ($HOME/opt/restclient)
#
#  One-time prerequisites:
#     brew install autoconf automake libtool pkgconf gengetopt jsoncpp tcl-tk@8 cppunit cmake
#     conda create -n cutiepie python=3.12
#     pip install sip numpy PyQt5 PyQtWebEngine matplotlib scipy pandas lmfit httplib2 requests opencv-python
#     restclient-cpp built from https://github.com/mrtazz/restclient-cpp into $RESTCLIENT

set -e

SRC="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="${CONDA_ENV:-/opt/miniconda3/envs/cutiepie}"
BUILD_DIR="${BUILD_DIR:-$HOME/cutiepie-build}"
PREFIX="${PREFIX:-$HOME/opt/cutiepie}"
RESTCLIENT="${RESTCLIENT:-$HOME/opt/restclient}"
TCL="$(brew --prefix tcl-tk@8)/lib"

export PATH="$CONDA_ENV/bin:$(brew --prefix)/bin:/usr/bin:/bin:/usr/sbin:/sbin"
# tcl-tk@8 must come first: the conda env ships its own tcl.pc.
export PKG_CONFIG_PATH="$TCL/pkgconfig:$CONDA_ENV/lib/pkgconfig:$(brew --prefix)/lib/pkgconfig"

# configure clones tcl++ into the current directory but looks for it in the
# source tree, so fetch it here.  Its Makefiles also use the GNU-only
# -rpath-link flag and <malloc.h>, neither of which exists on macOS.
prepare_tclplus() {
    if [ ! -f "$SRC/libtclplus/configure.ac" ]; then
        (cd "$SRC" && bash ./tcl++incorp)
    fi
    sed -i '' 's/-Wl,"-rpath-link=\$(libdir)"//' \
        "$SRC/libtclplus/exception/Makefile.am" "$SRC/libtclplus/tclplus/Makefile.am"
    sed -i '' 's|#include <malloc.h>|#include <stdlib.h>|' \
        "$SRC/libtclplus/tclplus/TCLList.cpp"
    (cd "$SRC/libtclplus" && autoreconf -fi)
}

if [ "$1" = "--configure" ] || [ ! -f "$BUILD_DIR/Makefile" ]; then
    prepare_tclplus
    (cd "$SRC" && autoreconf -fi)
    mkdir -p "$BUILD_DIR"
    (cd "$BUILD_DIR" && "$SRC/configure" --prefix="$PREFIX" \
        --with-restclient-cpp="$RESTCLIENT" \
        --with-tclplus-args="--with-tclconfig=$TCL --with-tkconfig=$TCL LIBS='-L$TCL -ltcl8.6'" \
        --with-incorp-build-cores="$(sysctl -n hw.ncpu)")
fi

make -C "$BUILD_DIR" -j"$(sysctl -n hw.ncpu)"
make -C "$BUILD_DIR" install

echo
echo "Installed in $PREFIX.  Run: $PREFIX/bin/_CutiePie"
