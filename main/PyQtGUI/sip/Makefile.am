
#  The Python module is built with sip-v6 from pyproject-posix.toml.in, whose
#  placeholders are written as [@]NAME[@] here so config.status leaves them be.

SIPBUILD = @builddir@/sipbuild

install-exec-hook:
	$(mkinstalldirs) @prefix@/QtGUI
	rm -f @libdir@/CPyConverter.so @libdir@/CPyConverter.cpython-*.so
	rm -rf $(SIPBUILD) && $(mkinstalldirs) $(SIPBUILD)
	sed -e 's|[@]SIPDIR[@]|$(abs_srcdir)|g' \
	    -e 's|[@]TOPSRCDIR[@]|$(abs_top_srcdir)|g' \
	    -e 's|[@]LIBDIR[@]|@libdir@|g' \
	    @srcdir@/pyproject-posix.toml.in > $(SIPBUILD)/pyproject.toml
	(cd $(SIPBUILD) && sip-build)
	cp $(SIPBUILD)/build/CPyConverter/build/lib.*/CPyConverter*.so @libdir@/

EXTRA_DIST = pyproject-posix.toml.in CPyConverter.sip CPyConverter.h dataAccess.h dataTypes.h

clean-local:
	rm -rf $(SIPBUILD)
