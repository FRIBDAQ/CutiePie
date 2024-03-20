clean-local:

install-exec-hook: 
	$(mkinstalldirs) @prefix@/QtGUI
	rm -f @prefix@/QtGUI/CPyConverter.so	
	python3 @srcdir@/configure.py @srcdir@ @libdir@ @top_srcdir@/mirrorclient && make -f Makefile.sip
	cp CPyConverter.so @libdir@/CPyConverter.so

EXTRA_DIST = configure.py CPyConverter.sip CPyConverter.h dataAccess.h dataTypes.h

clean-local:
	if test -e @builddir@/Makefile.sip; \
        then  make -f @builddir@/Makefile.sip clean ; fi
