"""The window-title version must track configure.ac in a source checkout and
the install-time stamp in an installed tree; the hardcoded fallback is the
last resort and must itself match configure.ac."""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import gui_stubs

CONFIGURE_AC = os.path.join(os.path.dirname(__file__), "..", "..", "..", "configure.ac")


def _ac_init_version():
    with open(CONFIGURE_AC, encoding="utf-8") as fh:
        return re.search(r"AC_INIT\(\s*\[?\s*CutiePie\s*\]?\s*,\s*\[?\s*([^,\]\)\s]+)",
                         fh.read()).group(1)


def _tree(tmp_path, configure_ac=None, version_file=None):
    """Lay out <root>/x/gui like the source tree; configure.ac sits at <root>,
    VERSION sits next to GUI.py."""
    gui_dir = tmp_path / "x" / "gui"
    gui_dir.mkdir(parents=True)
    if configure_ac is not None:
        (tmp_path / "configure.ac").write_text(configure_ac, encoding="utf-8")
    if version_file is not None:
        (gui_dir / "VERSION").write_text(version_file, encoding="utf-8")
    return str(gui_dir)


def test_source_checkout_reads_configure_ac(tmp_path):
    gui = gui_stubs.import_gui()
    gui_dir = _tree(tmp_path,
                    configure_ac="AC_INIT(CutiePie, v9.9-001, daqhelp@frib.msu.edu)\n",
                    version_file="v0.0-000\n")
    assert gui.cutiepie_version(gui_dir) == "v9.9-001"


def test_installed_tree_reads_the_version_stamp(tmp_path):
    gui = gui_stubs.import_gui()
    gui_dir = _tree(tmp_path, version_file="v9.9-002\n")
    assert gui.cutiepie_version(gui_dir) == "v9.9-002"


def test_blank_stamp_falls_through(tmp_path):
    gui = gui_stubs.import_gui()
    gui_dir = _tree(tmp_path, version_file="   \n")
    assert gui.cutiepie_version(gui_dir) == gui._VERSION_FALLBACK


def test_nothing_found_uses_the_fallback(tmp_path):
    gui = gui_stubs.import_gui()
    assert gui.cutiepie_version(_tree(tmp_path)) == gui._VERSION_FALLBACK


def test_default_lookup_is_the_real_gui_folder():
    gui = gui_stubs.import_gui()
    assert gui.cutiepie_version() == _ac_init_version()


def test_fallback_matches_configure_ac():
    """Bumping AC_INIT without the fallback is exactly how the installed title
    went stale once; the suite now refuses that."""
    gui = gui_stubs.import_gui()
    assert gui._VERSION_FALLBACK == _ac_init_version()


def test_install_hook_writes_the_stamp():
    makefile = os.path.join(os.path.dirname(__file__), "..", "..", "gui", "Makefile.am")
    with open(makefile, encoding="utf-8") as fh:
        hook = fh.read().split("install-exec-hook:", 1)[1]
    assert re.search(r"\$\(PACKAGE_VERSION\).*@prefix@/Script/VERSION", hook)
