"""gui/Makefile.am installs with a `*.py` glob but ships `make dist` from a
hand-kept EXTRA_DIST list, which has fallen behind twice. Pin the list to
the tree in both directions."""

import pathlib
import re

GUI = pathlib.Path(__file__).resolve().parents[2] / "gui"
PACKAGES = ("services", "adapters", "controllers")


def _extra_dist():
    text = (GUI / "Makefile.am").read_text()
    m = re.search(r"EXTRA_DIST\s*=\s*((?:[^\n]*\\\n)*[^\n]*)", text)
    assert m, "no EXTRA_DIST in gui/Makefile.am"
    return set(m.group(1).replace("\\\n", " ").split())


def _tree():
    files = {p.name for p in GUI.glob("*.py")}
    for pkg in PACKAGES:
        files |= {f"{pkg}/{p.name}" for p in (GUI / pkg).glob("*.py")}
    return files


def test_every_source_file_is_in_extra_dist():
    missing = sorted(_tree() - _extra_dist())
    assert not missing, f"add to EXTRA_DIST in gui/Makefile.am: {missing}"


def test_extra_dist_names_only_files_that_exist():
    absent = sorted(f for f in _extra_dist() if not (GUI / f).exists())
    assert not absent, f"EXTRA_DIST names deleted files: {absent}"
