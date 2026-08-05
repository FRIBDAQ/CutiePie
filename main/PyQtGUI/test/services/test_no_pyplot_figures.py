"""No figure may be created through the pyplot state machine. A pyplot-managed
figure registers itself in a process-global registry, which is what made a
deleted tab's figure outlive its tab; a Figure() built here is owned by the
widget that holds it and dies with it.

The allowlist is the migration's own to-do list: entries come out as the sites
are converted, and an empty allowlist means the job is finished. A NEW call in
a file that is not listed fails immediately."""

import ast
import os

GUI_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "gui")

# pyplot entry points that CREATE a figure (and so register one), plus close(),
# which only works on figures that were registered. Colormap and rcParams
# lookups are pure reads and are not listed.
PYPLOT_FIGURE_CALLS = {"figure", "subplots", "subplot_mosaic", "gcf", "gca",
                       "axes", "subplot", "close"}

# path relative to gui/ -> the calls still to be migrated
ALLOWLIST = {
    "PlotGUI.py": {"figure", "close"},
    "services/fit_manager.py": {"subplots"},
}


def _dotted(node):
    """"a.b.c" for an attribute/name chain, else None. Lets the scan see
    `matplotlib.pyplot.figure()` written out in full, not only `plt.figure()`."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _pyplot_calls(path):
    """Every pyplot figure call in a file, as {name}. Covers the three ways to
    reach it: an aliased module import, the fully written-out path, and a
    direct `from matplotlib.pyplot import figure`."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    # module aliases: import matplotlib.pyplot as plt / from matplotlib import pyplot
    aliases = {"matplotlib.pyplot"}
    bare_names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "matplotlib.pyplot" and a.asname:
                    aliases.add(a.asname)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "matplotlib":
                for a in node.names:
                    if a.name == "pyplot":
                        aliases.add(a.asname or "pyplot")
            elif node.module == "matplotlib.pyplot":
                for a in node.names:
                    if a.name in PYPLOT_FIGURE_CALLS:
                        bare_names[a.asname or a.name] = a.name

    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # plt.figure() / matplotlib.pyplot.figure()
        dotted = _dotted(node.func)
        if dotted and "." in dotted:
            prefix, _, attr = dotted.rpartition(".")
            if prefix in aliases and attr in PYPLOT_FIGURE_CALLS:
                found.add(attr)
        # figure(), imported by name
        elif isinstance(node.func, ast.Name) and node.func.id in bare_names:
            found.add(bare_names[node.func.id])
    return found


def _python_files():
    for root, _dirs, files in os.walk(GUI_DIR):
        if "__pycache__" in root:
            continue
        for name in sorted(files):
            if name.endswith(".py"):
                full = os.path.join(root, name)
                yield os.path.relpath(full, GUI_DIR), full


def test_no_unlisted_pyplot_figure_calls():
    offenders = {}
    for rel, full in _python_files():
        calls = _pyplot_calls(full) - ALLOWLIST.get(rel, set())
        if calls:
            offenders[rel] = sorted(calls)
    assert not offenders, (
        "pyplot figure calls outside the allowlist: %s. Build a Figure() and "
        "hand it to a FigureCanvas instead." % offenders)


def test_allowlist_has_no_stale_entries():
    """An entry that no longer matches anything means the migration finished a
    site and left the allowlist behind — the next new call there would be
    accepted silently."""
    stale = {}
    for rel, calls in ALLOWLIST.items():
        actual = _pyplot_calls(os.path.join(GUI_DIR, rel))
        unused = calls - actual
        if unused:
            stale[rel] = sorted(unused)
    assert not stale, "allowlist entries matching nothing: %s" % stale
