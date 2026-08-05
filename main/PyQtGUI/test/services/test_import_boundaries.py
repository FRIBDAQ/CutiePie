"""Guard: enforce the four-layer import DAG (ARCH.md §1, §3.5).

Layers (top to bottom):
    Presentation   — GUI.py, PlotGUI.py, popups, controllers/, adapters/
    Services       — gui/services/*.py, gui/view_state.py
    Domain         — fit/algo creators, factories, fit_function, fit_alpha_base
    Infrastructure — PyREST, CPyConverter, shm_parser, logger

Rules:
    1. Services must not import presentation modules at module scope.
    2. Domain must not import presentation or service modules at module scope.

Function-scope (deferred) imports are the approved escape hatch and are not
flagged.  ALLOWLIST tracks known module-scope violations being migrated; an
empty allowlist is the goal.
"""

import ast
import os

GUI_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "gui")

# ---- layer membership --------------------------------------------------- #

PRESENTATION_MODULES = frozenset({
    "GUI", "PlotGUI", "Main",
    "MenuGate", "MenuSumRegion", "MenuAndConfigGUI",
    "connectConfigGUI", "CopyPropertiesGUI",
    "OutputGUI", "OutputIntegrate", "SpecialFunctionsGUI",
    "otherOptions", "Functions1DGUI", "Functions2DGUI",
    "JoystickGUI", "WebWindow", "CsvPlotGUI",
    "CannyEdgePlot", "ImgSegPlot",
    "dialogs", "alpha_filter_dialog", "skel_plot",
})

PRESENTATION_PACKAGES = frozenset({"controllers", "adapters"})

SERVICE_MODULES = frozenset({
    "connection_manager", "plot_controller", "gate_manager",
    "fit_manager", "sum_region_manager", "spectrum_store",
    "display_slot", "thread_workers", "geometry_io",
    "dataframe_export", "peak_finder", "figure_overlay",
    "log_throttle", "tab_session", "shape_file",
    "view_state",
})

SERVICE_PACKAGES = frozenset({"services"})

# Known module-scope violations being migrated.
# path relative to gui/ -> set of forbidden module basenames still imported.
ALLOWLIST = {
    "imgseg_creator.py": {"ImgSegPlot"},
    "cannye_creator.py": {"CannyEdgePlot"},
}


# ---- AST helpers -------------------------------------------------------- #

def _walk_module_scope(stmts, out):
    """Collect imported module basenames from module-scope statements.

    Recurses into top-level if/try/class blocks (all execute at import time)
    but NOT into function bodies (deferred imports, approved).
    """
    for node in stmts:
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module.split(".")[0])
        elif isinstance(node, ast.If):
            _walk_module_scope(node.body, out)
            _walk_module_scope(node.orelse, out)
        elif isinstance(node, ast.Try):
            _walk_module_scope(node.body, out)
            for handler in node.handlers:
                _walk_module_scope(handler.body, out)
            _walk_module_scope(node.orelse, out)
            _walk_module_scope(node.finalbody, out)
        elif isinstance(node, ast.ClassDef):
            _walk_module_scope(node.body, out)


def _module_scope_imports(path):
    """Return the set of module basenames imported at module scope."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    out = set()
    _walk_module_scope(tree.body, out)
    return out


# ---- file iterators ----------------------------------------------------- #

def _service_files():
    """Yield (rel_path, abs_path) for every service-layer module."""
    svc_dir = os.path.join(GUI_DIR, "services")
    for name in sorted(os.listdir(svc_dir)):
        if name.endswith(".py") and name != "__init__.py":
            yield os.path.join("services", name), os.path.join(svc_dir, name)
    vs = os.path.join(GUI_DIR, "view_state.py")
    if os.path.exists(vs):
        yield "view_state.py", vs


def _domain_files():
    """Yield (rel_path, abs_path) for every domain-layer module."""
    for name in sorted(os.listdir(GUI_DIR)):
        if not name.endswith(".py"):
            continue
        if (name.startswith("fit_") or name.startswith("algo_")
                or name.endswith("_creator.py")):
            yield name, os.path.join(GUI_DIR, name)


# ---- tests -------------------------------------------------------------- #

def test_services_do_not_import_presentation():
    """Service modules must not import presentation at module scope."""
    offenders = {}
    for rel, full in _service_files():
        imports = _module_scope_imports(full)
        forbidden = (imports & PRESENTATION_MODULES) | (imports & PRESENTATION_PACKAGES)
        violations = forbidden - ALLOWLIST.get(rel, set())
        if violations:
            offenders[rel] = sorted(violations)
    assert not offenders, (
        "Service modules importing presentation: %s. Move the import inside "
        "a function (deferred) or invert the dependency with a signal."
        % offenders)


def test_domain_does_not_import_presentation_or_services():
    """Domain modules must not import presentation or services at module scope."""
    offenders = {}
    for rel, full in _domain_files():
        imports = _module_scope_imports(full)
        forbidden = (
            (imports & PRESENTATION_MODULES)
            | (imports & PRESENTATION_PACKAGES)
            | (imports & SERVICE_MODULES)
            | (imports & SERVICE_PACKAGES)
        )
        violations = forbidden - ALLOWLIST.get(rel, set())
        if violations:
            offenders[rel] = sorted(violations)
    assert not offenders, (
        "Domain modules importing presentation or services: %s. Domain modules "
        "must depend only on other domain modules and infrastructure."
        % offenders)


def test_sys_path_inventory():
    """sys.path manipulation must not spread beyond the known inventory.

    Main.py and GUI.py set up the import path at startup. Creator plugins
    repeat it so they stay importable when loaded standalone by users (the
    USERDIR/cwd plugin contract). No other module — especially not services
    or controllers — should touch sys.path. A new file appearing here must
    be registered in the inventory and justified."""
    KNOWN_FILES = frozenset({
        # entry points / composition root
        "Main.py",
        "GUI.py",
        # creator plugins (standalone-importable by design)
        "algo_skel_creator.py",
        "cannye_creator.py",
        "fit_alpha12_creator.py",
        "fit_alpha22_creator.py",
        "fit_alpha32_creator.py",
        "fit_alpha_linear_creator.py",
        "fit_alpha_multi_creator.py",
        "fit_alpha_multi_sigma_creator.py",
        "gmm_creator.py",
        "imgseg_creator.py",
        "kmean_creator.py",
    })
    hits = set()
    for root, _dirs, files in os.walk(GUI_DIR):
        if "__pycache__" in root:
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            full = os.path.join(root, name)
            with open(full, encoding="utf-8") as fh:
                source = fh.read()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Attribute)
                        and isinstance(node.value.value, ast.Name)
                        and node.value.value.id == "sys"
                        and node.value.attr == "path"
                        and node.attr in ("append", "insert", "extend")):
                    rel = os.path.relpath(full, GUI_DIR)
                    hits.add(rel)
    unknown = hits - KNOWN_FILES
    assert not unknown, (
        "New sys.path manipulation in: %s. Add to the inventory in this test "
        "if justified, or remove the sys.path call." % sorted(unknown))
    phantom = KNOWN_FILES - hits
    assert not phantom, (
        "Inventory lists files that no longer manipulate sys.path: %s. "
        "Remove them from KNOWN_FILES." % sorted(phantom))


def test_allowlist_has_no_stale_entries():
    """Allowlist entries that no longer match mean the violation was fixed --
    remove the entry so a regression is caught immediately."""
    stale = {}
    for rel, allowed in ALLOWLIST.items():
        full = os.path.join(GUI_DIR, rel)
        if not os.path.exists(full):
            stale[rel] = sorted(allowed) + ["(file does not exist)"]
            continue
        imports = _module_scope_imports(full)
        unused = allowed - imports
        if unused:
            stale[rel] = sorted(unused)
    assert not stale, "Allowlist entries matching nothing: %s" % stale


def _package_modules(pkg_dir):
    """Return the set of .py basenames (sans extension) in a package dir."""
    return {
        name[:-3]
        for name in os.listdir(pkg_dir)
        if name.endswith(".py") and name != "__init__.py"
    }


def test_services_roster_matches_all():
    """Every .py in gui/services/ must appear in services/__init__.__all__."""
    import importlib.util
    init = os.path.join(GUI_DIR, "services", "__init__.py")
    spec = importlib.util.spec_from_file_location("services", init)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    declared = set(mod.__all__)
    on_disk = _package_modules(os.path.join(GUI_DIR, "services"))
    unlisted = on_disk - declared
    assert not unlisted, (
        "Modules in gui/services/ not in __all__: %s. Add them to "
        "services/__init__.py so the boundary guard covers them."
        % sorted(unlisted))
    phantom = declared - on_disk
    assert not phantom, (
        "__all__ lists modules that do not exist: %s" % sorted(phantom))


def test_controllers_roster_matches_all():
    """Every .py in gui/controllers/ must appear in controllers/__init__.__all__."""
    import importlib.util
    init = os.path.join(GUI_DIR, "controllers", "__init__.py")
    spec = importlib.util.spec_from_file_location("controllers", init)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    declared = set(mod.__all__)
    on_disk = _package_modules(os.path.join(GUI_DIR, "controllers"))
    unlisted = on_disk - declared
    assert not unlisted, (
        "Modules in gui/controllers/ not in __all__: %s. Add them to "
        "controllers/__init__.py so the boundary guard covers them."
        % sorted(unlisted))
    phantom = declared - on_disk
    assert not phantom, (
        "__all__ lists modules that do not exist: %s" % sorted(phantom))
