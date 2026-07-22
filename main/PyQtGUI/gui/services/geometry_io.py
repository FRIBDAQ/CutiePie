"""Geometry file read/serialize — the Qt-free core of the geometry-IO cluster.

A "geometry" describes the tab's pad grid: how many rows/cols and, per pad, the
spectrum name, optional x/y view range, and log-scale flag. Two on-disk formats
are supported:

* **native** — a single-line Python dict literal (what :func:`serialize_geometry`
  writes), e.g. ``{"row": 2, "col": 2, "geo": {0: {"name": "h1", ...}}}``.
* **legacy** — a Xamine/dispwind ``.win`` file beginning with a ``Geometry R,C``
  line, parsed by :func:`_parse_old_geo`.

This module holds only the parsing/serialization; the QFileDialogs and the
MainWindow orchestration (``loadGeo``/``saveGeo``) stay in the presentation layer
and call in here. Extracted from ``GUI.py`` so the format logic is unit-testable
without PyQt5 (mirrors ``display_slot`` / ``notebook_process`` / ``logger``).
"""

import ast
import os
import re
import logging

_module_logger = logging.getLogger(__name__)


def read_geometry(filename, logger=None):
    """Read a geometry file and return the normalized structure::

        {"row": <nrows>, "col": <ncols>,
         "geo": {flatIndex: {"name": str,
                             "x": [min, max] | None,
                             "y": [min, max] | None,
                             "scale": bool}}}

    Returns ``None`` for an empty file or an unrecognized format. The format is
    sniffed from the first non-blank, non-comment line: ``Geometry ...`` → legacy
    ``.win`` parser; ``{`` → native dict literal.
    """
    log = logger or _module_logger
    log.info('read_geometry - filename: %s', filename)
    if os.stat(filename).st_size == 0:
        log.warning('read_geometry - empty geometry file: %s', filename)
        return None

    firstMeaningful = ""
    with open(filename) as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                firstMeaningful = stripped
                break

    if firstMeaningful.lower().startswith("geometry"):
        return _parse_old_geo(filename, log)
    if firstMeaningful.startswith("{"):
        # ast.literal_eval replaces the historical eval(): equivalent for every
        # legitimate dict-literal geometry, inert for hostile file contents
        # (decision taken 2026-07-08, was deferred).
        with open(filename, "r") as fh:
            text = fh.read()
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError, TypeError, RecursionError):
            log.warning('read_geometry - invalid geometry dict literal in %s', filename)
            return None
    log.warning('read_geometry - unrecognized geometry file format: %s', filename)
    return None


def _parse_old_geo(filename, logger=None):
    """Parse a legacy Xamine/dispwind ``.win`` geometry file into the same
    structure :func:`read_geometry` returns for the native format.

    Windows get sequential flat indices in file order (matching the original
    loader). ``COUNTSAXIS`` maps to log scale; ``Expanded`` supplies the x/y view
    range when present, otherwise the spectrum keeps its natural range. ``SCALE``,
    ``Refresh``, ``MAPPED`` and other per-window settings are ignored.
    """
    log = logger or _module_logger
    log.info('_parse_old_geo - filename: %s', filename)
    nrow = ncol = None
    properties = {}
    index  = 0
    name   = None
    scale  = False
    xRange = None
    yRange = None

    def _numbers(text):
        return [float(n) for n in re.findall(r'-?\d+(?:\.\d+)?', text)]

    try:
        with open(filename) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                keyword = line.split()[0].lower()
                if keyword == 'geometry':
                    nums = _numbers(line)
                    if len(nums) >= 2:
                        nrow, ncol = int(nums[0]), int(nums[1])
                elif keyword == 'window':
                    name, scale, xRange, yRange = None, False, None, None
                    match = re.search(r'"([^"]*)"', line)
                    if match:
                        name = match.group(1)
                elif keyword == 'countsaxis':
                    scale = True
                elif keyword == 'expanded':
                    nums = _numbers(line)
                    if len(nums) >= 2:
                        xRange = [nums[0], nums[1]]
                    if len(nums) >= 4:
                        yRange = [nums[2], nums[3]]
                elif keyword == 'endwindow':
                    if name:
                        properties[index] = {"name": name, "x": xRange,
                                             "y": yRange, "scale": scale}
                    index += 1
                    name, scale, xRange, yRange = None, False, None, None
    except OSError:
        log.warning('_parse_old_geo - could not read %s', filename, exc_info=True)
        return None

    if nrow is None or ncol is None:
        log.warning('_parse_old_geo - no "Geometry" line found in %s', filename)
        return None
    return {"row": nrow, "col": ncol, "geo": properties}


def serialize_geometry(row, col, properties):
    """Serialize a geometry to the native on-disk format (a dict-literal string).

    `properties` is the ``{flatIndex: {"name","x","y","scale"}}`` mapping the caller
    has gathered from the current pads. The result round-trips through
    :func:`read_geometry` (native branch). Kept here so the single canonical format
    lives with its parser.
    """
    return str({"row": int(row), "col": int(col), "geo": properties})


def serialize_session(tabs):
    """Serialize a multi-tab session to the native v2 dict-literal string.

    `tabs` is a list of {"name": str, "row": int, "col": int, "geo": {...}} in
    screen order; each per-tab "geo" uses exactly the single-tab property
    shape serialize_geometry writes. Round-trips through read_geometry_any.
    """
    return str({"version": 2, "kind": "cutiepie-session",
                "tabs": [{"name": str(t["name"]), "row": int(t["row"]),
                          "col": int(t["col"]), "geo": t["geo"]} for t in tabs]})


def validate_session(obj):
    """Return a list of human-readable problems with a parsed v2 session
    object; empty list = valid. Never raises."""
    if not isinstance(obj, dict):
        return ["session file is not a dict literal"]
    tabs = obj.get("tabs")
    if not isinstance(tabs, list) or not tabs:
        return ["session has no tabs (or the tabs list is empty)"]
    errors = []
    for i, tab in enumerate(tabs):
        if not isinstance(tab, dict):
            errors.append(f"tab {i}: not a dict")
            continue
        if not isinstance(tab.get("name"), str):
            errors.append(f"tab {i}: missing/invalid name")
        row, col = tab.get("row"), tab.get("col")
        if not (isinstance(row, int) and row >= 1 and isinstance(col, int) and col >= 1):
            errors.append(f"tab {i}: row/col must be integers >= 1")
        if not isinstance(tab.get("geo"), dict):
            errors.append(f"tab {i}: missing/invalid geo dict")
    return errors


def validate_single_geometry(obj):
    """Return a list of human-readable problems with a parsed single-tab
    geometry object; empty list = valid. Never raises. Used to reject a
    multi-tab session (or any malformed dict) that reached the single-tab
    loader — the shape ``_applyGeometryToCurrentTab`` requires is
    ``{"row": int>=1, "col": int>=1, "geo": dict}``."""
    if not isinstance(obj, dict):
        return ["geometry file is not a dict literal"]
    errors = []
    row, col = obj.get("row"), obj.get("col")
    if not (isinstance(row, int) and row >= 1 and isinstance(col, int) and col >= 1):
        errors.append("row/col must be integers >= 1")
    if not isinstance(obj.get("geo"), dict):
        errors.append("missing/invalid geo dict")
    return errors


def read_geometry_any(filename, logger=None):
    """Read any geometry vintage. Returns ("session", payload) for a v2
    multi-tab file, ("single", payload) for a v1/native or legacy .win file
    (payload exactly as read_geometry returns it), or None for
    unreadable/invalid files. Sessions are validated here so callers can
    replace the workspace only after a fully-good parse."""
    log = logger or _module_logger
    if os.stat(filename).st_size == 0:
        log.warning('read_geometry_any - empty geometry file: %s', filename)
        return None
    firstMeaningful = ""
    with open(filename) as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                firstMeaningful = stripped
                break
    if firstMeaningful.startswith("{"):
        with open(filename, "r") as fh:
            text = fh.read()
        try:
            obj = ast.literal_eval(text)
        except (ValueError, SyntaxError, TypeError, RecursionError):
            log.warning('read_geometry_any - invalid geometry dict literal in %s', filename)
            return None
        if isinstance(obj, dict) and "tabs" in obj:
            errors = validate_session(obj)
            if errors:
                log.warning('read_geometry_any - invalid session file %s: %s',
                            filename, "; ".join(errors))
                return None
            return ("session", obj)
        errors = validate_single_geometry(obj)
        if errors:
            log.warning('read_geometry_any - invalid single geometry %s: %s',
                        filename, "; ".join(errors))
            return None
        return ("single", obj)
    result = read_geometry(filename, log)
    return ("single", result) if result is not None else None
