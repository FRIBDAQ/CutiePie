"""Import a CSV spectrum into SpecTcl as a 1-d spectrum: read the file, work
out the axis, then create, fill and bind the spectrum over REST. Qt-free; the
dialogs around it are in controllers/csv_import_controller.py."""

import os
import re

import numpy as np

# SpecTcl needs a parameter to make a spectrum. Each imported spectrum gets
# one of its own, named import.001, import.002 and so on: gates are defined
# on parameters, so spectra sharing one would share their gates. No event
# ever sets these parameters, so analysis never increments an imported
# spectrum.
PARAMETER_PREFIX = "import"

# Characters a spectrum name cannot have here: the name is written into Tcl
# scripts inside braces and used as a glob pattern to look the spectrum up.
FORBIDDEN_IN_NAME = ' \t\r\n{}[]\\"$;*?'

# Longest Tcl script sent in one request (it travels in the URL).
MAX_SCRIPT_LENGTH = 3000

# Where the fill scripts keep their loop variables, see fill_scripts.
TCL_NAMESPACE = "::cutiepie_csv_import"


class CsvImportError(Exception):
    """SpecTcl did not take the spectrum; the message says at which step."""


# ------------------------------------------------------------ reading

def read_interspec_csv(path):
    """Read the spectrum out of an InterSpec CSV. The file opens with free
    header lines (acquisition settings, calibration, ROIs) and then has a
    table under a row naming its columns, e.g.

        Channel Data:
        Channel,Energy,Counts
        0,-751.0,0

    x is the Energy column, or Channel when the file has no energy; y is
    Counts (some exports call it Data). Returns (x, y) as float arrays.
    Raises ValueError when no such table is found."""
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        rows = [[c.strip() for c in line.split(",")] for line in fh]

    def col(names, prefixes):
        for prefix in prefixes:
            for i, name in enumerate(names):
                if name.startswith(prefix):
                    return i
        return None

    for n, row in enumerate(rows):
        names = [c.lower() for c in row]
        xi = col(names, ("energy", "channel"))
        yi = col(names, ("counts", "count", "data"))
        if xi is None or yi is None or xi == yi:
            continue
        x, y = [], []
        for r in rows[n + 1:]:
            try:
                xv, yv = float(r[xi]), float(r[yi])
            except (ValueError, IndexError):
                if x:
                    break               # the table ended
                continue                # blank or unit line before the data
            x.append(xv)
            y.append(yv)
        if x:
            return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    if not any(any(c.lower().startswith(("counts", "count", "data")) for c in row)
               for row in rows):
        raise ValueError(
            "No spectrum table found. An InterSpec CSV has a row naming the "
            "columns (Channel, Energy, Counts) followed by the data rows.")
    raise ValueError("The spectrum table has no numeric rows.")


def read_spectcl_csv(path):
    """Read plain rows of x,y. A single column is y against its row number."""
    arr = np.genfromtxt(path, delimiter=",", comments="#")
    if arr.ndim == 1:
        return np.arange(arr.size, dtype=float), arr.astype(float)
    return arr[:, 0].astype(float), arr[:, 1].astype(float)


def read_csv(path, csv_format):
    """The spectrum in `path`, written by `csv_format` ("InterSpec" or
    "SpecTcl"): (x, y) with the rows that are not numbers dropped, sorted by
    x. Raises ValueError (or what the reader raises) when it has none."""
    if csv_format == "InterSpec":
        x, y = read_interspec_csv(path)
    else:
        x, y = read_spectcl_csv(path)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size == 0:
        raise ValueError("The file has no rows of numbers.")
    order = np.argsort(x)
    return x[order], y[order]


# ------------------------------------------------------------ the spectrum

def default_spectrum_name(path):
    """The file name without its extension, made usable as a spectrum name."""
    stem = os.path.splitext(os.path.basename(path))[0]
    name = "".join("_" if c in FORBIDDEN_IN_NAME else c for c in stem)
    return name or "csv"


def spectrum_name_problem(name):
    """Why `name` cannot be used as the spectrum name; "" when it can."""
    if not name:
        return "The spectrum needs a name."
    bad = sorted({c for c in name if c in FORBIDDEN_IN_NAME})
    if bad:
        shown = " ".join("space" if c.isspace() else c for c in bad)
        return "The name cannot contain: " + shown
    return ""


def axis_for(x):
    """The axis of a 1-d spectrum whose bin centres are the x values:
    (low, high, bins, even). SpecTcl axes are linear, so the bins are equally
    wide from the first x to the last; `even` is False when some x value is
    then more than half a bin away from the centre of its own bin."""
    x = np.asarray(x, dtype=float)
    bins = int(x.size)
    if bins < 2:
        raise ValueError("A spectrum needs at least two rows.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("Two rows have the same x value.")
    width = (x[-1] - x[0]) / (bins - 1)
    centres = x[0] + width * np.arange(bins)
    even = bool(np.max(np.abs(x - centres)) <= 0.5 * width)
    return float(x[0] - width / 2), float(x[-1] + width / 2), bins, even


def channel_counts(y):
    """The channel contents SpecTcl takes: whole numbers, none below zero."""
    counts = np.rint(np.asarray(y, dtype=float))
    return [int(v) for v in np.clip(counts, 0, None)]


def _num(value):
    return "%.12g" % value


def parameter_name(number):
    return "%s.%03d" % (PARAMETER_PREFIX, number)


def _import_parameters(text):
    """The numbers of the import.NNN parameters named in a Tcl list."""
    return [int(n) for n in re.findall(
        r"(?<![^\s{])%s\.(\d+)(?![^\s}])" % re.escape(PARAMETER_PREFIX), text)]


def next_parameter(rest):
    """The first import.NNN after the ones SpecTcl already has."""
    listed = _run(rest, "parameter -list {%s.*}" % PARAMETER_PREFIX,
                  "listing the parameters")
    return parameter_name(max(_import_parameters(str(listed)), default=0) + 1)


def parameter_of(rest, name):
    """The import.NNN parameter spectrum `name` is defined on; "" when the
    spectrum does not exist or is defined on something else."""
    definition = str(_run(rest, "spectrum -list {%s}" % name,
                          "looking for the spectrum"))
    m = re.search(r"(?<![^\s{])%s\}?\s+1\s+\{?\s*(%s\.\d+)\s*\}?\s" % (
        re.escape(name), re.escape(PARAMETER_PREFIX)), definition)
    return m.group(1) if m else ""


def create_script(name, parameter, low, high, bins):
    return "spectrum {%s} 1 {%s} {{%s %s %d}}" % (
        name, parameter, _num(low), _num(high), bins)


def parameter_script(parameter, low, high, bins):
    """Make the parameter unless SpecTcl has it already."""
    return ("if {[llength [parameter -list {%s}]] == 0} "
            "{treeparameter -create {%s} %s %s %d {}}" % (
                parameter, parameter, _num(low), _num(high), bins))


def fill_scripts(name, counts, max_length=MAX_SCRIPT_LENGTH):
    """Tcl scripts that set the channels, several channels per script so a
    spectrum takes a handful of requests rather than one per channel. Runs of
    empty channels are left out: a new spectrum is empty already.

    The REST script service runs a script at global level, where a bare loop
    would overwrite variables of the user's. The loop variables are therefore
    declared in a namespace of their own, which is deleted afterwards. (Tcl's
    `apply` would do, but in SpecTcl `apply` is the command that applies a
    gate to a spectrum.)"""
    head = "namespace eval %s {variable v; variable channel " % TCL_NAMESPACE
    tail = (" {channel -set {%s} $channel $v; incr channel}}; "
            "namespace delete %s" % (name, TCL_NAMESPACE))
    scripts = []
    start, values, length = None, [], 0

    def flush():
        nonlocal start, values, length
        while values and values[-1] == "0":       # trailing empty channels
            values.pop()
        if values:
            scripts.append("%s%d; foreach v {%s}%s" % (
                head, start, " ".join(values), tail))
        start, values, length = None, [], 0

    for channel, count in enumerate(counts):
        if start is None:
            if count == 0:
                continue
            start = channel
        text = str(count)
        if values and len(head) + len(tail) + length + len(text) + 24 > max_length:
            flush()
            if count == 0:
                continue
            start = channel
        values.append(text)
        length += len(text) + 1
    flush()
    return scripts


def _run(rest, script, doing):
    reply = rest.script(script)
    if reply is None:
        raise CsvImportError(
            "No usable answer from SpecTcl while %s.\n"
            "Check that CutiePie is connected. Importing needs the REST "
            "script service of SpecTcl 5.10 or later." % doing)
    if reply.get("status") != "OK":
        raise CsvImportError("SpecTcl failed while %s:\n%s" % (
            doing, reply.get("detail") or reply.get("status") or "no reason given"))
    return reply.get("detail") or ""


def spectrum_exists(rest, name):
    """True when SpecTcl has a spectrum of that name."""
    return bool(str(_run(rest, "spectrum -list {%s}" % name,
                         "looking for the spectrum")).strip())


def import_spectrum(rest, name, x, y, replace=False):
    """Create spectrum `name` in SpecTcl from the CSV data, fill it and bind
    it to the display memory, which is what makes it show up in CutiePie.
    `replace` deletes a spectrum of that name first; if that one was an
    imported spectrum, its parameter is used again, so its gates stay. On a
    failure after the spectrum was made it is deleted again, so nothing half
    filled is left. Returns (low, high, bins, parameter). Raises
    CsvImportError."""
    low, high, bins, _even = axis_for(x)
    counts = channel_counts(y)

    parameter = parameter_of(rest, name) if replace else ""
    if not parameter:
        parameter = next_parameter(rest)
    _run(rest, parameter_script(parameter, low, high, bins), "making the parameter")
    if replace:
        _run(rest, "spectrum -delete {%s}" % name, "deleting the old spectrum")
    _run(rest, create_script(name, parameter, low, high, bins), "creating the spectrum")
    try:
        for script in fill_scripts(name, counts):
            _run(rest, script, "filling the spectrum")
        _run(rest, "sbind {%s}" % name, "binding the spectrum to the display")
    except CsvImportError:
        try:
            rest.script("spectrum -delete {%s}" % name)
        except Exception:
            pass
        raise
    return low, high, bins, parameter
