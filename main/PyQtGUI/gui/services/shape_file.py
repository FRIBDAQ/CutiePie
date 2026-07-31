"""Shape-file format guard for the AlphaEMGMulti / AlphaEMGMultiSigma fits.

The shape file is a TXT/CSV with one row per alpha line::

    isotope, half_life, energy_keV, alpha_intensity, sigma, tau1, tau2, eta, flag, chain

``_load_shapes`` in ``fit_alpha_multi_sigma_creator.py`` tolerantly skips blank,
comment (``#``), and malformed rows — so a wrong file (a calibration file, a
geometry ``.win``, junk) parses to ZERO shapes and the fit silently runs with no
components. This module is the up-front guard: it answers "does this file even
look like a shape file?" so the caller can abort with a clear message.

Kept Qt-free and dependency-light (no ``lmfit``) on purpose, so it is
unit-testable in the headless environment where the creator module — which pulls
in ``lmfit`` — cannot be imported.
"""

import csv
import json
import os


def _is_float(tok):
    try:
        float(tok)
        return True
    except (TypeError, ValueError):
        return False


def count_shape_rows(path):
    """Count rows that parse as a shape row, mirroring ``_load_shapes``' own
    accept criteria: a numeric energy (col 2) and numeric sigma/tau1/tau2
    (cols 4-6). Blank, comment (``#``) and short/malformed rows are skipped,
    exactly as the real loader skips them, so this never over- or under-counts
    relative to what the fit would actually use.

    Decoded with ``errors="replace"``: everything this reads is numeric except
    the isotope and chain names, so a file saved in latin-1 (an accent in an
    isotope label) still counts its rows instead of failing to decode. Raises
    ``OSError`` if the file cannot be opened — the two guards below catch it."""
    n = 0
    with open(path, "r", newline="", errors="replace") as f:
        for raw in csv.reader(f, skipinitialspace=True):
            if not raw or all(not str(x).strip() for x in raw):
                continue
            if str(raw[0]).strip().startswith("#"):
                continue
            cells = [str(x).strip() for x in raw]
            if len(cells) < 7:
                continue
            if (_is_float(cells[2]) and _is_float(cells[4])
                    and _is_float(cells[5]) and _is_float(cells[6])):
                n += 1
    return n


def looks_like_shape_file(path):
    """True if `path` reads as a shape file (>= 2 valid shape rows). Used as a
    swap guard by the calibration loader — a shape file selected in the
    calibration slot must not be mistaken for calibration numbers.

    Never raises, and now means it: an unreadable path is `OSError` and returns
    False, and undecodable bytes cannot raise at all because the read replaces
    them (a binary file then yields no valid rows, which is the same answer)."""
    try:
        return count_shape_rows(path) >= 2
    except OSError:
        return False


def validate_shape_file(path):
    """Return a list of human-readable problems; empty list = looks like a
    shape file. Never raises. Names common mis-selected formats (a binary file,
    a calibration JSON, a geometry/session or other JSON object) so the caller's
    dialog can be specific about what the user picked by mistake."""
    if not path or not os.path.isfile(path):
        return [f"file not found: {path}"]
    # read bytes, not text: the whole point of this function is to survive
    # whatever the file dialog hands it, and "All files (*)" hands it anything
    try:
        with open(path, "rb") as f:
            raw_head = f.read(4096)
    except OSError as e:
        return [f"could not read file: {e}"]

    head = raw_head.decode("utf-8", errors="replace")
    # A NUL says binary outright. Failing that, count how much of the head did
    # not decode: a shape file saved in latin-1 has one bad byte per accented
    # isotope name (a few percent at most), while binary content is typically
    # half the bytes, so the threshold sits far from either.
    if b"\x00" in raw_head or (head and head.count("�") / len(head) > 0.3):
        return ["looks like a binary file (a ROOT file, an image), "
                "not a shape file"]

    if head.lstrip().startswith("{"):
        obj = None
        try:
            obj = json.loads(head)
        except Exception:
            obj = None
        if isinstance(obj, dict) and (
                {"a", "b"} <= obj.keys() or {"calib_a", "calib_b"} <= obj.keys()):
            return ["looks like a calibration file (JSON a/b), not a shape file"]
        return ["looks like a geometry/session or JSON file, not a shape file"]

    try:
        n = count_shape_rows(path)
    except OSError as e:
        return [f"could not read file: {e}"]
    if n == 0:
        return ["no valid shape rows found (expected rows of: isotope, "
                "half_life, energy_keV, intensity, sigma, tau1, tau2, eta, "
                "flag, chain)"]
    return []
