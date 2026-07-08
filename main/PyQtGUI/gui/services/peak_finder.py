"""Peak finding — the Qt-free core of the peaks cluster.

The peak-analysis feature runs scipy's ``find_peaks`` over the counts of the
selected spectrum, restricted to the currently-visible x range, and reports the
found peaks (position + FWHM). The numeric half — clip the spectrum to the view
window, then locate peaks — is pure numpy/scipy and lives here; the MainWindow
keeps the Qt/matplotlib shell (reading the width widget and axes limits, drawing
the ``v``/vline/hline/text markers, wiring the per-peak checkboxes).

Qt-free by construction: it takes plain arrays and scalars and imports only
numpy/scipy. Extracted from ``GUI.py:analyzePeak`` / ``update_peak_output`` so
the find + output-formatting logic is unit-testable without PyQt5 (mirrors
``geometry_io`` / ``dataframe_export`` / ``display_slot``).
"""

import numpy as np
from scipy.signal import find_peaks


def find_peaks_in_range(x_axis, y_data, xmin, xmax, width):
    """Locate peaks in ``y_data`` restricted to the ``[xmin, xmax)`` x window.

    ``x_axis`` is the full bin-centre/edge axis (e.g. from ``createRange``) and
    ``y_data`` the matching counts. Bins whose x falls in the half-open window
    ``[xmin, xmax)`` are kept, then ``scipy.signal.find_peaks`` runs on the
    clipped counts with ``prominence=1`` and the given ``width``.

    Returns ``(datax, datay, peaks, properties)`` where ``datax``/``datay`` are
    the clipped arrays (NumPy), ``peaks`` the indices into them, and
    ``properties`` scipy's property dict. Moved verbatim from
    ``GUI.py:analyzePeak``.
    """
    x = []
    y = []
    # create new tmp list with subrange for fitting
    for i in range(len(x_axis)):
        if x_axis[i] >= xmin and x_axis[i] < xmax:
            x.append(x_axis[i])
            y.append(y_data[i])
    datax = np.array(x)
    datay = np.array(y)
    peaks, properties = find_peaks(datay, prominence=1, width=width)
    return datax, datay, peaks, properties


def format_peak_output(peaks, properties, datax):
    """Build the per-peak result lines shown in the peak-analysis output box.

    One line per peak: its position (``datax`` at the peak index) and FWHM (the
    scipy ``widths`` property). Moved verbatim from ``GUI.py:update_peak_output``.
    """
    x = datax.tolist()
    lines = []
    for i in range(len(peaks)):
        lines.append("Peak" + str(i + 1) + "\n\tpeak @ " + str(int(x[peaks[i]]))
                     + ", FWHM=" + str(int(properties['widths'][i])))
    return lines
