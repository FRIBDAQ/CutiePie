"""Spectrum DataFrame export — the Qt-free core of the Jupyter cluster.

When a Jupyter session is launched from the GUI, the current spectra are dumped
to a gzip-compressed CSV so the notebook can load them with pandas. The store
hands over a nested dict ``{spectrumName: {info1: , info2: , ...}}``; this module
reshapes it into a columnar table (one row per spectrum), flattening any NumPy
count arrays to plain Python lists so they survive the CSV round-trip, then
writes it.

Qt-free by construction: it takes the already-extracted spectrum dict and a
filepath as arguments and imports only pandas/numpy. The under/overflow
statistics columns come through the optional ``statistics_fetcher`` seam — a
callable mapping a spectrum name to its under/overflow dict (in practice the
``.get`` of the map ``ConnectionManager.getSpectrumStatistics`` builds from one
``/spectcl/specstats`` REST call) — so this module never touches REST itself. The MainWindow keeps the Qt shell — reading the filename widget, the
WebWindow lifecycle, notebook process control (``notebook_process.py``) — and
calls in here. Extracted from ``GUI.py:createDf`` so the reshape/flatten logic
is unit-testable without PyQt5 (mirrors ``geometry_io`` / ``display_slot`` /
``notebook_process``).
"""

import numpy as np
import pandas as pd

# Under/overflow statistics columns, filled per spectrum through the
# ``statistics_fetcher`` seam (sourced from the ``/spectcl/specstats`` REST
# reply). Any value the server doesn't report (e.g. the y pair on 1-D spectra)
# is NaN.
_STAT_COLUMNS = ['xunderflow', 'xoverflow', 'yunderflow', 'yoverflow']

# Column order of the exported table. Every spectrum contributes one row; the
# first column is the spectrum name, the rest come from the per-spectrum info
# dict returned by SpectrumStore.as_dict(), with the statistics columns
# bracketing ``data``: underflows before it, overflows after it.
_COLUMNS = ['name', 'dim', 'binx', 'minx', 'maxx', 'biny', 'miny', 'maxy',
            'xunderflow', 'yunderflow', 'data', 'xoverflow', 'yoverflow',
            'parameters', 'type']


def build_spectrum_dataframe(spectrum_dict, statistics_fetcher=None):
    """Reshape a ``{name: {info: value}}`` store dict into a pandas DataFrame.

    NumPy arrays (the ``data`` counts) are flattened to lists — 1-D via
    ``tolist()``, 2-D to a nested list — so the CSV holds plain sequences rather
    than ``ndarray`` reprs. Non-array values pass through unchanged. Moved
    verbatim from ``GUI.py:createDf``.

    ``statistics_fetcher(name)``, if given, must return the per-spectrum
    ``statistics`` mapping (or a false value when unavailable); its
    xunderflow/xoverflow/yunderflow/yoverflow entries fill the statistics
    columns, with NaN for anything missing.
    """
    formated_dict = {col: [] for col in _COLUMNS}
    for spectrum_name, info_dict in spectrum_dict.items():
        formated_dict["name"].append(spectrum_name)
        stats = statistics_fetcher(spectrum_name) if statistics_fetcher else None
        for col in _STAT_COLUMNS:
            formated_dict[col].append(stats.get(col, np.nan) if stats else np.nan)
        for key_info, val_info in info_dict.items():
            # ndarray data is parsed to a list (1d) or list of lists (2d) so the
            # data parsing is easier from the csv file.
            if isinstance(val_info, np.ndarray):
                to_list = []
                if len(val_info.shape) == 1:
                    to_list = val_info.tolist()
                if len(val_info.shape) == 2:
                    to_list = [[item for item in row] for row in val_info]
                val_info = to_list
            formated_dict[key_info].append(val_info)
    return pd.DataFrame.from_dict(formated_dict)


def export_spectrum_csv(spectrum_dict, filepath, statistics_fetcher=None):
    """Build the spectrum DataFrame and write it as a gzip-compressed CSV."""
    df = build_spectrum_dataframe(spectrum_dict, statistics_fetcher)
    df.to_csv(filepath, index=False, compression='gzip')
