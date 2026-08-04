"""Spectrum DataFrame export — the Qt-free core of the Jupyter cluster. When a
Jupyter session is launched from the GUI, the current spectra are dumped to a
gzip-compressed CSV so the notebook can load them with pandas."""

import os

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


def build_spectrum_dataframe(spectrum_dict, statistics_fetcher=None,
                             arrays_as_lists=True):
    """Reshape a ``{name: {info: value}}`` store dict into a pandas DataFrame.
    NumPy arrays (the ``data`` counts) are flattened to lists — 1-D via
    ``tolist()``, 2-D to a nested list — so the CSV holds plain sequences
    rather than ``ndarray`` reprs."""
    formated_dict = {col: [] for col in _COLUMNS}
    # Walk the SCHEMA, not the record. Appending whatever keys a record
    # happens to carry leaves the columns ragged when one is missing (pandas
    # then refuses to build the frame) and raises outright on a key the schema
    # does not know.
    for spectrum_name, info_dict in spectrum_dict.items():
        formated_dict["name"].append(spectrum_name)
        stats = statistics_fetcher(spectrum_name) if statistics_fetcher else None
        for col in _COLUMNS:
            if col == "name":
                continue
            if col in _STAT_COLUMNS:
                formated_dict[col].append(stats.get(col, np.nan) if stats else np.nan)
                continue
            val_info = info_dict.get(col, np.nan)
            # ndarray data is parsed to a list (1d) or list of lists (2d) so the
            # data parsing is easier from the csv file.
            if isinstance(val_info, np.ndarray) and arrays_as_lists:
                to_list = []
                if len(val_info.shape) == 1:
                    to_list = val_info.tolist()
                if len(val_info.shape) == 2:
                    to_list = [[item for item in row] for row in val_info]
                val_info = to_list
            formated_dict[col].append(val_info)
    return pd.DataFrame.from_dict(formated_dict)


_SIDECAR_SUFFIX = '-counts.npz'


def counts_sidecar_path(filepath):
    """Path of the counts sidecar belonging to the CSV at ``filepath``.
    Derived from the CSV name rather than recorded inside it, so the pair
    moves together by naming convention: ``df-run.gzip`` ->
    ``df-run-counts.npz``."""
    base, _ext = os.path.splitext(str(filepath))
    return base + _SIDECAR_SUFFIX


def _counts_key(row_index):
    """Archive member name for the counts of row ``row_index``. Keyed by
    POSITION, never by spectrum name: names in the field carry spaces,
    brackets and slashes, and a slash would turn an archive member into a
    path."""
    return f's{int(row_index)}'


def export_spectrum_csv(spectrum_dict, filepath, statistics_fetcher=None):
    """Write the spectrum table as a gzip CSV plus a binary counts sidecar.
    The counts go to a compressed ``.npz`` beside the CSV
    (:func:`counts_sidecar_path`) and the CSV's ``data`` column holds the
    archive key instead."""
    df = build_spectrum_dataframe(spectrum_dict, statistics_fetcher,
                                  arrays_as_lists=False)
    counts = {}
    if 'data' in df.columns:
        keys = []
        for i, value in enumerate(df['data']):
            keys.append(_counts_key(i))
            counts[_counts_key(i)] = np.asarray(value)
        df = df.assign(data=keys)
    np.savez_compressed(counts_sidecar_path(filepath), **counts)
    df.to_csv(filepath, index=False,
              compression={'method': 'gzip', 'compresslevel': 1})


def read_spectrum_export(filepath):
    """Load an export written by :func:`export_spectrum_csv` as a DataFrame
    whose ``data`` column holds real NumPy arrays.

    Pre-sidecar artifacts still open: their ``data`` cells hold the counts as a
    stringified list, which is passed through untouched so an old file reads
    exactly as it always did."""
    df = pd.read_csv(filepath, compression='gzip')
    if 'data' not in df.columns or not len(df):
        return df
    first = df['data'].iloc[0]
    if not (isinstance(first, str) and first.startswith('s')):
        return df                       # pre-sidecar artifact: counts inline
    side = counts_sidecar_path(filepath)
    if not os.path.exists(side):
        raise FileNotFoundError(
            f'counts sidecar missing for {filepath}: expected {side}. The CSV '
            f'and its sidecar must travel together.')
    with np.load(side, allow_pickle=False) as archive:
        df = df.assign(data=[archive[key] for key in df['data']])
    return df
