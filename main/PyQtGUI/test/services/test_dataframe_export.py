"""Headless net for the Jupyter dataframe-export cluster extracted from GUI.py.

Pins the reshape + ndarray-flatten + gzip-CSV behavior that was previously
buried in MainWindow.createDf: the nested store dict becomes a columnar table
(one row per spectrum), 1-D count arrays flatten to lists and 2-D to nested
lists, non-array fields pass through, and the file round-trips back through
pandas. Qt-free (pandas/numpy only), so it runs in the system python3 with no
PyQt5.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

import numpy as np
import pandas as pd
import pytest

from services.dataframe_export import build_spectrum_dataframe, export_spectrum_csv


def _spectrum(dim, data, **overrides):
    info = {'dim': dim, 'binx': 10, 'minx': 0, 'maxx': 10,
            'biny': 0, 'miny': 0, 'maxy': 0,
            'data': data, 'parameters': ['p.x'], 'type': '1'}
    info.update(overrides)
    return info


# ----------------------------------------------------------------- build shape

def test_columns_and_one_row_per_spectrum():
    d = {'h1': _spectrum(1, np.array([1, 2, 3])),
         'h2': _spectrum(1, np.array([4, 5]))}
    df = build_spectrum_dataframe(d)
    assert list(df.columns) == ['name', 'dim', 'binx', 'minx', 'maxx',
                                'biny', 'miny', 'maxy', 'data',
                                'parameters', 'type']
    assert len(df) == 2
    assert list(df['name']) == ['h1', 'h2']


def test_empty_dict_yields_empty_frame():
    df = build_spectrum_dataframe({})
    assert len(df) == 0
    assert 'name' in df.columns


# -------------------------------------------------------------- ndarray flatten

def test_1d_array_flattens_to_list():
    df = build_spectrum_dataframe({'h1': _spectrum(1, np.array([1, 2, 3]))})
    assert df['data'].iloc[0] == [1, 2, 3]
    assert isinstance(df['data'].iloc[0], list)


def test_2d_array_flattens_to_nested_list():
    df = build_spectrum_dataframe({'h1': _spectrum(2, np.array([[1, 2], [3, 4]]))})
    assert df['data'].iloc[0] == [[1, 2], [3, 4]]


def test_non_array_fields_pass_through():
    df = build_spectrum_dataframe({'h1': _spectrum(1, np.array([1]),
                                                   parameters=['p.x', 'p.y'])})
    assert df['parameters'].iloc[0] == ['p.x', 'p.y']
    assert df['type'].iloc[0] == '1'
    assert df['binx'].iloc[0] == 10


# ---------------------------------------------------------------- gzip round-trip

def test_export_writes_readable_gzip_csv(tmp_path):
    d = {'h1': _spectrum(1, np.array([1, 2, 3])),
         'h2': _spectrum(2, np.array([[1, 2], [3, 4]]))}
    out = str(tmp_path / 'spectra.csv')
    export_spectrum_csv(d, out)
    assert os.path.exists(out)
    back = pd.read_csv(out, compression='gzip')
    assert list(back['name']) == ['h1', 'h2']
    assert len(back) == 2
