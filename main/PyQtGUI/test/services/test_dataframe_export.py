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
                                'parameters', 'type',
                                'xunderflow', 'xoverflow',
                                'yunderflow', 'yoverflow']
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


# ------------------------------------------------------- under/overflow statistics

STAT_COLS = ['xunderflow', 'xoverflow', 'yunderflow', 'yoverflow']


def test_no_fetcher_yields_nan_statistics():
    df = build_spectrum_dataframe({'h1': _spectrum(1, np.array([1, 2]))})
    for col in STAT_COLS:
        assert np.isnan(df[col].iloc[0])


def test_fetcher_fills_statistics_columns():
    stats = {'h1': {'xunderflow': 3, 'xoverflow': 7,
                    'yunderflow': 1, 'yoverflow': 0}}
    df = build_spectrum_dataframe({'h1': _spectrum(2, np.array([[1, 2], [3, 4]]))},
                                  statistics_fetcher=stats.get)
    assert df['xunderflow'].iloc[0] == 3
    assert df['xoverflow'].iloc[0] == 7
    assert df['yunderflow'].iloc[0] == 1
    assert df['yoverflow'].iloc[0] == 0


def test_missing_statistics_keys_become_nan():
    # 1-D spectra report only the x pair — the y pair must degrade to NaN.
    stats = {'h1': {'xunderflow': 5, 'xoverflow': 2}}
    df = build_spectrum_dataframe({'h1': _spectrum(1, np.array([1, 2]))},
                                  statistics_fetcher=stats.get)
    assert df['xunderflow'].iloc[0] == 5
    assert df['xoverflow'].iloc[0] == 2
    assert np.isnan(df['yunderflow'].iloc[0])
    assert np.isnan(df['yoverflow'].iloc[0])


def test_fetcher_returning_none_or_empty_yields_nan():
    # Fetcher knows h1 but not h2 (None) and returns {} for h3.
    stats = {'h1': {'xunderflow': 1}, 'h3': {}}
    d = {'h1': _spectrum(1, np.array([1])),
         'h2': _spectrum(1, np.array([2])),
         'h3': _spectrum(1, np.array([3]))}
    df = build_spectrum_dataframe(d, statistics_fetcher=stats.get)
    assert df['xunderflow'].iloc[0] == 1
    for col in STAT_COLS:
        assert np.isnan(df[col].iloc[1])
        assert np.isnan(df[col].iloc[2])


def test_fetcher_called_once_per_spectrum_with_name():
    calls = []

    def fetcher(name):
        calls.append(name)
        return {}

    d = {'h1': _spectrum(1, np.array([1])),
         'h2': _spectrum(1, np.array([2]))}
    build_spectrum_dataframe(d, statistics_fetcher=fetcher)
    assert calls == ['h1', 'h2']


# ---------------------------------------------------------------- gzip round-trip

def test_export_writes_readable_gzip_csv(tmp_path):
    d = {'h1': _spectrum(1, np.array([1, 2, 3])),
         'h2': _spectrum(2, np.array([[1, 2], [3, 4]]))}
    out = str(tmp_path / 'spectra.csv')
    stats = {'h1': {'xunderflow': 4, 'xoverflow': 9}}
    export_spectrum_csv(d, out, statistics_fetcher=stats.get)
    assert os.path.exists(out)
    back = pd.read_csv(out, compression='gzip')
    assert list(back['name']) == ['h1', 'h2']
    assert len(back) == 2
    assert back['xunderflow'].iloc[0] == 4
    assert back['xoverflow'].iloc[0] == 9
    assert np.isnan(back['yunderflow'].iloc[0])
    assert np.isnan(back['xunderflow'].iloc[1])
