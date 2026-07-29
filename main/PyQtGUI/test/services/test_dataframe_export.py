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
                                'biny', 'miny', 'maxy',
                                'xunderflow', 'yunderflow', 'data',
                                'xoverflow', 'yoverflow',
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


def test_export_uses_fast_gzip_level(tmp_path):
    """The export must compress at the fast level, not the default 9.

    Pinned via the gzip header's XFL byte (offset 8), which the deflate spec
    sets to 2 for "best compression" and 4 for "fastest" — the only in-artifact
    evidence of the level, since the decompressed bytes are identical either
    way. Level 9 spent most of the export wall clock for a file a couple of MB
    smaller, which nobody was waiting on."""
    d = {'h1': _spectrum(2, np.arange(4096).reshape(64, 64))}
    out = str(tmp_path / 'spectra.csv')
    export_spectrum_csv(d, out)
    with open(out, 'rb') as fh:
        header = fh.read(10)
    assert header[:2] == b'\x1f\x8b'          # gzip magic
    assert header[8] == 4, "expected XFL=4 (fastest); 2 means level 9 is back"
    # and it is still a readable gzip CSV
    assert list(pd.read_csv(out, compression='gzip')['name']) == ['h1']


# ------------------------------------------------- counts sidecar (PERF_2 #2)

from services.dataframe_export import counts_sidecar_path, read_spectrum_export


def test_export_writes_a_counts_sidecar_next_to_the_csv(tmp_path):
    d = {'h1': _spectrum(1, np.array([1., 2., 3.]))}
    out = str(tmp_path / 'df-run.gzip')
    export_spectrum_csv(d, out)
    side = counts_sidecar_path(out)
    assert os.path.exists(side)
    assert side.endswith('-counts.npz')


def test_csv_no_longer_carries_stringified_counts(tmp_path):
    """The whole point of the sidecar: the counts must not be re-rendered as
    digits into a CSV cell, which is where the export spent its time."""
    d = {'h1': _spectrum(2, np.arange(2500).reshape(50, 50))}
    out = str(tmp_path / 'df-run.gzip')
    export_spectrum_csv(d, out)
    cell = str(pd.read_csv(out, compression='gzip')['data'].iloc[0])
    assert not cell.startswith('['), "counts were stringified into the CSV"
    assert '2499' not in cell


def test_round_trip_returns_the_original_arrays(tmp_path):
    one = np.array([1., 2., 3., 4.])
    two = np.arange(12.).reshape(3, 4)
    d = {'h1': _spectrum(1, one), 'h2': _spectrum(2, two)}
    out = str(tmp_path / 'df-run.gzip')
    export_spectrum_csv(d, out)
    back = read_spectrum_export(out)
    assert list(back['name']) == ['h1', 'h2']
    assert np.array_equal(back['data'].iloc[0], one)
    assert np.array_equal(back['data'].iloc[1], two)
    assert back['data'].iloc[1].shape == (3, 4)


def test_round_trip_survives_hostile_spectrum_names(tmp_path):
    """Names carry spaces, brackets and slashes in the field (SMOKE A9). The
    sidecar keys off the row position, never the name, so a name that is not a
    legal archive member cannot corrupt the mapping."""
    d = {'my spec': _spectrum(1, np.array([1., 2.])),
         'raw[0]': _spectrum(1, np.array([3., 4.])),
         'a/b': _spectrum(1, np.array([5., 6.]))}
    out = str(tmp_path / 'df-run.gzip')
    export_spectrum_csv(d, out)
    back = read_spectrum_export(out)
    assert list(back['name']) == ['my spec', 'raw[0]', 'a/b']
    assert np.array_equal(back['data'].iloc[2], np.array([5., 6.]))


def test_metadata_columns_are_unchanged_by_the_sidecar(tmp_path):
    d = {'h1': _spectrum(1, np.array([1., 2.]))}
    out = str(tmp_path / 'df-run.gzip')
    export_spectrum_csv(d, out, statistics_fetcher={'h1': {'xunderflow': 4}}.get)
    back = read_spectrum_export(out)
    assert back['binx'].iloc[0] == 10
    assert back['type'].iloc[0] == 1 or back['type'].iloc[0] == '1'
    assert back['xunderflow'].iloc[0] == 4


def test_reader_still_opens_a_pre_sidecar_export(tmp_path):
    """Old artifacts stay readable. Their counts sit in the CSV cell as a
    stringified list and are handed back exactly as pandas read them — the same
    thing an old notebook saw — rather than being parsed into an array, so
    nothing about reading an old file changes."""
    out = str(tmp_path / 'old.gzip')
    df = build_spectrum_dataframe({'h1': _spectrum(1, np.array([1., 2., 3.]))})
    df.to_csv(out, index=False, compression='gzip')
    back = read_spectrum_export(out)
    assert back['data'].iloc[0] == '[1.0, 2.0, 3.0]'
    assert back['binx'].iloc[0] == 10


def test_reader_reports_a_missing_sidecar_clearly(tmp_path):
    d = {'h1': _spectrum(1, np.array([1., 2.]))}
    out = str(tmp_path / 'df-run.gzip')
    export_spectrum_csv(d, out)
    os.unlink(counts_sidecar_path(out))
    with pytest.raises(FileNotFoundError, match='counts sidecar'):
        read_spectrum_export(out)


def test_builder_can_keep_arrays_as_arrays():
    """The sidecar path has no use for the list form: `tolist()` on a 2048^2
    spectrum inflates 34 MB of counts into ~16.7M Python floats (measured 191 MB
    peak) only for the export to convert them straight back. Callers that write
    the counts as binary opt out; the default stays lists so the CSV shape and
    every existing caller are unchanged."""
    arr = np.arange(6.).reshape(2, 3)
    d = {'h1': _spectrum(2, arr)}
    as_lists = build_spectrum_dataframe(d)
    assert isinstance(as_lists['data'].iloc[0], list)
    kept = build_spectrum_dataframe(d, arrays_as_lists=False)
    assert isinstance(kept['data'].iloc[0], np.ndarray)
    assert np.array_equal(kept['data'].iloc[0], arr)


# ---- B10: a partial or unexpected store record must not raise ---------------

def test_record_missing_a_column_does_not_raise():
    """The builder appended per-record values into fixed columns, so a record
    missing any key left the columns ragged and pandas raised `ValueError: All
    arrays must be of the same length`. `createDf` catches and logs, so the
    live symptom was a silently missing export."""
    d = {'full': _spectrum(1, np.array([1., 2.])), 'partial': {'dim': 1}}
    df = build_spectrum_dataframe(d)
    assert list(df['name']) == ['full', 'partial']
    assert df['binx'].iloc[0] == 10
    assert pd.isna(df['binx'].iloc[1])          # absent value reads as NaN


def test_record_with_an_unexpected_key_does_not_raise():
    """An unknown key used to raise KeyError against the fixed column dict."""
    d = {'h1': _spectrum(1, np.array([1., 2.]), gate=None, surprise='x')}
    df = build_spectrum_dataframe(d)
    assert list(df.columns) == list(_EXPECTED_COLUMNS)
    assert 'surprise' not in df.columns         # dropped, not crashed


def test_export_survives_a_partial_record(tmp_path):
    d = {'full': _spectrum(2, np.arange(4.).reshape(2, 2)), 'partial': {'dim': 2}}
    out = str(tmp_path / 'df-partial.gzip')
    export_spectrum_csv(d, out)
    back = read_spectrum_export(out)
    assert list(back['name']) == ['full', 'partial']


_EXPECTED_COLUMNS = ('name', 'dim', 'binx', 'minx', 'maxx', 'biny', 'miny',
                     'maxy', 'xunderflow', 'yunderflow', 'data', 'xoverflow',
                     'yoverflow', 'parameters', 'type')
