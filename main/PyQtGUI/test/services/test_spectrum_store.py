import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

import pytest
from services.spectrum_store import SpectrumStore


def test_set_and_get():
    store = SpectrumStore()
    store.set("h1", dim=1, binx=512, minx=0.0, maxx=1024.0,
              biny=0, miny=0.0, maxy=0.0, data=[], parameters=["p1"], type="1")
    assert store.get("h1", "dim") == 1
    assert store.get("h1", "binx") == 512
    assert store.get("h1", "parameters") == ["p1"]


def test_get_unknown_key_returns_none():
    store = SpectrumStore()
    store.set("h1", dim=2)
    assert store.get("h1", "nonexistent") is None


def test_get_unknown_name_returns_none():
    store = SpectrumStore()
    assert store.get("missing", "dim") is None


def test_invalid_key_ignored_by_set():
    store = SpectrumStore()
    store.set("h1", not_a_key="bad")
    assert not store.contains("h1")


def test_remove():
    store = SpectrumStore()
    store.set("h1", dim=1)
    store.remove("h1")
    assert not store.contains("h1")


def test_remove_missing_is_noop():
    store = SpectrumStore()
    store.remove("ghost")  # documented no-op; must not raise
    assert not store.contains("ghost")


def test_contains():
    store = SpectrumStore()
    assert not store.contains("h1")
    store.set("h1", dim=1)
    assert store.contains("h1")


def test_all_names_sorted():
    store = SpectrumStore()
    store.set("beta", dim=1)
    store.set("alpha", dim=2)
    assert store.all_names() == ["alpha", "beta"]


def test_as_dict_is_snapshot():
    store = SpectrumStore()
    store.set("h1", dim=1)
    d = store.as_dict()
    assert "h1" in d
    # Snapshot copy: later store changes must NOT appear in the returned dict,
    # and structural edits to the returned dict must NOT reach the store.
    store.set("h2", dim=2)
    assert "h2" not in d
    del d["h1"]
    assert store.contains("h1")


def test_get_record_returns_fields():
    store = SpectrumStore()
    store.set("h1", dim=1, binx=512)
    rec = store.get_record("h1")
    assert rec["dim"] == 1 and rec["binx"] == 512
    assert store.get_record("missing") is None


def test_upsert_merges_fields():
    store = SpectrumStore()
    store.set("h1", dim=1, binx=256)
    store.set("h1", binx=512)
    assert store.get("h1", "dim") == 1
    assert store.get("h1", "binx") == 512


# ---------------------------------------------------------------------------
# guard: `data` arrays are live shm views — replacing one with a
# non-aliasing array silently freezes the spectrum (regression class).
# ---------------------------------------------------------------------------

import numpy as np


def test_replacing_live_view_with_copy_is_refused():
    store = SpectrumStore()
    mirror = np.arange(12)
    store.set("h1", dim=1, data=mirror[0:-1])
    derived = (mirror[0:-1] * 2).copy()          # the mistake
    store.set("h1", data=derived)
    kept = store.get("h1", "data")
    assert kept is not derived
    assert np.shares_memory(kept, mirror)        # live view survived


def test_refused_data_write_still_updates_other_fields():
    store = SpectrumStore()
    mirror = np.arange(12)
    store.set("h1", dim=1, binx=10, data=mirror[0:-1])
    store.set("h1", binx=99, data=np.zeros(11))
    assert store.get("h1", "binx") == 99         # merge still happened
    assert np.shares_memory(store.get("h1", "data"), mirror)


def test_new_mirror_view_allowed_with_flag():
    store = SpectrumStore()
    old_mirror, new_mirror = np.arange(12), np.arange(24)
    store.set("h1", dim=1, data=old_mirror[0:-1])
    store.set("h1", data=new_mirror[0:-1], allow_data_replacement=True)
    assert np.shares_memory(store.get("h1", "data"), new_mirror)


def test_reslice_of_same_buffer_allowed_without_flag():
    store = SpectrumStore()
    mirror = np.arange(12)
    store.set("h1", dim=1, data=mirror[0:-1])
    store.set("h1", data=mirror[1:-1])           # still aliases the mirror
    assert np.shares_memory(store.get("h1", "data"), mirror)
    assert len(store.get("h1", "data")) == 10


def test_first_data_set_never_guarded():
    store = SpectrumStore()
    store.set("h1", dim=1)                       # record exists, no data yet
    view = np.arange(5)
    store.set("h1", data=view)
    assert store.get("h1", "data") is view


def test_non_ndarray_data_not_guarded():
    store = SpectrumStore()
    store.set("h1", dim=1, data=[])              # legacy list payloads
    store.set("h1", data=[1, 2, 3])
    assert store.get("h1", "data") == [1, 2, 3]
    store.set("h1", data=np.arange(3))           # list -> ndarray is fine too
    assert isinstance(store.get("h1", "data"), np.ndarray)
