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


def test_remove_missing_raises():
    store = SpectrumStore()
    with pytest.raises(KeyError):
        store.remove("ghost")


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


def test_as_dict_is_live():
    store = SpectrumStore()
    store.set("h1", dim=1)
    d = store.as_dict()
    assert "h1" in d
    store.set("h2", dim=2)
    assert "h2" in d


def test_upsert_merges_fields():
    store = SpectrumStore()
    store.set("h1", dim=1, binx=256)
    store.set("h1", binx=512)
    assert store.get("h1", "dim") == 1
    assert store.get("h1", "binx") == 512
