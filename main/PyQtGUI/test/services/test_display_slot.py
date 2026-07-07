import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

import pytest
from services.display_slot import DisplaySlot, SLOT_KEYS


# The exact per-pad template GUI.setGeo builds today (GUI.py:1924). C1's
# DisplaySlot must be a drop-in for this dict, so a fresh slot must equal it
# key-for-key — this test is the behavior-preservation anchor for the strangler.
SETGEO_TEMPLATE = {
    "name": [], "dim": [], "binx": [], "minx": [], "maxx": [], "biny": [],
    "miny": [], "maxy": [], "data": [], "parameters": [], "type": [],
    "log": [], "minz": [], "maxz": [], "spectrum": [], "axis": [], "cutoff": [],
}


def test_fresh_slot_matches_setgeo_template_exactly():
    assert DisplaySlot().as_dict() == SETGEO_TEMPLATE


def test_slot_keys_match_template_order():
    assert SLOT_KEYS == tuple(SETGEO_TEMPLATE.keys())


def test_every_field_defaults_to_empty_list():
    slot = DisplaySlot()
    for k in SLOT_KEYS:
        assert slot[k] == []           # falsy, matches old dict truthiness tests


def test_default_lists_are_not_shared_between_instances():
    a, b = DisplaySlot(), DisplaySlot()
    a["log"].append("x")               # mutate one instance's default
    assert b["log"] == []              # the other must be unaffected


def test_item_get_set_roundtrip_like_a_dict():
    slot = DisplaySlot()
    slot["log"] = True
    slot["minx"] = 0.0
    slot["maxx"] = 1024.0
    assert slot["log"] is True
    assert slot["minx"] == 0.0
    assert slot["maxx"] == 1024.0


def test_item_and_attribute_access_are_the_same_storage():
    slot = DisplaySlot()
    slot["log"] = True                 # write via mapping
    assert slot.log is True            # read via attribute
    slot.cutoff = (10.0, 20.0)         # write via attribute
    assert slot["cutoff"] == (10.0, 20.0)   # read via mapping


def test_contains_reports_whitelist_membership():
    slot = DisplaySlot()
    assert "log" in slot
    assert "axis" in slot
    assert "bogus" not in slot


def test_unknown_key_raises_keyerror_on_get_and_set():
    slot = DisplaySlot()
    with pytest.raises(KeyError):
        slot["bogus"]
    with pytest.raises(KeyError):
        slot["bogus"] = 1


def test_keys_values_items_cover_all_17_in_order():
    slot = DisplaySlot()
    slot["dim"] = 2
    assert list(slot.keys()) == list(SLOT_KEYS)
    assert list(slot.items())[1] == ("dim", 2)     # dim is 2nd in template order
    assert list(slot.values())[1] == 2


def test_data_is_an_empty_placeholder_by_default():
    # counts live ONLY in SpectrumStore (B4/C1) — the slot's data stays empty.
    assert DisplaySlot()["data"] == []


def test_getattr_setattr_roundtrip_over_every_whitelist_key():
    # C3 production idiom: getSpectrumViewInfo does getattr(slot, info[0]);
    # setSpectrumViewInfo / setGeo do setattr(slot, key, value). This must work
    # for all 17 whitelist keys (dynamic key access without the dict shim).
    slot = DisplaySlot()
    for i, k in enumerate(SLOT_KEYS):
        setattr(slot, k, i)                        # setattr path (writes)
    for i, k in enumerate(SLOT_KEYS):
        assert getattr(slot, k) == i               # getattr path (reads)
        assert slot[k] == i                        # shim agrees with typed access


def test_setattr_then_item_read_agree_after_setgeo_style_populate():
    # Replicates setGeo's typed populate (slot.name=…, setattr(slot,key,val))
    # and proves an item-read (old dict semantics) still sees the same values.
    slot = DisplaySlot()
    slot.name = "h1"
    for k, v in {"dim": 1, "binx": 512, "minx": 0.0, "maxx": 1024.0}.items():
        setattr(slot, k, v)
    assert (slot["name"], slot["dim"], slot["binx"]) == ("h1", 1, 512)
    assert slot["data"] == []                      # never populated (B4)


def test_slot_carries_opaque_artist_refs_without_touching_them():
    # axis/spectrum are held as opaque objects; the slot never calls Qt/mpl.
    sentinel_axis = object()
    sentinel_artist = object()
    slot = DisplaySlot()
    slot["axis"] = sentinel_axis
    slot["spectrum"] = sentinel_artist
    assert slot["axis"] is sentinel_axis
    assert slot["spectrum"] is sentinel_artist
