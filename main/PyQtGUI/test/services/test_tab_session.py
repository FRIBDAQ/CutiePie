"""Headless net for the per-tab session registry.

Pins the container semantics that Tabs used to implement by hand across six
parallel dicts/lists: contiguous renumbering on delete, swap, and the live
proxy views that let the old attribute names keep working during migration.
Qt-free, so it runs in the system python3 with no PyQt5.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

import pytest
from services.tab_session import TabSession, TabSessionRegistry


def _reg(n=3):
    r = TabSessionRegistry()
    for i in range(n):
        r.add(i, TabSession(widget="w%d" % i, layout=[1, 1]))
    return r


def test_session_defaults_are_independent_per_tab():
    a, b = TabSession(), TabSession()
    a.slots["x"] = 1
    a.layout.append(9)
    assert b.slots == {} and b.layout == [1, 1]


def test_default_layout_is_one_by_one():
    assert TabSession().layout == [1, 1]


def test_delete_renumbers_remaining_tabs_contiguously():
    r = _reg(3)
    r.delete(1)
    assert list(r.indices()) == [0, 1]
    assert r[0].widget == "w0"
    assert r[1].widget == "w2"      # old tab 2 slid down


def test_delete_keeps_every_field_together():
    r = _reg(3)
    r[2].slots["s"] = "third"
    r[2].click_bound = True
    r[2].selected_bak = 7
    r.delete(0)
    moved = r[1]
    assert moved.widget == "w2"
    assert moved.slots == {"s": "third"}
    assert moved.click_bound is True
    assert moved.selected_bak == 7


def test_swap_exchanges_whole_sessions():
    r = _reg(3)
    r[0].selected_bak = "A"
    r[2].selected_bak = "C"
    r.swap(0, 2)
    assert r[0].widget == "w2" and r[0].selected_bak == "C"
    assert r[2].widget == "w0" and r[2].selected_bak == "A"


def test_len_counts_tabs():
    assert len(_reg(4)) == 4


def test_mapping_view_reads_and_writes_through():
    r = _reg(2)
    view = r.mapping_view("click_bound")
    assert view[0] is False
    view[1] = True
    assert r[1].click_bound is True
    assert dict(view.items()) == {0: False, 1: True}
    assert list(view.keys()) == [0, 1]
    assert len(view) == 2


def test_mapping_view_follows_a_delete():
    r = _reg(3)
    view = r.mapping_view("widget")
    r.delete(0)
    assert dict(view) == {0: "w1", 1: "w2"}


def test_sequence_view_reads_and_writes_through():
    r = _reg(3)
    view = r.sequence_view("selected_bak")
    assert list(view) == [None, None, None]
    view[1] = 5
    assert r[1].selected_bak == 5
    assert len(view) == 3


def test_sequence_view_holds_mutable_field_identity():
    # GUI.py does `self.wTab.layout[i] = [nRow, nCol]` AND reads
    # `nRow, nCol = self.wTab.layout[i]` -- both must hit the same list.
    r = _reg(2)
    view = r.sequence_view("layout")
    view[0] = [2, 3]
    assert r[0].layout == [2, 3]
    assert list(view[0]) == [2, 3]


def test_sequence_view_raises_indexerror_not_keyerror_out_of_range():
    # MutableSequence iterates by walking indices until IndexError, so a
    # KeyError here would escape out of any `for` over the view.
    view = _reg(2).sequence_view("layout")
    with pytest.raises(IndexError):
        view[5]
    with pytest.raises(IndexError):
        view[5] = [1, 1]


def test_sequence_view_supports_negative_indices():
    view = _reg(3).sequence_view("widget")
    assert view[-1] == "w2"
    assert view[-3] == "w0"
    with pytest.raises(IndexError):
        view[-4]


def test_zoom_info_accepts_empty_list_and_none_and_pair():
    # three live shapes, all preserved; [] and None are both falsy for callers
    r = _reg(1)
    view = r.mapping_view("zoom_info")
    assert view[0] == []
    view[0] = None
    assert r[0].zoom_info is None
    view[0] = [3, "spec"]
    assert r[0].zoom_info == [3, "spec"]


def test_delete_returns_session_before_renumbering():
    # deleteTab must close the doomed tab's figure BEFORE indices shift, or it
    # closes the wrong one. delete() returning the removed session is what makes
    # that orderable.
    r = _reg(3)
    removed = r.delete(1)
    assert removed.widget == "w1"
    assert r[1].widget == "w2"


def test_swap_is_a_no_op_when_indices_are_equal():
    r = _reg(2)
    r.swap(1, 1)
    assert r[1].widget == "w1"


def test_delete_of_unknown_index_raises_keyerror():
    with pytest.raises(KeyError):
        _reg(2).delete(9)
