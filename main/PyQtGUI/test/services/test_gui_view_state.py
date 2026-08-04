"""Characterization tests for MainWindow's view-state seams.

These are the accessors every service is constructed against — the object
ARCH.md §7 D9 will become. They are pinned HERE, against the current structure,
so the extraction can be verified by this file passing unedited.

The first tests to reach `MainWindow` at all (AUDIT.md H14). They run on a
`gui_stubs.bare_window()`: a real MainWindow whose `__init__` was never run,
with only the collaborators each method actually reads injected.

Three pins below record behavior that has already produced defects, and exist
so a refactor cannot quietly drop them:

* `nameFromIndex` answers with the ENLARGED spectrum for any index while a pad
  is enlarged (the stale-index pitfall behind H11),
* a pad the geometry blanked reads as the string "empty", not None, and the
  store does not hold it (the premise of BUGS.md E24),
* the two metadata tiers are independent: store minx/maxx are the axis
  definition, the per-tab slot's are the current view range (BUGS.md E7).
"""

import os
import sys
import types
import time

import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib.figure import Figure

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import gui_stubs
from services.spectrum_store import SpectrumStore
from services.display_slot import DisplaySlot


class FakeTabs:
    """wTab: the per-tab slot dicts and the enlarged-pad record."""

    def __init__(self):
        # two tabs, so a per-tab accessor can be caught answering for the
        # wrong one
        self.slots = {0: {}, 1: {}}
        self.zoom = {0: None, 1: None}
        self.current = 0

    def currentIndex(self):
        return self.current

    def tabSlots(self, index):
        return self.slots[index]

    def zoomInfo(self, index):
        return self.zoom[index]

    def setZoomInfo(self, index, value):
        self.zoom[index] = value


class FakePlot:
    """currentPlot: only the pad-index to spectrum-name map is read here."""

    def __init__(self):
        self.h_dict_geo = {}


@pytest.fixture
def win(monkeypatch):
    w = gui_stubs.bare_window()
    w.spectra = SpectrumStore()
    w.wTab = FakeTabs()
    w.currentPlot = FakePlot()
    # These accessors moved to ViewState (FACTORIZATION.md stage 9). The
    # unchanged test bodies keep driving them on the window: the methods are
    # bound onto the instance, and the gate-name cache is proxied, exactly as
    # MainWindow's own call sites now reach them through self.view_state.
    from view_state import ViewState
    w.gate_fetches = []
    w.view_state = ViewState(
        spectra=w.spectra,
        tabs=w.wTab,
        get_current_plot=lambda: w.currentPlot,
        applylistgate=lambda name: w.gate_fetches.append(name),
        gate_name_fetched=types.SimpleNamespace(emit=lambda *a: None),
        logger=w.logger,
    )
    for _name in ("getSpectrumStoreInfo", "setSpectrumViewInfo", "getSpectrumViewInfo",
                  "getSpectrumViewDict", "getSpectrumStoreDict", "nameFromIndex",
                  "setGeo", "getGeo", "setEnlargedSpectrum", "getEnlargedSpectrum",
                  "getAppliedGateName", "_refreshGateNameAsync"):
        setattr(w, _name, getattr(w.view_state, _name))
    for _attr in ("_gate_name_cache", "_gate_name_inflight", "_GATE_NAME_TTL"):
        monkeypatch.setattr(
            type(w), _attr,
            property(lambda self, a=_attr: getattr(self.view_state, a),
                     lambda self, v, a=_attr: setattr(self.view_state, a, v)),
            raising=False)
    return w


def add_spectrum(win, name="h1", index=0, dim=1, **fields):
    """Populate both tiers the way a connect + setGeo does."""
    record = dict(dim=dim, binx=512, minx=0.0, maxx=1024.0,
                  biny=0, miny=0.0, maxy=0.0, data=[], parameters=["p1"],
                  type="1")
    record.update(fields)
    win.spectra.set(name, **record)
    win.setGeo(index, name)
    return record


# --------------------------------------------------------------- nameFromIndex

def test_name_from_index_reads_the_geometry_map(win):
    add_spectrum(win, "h1", index=0)
    assert win.nameFromIndex(0) == "h1"


def test_name_from_index_returns_none_for_a_pad_outside_the_grid(win):
    add_spectrum(win, "h1", index=0)
    assert win.nameFromIndex(7) is None


def test_name_from_index_returns_the_literal_empty_for_a_blanked_pad(win):
    # PIN (BUGS.md E24): InitializeCanvas refills the map with "empty" on a
    # geometry change. The pad reads as that string, NOT as None, and the store
    # does not hold it — so a dimension lookup through here yields None and
    # every caller has to cope with a third answer.
    win.currentPlot.h_dict_geo[3] = "empty"
    assert win.nameFromIndex(3) == "empty"
    assert win.spectra.get("empty", "dim") is None


def test_name_from_index_short_circuits_to_the_enlarged_spectrum(win):
    # PIN (AUDIT.md H11): while a pad is enlarged this answers with the
    # enlarged spectrum for ANY index, including one it does not own. Fit
    # records that resolve a stored index through here get the wrong spectrum.
    add_spectrum(win, "h1", index=0)
    add_spectrum(win, "h2", index=1, dim=2)
    win.setEnlargedSpectrum(1, "h2")
    assert win.nameFromIndex(0) == "h2"
    assert win.nameFromIndex(99) == "h2"


# -------------------------------------------------------- getSpectrumStoreInfo

def test_store_info_by_index_and_by_name(win):
    add_spectrum(win, "h1", index=0, dim=2)
    assert win.getSpectrumStoreInfo("dim", index=0) == 2
    assert win.getSpectrumStoreInfo("dim", name="h1") == 2


def test_store_info_with_no_identifier_uses_the_enlarged_spectrum(win):
    add_spectrum(win, "h1", index=0)
    add_spectrum(win, "h2", index=1, dim=2)
    win.setEnlargedSpectrum(1, "h2")
    assert win.getSpectrumStoreInfo("dim") == 2


def test_store_info_with_no_identifier_and_no_enlarged_pad_returns_none(win):
    add_spectrum(win, "h1", index=0)
    assert win.getSpectrumStoreInfo("dim") is None


def test_store_info_for_an_unknown_spectrum_returns_none(win):
    assert win.getSpectrumStoreInfo("dim", name="nope") is None


# --------------------------------------------------------- getSpectrumViewInfo

def test_view_info_reads_the_slot_for_the_index(win):
    add_spectrum(win, "h1", index=0)
    win.setSpectrumViewInfo(minx=12.0, index=0)
    assert win.getSpectrumViewInfo("minx", index=0) == 12.0


def test_view_info_rejects_a_key_outside_the_slot_whitelist(win):
    add_spectrum(win, "h1", index=0)
    assert win.getSpectrumViewInfo("not_a_slot_key", index=0) is None


def test_view_info_returns_none_for_a_pad_with_no_slot(win):
    add_spectrum(win, "h1", index=0)
    assert win.getSpectrumViewInfo("minx", index=5) is None


def test_view_info_ignores_the_passed_index_while_a_pad_is_enlarged(win):
    # PIN: the enlarged branch overrides the caller's index entirely. This is
    # the display-tier half of the H11 pitfall pinned above.
    add_spectrum(win, "h1", index=0)
    add_spectrum(win, "h2", index=1)
    win.setSpectrumViewInfo(minx=11.0, index=0)
    win.setSpectrumViewInfo(minx=22.0, index=1)
    win.setEnlargedSpectrum(1, "h2")
    assert win.getSpectrumViewInfo("minx", index=0) == 22.0


# --------------------------------------------------------- setSpectrumViewInfo

def test_set_view_info_writes_only_whitelisted_keys(win):
    add_spectrum(win, "h1", index=0)
    win.setSpectrumViewInfo(minx=3.0, not_a_slot_key="x", index=0)
    slot = win.wTab.tabSlots(0)[0]
    assert slot.minx == 3.0
    assert not hasattr(slot, "not_a_slot_key")


def test_set_view_info_with_no_identifier_writes_nothing(win):
    add_spectrum(win, "h1", index=0)
    before = win.getSpectrumViewInfo("minx", index=0)
    win.setSpectrumViewInfo(minx=99.0)
    assert win.getSpectrumViewInfo("minx", index=0) == before


def test_set_view_info_does_not_create_a_slot_for_an_unknown_pad(win):
    win.setSpectrumViewInfo(minx=99.0, index=4)
    assert 4 not in win.wTab.tabSlots(0)


def test_set_view_info_sets_axis_alongside_spectrum(win):
    # the artist and its axes are written together, so a pad can never hold one
    # without the other
    add_spectrum(win, "h1", index=0)
    ax = Figure().add_subplot(111)
    line, = ax.plot([0, 1], [1, 2])
    win.setSpectrumViewInfo(spectrum=line, index=0)
    assert win.getSpectrumViewInfo("spectrum", index=0) is line
    assert win.getSpectrumViewInfo("axis", index=0) is ax


# ---------------------------------------------------------------------- setGeo

def test_set_geo_creates_a_slot_and_copies_the_store_record(win):
    add_spectrum(win, "h1", index=0, dim=2, binx=256)
    slot = win.wTab.tabSlots(0)[0]
    assert isinstance(slot, DisplaySlot)
    assert win.getGeo()[0] == "h1"
    assert slot.name == "h1" and slot.dim == 2 and slot.binx == 256


def test_set_geo_never_copies_counts_into_the_display_tier(win):
    # ARCH.md invariant / BUGS.md B4: the canonical array lives only in the
    # store; the slot keeps the empty placeholder it was built with.
    win.spectra.set("h1", dim=1, binx=4, minx=0.0, maxx=4.0, biny=0,
                    miny=0.0, maxy=0.0, data=[1, 2, 3, 4], parameters=[],
                    type="1")
    win.setGeo(0, "h1")
    assert win.wTab.tabSlots(0)[0].data == []


def test_set_geo_for_a_spectrum_the_store_lacks_keeps_the_name_only(win):
    win.setGeo(0, "ghost")
    slot = win.wTab.tabSlots(0)[0]
    assert win.getGeo()[0] == "ghost"
    assert slot.name == "ghost"
    assert slot.dim == []            # untouched DisplaySlot default


def test_set_geo_reuses_an_existing_slot(win):
    add_spectrum(win, "h1", index=0)
    first = win.wTab.tabSlots(0)[0]
    add_spectrum(win, "h2", index=0)
    assert win.wTab.tabSlots(0)[0] is first


# ------------------------------------------------------------ the two tiers

def test_the_two_metadata_tiers_are_independent(win):
    # PIN (BUGS.md E7): same field names, different meanings. The store keeps
    # the axis DEFINITION; the slot keeps the current VIEW range, which every
    # zoom rewrites. Data-coordinate math must read the store tier.
    add_spectrum(win, "h1", index=0, minx=0.0, maxx=1024.0)
    win.setSpectrumViewInfo(minx=200.0, maxx=300.0, index=0)      # a zoom
    assert win.getSpectrumStoreInfo("minx", index=0) == 0.0
    assert win.getSpectrumStoreInfo("maxx", index=0) == 1024.0
    assert win.getSpectrumViewInfo("minx", index=0) == 200.0
    assert win.getSpectrumViewInfo("maxx", index=0) == 300.0


# ------------------------------------------------------- enlarged-pad record

def test_enlarged_spectrum_round_trips_and_clears(win):
    assert win.getEnlargedSpectrum() is None
    win.setEnlargedSpectrum(2, "h2")
    assert win.getEnlargedSpectrum() == [2, "h2"]
    win.setEnlargedSpectrum(None, None)
    assert win.getEnlargedSpectrum() is None


# --------------------------------------------------------- getAppliedGateName

@pytest.fixture
def gate_win(win):
    win._gate_name_cache = {}
    win.refresh_calls = []
    # the recorder goes on ViewState as well: getAppliedGateName calls the
    # refresh on its OWN object now, not through the window
    win._refreshGateNameAsync = win.refresh_calls.append
    win.view_state._refreshGateNameAsync = win.refresh_calls.append
    add_spectrum(win, "h1", index=0)
    return win


def test_gate_name_returns_a_fresh_cache_entry_without_refetching(gate_win):
    gate_win._gate_name_cache["h1"] = ("gateA", time.monotonic())
    assert gate_win.getAppliedGateName(index=0) == "gateA"
    assert gate_win.refresh_calls == []


def test_gate_name_serves_a_stale_entry_and_revalidates(gate_win):
    # PERFORMANCE.md P3: the hover path must never block on HTTP, so an expired
    # entry answers immediately with the stale value and refetches behind it.
    stale = time.monotonic() - (gate_win._GATE_NAME_TTL + 1.0)
    gate_win._gate_name_cache["h1"] = ("gateA", stale)
    assert gate_win.getAppliedGateName(index=0) == "gateA"
    assert gate_win.refresh_calls == ["h1"]


def test_gate_name_on_a_cold_cache_returns_none_and_revalidates(gate_win):
    assert gate_win.getAppliedGateName(index=0) is None
    assert gate_win.refresh_calls == ["h1"]


def test_gate_name_by_name_matches_by_index(gate_win):
    gate_win._gate_name_cache["h1"] = ("gateA", time.monotonic())
    assert gate_win.getAppliedGateName(name="h1") == "gateA"


def test_gate_name_with_no_identifier_returns_none_and_does_not_refetch(gate_win):
    assert gate_win.getAppliedGateName() is None
    assert gate_win.refresh_calls == []


# ------------------------------------------------------------ the whole-dict accessors

def test_the_store_dict_is_the_canonical_registry(win):
    """`getSpectrumStoreDict` is what the Jupyter export dumps, so it has to be
    the STORE's own view — every spectrum, whether or not a pad shows it."""
    add_spectrum(win, "h1", index=0)
    win.spectra.set("h2", dim=1, binx=8, minx=0.0, maxx=8.0, biny=0, miny=0.0,
                    maxy=0.0, data=[], parameters=["p1"], type="1")
    d = win.getSpectrumStoreDict()
    assert set(d) == {"h1", "h2"}


def test_the_store_dict_is_not_the_live_store(win):
    """A caller iterating it must not be able to mutate the registry by
    accident — the export walks and reshapes it."""
    add_spectrum(win, "h1", index=0)
    d = win.getSpectrumStoreDict()
    d["injected"] = {}
    assert win.spectra.contains("injected") is False


def test_the_view_dict_is_the_current_tabs_slots(win):
    """`getSpectrumViewDict` is the DISPLAY tier for the tab on screen, keyed
    by pad index — not by name, and not the store."""
    add_spectrum(win, "h1", index=0)
    add_spectrum(win, "h2", index=3)
    d = win.getSpectrumViewDict()
    assert set(d) == {0, 3}
    assert d[0].name == "h1" and d[3].name == "h2"


def test_the_view_dict_follows_the_tab_switch(win):
    """Two tabs, two slot dicts: the accessor must answer for whichever tab is
    current, or a per-tab setting reads off the wrong pad."""
    add_spectrum(win, "h1", index=0)
    win.wTab.current = 1
    assert win.getSpectrumViewDict() == {}
    win.wTab.current = 0
    assert set(win.getSpectrumViewDict()) == {0}


def test_a_second_fetch_while_one_is_in_flight_starts_no_thread(win, monkeypatch):
    """One fetch per spectrum, however many hover events arrive meanwhile.

    The in-flight set is what prevents the second one, and asserting on the SET
    cannot show it: re-adding a member is a no-op, so the set reads the same
    with the guard removed. What the guard actually saves is the thread, so
    that is what this counts.
    """
    import view_state as vs
    started = []

    class FakeThread:
        def __init__(self, target=None, daemon=None, name=None):
            self._target = target

        def start(self):
            started.append(1)

    monkeypatch.setattr(vs.threading, "Thread", FakeThread)
    win._refreshGateNameAsync("h1")
    win._refreshGateNameAsync("h1")
    assert len(started) == 1
