"""Characterization tests for MainWindow's peak-scan cluster. Peak Finder 1:
the Scan button, the checkable peak list, the four red marker artists per
peak, and Clear."""

import logging
import os
import sys
import types

import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib.figure import Figure

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import gui_stubs


# --------------------------------------------------------------- fake widgets

class FakeItem:
    """QListWidgetItem stand-in: check state and the user-checkable flag."""

    def __init__(self, label=""):
        self.label = label
        self._flags = 0
        self._state = 0

    def flags(self):
        return self._flags

    def setFlags(self, f):
        self._flags = f

    def checkState(self):
        return self._state

    def setCheckState(self, s):
        self._state = s
        if self.owner is not None and not self.owner.signals_blocked:
            self.owner.emit_item_changed(self)

    owner = None


class FakeListWidget:
    """QListWidget stand-in that honours blockSignals.

    itemChanged is the signal the production code blocks while it flips check
    states in bulk; a double that ignored blockSignals would hide the very
    re-entry the block exists to prevent.
    """

    def __init__(self):
        self.items = []
        self.signals_blocked = False
        self.on_item_changed = None
        self.emissions = 0

    def blockSignals(self, on):
        self.signals_blocked = bool(on)

    def count(self):
        return len(self.items)

    def item(self, row):
        return self.items[row]

    def row(self, item):
        return self.items.index(item)

    def addItem(self, item):
        item.owner = self
        self.items.append(item)

    def clear(self):
        for item in self.items:
            item.owner = None
        self.items = []

    def emit_item_changed(self, item):
        self.emissions += 1
        if self.on_item_changed is not None:
            self.on_item_changed(item)


class FakeResults:
    def __init__(self):
        self.lines = []

    def append(self, s):
        self.lines.append(s)

    def clear(self):
        self.lines = []


class FakeText:
    def __init__(self, value=""):
        self.value = value

    def text(self):
        return self.value


class FakeCombo:
    def __init__(self, value):
        self.value = value

    def currentText(self):
        return self.value


class FakePeakTab:
    def __init__(self):
        self.peak_list = FakeListWidget()
        self.peak_results = FakeResults()
        self.peak_width = FakeText("2")
        self.peak_algo = FakeCombo("Raw counts (legacy)")


class FakeExtraPopup:
    def __init__(self):
        self.peak = FakePeakTab()


class FakeCanvas:
    def __init__(self):
        self.draws = 0

    def draw(self):
        self.draws += 1

    def draw_idle(self):
        self.draws += 1


class FakePlot:
    def __init__(self):
        self.canvas = FakeCanvas()
        self.selected_plot_index = 0


# --------------------------------------------------------------- the spectrum

# A three-peak spectrum on a 0..100 axis, wide enough that the default width
# finds all three and the clip window can exclude some of them.
BINS = 101
COUNTS = np.zeros(BINS)
for centre in (20, 50, 80):
    for offset, height in ((-2, 20), (-1, 60), (0, 100), (1, 60), (2, 20)):
        COUNTS[centre + offset] = height


    # The cluster's state lives on the controller; these proxies let the
    # unchanged test bodies keep reading it on the window.
_SCAN_STATE = ("datax", "datay", "peaks", "properties", "isChecked",
               "peak_pos", "peak_vl", "peak_hl", "peak_txt")


def _proxy(attr):
    return property(lambda self: getattr(self.peak_scan_controller, attr),
                    lambda self, v: setattr(self.peak_scan_controller, attr, v))


@pytest.fixture
def win(monkeypatch):
    gui = gui_stubs.import_gui()
    w = gui.MainWindow.__new__(gui.MainWindow)
    w.logger = logging.getLogger("test.peakscan")
    w.extraPopup = FakeExtraPopup()
    w.currentPlot = FakePlot()

    figure = Figure()
    w.axes = figure.add_subplot(111)
    w.axes.set_xlim(0, 100)

    store = {"binx": BINS, "minx": 0, "maxx": 100, "data": COUNTS}
    w.store = store
    w.getSpectrumViewInfo = lambda field, index=None: w.axes
    w.getSpectrumStoreInfo = lambda field, index=None: store[field]
    w.plot_controller = types.SimpleNamespace(
        createRange=lambda binx, minx, maxx: np.linspace(minx, maxx, binx))

    from controllers import peak_scan_controller as psc
    monkeypatch.setattr(psc, "QListWidgetItem", FakeItem)
    w.peak_scan_controller = psc.PeakScanController(
        peak_tab=w.extraPopup.peak,
        get_current_plot=lambda: w.currentPlot,
        get_selected_index=lambda: w.currentPlot.selected_plot_index,
        # the seams stay late-bound: a test that replaces the window's lookup
        # (the no-axes case) must reach the controller through the same door
        get_store_info=lambda field, index=None: w.getSpectrumStoreInfo(field, index=index),
        get_view_info=lambda field, index=None: w.getSpectrumViewInfo(field, index=index),
        plot_controller=w.plot_controller,
        logger=w.logger,
    )
    for attr in _SCAN_STATE:
        monkeypatch.setattr(type(w), attr, _proxy(attr), raising=False)

    # the methods the tests drive directly are the ones that stayed on
    # MainWindow as connect targets; the rest are reached through the same
    # names via the controller
    for name in ("_syncPeakMarker", "populatePeakList", "removePeak",
                 "removeAllPeaks", "resetPeakDict", "drawSinglePeaks",
                 "update_peak_output"):
        setattr(w, name, getattr(w.peak_scan_controller, name))

    # the list re-enters peakItemChanged on every unblocked state change,
    # exactly as the real itemChanged connection does
    w.extraPopup.peak.peak_list.on_item_changed = lambda item: w.peakItemChanged(item)
    return w


def scan(win):
    """Run one full Scan, the way the button does."""
    win.analyzePeak()


def artist_rows(win):
    return sorted(win.peak_pos.keys())


def census(win):
    """What is actually ON the pad: (markers, vline/hline collections,
    labels). The artist dicts say what the code THINKS it drew."""
    return (len(win.axes.lines), len(win.axes.collections), len(win.axes.texts))


# ------------------------------------------------------------ analyzePeak/E7

def test_scan_finds_the_peaks_in_the_visible_window(win):
    scan(win)
    assert len(win.peaks) == 3


def test_the_x_axis_comes_from_the_store_tier_not_the_view(win):
    """The axes are zoomed to a quarter of the spectrum; the bin count and bounds
    must still come from the store, with the axes supplying only the window to
    clip to."""
    asked = []
    win.getSpectrumStoreInfo = lambda field, index=None: (
        asked.append(field) or win.store[field])
    win.axes.set_xlim(40, 60)
    scan(win)
    for field in ("binx", "minx", "maxx"):
        assert field in asked, f"{field} was not read from the store tier"
    # only the peak inside the zoom window survives the clip
    assert len(win.peaks) == 1
    assert 40 <= win.datax[win.peaks[0]] <= 60


def test_the_clipped_data_is_what_the_markers_are_drawn_from(win):
    win.axes.set_xlim(40, 60)
    scan(win)
    assert win.datax.min() >= 40 and win.datax.max() < 60
    assert len(win.datax) == len(win.datay)


def test_the_algorithm_combo_selects_the_finder(win):
    """Asserted by the answers differing, not by the call. On a noisy spectrum
    the legacy raw-counts search reports every ripple above prominence 1, while
    Mariscotti's significance test rejects them — so a combo that is read gives
    two different peak counts and a combo that is ignored gives one."""
    rng = np.random.default_rng(7)
    win.store["data"] = COUNTS + rng.integers(0, 5, BINS)

    win.extraPopup.peak.peak_algo = FakeCombo("Raw counts (legacy)")
    scan(win)
    raw = len(win.peaks)

    win.extraPopup.peak.peak_algo = FakeCombo("Mariscotti (2nd difference)")
    scan(win)
    mariscotti = len(win.peaks)

    assert raw > mariscotti, "the algorithm combo is not reaching the finder"


def test_an_unknown_algorithm_falls_back_instead_of_failing(win):
    win.extraPopup.peak.peak_algo = FakeCombo("no such algorithm")
    scan(win)
    assert len(win.peaks) == 3


def test_a_bad_width_entry_does_not_raise(win):
    """Best effort: the user sees nothing happen, the GUI stays up, the log
    carries the reason. Scanned once first, so the assertion cannot be
    satisfied by the fixture's own empty state."""
    scan(win)
    good = win.peaks
    win.extraPopup.peak.peak_width = FakeText("not a number")
    scan(win)
    assert win.peaks is good, "a failed scan overwrote the previous results"


def test_scanning_a_pad_with_no_axes_does_not_raise(win):
    """The other failure family, and not a ValueError: a pad that has a slot
    but was never drawn answers None for its axis, so the window read is what
    blows up. Best-effort means best-effort for that too."""
    scan(win)
    good = win.peaks
    win.getSpectrumViewInfo = lambda field, index=None: None
    scan(win)
    assert win.peaks is good


def test_a_failed_scan_leaves_the_previous_list_intact(win):
    """The width is parsed before anything is cleared, so a rescan that fails
    must not take the standing scan down with it."""
    scan(win)
    win.extraPopup.peak.peak_width = FakeText("")
    scan(win)
    assert win.extraPopup.peak.peak_list.count() == 3
    assert artist_rows(win) == [0, 1, 2]


def test_the_results_pane_gets_one_line_per_output_row(win):
    scan(win)
    assert win.extraPopup.peak.peak_results.lines
    assert len(win.extraPopup.peak.peak_results.lines) == len(win.peaks)


# --------------------------------------------------------- list + marker sync

def test_every_found_peak_arrives_checked_and_drawn(win):
    scan(win)
    peak_list = win.extraPopup.peak.peak_list
    assert peak_list.count() == 3
    assert all(item.checkState() == 2 for item in peak_list.items)
    assert artist_rows(win) == [0, 1, 2]


def test_each_drawn_peak_owns_all_four_artists(win):
    scan(win)
    for row in range(3):
        assert row in win.peak_pos
        assert row in win.peak_vl
        assert row in win.peak_hl
        assert row in win.peak_txt


def test_unchecking_one_row_removes_only_its_markers(win):
    scan(win)
    win.extraPopup.peak.peak_list.item(1).setCheckState(0)
    assert artist_rows(win) == [0, 2]
    assert win.isChecked[1] is False


def test_rechecking_a_row_puts_its_markers_back(win):
    scan(win)
    item = win.extraPopup.peak.peak_list.item(1)
    item.setCheckState(0)
    item.setCheckState(2)
    assert artist_rows(win) == [0, 1, 2]


def test_checking_an_already_checked_row_does_not_stack_artists(win):
    """The isChecked guard. Without it the second draw overwrites the four
    handles and the first set of artists is orphaned on the canvas."""
    scan(win)
    first = win.peak_pos[0]
    win._syncPeakMarker(0, True)
    assert win.peak_pos[0] is first


def test_unchecking_an_already_unchecked_row_is_a_no_op(win):
    scan(win)
    win._syncPeakMarker(0, False)
    win._syncPeakMarker(0, False)
    assert 0 not in win.peak_pos


def test_a_marker_removal_failure_still_clears_the_state(win):
    """The cleanup is best-effort, but the row must not stay stuck 'checked'
    or it can never be drawn again."""
    scan(win)
    win.peak_pos[0] = None          # what a stale handle looks like
    win._syncPeakMarker(0, False)
    assert win.isChecked[0] is False


# ------------------------------------------------------------- bulk operations

def test_check_all_draws_every_row(win):
    scan(win)
    win.setAllPeaksChecked(False)
    win.setAllPeaksChecked(True)
    assert artist_rows(win) == [0, 1, 2]


def test_uncheck_all_removes_every_marker(win):
    scan(win)
    win.setAllPeaksChecked(False)
    assert win.peak_pos == {}
    assert win.peak_vl == {} and win.peak_hl == {} and win.peak_txt == {}


def test_bulk_check_blocks_the_item_signal_while_flipping(win):
    """Signal storm. Each unblocked setCheckState re-enters peakItemChanged,
    which draws the canvas — n redraws instead of one."""
    scan(win)
    win.extraPopup.peak.peak_list.emissions = 0
    win.setAllPeaksChecked(False)
    assert win.extraPopup.peak.peak_list.emissions == 0


def test_bulk_check_redraws_the_canvas_once(win):
    scan(win)
    before = win.currentPlot.canvas.draws
    win.setAllPeaksChecked(False)
    assert win.currentPlot.canvas.draws == before + 1


def test_single_item_change_redraws_the_canvas(win):
    scan(win)
    before = win.currentPlot.canvas.draws
    win.extraPopup.peak.peak_list.item(0).setCheckState(0)
    assert win.currentPlot.canvas.draws == before + 1


# ------------------------------------------------------- rescan and clear

def test_a_second_scan_does_not_leak_the_first_scans_markers(win):
    """The stale-artist rule, asserted on the PAD. populatePeakList clears the
    old markers before it rebuilds the list."""
    scan(win)
    after_first = census(win)
    scan(win)
    assert census(win) == after_first, "the previous scan's artists are still on the pad"


def test_a_second_scan_replaces_the_list_rather_than_appending(win):
    scan(win)
    scan(win)
    assert win.extraPopup.peak.peak_list.count() == 3


def test_rebuilding_the_list_does_not_re_enter_on_every_row(win):
    scan(win)
    win.extraPopup.peak.peak_list.emissions = 0
    scan(win)
    assert win.extraPopup.peak.peak_list.emissions == 0


def test_clear_empties_the_results_the_list_and_the_markers(win):
    scan(win)
    win.peakAnalClear()
    assert win.extraPopup.peak.peak_results.lines == []
    assert win.extraPopup.peak.peak_list.count() == 0
    assert win.peak_pos == {} and win.peak_vl == {}
    assert win.peak_hl == {} and win.peak_txt == {}


def test_clear_then_scan_starts_from_a_clean_slate(win):
    scan(win)
    win.peakAnalClear()
    scan(win)
    assert artist_rows(win) == [0, 1, 2]
    assert win.extraPopup.peak.peak_list.count() == 3


def test_clearing_twice_is_a_no_op(win):
    """The second Clear runs against the state the first one left, not against
    the fixture's untouched dicts."""
    scan(win)
    win.peakAnalClear()
    win.peakAnalClear()
    assert win.extraPopup.peak.peak_list.count() == 0
    assert win.peak_pos == {}


def test_reset_peak_dict_empties_all_four_artist_maps(win):
    scan(win)
    win.resetPeakDict()
    assert win.peak_pos == {} and win.peak_vl == {}
    assert win.peak_hl == {} and win.peak_txt == {}


def test_clear_drops_a_handle_the_uncheck_could_not_remove(win):
    """Why Clear resets the dicts even though the uncheck already emptied them:
    a row whose removal failed leaves its key behind, and the next scan would
    inherit it. Seeded with a stale handle no list row owns."""
    scan(win)
    win.peak_pos[99] = None
    win.peak_txt[99] = None
    win.peakAnalClear()
    assert win.peak_pos == {} and win.peak_txt == {}


def test_remove_all_peaks_goes_through_the_bulk_uncheck(win):
    scan(win)
    win.removeAllPeaks()
    assert win.peak_pos == {}
    assert all(item.checkState() == 0
               for item in win.extraPopup.peak.peak_list.items)


def test_remove_peak_drops_every_artist_for_that_row(win):
    """Both halves matter: the dict keys go, and so do the artists. Deleting
    the key while the artist stays drawn is the leak shape this cluster's
    predecessor shipped."""
    scan(win)
    before = census(win)
    win.removePeak(1)
    assert 1 not in win.peak_pos and 1 not in win.peak_vl
    assert 1 not in win.peak_hl and 1 not in win.peak_txt
    assert 0 in win.peak_pos and 2 in win.peak_pos
    # one marker line, two collections (vlines + hlines), one label
    assert census(win) == (before[0] - 1, before[1] - 2, before[2] - 1)


# --------------------------------------------------------------- drawing math

def test_draw_single_peaks_puts_the_marker_at_the_peak_x(win):
    scan(win)
    win.setAllPeaksChecked(False)
    win.drawSinglePeaks(win.peaks, win.properties, win.datay, 0)
    line = win.peak_pos[0][0]
    assert line.get_xdata()[0] == pytest.approx(win.datax[win.peaks[0]])


def test_the_label_text_is_the_peak_position(win):
    scan(win)
    win.setAllPeaksChecked(False)
    win.drawSinglePeaks(win.peaks, win.properties, win.datay, 1)
    assert win.peak_txt[1].get_text() == str(int(win.datax[win.peaks[1]]))


def test_update_peak_output_appends_without_clearing(win):
    scan(win)
    before = len(win.extraPopup.peak.peak_results.lines)
    win.update_peak_output(win.peaks, win.properties)
    assert len(win.extraPopup.peak.peak_results.lines) == before * 2
