"""Characterization tests for the Peak Finder 2 helpers every other part of the
cluster calls: drawing a fit, fetching its spectrum, testing whether its pad is
still alive, the shape menus, the Config cap and the redraw sweep."""

import logging
import os
import sys
import types

import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import gui_stubs


# --------------------------------------------------------------- fake widgets

class FakeLabel:
    def __init__(self):
        self._text = ""

    def text(self):
        return self._text

    def setText(self, t):
        self._text = t


class FakeCombo:
    def __init__(self, value):
        self.value = value

    def currentText(self):
        return self.value


class FakePeakTab:
    def __init__(self):
        self.peak2_status = FakeLabel()
        self.peak2_signal = FakeCombo("Gaussian")
        self.peak2_bg = FakeCombo("Linear")
        self.peak2_cb_tail = FakeCombo("left")


class FakeExtraPopup:
    def __init__(self):
        self.peak = FakePeakTab()


class FakeStore:
    """Name-keyed, like SpectrumStore: an index means nothing to it."""

    def __init__(self, records):
        self.records = records
        self.asked = []

    def contains(self, name):
        return name in self.records

    def get(self, name, field):
        self.asked.append((name, field))
        return self.records[name][field]


# ------------------------------------------------------------------- fixtures

def make_result(mus=(50.0,), areas=(100.0,)):
    """A composite fit result of the shape _peak2_draw and the mu helper read."""
    xx = np.linspace(40.0, 60.0, 41)
    y_bg = np.ones_like(xx) * 2.0
    y_comp = [np.exp(-0.5 * ((xx - mu) / 2.0) ** 2) * 10.0 for mu in mus]
    y_fit = y_bg + sum(y_comp)
    return {
        "xx": xx, "y_fit": y_fit, "y_bg": y_bg,
        "y_comp": y_comp if len(y_comp) > 1 else [],
        "components": [{"mu": mu, "area": area, "sigma": 2.0}
                       for mu, area in zip(mus, areas)],
    }


# createRange returns bins+1 edges, and the counts array carries the underflow
# channel at index 0 — so edges and counts are the same length, and dropping one
# from each end is what lines them up.
SPECTRUM = {"binx": 10, "minx": 0.0, "maxx": 10.0,
            "data": np.arange(11, dtype=float)}


@pytest.fixture
def win(monkeypatch):
    gui = gui_stubs.import_gui()
    w = gui.MainWindow.__new__(gui.MainWindow)
    w.logger = logging.getLogger("test.peakfit2.floor")
    w.extraPopup = FakeExtraPopup()

    w.store = FakeStore({"alpha": dict(SPECTRUM), "beta": dict(SPECTRUM)})
    w.spectra = w.store
    w.getSpectrumStoreInfo = lambda field, index=None, name=None: w.store.get(name, field)
    w.plot_controller = types.SimpleNamespace(
        createRange=lambda bins, vmin, vmax: np.linspace(
            float(vmin), float(vmax), int(bins) + 1))

    # a REAL canvas, counting its own redraws: figure.delaxes reaches into the
    # canvas, so a stand-in cannot exercise the detached-axes case
    w.figure = Figure()
    canvas = FigureCanvasAgg(w.figure)
    canvas.draws = 0
    canvas.draw_idle = lambda: setattr(canvas, "draws", canvas.draws + 1)
    w.axes = w.figure.add_subplot(111)
    w.axes_by_index = {0: w.axes}
    w.getSpectrumViewInfo = lambda field, index=None: w.axes_by_index.get(index)

    w.peak2_fits = []

    settings = {}
    w.settings = settings

    class FakeSettings:
        def value(self, key, default=None, type=None):
            return settings.get(key, default)

        def setValue(self, key, value):
            settings[key] = value

    from controllers import peak_fit2_controller as pfc
    monkeypatch.setattr(gui, "QSettings", FakeSettings, raising=False)
    monkeypatch.setattr(pfc, "QSettings", FakeSettings)
    w.peak_fit2_controller = pfc.PeakFit2Controller(
        peak_tab=w.extraPopup.peak,
        spectra=w.spectra,
        # late-bound: several tests replace the window's lookups after the
        # fixture has run, and must still reach the controller through them
        get_store_info=lambda field, index=None, name=None: w.getSpectrumStoreInfo(
            field, index=index, name=name),
        get_view_info=lambda field, index=None: w.getSpectrumViewInfo(field, index=index),
        plot_controller=w.plot_controller,
        logger=w.logger,
    )
    # The cluster's state lives on the controller; these proxies let the
    # unchanged test bodies keep reading it on the window.
    for attr in ("peak2_fits", "peak2_count", "peak2_armed", "peak2_fix_armed",
                 "peak2_drag", "peak2_conns"):
        monkeypatch.setattr(
            type(w), attr,
            property(lambda self, a=attr: getattr(self.peak_fit2_controller, a),
                     lambda self, v, a=attr: setattr(self.peak_fit2_controller, a, v)),
            raising=False)
    # 8d dropped every shim that is not a .connect() target, so the fixture
    # binds those names onto the window instance: the test bodies still drive
    # MainWindow exactly as they did before the move, and the ones MainWindow
    # still defines keep routing through its own shim.
    for _name in dir(w.peak_fit2_controller):
        if (_name.startswith(("_peak2_", "peakFit2", "onPeakFit2"))
                and not hasattr(type(w), _name)
                and _name not in w.__dict__):     # keep this fixture's own doubles
            setattr(w, _name, getattr(w.peak_fit2_controller, _name))
    return w


# ------------------------------------------------------- _peak2_spectrum_arrays

def test_spectrum_arrays_resolves_by_name(win):
    """The store is asked for this NAME, and nothing consults a pad index on the
    way."""
    win.store.asked = []
    win._peak2_spectrum_arrays("beta")
    assert {name for name, _ in win.store.asked} == {"beta"}


def test_spectrum_arrays_returns_bin_centres_and_matching_counts(win):
    xc, y = win._peak2_spectrum_arrays("alpha")
    assert len(xc) == len(y)
    edges = np.linspace(0.0, 10.0, 11)
    assert xc == pytest.approx(edges[:-1] + 0.5 * np.diff(edges))


def test_spectrum_arrays_drops_the_underflow_channel(win):
    """The counts are offset by one against the edges: bin 0 of the shm array is
    the underflow channel, so the first REAL count pairs with the first centre."""
    xc, y = win._peak2_spectrum_arrays("alpha")
    assert y[0] == SPECTRUM["data"][1]


def test_spectrum_arrays_says_none_for_a_removed_spectrum(win):
    assert win._peak2_spectrum_arrays("gone") is None


def test_spectrum_arrays_asks_the_store_before_trying_the_fields(win):
    """A removed spectrum is an expected state, not an error: the containment
    test has to come first, so absence never reaches the field reads."""
    win.store.asked = []
    win._peak2_spectrum_arrays("gone")
    assert win.store.asked == []


def test_spectrum_arrays_survives_a_store_that_raises(win):
    def boom(field, index=None, name=None):
        raise RuntimeError("store went away mid-read")
    win.getSpectrumStoreInfo = boom
    assert win._peak2_spectrum_arrays("alpha") is None


# ----------------------------------------------------- _peak2_live_axes (both)

def test_live_axes_returns_the_axes_of_a_drawn_fit(win):
    (line,) = win.axes.plot([0, 1], [0, 1])
    assert win._peak2_live_axes({"artists": (line,)}) is win.axes


def test_live_axes_says_none_when_the_record_has_no_artists(win):
    assert win._peak2_live_axes({"artists": ()}) is None
    assert win._peak2_live_axes({}) is None


def test_live_axes_says_none_after_the_pad_was_cleared(win):
    """Spectrum removal calls ax.clear(), which nulls the artist's .axes."""
    (line,) = win.axes.plot([0, 1], [0, 1])
    win.axes.clear()
    assert win._peak2_live_axes({"artists": (line,)}) is None


def test_live_axes_says_none_after_the_axes_was_detached(win):
    """The subtle one. A geometry change runs figure.delaxes, which leaves
    both artist.axes and axes.figure pointing at live objects — only
    membership of figure.axes changes."""
    (line,) = win.axes.plot([0, 1], [0, 1])
    win.figure.delaxes(win.axes)
    assert win._peak2_live_axes({"artists": (line,)}) is None


# ------------------------------------------------------------- _peak2_status

def test_status_shows_the_message(win):
    win._peak2_status("[armed] click a peak")
    assert win.extraPopup.peak.peak2_status.text() == "[armed] click a peak"


def test_status_replaces_rather_than_accumulates(win):
    win._peak2_status("[armed]")
    win._peak2_status("[config] Max window: 60 bins.")
    assert win.extraPopup.peak.peak2_status.text() == "[config] Max window: 60 bins."


# -------------------------------------------------------- _peak2_current_spec

def test_current_spec_maps_the_menu_labels_to_model_keys(win):
    win.extraPopup.peak.peak2_signal = FakeCombo("Crystal ball")
    win.extraPopup.peak.peak2_bg = FakeCombo("Cubic")
    spec = win._peak2_current_spec()
    assert spec["signal"] == "crystal_ball"
    assert spec["background"] == "poly3"


def test_current_spec_defaults_an_unknown_label_rather_than_failing(win):
    win.extraPopup.peak.peak2_signal = FakeCombo("Something new")
    win.extraPopup.peak.peak2_bg = FakeCombo("Quartic")
    spec = win._peak2_current_spec()
    assert spec["signal"] == "gaussian"
    assert spec["background"] == "poly1"


def test_current_spec_is_always_single_component(win):
    """Multi-component fits only ever arise from the auto-add on drag, never
    from the menus."""
    assert win._peak2_current_spec()["n_components"] == 1


def test_current_spec_carries_the_tail_side_through(win):
    win.extraPopup.peak.peak2_cb_tail = FakeCombo("right")
    assert win._peak2_current_spec()["tail_side"] == "right"


# ------------------------------------------------------------- _peak2_result_mu

def test_result_mu_of_a_single_component_fit_is_its_mu(win):
    assert win._peak2_result_mu(make_result(mus=(50.0,))) == 50.0


def test_result_mu_quotes_the_strongest_component_not_the_first(win):
    """The component order is the seeding order, which means nothing to the
    user; area is what makes a component the one the fit is about."""
    r = make_result(mus=(45.0, 55.0), areas=(10.0, 900.0))
    assert win._peak2_result_mu(r) == 55.0


def test_result_mu_ties_break_toward_the_first_component(win):
    r = make_result(mus=(45.0, 55.0), areas=(100.0, 100.0))
    assert win._peak2_result_mu(r) == 45.0


# ------------------------------------------------------------- _peak2_draw

def test_draw_puts_the_curve_first_and_the_fill_third(win):
    """Positional contract: drag-grab and edit-hit index this tuple."""
    arts = win._peak2_draw(win.axes, make_result())
    assert arts[0] in win.axes.lines
    assert arts[2] in win.axes.collections


def test_draw_of_a_single_component_makes_four_artists(win):
    arts = win._peak2_draw(win.axes, make_result())
    assert len(arts) == 4


def test_draw_adds_one_dashed_overlay_per_component_when_multi(win):
    arts = win._peak2_draw(win.axes, make_result(mus=(45.0, 55.0), areas=(1.0, 2.0)))
    assert len(arts) == 6


def test_draw_tags_every_artist(win):
    for art in win._peak2_draw(win.axes, make_result()):
        assert art.get_gid() == "peakfit2"


def test_draw_puts_everything_on_the_axes_it_was_given(win):
    other = win.figure.add_subplot(212)
    before = len(other.lines)
    win._peak2_draw(win.axes, make_result())
    assert len(other.lines) == before


# -------------------------------------------------- _peak2_max_window_bins

def test_max_window_bins_is_none_when_unset(win):
    assert win._peak2_max_window_bins() is None


def test_max_window_bins_reads_the_configured_cap(win):
    win.settings["PeakFinder2/max_window_bins"] = "60"
    assert win._peak2_max_window_bins() == 60


def test_max_window_bins_rejects_a_non_positive_cap(win):
    win.settings["PeakFinder2/max_window_bins"] = "0"
    assert win._peak2_max_window_bins() is None


def test_max_window_bins_rejects_junk_rather_than_raising(win):
    win.settings["PeakFinder2/max_window_bins"] = "sixty"
    assert win._peak2_max_window_bins() is None


# --------------------------------------------------------- peakFit2RedrawAll

def test_redraw_all_replaces_every_records_artists(win):
    rec = {"index": 0, "result": make_result(), "artists": ()}
    rec["artists"] = win._peak2_draw(win.axes, rec["result"])
    win.peak2_fits = [rec]
    old = rec["artists"]
    win.peakFit2RedrawAll()
    assert rec["artists"] != old
    assert all(a not in win.axes.lines for a in old if hasattr(a, "get_xdata"))


def test_redraw_all_does_not_accumulate_artists(win):
    rec = {"index": 0, "result": make_result(), "artists": ()}
    rec["artists"] = win._peak2_draw(win.axes, rec["result"])
    win.peak2_fits = [rec]
    after_first = (len(win.axes.lines), len(win.axes.collections))
    win.peakFit2RedrawAll()
    win.peakFit2RedrawAll()
    assert (len(win.axes.lines), len(win.axes.collections)) == after_first


def test_redraw_all_empties_the_record_when_its_pad_is_gone(win):
    rec = {"index": 7, "result": make_result(), "artists": ()}
    rec["artists"] = win._peak2_draw(win.axes, rec["result"])
    win.peak2_fits = [rec]
    win.peakFit2RedrawAll()          # index 7 resolves to no axis
    assert rec["artists"] == ()


def test_redraw_all_survives_a_stale_artist_handle(win):
    rec = {"index": 0, "result": make_result(), "artists": (object(),)}
    win.peak2_fits = [rec]
    win.peakFit2RedrawAll()
    assert len(rec["artists"]) == 4


def test_redraw_all_draws_each_canvas_once_not_once_per_fit(win):
    recs = []
    for _ in range(3):
        rec = {"index": 0, "result": make_result(), "artists": ()}
        rec["artists"] = win._peak2_draw(win.axes, rec["result"])
        recs.append(rec)
    win.peak2_fits = recs
    win.figure.canvas.draws = 0
    win.peakFit2RedrawAll()
    assert win.figure.canvas.draws == 1


def test_redraw_all_on_an_empty_record_list_is_a_no_op(win):
    win.peak2_fits = []
    win.peakFit2RedrawAll()
    assert win.figure.canvas.draws == 0
