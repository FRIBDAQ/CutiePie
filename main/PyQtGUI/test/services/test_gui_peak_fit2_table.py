"""Characterization tests for Peak Finder 2's results table, shape menus and edit
popup: adding and updating rows, the selection, highlight, delete, clear, the
menu re-entrancy guard, and the modal editor."""

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

USER_ROLE = 0x0100          # QtCore.Qt.UserRole


# --------------------------------------------------------------- fake widgets

class FakeItem:
    def __init__(self, text=""):
        self._text = text
        self._data = {}
        self.tooltip = ""

    def text(self):
        return self._text

    def setText(self, t):
        self._text = t

    def data(self, role):
        return self._data.get(role)

    def setData(self, role, value):
        self._data[role] = value

    def setToolTip(self, t):
        self.tooltip = t


class FakeTable:
    """QTableWidget stand-in: rows of FakeItem, a current row, sorting flag."""

    def __init__(self):
        self.rows = []
        self.current = -1
        self.sorting = True
        self.sorting_calls = []

    def rowCount(self):
        return len(self.rows)

    def insertRow(self, ri):
        self.rows.insert(ri, {})

    def removeRow(self, ri):
        del self.rows[ri]

    def setRowCount(self, n):
        self.rows = self.rows[:n]

    def item(self, ri, ci):
        if ri < 0 or ri >= len(self.rows):
            return None
        return self.rows[ri].get(ci)

    def setItem(self, ri, ci, item):
        self.rows[ri][ci] = item

    def currentRow(self):
        return self.current

    def setSortingEnabled(self, on):
        self.sorting = bool(on)
        self.sorting_calls.append(bool(on))


class FakeCombo:
    def __init__(self, value=""):
        self.value = value
        self.blocked = False
        self.sets_while_unblocked = 0

    def currentText(self):
        return self.value

    def setCurrentText(self, v):
        self.value = v
        if not self.blocked:
            self.sets_while_unblocked += 1

    def blockSignals(self, on):
        self.blocked = bool(on)


class FakeLabel:
    def __init__(self):
        self._text = ""

    def text(self):
        return self._text

    def setText(self, t):
        self._text = t


class FakePeakTab:
    def __init__(self):
        self.peak2_table = FakeTable()
        self.peak2_status = FakeLabel()
        self.peak2_signal = FakeCombo("Gaussian")
        self.peak2_bg = FakeCombo("Linear")
        self.peak2_cb_tail = FakeCombo("low")


class FakeExtraPopup:
    def __init__(self):
        self.peak = FakePeakTab()


def fit_result(lo=40.0, hi=60.0, mus=(50.0,), ok=True, error=None, spec=None):
    xx = np.linspace(lo, hi, 21)
    y_bg = np.ones_like(xx)
    y_comp = [np.exp(-0.5 * ((xx - mu) / 2.0) ** 2) * 10.0 for mu in mus]
    # the full shape the results-table formatter reads (uncertainties, the
    # background parameters and the window it was fitted over)
    r = {"ok": ok, "xx": xx, "y_bg": y_bg, "y_fit": y_bg + sum(y_comp),
         "y_comp": y_comp if len(y_comp) > 1 else [],
         "components": [{"mu": mu, "dmu": 0.01, "A": 10.0, "dA": 0.1,
                         "sigma": 2.0, "dsigma": 0.02,
                         "fwhm": 4.71, "dfwhm": 0.05,
                         "area": 100.0 + i, "darea": 1.0}
                        for i, mu in enumerate(mus)],
         "bg_params": {"m": 0.0, "b": 1.0},
         "redchi": 1.05, "win_lo": lo, "win_hi": hi,
         "spec": spec or {"signal": "gaussian", "background": "poly1",
                          "n_components": len(mus), "tail_side": "low"}}
    if error is not None:
        r["error"] = error
    return r


SPECTRUM = {"dim": 1, "binx": 100, "minx": 0.0, "maxx": 100.0,
            "data": np.arange(101, dtype=float)}


@pytest.fixture
def win(monkeypatch):
    gui = gui_stubs.import_gui()
    w = gui.MainWindow.__new__(gui.MainWindow)
    w.logger = logging.getLogger("test.peakfit2.table")
    w.extraPopup = FakeExtraPopup()

    figure = Figure()
    FigureCanvasAgg(figure)
    w.ax = figure.add_subplot(111)
    w.ax.set_xlim(0, 100)
    w.currentPlot = types.SimpleNamespace(figure=figure, canvas=figure.canvas,
                                          isEnlarged=False)
    w.wTab = types.SimpleNamespace(currentIndex=lambda: 0,
                                   plot=lambda i: types.SimpleNamespace(canvas=figure.canvas),
                                   selectedPad=lambda i: 0)

    w.store = dict(SPECTRUM)
    w.getSpectrumStoreInfo = lambda field, index=None, name=None: w.store[field]
    w.plot_controller = types.SimpleNamespace(
        createRange=lambda bins, vmin, vmax: np.linspace(
            float(vmin), float(vmax), int(bins) + 1))

    w.peak2_fits = []
    w.peak2_count = 0
    w.settings = {}

    class FakeSettings:
        def value(self, key, default=None, type=None):
            return w.settings.get(key, default)

        def setValue(self, key, value):
            w.settings[key] = value

    # the modal editor, scripted: what the user typed and whether they accepted
    w.dialog_script = None          # dict(mu=…, sigma=…, fwhm=…, accept=bool)
    w.dialog_titles = []

    w.refits = []

    def fake_composite(xc, y, lo, hi, spec, fixed=None, seeds=None):
        w.refits.append({"lo": lo, "hi": hi, "spec": spec,
                         "fixed": fixed, "seeds": seeds})
        return w.refit_result

    w.refit_result = fit_result()

    monkeypatch.setattr(gui, "QSettings", FakeSettings, raising=False)
    # QTableWidgetItem has no behavior under the stub, so the numeric-sorting
    # cell class it subclasses cannot store anything. The sort ORDER is Qt's
    # job; what this cluster owns is what goes into each cell.
    monkeypatch.setattr(gui, "_NumericItem", FakeItem, raising=False)
    _numeric_item_owner = ("gui", "pfc")     # patched in both, see below

    from controllers import peak_fit2_controller as pfc
    monkeypatch.setattr(pfc, "QSettings", FakeSettings)
    monkeypatch.setattr(pfc, "_NumericItem", FakeItem, raising=False)
    # the fit engine is reached from whichever module currently owns these
    # methods, so both namespaces are covered
    monkeypatch.setattr(pfc, "fit_composite", fake_composite)
    monkeypatch.setattr(gui, "fit_composite", fake_composite, raising=False)
    w.peak_fit2_controller = pfc.PeakFit2Controller(
        peak_tab=w.extraPopup.peak,
        spectra=types.SimpleNamespace(contains=lambda name: name == "alpha"),
        get_store_info=lambda field, index=None, name=None: w.getSpectrumStoreInfo(
            field, index=index, name=name),
        get_view_info=lambda field, index=None: w.ax,
        plot_controller=w.plot_controller,
        tabs=w.wTab,
        get_current_plot=lambda: w.currentPlot,
        get_gate_popup=lambda: types.SimpleNamespace(isVisible=lambda: False),
        get_sum_popup=lambda: types.SimpleNamespace(isVisible=lambda: False),
        name_from_index=lambda index: "alpha",
        parent_widget=w,
        logger=w.logger,
    )
    for attr in ("peak2_armed", "peak2_fix_armed", "peak2_drag", "peak2_conns"):
        monkeypatch.setattr(
            type(w), attr,
            property(lambda self, a=attr: getattr(self.peak_fit2_controller, a),
                     lambda self, v, a=attr: setattr(self.peak_fit2_controller, a, v)),
            raising=False)

    # The modal editor is a QDialog built inline. Replace the dialog class so a
    # test can say what the user typed and whether they pressed Apply; the real
    # widget construction is Qt's, not this cluster's, behavior.
    class FakeDialog:
        Accepted = 1
        Rejected = 0

        def __init__(self, parent=None):
            self.title = ""
            self.layout = None

        def setWindowTitle(self, t):
            self.title = t
            w.dialog_titles.append(t)

        def setLayout(self, lay):
            self.layout = lay

        def accept(self):
            pass

        def reject(self):
            pass

        def exec_(self):
            script = w.dialog_script or {}
            for field, edit in w.edits.items():
                if field in script:
                    edit.setText(str(script[field]))
            return (FakeDialog.Accepted if script.get("accept")
                    else FakeDialog.Rejected)

    w.edits = {}

    class FakeLineEdit:
        def __init__(self, text=""):
            self._text = text
            self._slots = []
            # μ is created first, then σ, then FWHM
            w.edits[("mu", "sigma", "fwhm")[len(w.edits) % 3]] = self

        def text(self):
            return self._text

        def setText(self, t):
            self._text = str(t)
            for s in self._slots:
                s(self._text)

        @property
        def textEdited(self):
            outer = self

            class Sig:
                def connect(self, slot):
                    outer._slots.append(slot)
            return Sig()

    class FakeButton:
        def __init__(self, text=""):
            self._text = text

        @property
        def clicked(self):
            class Sig:
                def connect(self, slot):
                    pass
            return Sig()

    class FakeLayout:
        def __init__(self, *a, **k):
            pass

        def addRow(self, *a):
            pass

        def addWidget(self, *a):
            pass

        def addLayout(self, *a):
            pass

    for name, obj in (("QDialog", FakeDialog), ("QLineEdit", FakeLineEdit),
                      ("QPushButton", FakeButton), ("QFormLayout", FakeLayout),
                      ("QHBoxLayout", FakeLayout), ("QVBoxLayout", FakeLayout)):
        monkeypatch.setattr(pfc, name, obj, raising=False)
        monkeypatch.setattr(gui, name, obj, raising=False)
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


def add_fit(win, number=1, mus=(50.0,), name="alpha", lo=40.0, hi=60.0):
    r = fit_result(lo=lo, hi=hi, mus=mus)
    rec = {"number": number, "index": 0, "name": name, "result": r,
           "artists": win.peak_fit2_controller._peak2_draw(win.ax, r)}
    win.peak2_fits.append(rec)
    win._peak2_add_row(number, r)
    return rec


def select_row(win, ri):
    win.extraPopup.peak.peak2_table.current = ri


def status(win):
    return win.extraPopup.peak.peak2_status.text()


def table(win):
    return win.extraPopup.peak.peak2_table


# ----------------------------------------------------------------- add/update

def test_a_fit_becomes_one_row(win):
    add_fit(win)
    assert table(win).rowCount() == 1


def test_the_number_cell_carries_the_fit_number_as_its_sort_value(win):
    """This value is also the row-to-fit lookup key for delete and highlight, so
    sorting the table cannot break either."""
    add_fit(win, number=7)
    assert table(win).item(0, 0).data(USER_ROLE) == 7.0


def test_every_cell_carries_a_numeric_sort_value(win):
    add_fit(win)
    row = table(win).rows[0]
    assert all(item.data(USER_ROLE) is not None for item in row.values())


def test_every_cell_carries_the_hover_detail(win):
    add_fit(win)
    row = table(win).rows[0]
    assert all(item.tooltip for item in row.values())


def test_sorting_is_suspended_during_the_insert(win):
    """Re-sorting mid-insert would move the row out from under the writes."""
    add_fit(win)
    assert table(win).sorting_calls[:2] == [False, True]
    assert table(win).sorting is True


def test_an_update_rewrites_that_fits_row_in_place(win):
    add_fit(win, number=3)
    win._peak2_update_row(3, fit_result(mus=(55.0,)), tag="edited")
    assert table(win).rowCount() == 1
    assert table(win).item(0, 0).data(USER_ROLE) == 3.0


def test_an_update_finds_the_row_by_number_not_by_position(win):
    add_fit(win, number=1)
    add_fit(win, number=2)
    before = table(win).item(0, 1).text()
    win._peak2_update_row(2, fit_result(mus=(70.0,)))
    assert table(win).item(0, 1).text() == before


def test_updating_an_unknown_number_changes_nothing(win):
    add_fit(win, number=1)
    before = [i.text() for i in table(win).rows[0].values()]
    win._peak2_update_row(99, fit_result())
    assert [i.text() for i in table(win).rows[0].values()] == before


# ------------------------------------------------------------- the selection

def test_no_selection_reads_as_none(win):
    add_fit(win)
    select_row(win, -1)
    assert win._peak2_selected_number() is None


def test_the_selected_row_reports_its_fit_number(win):
    add_fit(win, number=4)
    select_row(win, 0)
    assert win._peak2_selected_number() == 4


def test_a_row_with_no_number_cell_reads_as_none(win):
    add_fit(win)
    table(win).rows[0][0] = None
    select_row(win, 0)
    assert win._peak2_selected_number() is None


# ------------------------------------------------------------------- highlight

def test_selecting_a_row_highlights_that_curve(win):
    rec = add_fit(win, number=1)
    select_row(win, 0)
    win._peak2_row_selected()
    curve = rec["artists"][0]
    assert curve.get_linewidth() == pytest.approx(3.2)
    assert curve.get_color() == "tab:orange"


def test_selecting_a_row_restores_the_others(win):
    """Select the first fit so it IS highlighted, then select the second: the
    first has to go back to the default. Without that step the assertion is
    satisfied by the colour the curve was drawn with."""
    first = add_fit(win, number=1)
    add_fit(win, number=2)
    select_row(win, 0)
    win._peak2_row_selected()
    assert first["artists"][0].get_color() == "tab:orange"
    select_row(win, 1)
    win._peak2_row_selected()
    assert first["artists"][0].get_color() == "tab:red"
    assert first["artists"][0].get_linewidth() == pytest.approx(1.8)


def test_selecting_a_row_syncs_the_shape_menus(win):
    """The menus reflect the fit you would act on."""
    add_fit(win, number=1)
    win.peak2_fits[0]["result"]["spec"] = {"signal": "crystal_ball",
                                           "background": "poly3",
                                           "tail_side": "right"}
    select_row(win, 0)
    win._peak2_row_selected()
    assert win.extraPopup.peak.peak2_signal.currentText() == "Crystal ball"
    assert win.extraPopup.peak.peak2_bg.currentText() == "Cubic"


def test_the_menu_sync_never_fires_the_shape_slot(win):
    """The re-entrancy guard. Unblocked, selecting a row re-fits the fit you just
    selected, silently."""
    add_fit(win, number=1)
    select_row(win, 0)
    win._peak2_row_selected()
    assert win.extraPopup.peak.peak2_signal.sets_while_unblocked == 0
    assert win.refits == []


def test_a_record_with_no_artists_is_skipped_by_the_highlight(win):
    rec = add_fit(win, number=1)
    rec["artists"] = ()
    select_row(win, 0)
    win._peak2_row_selected()          # must not raise


# ---------------------------------------------------------------------- delete

def test_delete_with_no_selection_says_so(win):
    add_fit(win)
    select_row(win, -1)
    win._peak2_delete_selected()
    assert "[delete]" in status(win) and "Select a fit row" in status(win)
    assert table(win).rowCount() == 1


def test_delete_removes_the_row_the_record_and_the_artists(win):
    rec = add_fit(win, number=1)
    artists = rec["artists"]
    select_row(win, 0)
    win._peak2_delete_selected()
    assert table(win).rowCount() == 0
    assert win.peak2_fits == []
    assert all(a not in win.ax.lines for a in artists if hasattr(a, "get_xdata"))


def test_delete_takes_only_the_selected_fit(win):
    add_fit(win, number=1)
    keep = add_fit(win, number=2)
    select_row(win, 0)
    win._peak2_delete_selected()
    assert [r["number"] for r in win.peak2_fits] == [2]
    assert table(win).rowCount() == 1
    assert keep["artists"][0] in win.ax.lines


def test_delete_names_the_peak_it_removed(win):
    add_fit(win, number=5)
    select_row(win, 0)
    win._peak2_delete_selected()
    assert "Peak 5" in status(win)


def test_delete_releases_the_handler_when_the_canvas_empties(win):
    """A canvas with no fits left and no arming does not need the press
    handler."""
    add_fit(win, number=1)
    win._peak2_connect(win.currentPlot.canvas)
    select_row(win, 0)
    win._peak2_delete_selected()
    assert win.peak2_conns == {}


def test_delete_keeps_the_handler_while_armed(win):
    add_fit(win, number=1)
    win._peak2_connect(win.currentPlot.canvas)
    win.peak2_armed = True
    select_row(win, 0)
    win._peak2_delete_selected()
    assert win.currentPlot.canvas in win.peak2_conns


# ------------------------------------------------------------------- clear

def test_clear_empties_the_table_the_records_and_the_pad(win):
    rec = add_fit(win, number=1)
    artists = rec["artists"]
    win.peakFit2Clear()
    assert table(win).rowCount() == 0
    assert win.peak2_fits == []
    assert all(a not in win.ax.lines for a in artists if hasattr(a, "get_xdata"))


def test_clear_resets_the_peak_numbering(win):
    add_fit(win, number=1)
    win.peak2_count = 1
    win.peakFit2Clear()
    assert win.peak2_count == 0


def test_clear_blanks_the_status_line(win):
    add_fit(win, number=1)
    win._peak2_status("something")
    win.peakFit2Clear()
    assert status(win) == ""


def test_clear_releases_the_handler(win):
    add_fit(win, number=1)
    win._peak2_connect(win.currentPlot.canvas)
    win.peakFit2Clear()
    assert win.peak2_conns == {}


def test_clear_on_an_empty_panel_does_not_raise(win):
    win.peakFit2Clear()
    win.peakFit2Clear()
    assert win.peak2_fits == []


# -------------------------------------------------------------- shape menus

def test_the_menus_restore_the_last_used_selection(win):
    win.settings["PeakFinder2/signal_shape"] = "Crystal ball"
    win.settings["PeakFinder2/background_shape"] = "Quadratic"
    win._peak2_load_shape_menus()
    assert win.extraPopup.peak.peak2_signal.currentText() == "Crystal ball"
    assert win.extraPopup.peak.peak2_bg.currentText() == "Quadratic"


def test_restoring_the_menus_never_fires_the_shape_slot(win):
    win.settings["PeakFinder2/signal_shape"] = "Crystal ball"
    win._peak2_load_shape_menus()
    assert win.extraPopup.peak.peak2_signal.sets_while_unblocked == 0


def test_an_unset_menu_keeps_its_default(win):
    win._peak2_load_shape_menus()
    assert win.extraPopup.peak.peak2_signal.currentText() == "Gaussian"


def test_changing_a_menu_persists_all_three(win):
    win.extraPopup.peak.peak2_signal.value = "Crystal ball"
    win._peak2_shape_changed()
    assert win.settings["PeakFinder2/signal_shape"] == "Crystal ball"
    assert "PeakFinder2/background_shape" in win.settings
    assert "PeakFinder2/cb_tail_side" in win.settings


def test_changing_a_menu_with_no_row_selected_does_not_refit(win):
    """With nothing selected the menu only sets the default for the next fit."""
    add_fit(win, number=1)
    select_row(win, -1)
    win._peak2_shape_changed()
    assert win.refits == []


def test_changing_a_menu_refits_the_selected_fit(win):
    add_fit(win, number=1)
    select_row(win, 0)
    win.extraPopup.peak.peak2_signal.value = "Crystal ball"
    win._peak2_shape_changed()
    assert len(win.refits) == 1


# ------------------------------------------------------------ refit-selected

def test_a_refit_keeps_the_fits_own_window(win):
    rec = add_fit(win, number=1, lo=42.0, hi=58.0)
    win._peak2_refit_selected(rec)
    assert (win.refits[0]["lo"], win.refits[0]["hi"]) == (42.0, 58.0)


def test_a_refit_keeps_the_component_count(win):
    rec = add_fit(win, number=1, mus=(45.0, 55.0))
    win._peak2_refit_selected(rec)
    assert win.refits[0]["spec"]["n_components"] == 2


def test_a_refit_seeds_mu_from_the_existing_components(win):
    """So the refit stays on the same peaks instead of wandering."""
    rec = add_fit(win, number=1, mus=(45.0, 55.0))
    win._peak2_refit_selected(rec)
    assert win.refits[0]["seeds"] == {"mu1": 45.0, "mu2": 55.0}


def test_a_refit_replaces_the_artists_and_the_row(win):
    rec = add_fit(win, number=1)
    old = rec["artists"]
    win._peak2_refit_selected(rec)
    assert rec["artists"] != old
    assert table(win).rowCount() == 1


def test_a_failed_refit_keeps_the_previous_fit(win):
    rec = add_fit(win, number=1)
    old_result, old_artists = rec["result"], rec["artists"]
    win.refit_result = fit_result(ok=False, error="did not converge")
    win._peak2_refit_selected(rec)
    assert "[shape]" in status(win) and "unchanged" in status(win)
    assert rec["result"] is old_result and rec["artists"] is old_artists


def test_a_refit_of_a_removed_spectrum_is_reported(win):
    rec = add_fit(win, number=1, name="gone")
    old_artists = rec["artists"]
    win._peak2_refit_selected(rec)
    assert "unavailable" in status(win)
    assert rec["artists"] is old_artists


# --------------------------------------------------------------- edit popup

def test_the_editor_offers_the_components_current_values(win):
    rec = add_fit(win, number=1, mus=(50.0,))
    win.dialog_script = {"accept": False}
    win._peak2_open_edit(rec, 50.0)
    assert win.edits["mu"].text() == "50"


def test_cancel_changes_nothing(win):
    rec = add_fit(win, number=1)
    old_result = rec["result"]
    win.dialog_script = {"mu": 51.0, "accept": False}
    win._peak2_open_edit(rec, 50.0)
    assert rec["result"] is old_result
    assert win.refits == []


def test_applying_with_nothing_changed_says_so(win):
    rec = add_fit(win, number=1)
    win.dialog_script = {"accept": True}
    win._peak2_open_edit(rec, 50.0)
    assert "nothing changed" in status(win)
    assert win.refits == []


def test_only_the_edited_field_becomes_fixed(win):
    """Fields the user did not touch stay free."""
    rec = add_fit(win, number=1)
    win.dialog_script = {"mu": 51.0, "accept": True}
    win._peak2_open_edit(rec, 50.0)
    assert win.refits[0]["fixed"] == {"mu1": 51.0}


def test_editing_the_width_fixes_sigma(win):
    rec = add_fit(win, number=1)
    win.dialog_script = {"sigma": 3.0, "accept": True}
    win._peak2_open_edit(rec, 50.0)
    assert win.refits[0]["fixed"] == {"sigma1": 3.0}


def test_editing_fwhm_reaches_sigma_through_the_link(win):
    """σ and FWHM are linked live, so a width edit either way is caught."""
    rec = add_fit(win, number=1)
    win.dialog_script = {"fwhm": 9.42, "accept": True}
    win._peak2_open_edit(rec, 50.0)
    assert "sigma1" in win.refits[0]["fixed"]
    assert win.refits[0]["fixed"]["sigma1"] == pytest.approx(4.0, rel=1e-3)


def test_an_invalid_number_is_refused(win):
    rec = add_fit(win, number=1)
    old_result = rec["result"]
    win.dialog_script = {"mu": "wide", "accept": True}
    win._peak2_open_edit(rec, 50.0)
    assert "invalid number" in status(win)
    assert rec["result"] is old_result


def test_a_mu_outside_the_window_is_refused(win):
    """K13(f)'s range guard: pinning μ outside the fit window cannot converge."""
    rec = add_fit(win, number=1, lo=40.0, hi=60.0)
    win.dialog_script = {"mu": 999.0, "accept": True}
    win._peak2_open_edit(rec, 50.0)
    assert "unchanged" in status(win)
    assert win.refits == []


def test_a_multi_component_edit_targets_the_nearest_component(win):
    """The right-clicked x picks which component is edited."""
    rec = add_fit(win, number=1, mus=(45.0, 55.0))
    win.dialog_script = {"mu": 56.0, "accept": True}
    win._peak2_open_edit(rec, 54.0)
    assert win.refits[0]["fixed"] == {"mu2": 56.0}


def test_a_multi_component_edit_names_the_component_in_the_title(win):
    rec = add_fit(win, number=1, mus=(45.0, 55.0))
    win.dialog_script = {"accept": False}
    win._peak2_open_edit(rec, 54.0)
    assert "component 2/2" in win.dialog_titles[-1]


def test_a_multi_component_edit_seeds_the_others_where_they_are(win):
    rec = add_fit(win, number=1, mus=(45.0, 55.0))
    win.dialog_script = {"mu": 56.0, "accept": True}
    win._peak2_open_edit(rec, 54.0)
    assert win.refits[0]["seeds"] == {"mu1": 45.0, "mu2": 55.0}


def test_a_right_click_with_no_x_edits_the_first_component(win):
    rec = add_fit(win, number=1, mus=(45.0, 55.0))
    win.dialog_script = {"accept": False}
    win._peak2_open_edit(rec, None)
    assert win.edits["mu"].text() == "45"


def test_a_failed_edit_refit_keeps_the_previous_fit(win):
    rec = add_fit(win, number=1)
    old_result = rec["result"]
    win.refit_result = fit_result(ok=False, error="singular matrix")
    win.dialog_script = {"mu": 51.0, "accept": True}
    win._peak2_open_edit(rec, 50.0)
    assert "[failed]" in status(win)
    assert rec["result"] is old_result


def test_a_successful_edit_updates_the_row_and_reports(win):
    rec = add_fit(win, number=2)
    win.dialog_script = {"mu": 51.0, "accept": True}
    win._peak2_open_edit(rec, 50.0)
    assert "Peak 2 (edited)" in status(win)
    assert table(win).rowCount() == 1
