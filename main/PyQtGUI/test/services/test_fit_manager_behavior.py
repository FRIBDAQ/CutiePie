"""Characterization tests for FitManager.

These pin FitManager's CURRENT observable behavior — popup-widget effects,
dialog calls, fit-input preparation, artist labeling/tagging — before the
step-1 inversion (widget writes -> signals, widget reads -> method arguments).

Headless strategy: qt_stubs.install_missing_runtime_stubs() provides the
PyQt5 import surface; every Qt object FitManager *uses* at runtime
(QMessageBox, QSettings, QTextEdit) is monkeypatched at the module attribute
with a recording double, so the same tests run under stub AND real Qt (no
QApplication needed). Matplotlib runs on the Agg backend; axes are real.
"""

import importlib
import logging
import os
import sys
import threading

import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import qt_stubs


_AFFECTED_MODULES = (
    "PyQt5", "PyQt5.QtCore", "PyQt5.QtWidgets",
    "CPyConverter", "httplib2",
    "alpha_filter_dialog", "services.fit_manager",
)


@pytest.fixture(scope="module")
def fm_mod():
    saved = {name: sys.modules.get(name) for name in _AFFECTED_MODULES}
    installed = qt_stubs.install_missing_runtime_stubs()
    if installed:
        for name in ("services.fit_manager", "alpha_filter_dialog"):
            sys.modules.pop(name, None)
    module = importlib.import_module("services.fit_manager")
    yield module
    for name, prev in saved.items():
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


class FakeFit:
    """Records fit.start() calls; plots one line on the ax by default."""

    def __init__(self, return_none=False, side_effect=None):
        self.calls = []
        self.return_none = return_none
        self.side_effect = side_effect

    def start(self, x, y, xmin, xmax, fitpar, ax, resultsText):
        self.calls.append(dict(x=np.asarray(x), y=np.asarray(y),
                               xmin=xmin, xmax=xmax, fitpar=list(fitpar),
                               ax=ax, resultsText=resultsText))
        if self.side_effect is not None:
            self.side_effect()
        if self.return_none:
            return None
        line, = ax.plot(x, y)
        return line


class FakeFitFactory:
    def __init__(self, configs=None, fit=None):
        self._configs = configs or {}
        self.fit = fit or FakeFit()
        self.created = []

    def create(self, name, **config):
        self.created.append((name, dict(config)))
        return self.fit


class Env:
    def __init__(self, module, monkeypatch, configs=None, fit=None):
        from services.spectrum_store import SpectrumStore
        self.mod = module
        self.store = SpectrumStore()
        self.factory = FakeFitFactory(configs=configs, fit=fit)
        self.msgbox = qt_stubs.fresh_message_box()
        self.text_edits = []

        env = self

        class TrackingTextEdit(qt_stubs.FakeTextEdit):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                env.text_edits.append(self)

        qt_stubs.FakeQSettings.store = {}
        monkeypatch.setattr(module, "QMessageBox", self.msgbox)
        monkeypatch.setattr(module, "QSettings", qt_stubs.FakeQSettings)
        monkeypatch.setattr(module, "QTextEdit", TrackingTextEdit)
        self.fm = module.FitManager(
            fit_factory=self.factory, spectra=self.store,
            parent_widget=None,
            logger=logging.getLogger("test.fit_manager"),
        )
        # output signals, recorded from construction on
        self.busy = qt_stubs.record_signal(self.fm.fitBusyChanged)
        self.abort_en = qt_stubs.record_signal(self.fm.abortEnabledChanged)
        self.results = qt_stubs.record_signal(self.fm.fitResultsAppended)
        self.labels = qt_stubs.record_signal(self.fm.fitLabelsTextChanged)


@pytest.fixture
def env(fm_mod, monkeypatch):
    return Env(fm_mod, monkeypatch)


def make_ax():
    fig = Figure()
    FigureCanvasAgg(fig)
    return fig.add_subplot(111)


def add_1d_spectrum(store, name="h1", binx=10, minx=0.0, maxx=10.0):
    store.set(name, dim=1, binx=binx, minx=minx, maxx=maxx,
              biny=None, miny=None, maxy=None,
              data=np.arange(binx + 1, dtype=float), parameters=[], type="1")


# ---------------------------------------------------------------- pure logic

def test_create_range(fm_mod):
    r = fm_mod.FitManager._create_range(10, 0, 10)
    assert len(r) == 11 and r[0] == 0.0 and r[-1] == 10.0


def test_load_calibration_json_ab(env, tmp_path):
    p = tmp_path / "cal.json"
    p.write_text('{"a": 2.5, "b": -3.0}')
    assert env.fm.load_calibration_any(str(p)) == (2.5, -3.0)


def test_load_calibration_json_calib_keys(env, tmp_path):
    p = tmp_path / "cal.json"
    p.write_text('{"calib_a": 1.5, "calib_b": 0.25}')
    assert env.fm.load_calibration_any(str(p)) == (1.5, 0.25)


def test_load_calibration_key_value_text(env, tmp_path):
    p = tmp_path / "cal.txt"
    p.write_text("a = 4.0\nb = -1e-2\n")
    assert env.fm.load_calibration_any(str(p)) == (4.0, -0.01)


def test_load_calibration_first_two_numbers_fallback(env, tmp_path):
    p = tmp_path / "cal.txt"
    p.write_text("slope 3.5 offset 7\n")
    assert env.fm.load_calibration_any(str(p)) == (3.5, 7.0)


def test_load_calibration_unparseable_raises(env, tmp_path):
    p = tmp_path / "cal.txt"
    p.write_text("nothing here")
    with pytest.raises(ValueError):
        env.fm.load_calibration_any(str(p))


def test_prepare_fit_config_non_emg_is_factory_copy(env):
    env.factory._configs["Gaus"] = {"amp": 1}
    config = env.fm.prepare_fit_config("Gaus")
    assert config == {"amp": 1}
    config["amp"] = 99
    assert env.factory._configs["Gaus"] == {"amp": 1}   # copy, not alias


# ------------------------------------------------------------- axis limits

def test_axis_limits_from_arguments(env):
    ax = make_ax()
    ax.set_xlim(0, 100)
    assert env.fm.axisLimitsForFit(ax, "10", "20") == (10.0, 20.0)
    assert env.msgbox.calls == []


def test_axis_limits_fall_back_to_xlim(env):
    ax = make_ax()
    ax.set_xlim(5, 50)
    # empty min -> xlim; invalid max -> xlim + warning log
    left, right = env.fm.axisLimitsForFit(ax, "", "junk")
    assert (left, right) == (5.0, 50.0)


def test_axis_limits_swap_when_reversed(env):
    ax = make_ax()
    ax.set_xlim(0, 100)
    assert env.fm.axisLimitsForFit(ax, "30", "10") == (10.0, 30.0)
    assert env.msgbox.calls[0][0] == "about"          # swap warning dialog


# ---------------------------------------------------------- abort + labels

def test_on_abort_clicked(env):
    env.fm.on_abort_clicked()
    assert env.fm._abort_fit is True
    assert env.abort_en == [(False,)]
    assert ("[abort] Requested…",) in env.results


def test_set_fit_line_label_fills_first_gap(env):
    ax = make_ax()
    for idx in (0, 2):
        line, = ax.plot([0, 1], [0, 1])
        line.set_label(f"fit-_-{idx}")
    new_line, = ax.plot([0, 1], [1, 0])
    results = qt_stubs.FakeTextEdit()
    results.append("chi2=1")
    env.fm.setFitLineLabel(ax, new_line, results, "h1")
    assert new_line.get_label() == "fit-_-1"          # fills the 0,2 gap
    assert env.results == [("Fit 1 (h1) :",), ("chi2=1",), (" ",)]


def test_set_fit_line_label_none_line_is_noop(env):
    env.fm.setFitLineLabel(make_ax(), None, qt_stubs.FakeTextEdit(), "h1")
    assert env.results == []


def test_list_and_print_fit_line_labels(env):
    ax = make_ax()
    for idx in (0, 1):
        line, = ax.plot([0, 1], [0, 1])
        line.set_label(f"fit-_-{idx}")
    ax.plot([0, 1], [1, 1])                           # unlabeled: ignored
    assert env.fm.listFitLineLabels(ax) == ["0", "1"]
    env.fm.printFitLineLabels(0, "h1", ax)
    assert env.labels[-1] == ("0 1 ",)                    # trailing space


def test_delete_fit_removes_only_known_indices(env):
    ax = make_ax()
    lines = {}
    for idx in (0, 1):
        line, = ax.plot([0, 1], [0, 1])
        line.set_label(f"fit-_-{idx}")
        lines[idx] = line
    env.fm.deleteFit(0, "h1", ax, "0")
    assert env.fm.listFitLineLabels(ax) == ["1"]
    # a request containing any unknown index deletes NOTHING
    env.fm.deleteFit(0, "h1", ax, "1 7")
    assert env.fm.listFitLineLabels(ax) == ["1"]


def test_tag_and_clear_fit_artists(env):
    ax = make_ax()
    keep, = ax.plot([0, 1], [0, 0])                   # pre-existing artist
    before = env.fm._snap_ax_ids(ax)
    new, = ax.plot([0, 1], [1, 1])
    env.fm._tag_new_fit_artists(ax, before)
    assert new.get_gid() == "fit" and keep.get_gid() is None

    labeled, = ax.plot([0, 1], [2, 2])
    labeled.set_label("fit-_-3")
    env.fm._clear_all_fit_artists_in_figure(ax.figure)
    remaining = list(ax.lines)
    assert keep in remaining
    assert new not in remaining and labeled not in remaining
    assert env.labels[-1] == ("3",)                   # labels snapshot pre-clear


# ------------------------------------------------------------------- fit()

def fit_env(fm_mod, monkeypatch, **fit_kwargs):
    env = Env(fm_mod, monkeypatch,
              configs={"Gaus": {"amp": 1}}, fit=FakeFit(**fit_kwargs))
    add_1d_spectrum(env.store)
    return env


def test_fit_1d_happy_path(fm_mod, monkeypatch):
    env = fit_env(fm_mod, monkeypatch)
    ax = make_ax()
    busy_during = []
    env.factory.fit.side_effect = lambda: busy_during.append(env.busy[-1])

    env.fm.fit(0, "h1", ax, "Gaus", ["1.5"] + [""] * 19, "2", "8")

    assert env.factory.created == [("Gaus", {"amp": 1})]
    call = env.factory.fit.calls[0]
    # bin centers strictly inside (xmin, xmax]; y offset by one bin (ytmp[i])
    assert np.allclose(call["x"], [2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    assert np.allclose(call["y"], [3, 4, 5, 6, 7, 8])
    assert (call["xmin"], call["xmax"]) == (2.0, 8.0)
    assert call["fitpar"] == [1.5] + [None] * 19
    # busy during the fit, restored after
    assert busy_during == [(True,)]
    assert env.busy == [(True,), (False,)]
    # fit line labeled and tagged; results routed to popup and result window
    line = [l for l in ax.lines if l.get_label().startswith("fit-_-")][0]
    assert line.get_label() == "fit-_-0" and line.get_gid() == "fit-0"
    assert env.results[0] == ("Fit 0 (h1) :",)
    result_win = env.text_edits[0]
    assert result_win.read_only is True and result_win.shown == 1
    assert result_win.window_title == "Fit results — Gaus : h1"


def test_fit_emg12_injects_bin_width_and_wmode(fm_mod, monkeypatch):
    env = fit_env(fm_mod, monkeypatch)
    ax = make_ax()
    env.fm.fit(0, "h1", ax, "AlphaEMG12", [""] * 20, "0", "10")
    fitpar = env.factory.fit.calls[0]["fitpar"]
    assert fitpar[4] == 1.0                            # bw = (maxx-minx)/binx
    assert fitpar[5] == 2                              # AlphaEMG12 wmode


def test_fit_without_context_never_goes_busy(fm_mod, monkeypatch):
    # (fixed 2026-07-04): the missing-context early return used to fire
    # AFTER busy(True) and outside the try/finally, leaving the fit button
    # disabled. The context check now precedes the busy toggle entirely.
    env = fit_env(fm_mod, monkeypatch)
    env.fm.fit(None, None, None, "Gaus")
    assert env.factory.created == []
    assert env.busy == []                             # no busy state entered


def test_fit_2d_shows_warning_and_restores_buttons(fm_mod, monkeypatch):
    env = fit_env(fm_mod, monkeypatch)
    env.store.set("h2", dim=2, binx=4, minx=0.0, maxx=4.0, data=np.zeros((4, 4)))
    env.fm.fit(0, "h2", make_ax(), "Gaus")
    assert any(kind == "about" and "2D fitting" in text
               for kind, _, text in env.msgbox.calls)
    assert env.busy == [(True,), (False,)]


def test_fit_cancelled_config_warns_before_busy(fm_mod, monkeypatch):
    env = fit_env(fm_mod, monkeypatch)

    def boom(fit_funct, force_prompt=False):
        raise ValueError("no shape file")

    monkeypatch.setattr(env.fm, "prepare_fit_config", boom)
    env.fm.fit(0, "h1", make_ax(), "Gaus")
    assert env.msgbox.calls == [("warning", "Fit cancelled", "no shape file")]
    assert env.busy == []                             # never went busy
    assert env.factory.created == []


def test_fit_abort_appends_message_and_skips_labeling(fm_mod, monkeypatch):
    env = fit_env(fm_mod, monkeypatch, return_none=True)
    env.factory.fit.side_effect = env.fm.on_abort_clicked
    ax = make_ax()
    env.fm.fit(0, "h1", ax, "Gaus", [""] * 20, "0", "10")
    result_win = env.factory.fit.calls[0]["resultsText"]
    assert ("append", "[abort] Fit stopped by user.") in result_win.ops
    assert env.fm.listFitLineLabels(ax) == []         # no label assigned
    assert env.busy[-1] == (False,)                   # finally restored


def test_fit_csv_guard_requires_loaded_csv(env):
    env.fm.on_fit_csv_clicked()
    assert env.msgbox.calls[0][:2] == ("warning", "No CSV loaded")
    assert env.factory.created == []


def test_alpha_filter_popup_closed_for_other_models(env):
    closes = []
    env.fm._alphaFilterDlg = type("D", (), {"close": lambda self: closes.append(1)})()
    env.fm._maybe_show_alpha_filter_popup(object(), "Gaus")
    assert closes == [1]
    assert env.fm._alphaFilterDlg is None


def test_delete_fit_index_is_exact_not_a_prefix(fm_mod):
    # "fit-_-1" is a substring of "fit-_-10"; deleting fit 1 must not
    # delete fit 10.
    fm = fm_mod.FitManager(fit_factory=FakeFitFactory(), spectra=None,
                           parent_widget=None,
                           logger=logging.getLogger("t.fm.h7"))
    fig = Figure(); FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    l1, = ax.plot([0, 1], [0, 1]); l1.set_label("fit-_-1")
    l10, = ax.plot([0, 1], [1, 0]); l10.set_label("fit-_-10")
    fm.deleteFit(index=0, name="h", ax=ax, fit_idx_text="1")
    labels = [ln.get_label() for ln in ax.lines]
    assert "fit-_-1" not in labels
    assert "fit-_-10" in labels          # pre-fix: removed as collateral


# ---------------------------------------------------- save / load fit curve

def test_save_fit_curve_no_curve_warns(env, tmp_path):
    env.fm._lastFitCurve = None
    out = env.fm.save_fit_curve(path=str(tmp_path / "x.csv"))
    assert out is None
    assert any(c[0] == "warning" for c in env.msgbox.calls)


def test_save_fit_curve_writes_header_and_columns(env, tmp_path):
    x = np.linspace(0.0, 5.0, 6)
    y = x ** 2
    env.fm._lastFitCurve = {"x": x, "y": y, "model": "AlphaEMGMultiSigma", "name": "h1"}
    p = tmp_path / "curve.csv"
    out = env.fm.save_fit_curve(path=str(p))
    assert out == str(p)
    text = p.read_text()
    assert "# CutiePie fit curve" in text
    assert "model = AlphaEMGMultiSigma" in text
    assert "npoints = 6" in text
    # data round-trips via the same loader the GUI uses
    arr = np.genfromtxt(str(p), delimiter=",", comments="#")
    assert np.allclose(arr[:, 0], x)
    assert np.allclose(arr[:, 1], y)


def test_save_then_load_round_trips_onto_axis(env, tmp_path):
    x = np.linspace(0.0, 10.0, 50)
    y = np.sin(x)
    env.fm._lastFitCurve = {"x": x, "y": y, "model": "AlphaEMGMultiSigma", "name": "h1"}
    p = tmp_path / "curve.csv"
    env.fm.save_fit_curve(path=str(p))

    ax = make_ax()
    line = env.fm.load_fit_curve(ax=ax, path=str(p))
    assert line is not None
    assert np.allclose(line.get_xdata(), x)
    assert np.allclose(line.get_ydata(), y)


def test_loaded_curve_is_a_deletable_fit_artist(env, tmp_path):
    x = np.arange(5, dtype=float)
    env.fm._lastFitCurve = {"x": x, "y": x, "model": "AlphaEMGMultiSigma", "name": "h1"}
    p = tmp_path / "curve.csv"
    env.fm.save_fit_curve(path=str(p))

    ax = make_ax()
    line = env.fm.load_fit_curve(ax=ax, path=str(p))
    assert line.get_label() == "fit-_-0"
    assert line.get_gid() == "fit-0"
    # visible to the fit-line machinery
    assert env.fm.listFitLineLabels(ax) == ["0"]
    # and removable through the normal delete path
    env.fm.deleteFit(index=0, name="h1", ax=ax, fit_idx_text="0")
    assert env.fm.listFitLineLabels(ax) == []


def test_load_takes_next_free_fit_index(env, tmp_path):
    x = np.arange(4, dtype=float)
    env.fm._lastFitCurve = {"x": x, "y": x, "model": "AlphaEMGMultiSigma", "name": "h1"}
    p = tmp_path / "curve.csv"
    env.fm.save_fit_curve(path=str(p))

    ax = make_ax()
    existing, = ax.plot([0, 1], [0, 1]); existing.set_label("fit-_-0")
    line = env.fm.load_fit_curve(ax=ax, path=str(p))
    assert line.get_label() == "fit-_-1"


def test_load_without_axis_warns(env):
    out = env.fm.load_fit_curve(ax=None, path="whatever.csv")
    assert out is None
    assert any(c[0] == "warning" for c in env.msgbox.calls)


# ------------------------------------- delete removes ALL of a fit's artists

class MultiArtistFit(FakeFit):
    """Mimics AlphaEMGMultiSigma: draws a total line + a dashed subpeak line
    (isotope-labelled, NOT fit-_-N) + a text annotation, returns the total."""

    def start(self, x, y, xmin, xmax, fitpar, ax, resultsText):
        self.calls.append(dict(x=np.asarray(x), y=np.asarray(y),
                               xmin=xmin, xmax=xmax, fitpar=list(fitpar),
                               ax=ax, resultsText=resultsText))
        (total,) = ax.plot(x, y, color="tab:orange", label="fit total")
        (sub,) = ax.plot(x, y * 0.5, ls="--", label="Bi211")   # isotope-labelled
        ax.text(float(x[0]), float(y[0]), "Bi211 6623")
        ax.legend()
        total.component_data = {"x": np.asarray(x), "ytot": np.asarray(y)}
        return total


def test_delete_removes_all_artists_of_a_multi_component_fit(fm_mod, monkeypatch):
    # Regression: deleteFit used to remove only the fit-_-N total line, leaving
    # subpeak curves, text labels and the legend orphaned on the pad. Uses the
    # model-agnostic Gaus path so the artist/tag/delete contract is exercised
    # without the EMG-only calibration/shape prompts.
    env = Env(fm_mod, monkeypatch,
              configs={"Gaus": {"amp": 1}}, fit=MultiArtistFit())
    add_1d_spectrum(env.store)
    ax = make_ax()

    env.fm.fit(0, "h1", ax, "Gaus", [""] * 20, "2", "8")

    # the fit drew a total + a subpeak + a text (+ legend)
    assert env.fm.listFitLineLabels(ax) == ["0"]
    assert len(ax.lines) == 2 and len(ax.texts) == 1

    env.fm.deleteFit(0, "h1", ax, "0")

    # everything the fit added must be gone, not just the total line
    assert env.fm.listFitLineLabels(ax) == []
    assert len(ax.lines) == 0, "subpeak curve left behind"
    assert len(ax.texts) == 0, "text annotation left behind"
    assert ax.get_legend() is None, "stale legend left behind"


# ------------------------------------------- multi-component save / load

def _components_curve():
    x = np.linspace(0.0, 10.0, 40)
    total = np.exp(-((x - 5) ** 2))
    bi = 0.6 * total
    po = 0.4 * total
    return {
        "x": x, "y": total, "model": "AlphaEMGMultiSigma", "name": "h1",
        "components": [("fit total", total), ("Bi211", bi), ("Po215", po)],
        "chains": {"Bi211": "A227", "Po215": "A227"},
        "colors": {"fit total": "tab:orange", "Bi211": "C0", "Po215": "C1"},
    }


def test_save_writes_multi_component_file(env, tmp_path):
    env.fm._lastFitCurve = _components_curve()
    p = tmp_path / "multi.csv"
    env.fm.save_fit_curve(path=str(p))
    text = p.read_text()
    assert "multi-component" in text
    assert "meta = " in text
    assert "Bi211" in text and "Po215" in text
    # numeric block still readable by the plain CSV loader
    arr = np.genfromtxt(str(p), delimiter=",", comments="#")
    assert arr.shape[1] == 4                       # x + total + 2 isotopes


def test_read_round_trips_components_chains_and_colors(env, tmp_path):
    env.fm._lastFitCurve = _components_curve()
    p = tmp_path / "multi.csv"
    env.fm.save_fit_curve(path=str(p))

    struct = env.fm._read_fit_curve_file(str(p))
    assert struct["multi"] is True
    names = [n for n, _ in struct["components"]]
    assert names == ["fit total", "Bi211", "Po215"]
    assert struct["chains"] == {"Bi211": "A227", "Po215": "A227"}
    assert struct["colors"]["Bi211"] == "C0"
    total = dict(struct["components"])["fit total"]
    assert np.allclose(total, _components_curve()["y"])


def test_old_total_only_file_reads_as_non_multi(env, tmp_path):
    # a fit with no per-isotope components saves the 2-column total-only format
    env.fm._lastFitCurve = {"x": np.arange(5.0), "y": np.arange(5.0),
                            "model": "Gaus", "name": "h1"}
    p = tmp_path / "total.csv"
    env.fm.save_fit_curve(path=str(p))
    assert "multi-component" not in p.read_text()
    struct = env.fm._read_fit_curve_file(str(p))
    assert struct["multi"] is False
    assert [n for n, _ in struct["components"]] == ["fit total"]


def test_plot_components_tags_one_deletable_group(env):
    ax = make_ax()
    curve = _components_curve()
    i, lines = env.fm._plot_fit_components(ax, curve["x"], curve["components"],
                                           curve["colors"])
    assert i == 0
    assert len(lines) == 3
    # every line shares the one per-index gid
    assert all(l.get_gid() == "fit-0" for l in lines)
    # exactly one carrier holds the fit-_-0 label (the total)
    carriers = [l for l in lines if l.get_label() == "fit-_-0"]
    assert len(carriers) == 1
    assert env.fm.listFitLineLabels(ax) == ["0"]
    # deletes as a single unit
    env.fm.deleteFit(0, "h1", ax, "0")
    assert len(ax.lines) == 0


def test_load_multi_component_uses_picker_subset(env, tmp_path, monkeypatch):
    env.fm._lastFitCurve = _components_curve()
    p = tmp_path / "multi.csv"
    env.fm.save_fit_curve(path=str(p))

    # user checks only the total and Bi211
    monkeypatch.setattr(env.fm, "_prompt_component_selection",
                        lambda names, chains: ["fit total", "Bi211"])
    ax = make_ax()
    env.fm.load_fit_curve(ax=ax, path=str(p))
    assert len(ax.lines) == 2                       # Po215 was unchecked
    assert env.fm.listFitLineLabels(ax) == ["0"]    # one fit group


def test_load_multi_component_cancel_draws_nothing(env, tmp_path, monkeypatch):
    env.fm._lastFitCurve = _components_curve()
    p = tmp_path / "multi.csv"
    env.fm.save_fit_curve(path=str(p))
    monkeypatch.setattr(env.fm, "_prompt_component_selection",
                        lambda names, chains: None)   # cancelled
    ax = make_ax()
    out = env.fm.load_fit_curve(ax=ax, path=str(p))
    assert out is None
    assert len(ax.lines) == 0
