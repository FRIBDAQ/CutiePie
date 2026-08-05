"""Characterization tests for MainWindow's Copy Properties cluster.
`copyPopup`, `applyCopy`, `selectAll`, `histAllAttr` and `closeCopy` — 240
lines that have produced six defects, more than any other cluster in the
class."""

import logging
import os
import sys

import matplotlib
matplotlib.use("Agg", force=True)
from matplotlib.figure import Figure

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import gui_stubs
from services.spectrum_store import SpectrumStore


# ------------------------------------------------------------------ widgets

class FakeCheck:
    def __init__(self, checked=False, text=""):
        self._checked = checked
        self._text = text
        self.enabled = True

    def isChecked(self):
        return self._checked

    def setChecked(self, value):
        self._checked = value

    def text(self):
        return self._text

    def setEnabled(self, value):
        self.enabled = value

    def isEnabled(self):
        return self.enabled


class FakeLabel:
    def __init__(self, text=""):
        self._text = text
        self.enabled = True

    def text(self):
        return self._text

    def setText(self, value):
        self._text = value

    def setEnabled(self, value):
        self.enabled = value


class FakePalette:
    def __init__(self, colour="#ff0000"):
        self.colour = colour

    def color(self, role):
        return self

    def name(self):
        return self.colour


class StubSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)


class FakeButton:
    """A target button in the popup's form, or one of the action buttons."""

    def __init__(self, text, checked=False, colour="#ff0000"):
        self.clicked = StubSignal()
        self._text = text
        self._checked = checked
        self._props = {}
        self.style = ""
        self._palette = FakePalette(colour)
        self.deleted = False

    def text(self):
        return self._text

    def setText(self, value):
        self._text = value

    def isChecked(self):
        return self._checked

    def setChecked(self, value):
        self._checked = value

    def setCheckable(self, value):
        pass

    def setStyleSheet(self, value):
        self.style = value

    def setProperty(self, key, value):
        self._props[key] = value

    def property(self, key):
        return self._props.get(key)

    def palette(self):
        return self._palette

    def deleteLater(self):
        self.deleted = True


class FakeForm:
    def __init__(self):
        self.rows = []

    def rowCount(self):
        return len(self.rows)

    def removeRow(self, i):
        self.rows.pop(i)

    def addRow(self, label, widget):
        self.rows.append((label, widget))


class FakeCopyAttr:
    """CopyPropertiesGUI: the five property checkboxes, the source labels,
    the target-button form, and the show/close lifecycle."""

    def __init__(self):
        self.axisLimitX = FakeCheck()
        self.axisLimitY = FakeCheck()
        self.axisScale = FakeCheck()
        self.histoScaleminZ = FakeCheck()
        self.histoScalemaxZ = FakeCheck()
        self.histoLabel = FakeLabel()
        self.axisSLabel = FakeLabel()
        self.axisLimLabelX = FakeLabel()
        self.axisLimLabelY = FakeLabel()
        self.histoScaleValueminZ = FakeLabel()
        self.histoScaleValuemaxZ = FakeLabel()
        self.copy_log = FakeForm()
        self.buttons = []
        self.visible = False

    # the popup builds target buttons with `self` (MainWindow) as parent and
    # then hands them to the form; findChildren is what applyCopy walks
    def findChildren(self, _cls):
        return list(self.buttons)

    def isVisible(self):
        return self.visible

    def show(self):
        self.visible = True

    def close(self):
        self.visible = False


class FakeTabs:
    def __init__(self):
        self.slots = {0: {}}
        self.zoom = {0: None}
        self.layout = [2, 3]

    def currentIndex(self):
        return 0

    def tabSlots(self, index):
        return self.slots[index]

    def zoomInfo(self, index):
        return self.zoom[index]

    def setZoomInfo(self, index, value):
        self.zoom[index] = value

    def tabLayout(self, index):
        return self.layout


class FakePlot:
    def __init__(self):
        self.h_dict_geo = {}
        self.selected_plot_index = 0
        self.histo_autoscale = FakeCheck(checked=True)


class FakePlotController:
    """Records what applyCopy asks of the renderer."""

    def __init__(self):
        self.scale_calls = []
        self.update_calls = 0

    def setAxisScale(self, ax, index, *scale):
        self.scale_calls.append((index, scale))

    def updatePlot(self):
        self.update_calls += 1

    def plotPosition(self, index, layout):
        ncol = layout[1]
        return index // ncol, index % ncol


@pytest.fixture
def win(monkeypatch):
    """A bare MainWindow wired to a real CopyPropertiesController. The only
    thing that changed when the cluster moved out: this fixture now builds the
    controller that __init__ builds in production, since __init__ is
    deliberately not run."""
    gui = gui_stubs.import_gui()
    w = gui.MainWindow.__new__(gui.MainWindow)
    w.logger = logging.getLogger("test.copyproperties")
    w.spectra = SpectrumStore()
    w.wTab = FakeTabs()
    w.currentPlot = FakePlot()

    # These accessors live on ViewState; bind them back onto the window so
    # the unchanged test bodies keep driving MainWindow.
    from view_state import ViewState
    import types as _types
    w.view_state = ViewState(
        spectra=w.spectra,
        tabs=w.wTab,
        get_current_plot=lambda: w.currentPlot,
        applylistgate=lambda name: None,
        gate_name_fetched=_types.SimpleNamespace(emit=lambda *a: None),
        logger=w.logger,
    )
    for _name in ("getSpectrumStoreInfo", "setSpectrumViewInfo", "getSpectrumViewInfo",
                  "getSpectrumViewDict", "getSpectrumStoreDict", "nameFromIndex",
                  "setGeo", "getGeo", "setEnlargedSpectrum", "getEnlargedSpectrum",
                  "getAppliedGateName"):
        if _name not in w.__dict__:
            setattr(w, _name, getattr(w.view_state, _name))
    w.copyAttr = FakeCopyAttr()
    w.plot_controller = FakePlotController()
    w.figure = Figure()

    from controllers import copy_properties_controller as cpc
    # the popup constructs QPushButton(name, parent); the fake stands in so the
    # buttons it builds are the ones findChildren hands back
    monkeypatch.setattr(cpc, "QPushButton",
                        lambda text, parent=None: FakeButton(text))
    w.copy_props = cpc.CopyPropertiesController(
        copy_attr=w.copyAttr,
        get_selected_index=lambda: w.currentPlot.selected_plot_index,
        set_autoscale=lambda on: w.currentPlot.histo_autoscale.setChecked(on),
        get_store_info=w.getSpectrumStoreInfo,
        get_view_info=w.getSpectrumViewInfo,
        set_view_info=w.setSpectrumViewInfo,
        get_geo=w.getGeo,
        name_from_index=w.nameFromIndex,
        plot_position=w.plotPosition,
        plot_controller=w.plot_controller,
        parent_widget=w,
        logger=w.logger,
    )
    return w


def add_pad(win, name, index, dim=1, with_axes=True):
    """Populate both tiers and give the pad an axes, the way a draw does."""
    win.spectra.set(name, dim=dim, binx=512, minx=0.0, maxx=1024.0,
                    biny=512, miny=0.0, maxy=1024.0, data=[],
                    parameters=["p1"], type=str(dim))
    win.setGeo(index, name)
    if with_axes:
        ax = win.figure.add_subplot(6, 1, index + 1)
        win.setSpectrumViewInfo(axis=ax, index=index)
        return ax
    return None


def action_buttons():
    # CopyPropertiesGUI builds exactly four: Ok, Apply, Cancel, and one
    # Select all whose TEXT toggles to "Deselect all". There is no fifth
    # button, which is why selectAll flips a label rather than reading state.
    return [FakeButton(t) for t in ("Ok", "Cancel", "Apply", "Select all")]


def target_button(index, checked=True, text="hTarget"):
    b = FakeButton(text, checked=checked)
    b.setProperty("padIndex", index)
    return b


# ================================================================= copyPopup

def test_popup_does_not_open_when_the_pad_has_no_spectrum(win):
    win.currentPlot.selected_plot_index = 4          # nothing there
    win.copyPopup()
    assert win.copyAttr.visible is False


def test_popup_lists_only_same_dim_targets_and_never_the_source(win):
    add_pad(win, "src", 0, dim=1)
    add_pad(win, "same", 1, dim=1)
    add_pad(win, "other", 2, dim=2)
    win.copyPopup()
    names = [w.text() for _, w in win.copyAttr.copy_log.rows]
    assert names == ["same"]


def test_each_target_button_carries_its_pad_index(win):
    # PIN: the index rides on the button. applyCopy used to
    # recover it by scraping the label and multiplying by the column count read
    # at Apply time, so a geometry change re-pointed every target.
    add_pad(win, "src", 0)
    add_pad(win, "t4", 4)
    win.copyPopup()
    _, button = win.copyAttr.copy_log.rows[0]
    assert button.property("padIndex") == 4


def test_target_rows_are_labelled_by_grid_position(win):
    add_pad(win, "src", 0)
    add_pad(win, "t4", 4)
    win.copyPopup()
    label, _ = win.copyAttr.copy_log.rows[0]
    assert label == "row: 1 col: 1"          # index 4 on a 2x3 grid


def test_popup_rebuilds_its_form_instead_of_stacking_rows(win):
    add_pad(win, "src", 0)
    add_pad(win, "t1", 1)
    win.copyPopup()
    win.copyPopup()
    assert len(win.copyAttr.copy_log.rows) == 1


def test_popup_reports_the_stored_view_range(win):
    add_pad(win, "src", 0)
    win.setSpectrumViewInfo(minx=10.0, maxx=90.0, miny=1.0, maxy=42.0, index=0)
    win.copyPopup()
    assert win.copyAttr.axisLimLabelX.text() == "[10.0,90.0]"
    assert win.copyAttr.axisLimLabelY.text() == "[1.0,42.0]"


def test_popup_falls_back_to_the_axes_when_the_y_range_was_never_stored(win):
    # PIN: a 1-D spectrum added on a binding trace carries
    # miny/maxy None, and setAxisScale only writes them while autoscale is on.
    # Formatting None with :.1f used to raise and the popup never appeared.
    ax = add_pad(win, "src", 0)
    ax.set_ylim(0.0, 87.0)
    win.setSpectrumViewInfo(minx=0.0, maxx=1024.0, miny=None, maxy=None, index=0)
    win.copyPopup()
    assert win.copyAttr.visible is True
    assert win.copyAttr.axisLimLabelY.text() == "[0.0,87.0]"


def test_popup_refuses_to_open_when_no_range_can_be_had_at_all(win):
    # the same guard's other half: no axes either, so nothing can be reported
    add_pad(win, "src", 0, with_axes=False)
    win.setSpectrumViewInfo(minx=None, maxx=None, miny=None, maxy=None, index=0)
    win.copyPopup()
    assert win.copyAttr.visible is False


def test_popup_fills_the_z_boxes_from_a_2d_source(win):
    ax = add_pad(win, "src", 0, dim=2)
    image = ax.imshow([[0, 1], [2, 3]])
    image.set_clim(3.0, 77.0)
    win.setSpectrumViewInfo(spectrum=image, minx=0.0, maxx=1.0,
                            miny=0.0, maxy=1.0, index=0)
    win.copyPopup()
    assert win.copyAttr.histoScaleValueminZ.text() == "3.0"
    assert win.copyAttr.histoScaleValuemaxZ.text() == "77.0"


# ================================================================= applyCopy

def prime_apply(win, source_range=("[0.0,100.0]", "[1.0,50.0]"),
                scale="Linear", z=("0.0", "10.0")):
    """Fill the popup labels the way copyPopup would have."""
    win.copyAttr.axisLimLabelX.setText(source_range[0])
    win.copyAttr.axisLimLabelY.setText(source_range[1])
    win.copyAttr.axisSLabel.setText(scale)
    win.copyAttr.histoScaleValueminZ.setText(z[0])
    win.copyAttr.histoScaleValuemaxZ.setText(z[1])


def test_apply_copies_the_checked_properties_to_both_tiers(win):
    # PIN: updatePlot's limits path is autoscale-gated, so a
    # view-tier write alone never reaches the screen. Both must happen.
    add_pad(win, "src", 0)
    target_ax = add_pad(win, "dst", 1)
    prime_apply(win)
    win.copyAttr.axisLimitX._checked = True
    win.copyAttr.axisLimitY._checked = True
    win.copyAttr.buttons = action_buttons() + [target_button(1)]
    win.applyCopy()
    assert win.getSpectrumViewInfo("minx", index=1) == 0.0
    assert win.getSpectrumViewInfo("maxx", index=1) == 100.0
    assert target_ax.get_xlim() == (0.0, 100.0)
    assert target_ax.get_ylim() == (1.0, 50.0)


def test_apply_reads_the_property_checkboxes_by_name(win):
    # PIN: binding these to findChildren() construction order
    # meant a reordered or added checkbox silently re-pointed the flags.
    add_pad(win, "src", 0)
    target_ax = add_pad(win, "dst", 1)
    prime_apply(win)
    win.copyAttr.axisLimitY._checked = True          # only Y
    win.copyAttr.buttons = action_buttons() + [target_button(1)]
    before_x = target_ax.get_xlim()
    win.applyCopy()
    assert target_ax.get_xlim() == before_x
    assert target_ax.get_ylim() == (1.0, 50.0)


def test_apply_targets_only_checked_buttons(win):
    add_pad(win, "src", 0)
    add_pad(win, "a", 1)
    add_pad(win, "b", 2)
    prime_apply(win)
    win.copyAttr.axisLimitX._checked = True
    win.copyAttr.buttons = action_buttons() + [target_button(1),
                                               target_button(2, checked=False)]
    win.setSpectrumViewInfo(minx=555.0, index=2)                  # sentinel
    win.applyCopy()
    assert win.getSpectrumViewInfo("minx", index=1) == 0.0
    assert win.getSpectrumViewInfo("minx", index=2) == 555.0      # untouched


def test_apply_skips_a_button_with_no_pad_index(win):
    # C17(d): the action buttons carry none, and neither does anything else
    # that slipped into the popup. Skipped with a warning, never guessed at.
    add_pad(win, "src", 0)
    add_pad(win, "dst", 1)
    prime_apply(win)
    win.copyAttr.axisLimitX._checked = True
    stray = FakeButton("stray", checked=True)          # no padIndex property
    win.copyAttr.buttons = action_buttons() + [stray]
    win.setSpectrumViewInfo(minx=555.0, index=1)                  # sentinel
    win.applyCopy()
    assert win.getSpectrumViewInfo("minx", index=1) == 555.0      # untouched


def test_apply_turns_autoscale_off_first(win):
    # otherwise the trailing updatePlot recomputes y from the data and discards
    # everything just copied
    add_pad(win, "src", 0)
    add_pad(win, "dst", 1)
    prime_apply(win)
    win.copyAttr.axisLimitY._checked = True
    win.copyAttr.buttons = action_buttons() + [target_button(1)]
    assert win.currentPlot.histo_autoscale.isChecked() is True
    win.applyCopy()
    assert win.currentPlot.histo_autoscale.isChecked() is False


def test_apply_clamps_a_non_positive_bottom_onto_a_log_target(win):
    # PIN: a linear source reports a bottom at or below zero and
    # a log axis rejects it, so only the top of the range used to copy. The
    # clamp lands in BOTH tiers, so what is stored equals what is drawn.
    add_pad(win, "src", 0)
    target_ax = add_pad(win, "dst", 1)
    target_ax.set_yscale("log")
    prime_apply(win, source_range=("[0.0,100.0]", "[0.0,50.0]"))
    win.copyAttr.axisLimitY._checked = True
    win.copyAttr.buttons = action_buttons() + [target_button(1)]
    win.applyCopy()
    assert win.getSpectrumViewInfo("miny", index=1) == 0.001
    assert target_ax.get_ylim()[0] == pytest.approx(0.001)


def test_apply_leaves_a_linear_target_bottom_alone(win):
    add_pad(win, "src", 0)
    target_ax = add_pad(win, "dst", 1)
    prime_apply(win, source_range=("[0.0,100.0]", "[0.0,50.0]"))
    win.copyAttr.axisLimitY._checked = True
    win.copyAttr.buttons = action_buttons() + [target_button(1)]
    win.applyCopy()
    assert win.getSpectrumViewInfo("miny", index=1) == 0.0


def test_apply_stores_the_raw_value_when_the_pad_has_no_axes(win):
    # PIN: with no axes the scale is unknowable, so the raw
    # value is stored and setAxisScale clamps it on read. An undrawn pad's
    # slot holds the DisplaySlot empty-list default rather than None, and
    # testing that against None sent `[]` into get_yscale(); the blanket
    # except then swallowed the AttributeError and the whole Apply was lost,
    # every target included.
    add_pad(win, "src", 0)
    add_pad(win, "dst", 1, with_axes=False)
    prime_apply(win, source_range=("[0.0,100.0]", "[7.0,50.0]"))
    win.copyAttr.axisLimitY._checked = True
    win.copyAttr.buttons = action_buttons() + [target_button(1)]
    win.applyCopy()
    assert win.getSpectrumViewInfo("miny", index=1) == 7.0
    assert win.getSpectrumViewInfo("maxy", index=1) == 50.0
    assert win.plot_controller.scale_calls == []
    assert win.plot_controller.update_calls == 1        # the Apply completed


def test_apply_copies_z_only_for_a_2d_source(win):
    ax = add_pad(win, "src", 0, dim=2)
    dst_ax = add_pad(win, "dst", 1, dim=2)
    image = dst_ax.imshow([[0, 1], [2, 3]])
    win.setSpectrumViewInfo(spectrum=image, index=1)
    prime_apply(win, z=("5.0", "55.0"))
    win.copyAttr.histoScaleminZ._checked = True
    win.copyAttr.buttons = action_buttons() + [target_button(1)]
    win.applyCopy()
    assert win.getSpectrumViewInfo("minz", index=1) == 5.0
    assert image.get_clim() == (5.0, 55.0)


def test_apply_ignores_z_for_a_1d_source(win):
    add_pad(win, "src", 0, dim=1)
    add_pad(win, "dst", 1, dim=1)
    prime_apply(win, z=("5.0", "55.0"))
    win.copyAttr.histoScaleminZ._checked = True
    win.copyAttr.buttons = action_buttons() + [target_button(1)]
    win.applyCopy()
    assert win.getSpectrumViewInfo("minz", index=1) == []      # slot default


def test_apply_routes_the_scale_through_the_plot_controller(win):
    add_pad(win, "src", 0)
    add_pad(win, "dst", 1)
    prime_apply(win, scale="Log")
    win.copyAttr.axisScale._checked = True
    win.copyAttr.buttons = action_buttons() + [target_button(1)]
    win.applyCopy()
    assert win.getSpectrumViewInfo("log", index=1) is True
    assert win.plot_controller.scale_calls == [(1, ("log",))]


def test_apply_redraws_once_at_the_end(win):
    add_pad(win, "src", 0)
    add_pad(win, "a", 1)
    add_pad(win, "b", 2)
    prime_apply(win)
    win.copyAttr.axisLimitX._checked = True
    win.copyAttr.buttons = action_buttons() + [target_button(1), target_button(2)]
    win.applyCopy()
    assert win.plot_controller.update_calls == 1


def test_apply_survives_a_target_pad_the_store_no_longer_knows(win):
    # PIN: a geometry change blanks the pad-to-name map while the
    # slots keep their artists. The target is skipped, the slot is still
    # written, and the exception must not escape into the Qt slot.
    add_pad(win, "src", 0)
    add_pad(win, "dst", 1)
    prime_apply(win, scale="Log")
    win.copyAttr.axisScale._checked = True
    win.copyAttr.buttons = action_buttons() + [target_button(1)]
    win.currentPlot.h_dict_geo[1] = "empty"           # what InitializeCanvas leaves
    win.applyCopy()
    assert win.plot_controller.scale_calls == [(1, ("log",))]
    assert win.plot_controller.update_calls == 1


def test_apply_never_lets_an_exception_escape_the_slot(win):
    add_pad(win, "src", 0)
    add_pad(win, "dst", 1)
    prime_apply(win)
    win.copyAttr.axisLimLabelX.setText("not a literal")      # literal_eval raises
    win.copyAttr.axisLimitX._checked = True
    win.copyAttr.buttons = action_buttons() + [target_button(1)]
    win.applyCopy()                                   # logged, not raised
    assert win.plot_controller.update_calls == 0


# ============================================== selectAll / histAllAttr / close

def test_select_all_checks_every_target_and_flips_its_own_label(win):
    targets = [target_button(1, checked=False), target_button(2, checked=False)]
    buttons = action_buttons() + targets
    win.copyAttr.buttons = buttons
    win.selectAll()
    assert all(t.isChecked() for t in targets)
    assert [b.text() for b in buttons
            if b.text() in ("Select all", "Deselect all")] == ["Deselect all"]


def test_deselect_all_clears_every_target(win):
    targets = [target_button(1), target_button(2)]
    win.copyAttr.buttons = action_buttons() + targets
    win.selectAll()          # label becomes "Deselect all"
    win.selectAll()          # label back to "Select all", and everything clears
    assert not any(t.isChecked() for t in targets)


def test_hist_all_attr_sets_every_property_checkbox(win):
    add_pad(win, "src", 0)
    win.histAllAttr(FakeCheck(checked=True, text="Select all properties"))
    assert win.copyAttr.axisLimitX.isChecked()
    assert win.copyAttr.axisLimitY.isChecked()
    assert win.copyAttr.axisScale.isChecked()


def test_hist_all_attr_disables_the_z_boxes_for_a_1d_source(win):
    add_pad(win, "src", 0, dim=1)
    win.histAllAttr(FakeCheck(checked=False, text="Select all properties"))
    assert win.copyAttr.histoScaleminZ.enabled is False
    assert win.copyAttr.histoScaleValuemaxZ.enabled is False


def test_hist_all_attr_enables_the_z_boxes_for_a_2d_source(win):
    add_pad(win, "src", 0, dim=2)
    win.histAllAttr(FakeCheck(checked=False, text="Select all properties"))
    assert win.copyAttr.histoScaleminZ.enabled is True


def test_close_deletes_the_target_buttons_only(win):
    actions = action_buttons()
    targets = [target_button(1)]
    win.copyAttr.buttons = actions + targets
    win.closeCopy()
    assert all(not b.deleted for b in actions)
    assert all(t.deleted for t in targets)
    assert win.copyAttr.visible is False


def test_connect_copy_toggles_the_button_colour(win):
    red = FakeButton("t", colour="#ff0000")
    win.connectCopy(red)
    assert "green" in red.style
    green = FakeButton("t", colour="#008000")
    win.connectCopy(green)
    assert "red" in green.style


# ------------------------------------------------- z fields follow the source

def z_widgets(win):
    return (win.copyAttr.histoScaleminZ, win.copyAttr.histoScalemaxZ,
            win.copyAttr.histoScaleValueminZ, win.copyAttr.histoScaleValuemaxZ)


def test_popup_disables_the_z_fields_for_a_1d_source(win):
    # a 1D pad has no colour scale, and applyCopy has always refused to copy
    # one — the dialog used to offer it anyway, pre-filled with 0/256
    add_pad(win, "src", 0)
    # exactly what a 2D source leaves behind in the shared dialog
    win.copyAttr.histoScaleminZ.setChecked(True)
    win.copyAttr.histoScaleValueminZ.setText("3.0")
    win.copyAttr.histoScaleValuemaxZ.setText("77.0")
    win.copyPopup()
    assert all(not w.enabled for w in z_widgets(win))
    assert not win.copyAttr.histoScaleminZ.isChecked()
    assert win.copyAttr.histoScaleValueminZ.text() == ""
    assert win.copyAttr.histoScaleValuemaxZ.text() == ""


def test_popup_enables_the_z_fields_for_a_2d_source(win):
    ax = add_pad(win, "src", 0, dim=2)
    image = ax.imshow([[0, 1], [2, 3]])
    image.set_clim(3.0, 77.0)
    win.setSpectrumViewInfo(spectrum=image, minx=0.0, maxx=1.0,
                            miny=0.0, maxy=1.0, index=0)
    win.copy_props._set_z_fields_enabled(False)       # as a 1D pad left them
    win.copyPopup()
    assert all(w.enabled for w in z_widgets(win))
    assert win.copyAttr.histoScaleValueminZ.text() == "3.0"


def test_select_all_properties_skips_the_disabled_z_boxes(win):
    add_pad(win, "src", 0)
    win.copyPopup()
    master = FakeCheck(checked=True, text="Select all properties")
    win.copy_props.histAllAttr(master)
    assert win.copyAttr.axisLimitX.isChecked()
    assert not win.copyAttr.histoScaleminZ.isChecked()
    assert not win.copyAttr.histoScalemaxZ.isChecked()


def test_apply_still_copies_x_and_y_with_the_z_boxes_blank(win):
    # the z boxes are read inside applyCopy's try; parsing a blank one raised
    # into the catch-all and took x, y and scale down with it
    add_pad(win, "src", 0)
    ax = add_pad(win, "dst", 1)
    win.currentPlot.selected_plot_index = 0
    win.copyPopup()
    assert win.copyAttr.histoScaleValueminZ.text() == ""
    win.copyAttr.axisLimLabelX.setText("[0.0,100.0]")
    win.copyAttr.axisLimLabelY.setText("[1.0,50.0]")
    win.copyAttr.axisLimitX._checked = True
    win.copyAttr.axisLimitY._checked = True
    win.copyAttr.buttons = action_buttons() + [target_button(1)]

    win.copy_props.applyCopy()

    assert win.getSpectrumViewInfo("minx", index=1) == 0.0
    assert ax.get_xlim() == (0.0, 100.0)
    assert ax.get_ylim() == (1.0, 50.0)
