"""Copy Properties: read one pad's display settings and apply the ticked ones to
the selected targets."""

import ast
import logging
from numbers import Number

from PyQt5.QtWidgets import QPushButton
from PyQt5.QtGui import QPalette


class CopyPropertiesController:

    def __init__(self, copy_attr,
                 get_selected_index, set_autoscale,
                 get_store_info, get_view_info, set_view_info,
                 get_geo, name_from_index, plot_position,
                 plot_controller, parent_widget=None, logger=None):
        self._copy_attr          = copy_attr            # CopyPropertiesGUI
        self._get_selected_index = get_selected_index   # () -> int|None
        self._set_autoscale      = set_autoscale        # (bool) -> None
        self._get_store_info     = get_store_info       # (key, index=|name=) -> value
        self._get_view_info      = get_view_info        # (key, index=) -> value
        self._set_view_info      = set_view_info        # (key=val, index=) -> None
        self._get_geo            = get_geo              # () -> {index: name}
        self._name_from_index    = name_from_index      # (index) -> str|None
        self._plot_position      = plot_position        # (index) -> (row, col)
        self._plot_controller    = plot_controller
        self._parent_widget      = parent_widget        # parent for the target buttons
        self.logger              = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Popup lifecycle
    # ------------------------------------------------------------------

    # open copy properties popup
    def copyPopup(self):
        self.logger.info('copyPopup - selected_plot_index: %s', self._get_selected_index())
        if self._copy_attr.isVisible():
            self._copy_attr.close()
        index = self._get_selected_index()
        name = self._name_from_index(index)
        dim = self._get_store_info("dim", index=index)

        if dim is None:
            self.logger.debug('copyPopup - dim is None', exc_info=True)
            return

        # setting up info for source histogram
        self._copy_attr.histoLabel.setText(name)
        # z is the colour scale of a 2D image; a 1D pad has none, and applyCopy
        # has always refused to copy it. Offer the boxes only where they mean
        # something, and blank them so a previous 2D source's numbers cannot be
        # read as this pad's.
        self._set_z_fields_enabled(dim == 2)
        if dim == 2:
            spectrum = self._get_view_info("spectrum", index=index)
            zmin, zmax = spectrum.get_clim()
            self._copy_attr.histoScaleValueminZ.setText(f"{zmin}")
            self._copy_attr.histoScaleValuemaxZ.setText(f"{zmax}")
        else:
            self._copy_attr.histoScaleValueminZ.setText("")
            self._copy_attr.histoScaleValuemaxZ.setText("")
        self._copy_attr.axisSLabel.setText("Log" if self._get_view_info("log", index=index) else "Linear")
        xmin = self._get_view_info("minx", index=index)
        xmax = self._get_view_info("maxx", index=index)
        ymin = self._get_view_info("miny", index=index)
        ymax = self._get_view_info("maxy", index=index)
        # A pad can legitimately have no stored y range. The display tier gets
        # one only as a side effect of setAxisScale, which the render tick
        # calls just while autoscale is on, and a 1D spectrum that arrived on
        # a binding trace starts out with miny/maxy None (the connect-time
        # path fills them from the shared-memory header, the trace path has
        # nothing to fill them from).
        ax = self._get_view_info("axis", index=index)
        if ax:
            if not isinstance(xmin, Number) or not isinstance(xmax, Number):
                xmin, xmax = ax.get_xlim()
            if not isinstance(ymin, Number) or not isinstance(ymax, Number):
                ymin, ymax = ax.get_ylim()
        if not all(isinstance(v, Number) for v in (xmin, xmax, ymin, ymax)):
            self.logger.warning('copyPopup - pad %s has no usable axis range (x: %s, %s  y: %s, %s); not opening',
                                index, xmin, xmax, ymin, ymax)
            return
        self._copy_attr.axisLimLabelX.setText(f"[{xmin:.1f},{xmax:.1f}]")
        self._copy_attr.axisLimLabelY.setText(f"[{ymin:.1f},{ymax:.1f}]")

        # reset QFormLayout
        rowCount = self._copy_attr.copy_log.rowCount()
        for i in range(rowCount):
            self._copy_attr.copy_log.removeRow(0)

        try:
            for idx, nameTarget in self._get_geo().items():
                if dim == self._get_store_info("dim", index=idx) and idx != index:
                    instance = QPushButton(nameTarget, self._parent_widget)
                    instance.setCheckable(True)
                    instance.setStyleSheet('QPushButton {color: red;}')
                    # the pad this button stands for, carried on the button
                    # itself. applyCopy used to recover it by scraping the row
                    # and column back out of the label text and multiplying by
                    # the column count read at Apply time, so re-applying a
                    # geometry while the popup was open sent the properties to
                    # different pads than the ones the user picked.
                    instance.setProperty("padIndex", int(idx))
                    row, col = self._plot_position(idx)
                    self._copy_attr.copy_log.addRow("row: "+str(row)+" col: "+str(col), instance)
                    instance.clicked.connect(lambda state, instance=instance: self.connectCopy(instance))
        except KeyError:
            self.logger.warning('copyPopup - KeyError occured', exc_info=True)
        self._copy_attr.show()

    def _set_z_fields_enabled(self, on):
        """Enable or disable the Min Z / Max Z pair. Disabling also unchecks:
        a checked-but-disabled box would still read as checked at Apply."""
        for w in (self._copy_attr.histoScaleminZ, self._copy_attr.histoScalemaxZ):
            if not on:
                w.setChecked(False)
            w.setEnabled(on)
        self._copy_attr.histoScaleValueminZ.setEnabled(on)
        self._copy_attr.histoScaleValuemaxZ.setEnabled(on)

    # callback for copyAttr.okAttr
    def okCopy(self):
        self.logger.info('okCopy')
        self.applyCopy()
        self.closeCopy()

    # callback for copyAttr.cancelAttr
    def closeCopy(self):
        self.logger.info('closeCopy')
        discard = ["Ok", "Cancel", "Apply", "Select all", "Deselect all"]
        for instance in self._copy_attr.findChildren(QPushButton):
            if instance.text() not in discard:
                instance.deleteLater()

        self._copy_attr.close()

    # ------------------------------------------------------------------
    # Applying
    # ------------------------------------------------------------------

    # callback for copyAttr.applyAttr
    def applyCopy(self):
        self.logger.info('applyCopy')
        try:
            # read each property checkbox by name; a positional list built from
            # findChildren() would silently re-point if CopyProperties ever
            # reorders or gains a checkbox. histoAll is the master toggle, not a
            # property, so it is not one of these.
            copy_xlim  = self._copy_attr.axisLimitX.isChecked()
            copy_ylim  = self._copy_attr.axisLimitY.isChecked()
            copy_scale = self._copy_attr.axisScale.isChecked()
            copy_minz  = self._copy_attr.histoScaleminZ.isChecked()
            copy_maxz  = self._copy_attr.histoScalemaxZ.isChecked()

            self.logger.debug('applyCopy - x: %s, y: %s, scale: %s, minz: %s, maxz: %s',
                              copy_xlim, copy_ylim, copy_scale, copy_minz, copy_maxz)

            dim = self._get_store_info("dim", index=self._get_selected_index())
            indexes = []

            # creating list of target histograms
            discard = ["Ok", "Cancel", "Apply", "Select all", "Deselect all"]
            for instance in self._copy_attr.findChildren(QPushButton):
                if instance.text() not in discard and instance.isChecked():
                    # the pad index copyPopup stored on the button, not one
                    # re-derived from the label and the current column count
                    padIndex = instance.property("padIndex")
                    if padIndex is None:
                        self.logger.warning('applyCopy - target button %s carries no pad index, skipped',
                                            instance.text())
                        continue
                    indexes.append(int(padIndex))

            self.logger.debug('applyCopy - indexes : %s', indexes)

            # src values to copy to destination
            xlim_src = ast.literal_eval(self._copy_attr.axisLimLabelX.text())
            ylim_src = ast.literal_eval(self._copy_attr.axisLimLabelY.text())
            scale_src = self._copy_attr.axisSLabel.text()
            scale_src_bool = True if scale_src == "Log" else False
            # read only where it exists: the boxes are blank for a 1D source,
            # and parsing them unconditionally would abort the whole copy —
            # x, y and scale included — on a float("") deep in this try
            zlim_src = None
            if dim == 2:
                zlim_src = [float(self._copy_attr.histoScaleValueminZ.text()),
                            float(self._copy_attr.histoScaleValuemaxZ.text())]

            self.logger.debug('applyCopy - xlim_src, ylim_src, scale_src, zlim_src : %s, %s, %s, %s',
                              xlim_src, ylim_src, scale_src, zlim_src)

            # autoscale off, or the trailing updatePlot recomputes y/z from the
            # data and discards the copied values (same pattern as
            # zoomInOut / cutoffButtonCallback)
            self._set_autoscale(False)

            # copy to destination
            for index in indexes:
                # the target axes are read up front because the y bottom has
                # to be clamped before it is stored, not only before it is
                # drawn: a linear source pad reports a zero or slightly
                # negative bottom, and a log-scaled target rejects that
                # outright (matplotlib warns and keeps its own, so only half
                # the range copies). Clamp to the same floor setAxisScale uses
                # for its log branch.
                ax = self._get_view_info("axis", index=index)
                ymin_dst, ymax_dst = ylim_src[0], ylim_src[1]
                if ax and ax.get_yscale() == "log" and ymin_dst <= 0:
                    ymin_dst = 0.001

                # set the limits for x,y
                if copy_xlim:
                    self._set_view_info(minx=xlim_src[0], index=index)
                    self._set_view_info(maxx=xlim_src[1], index=index)
                if copy_ylim:
                    self._set_view_info(miny=ymin_dst, index=index)
                    self._set_view_info(maxy=ymax_dst, index=index)
                # set log/lin scale
                if copy_scale:
                    self._set_view_info(log=scale_src_bool, index=index)
                # set minZ/maxZ (either box copies both bounds)
                if dim == 2 and (copy_minz or copy_maxz):
                    self._set_view_info(minz=zlim_src[0], index=index)
                    self._set_view_info(maxz=zlim_src[1], index=index)
                # apply to the target axes directly: updatePlot's only
                # limits-application path is autoscale-gated, so view-tier
                # writes alone never reach the screen (okCutoff precedent)
                if not ax:
                    continue
                if copy_xlim:
                    ax.set_xlim(xlim_src[0], xlim_src[1])
                if copy_ylim:
                    ax.set_ylim(ymin_dst, ymax_dst)
                if dim == 2 and (copy_minz or copy_maxz):
                    spectrum = self._get_view_info("spectrum", index=index)
                    if spectrum is not None:
                        spectrum.set_clim(zlim_src[0], zlim_src[1])
                if copy_scale:
                    self._plot_controller.setAxisScale(ax, index, "log")
            self._plot_controller.updatePlot()
        except Exception:
            self.logger.exception('applyCopy - copy properties failed')

    # ------------------------------------------------------------------
    # Selection widgets
    # ------------------------------------------------------------------

    # callback to change color when press spectrum button name
    def connectCopy(self, instance):
        self.logger.info('connectCopy')
        if (instance.palette().color(QPalette.Text).name() == "#008000"):
            instance.setStyleSheet('QPushButton {color: red;}')
        else:
            instance.setStyleSheet('QPushButton {color: green;}')

    def selectAll(self):
        self.logger.info('selectAll')
        flag = False
        basic = ["Ok", "Cancel", "Apply"]
        discard = ["Ok", "Cancel", "Apply", "Select all", "Deselect all"]
        for instance in self._copy_attr.findChildren(QPushButton):
            if instance.text() not in discard:
                instance.setChecked(True)
                instance.setStyleSheet('QPushButton {color: green;}')
            else:
                if instance.text() not in basic:
                    if instance.text() == "Select all":
                        instance.setText("Deselect all")
                    else:
                        instance.setText("Select all")
                        flag = True

        if flag == True:
            for instance in self._copy_attr.findChildren(QPushButton):
                if instance.text() not in discard:
                    instance.setChecked(False)
                    instance.setStyleSheet('QPushButton {color: red;}')
                    flag = False

    def histAllAttr(self, b):
        self.logger.info('histAllAttr - b.text(): %s', b.text())
        if b.text() == "Select all properties":
            if b.isChecked() == True:
                self._copy_attr.axisLimitX.setChecked(True)
                self._copy_attr.axisLimitY.setChecked(True)
                self._copy_attr.axisScale.setChecked(True)
                # "all" means all the ones this source actually has: on a 1D
                # pad the z boxes are disabled and stay unchecked
                if self._copy_attr.histoScaleminZ.isEnabled():
                    self._copy_attr.histoScaleminZ.setChecked(True)
                if self._copy_attr.histoScalemaxZ.isEnabled():
                    self._copy_attr.histoScalemaxZ.setChecked(True)
            else:
                self._copy_attr.axisLimitX.setChecked(False)
                self._copy_attr.axisLimitY.setChecked(False)
                self._copy_attr.axisScale.setChecked(False)
                self._copy_attr.histoScaleminZ.setChecked(False)
                self._copy_attr.histoScalemaxZ.setChecked(False)

        dim = self._get_store_info("dim", index=self._get_selected_index())

        if dim == 1:
            self._copy_attr.histoScaleminZ.setEnabled(False)
            self._copy_attr.histoScaleValueminZ.setEnabled(False)
            self._copy_attr.histoScalemaxZ.setEnabled(False)
            self._copy_attr.histoScaleValuemaxZ.setEnabled(False)
        else:
            self._copy_attr.histoScaleminZ.setEnabled(True)
            self._copy_attr.histoScaleValueminZ.setEnabled(True)
            self._copy_attr.histoScalemaxZ.setEnabled(True)
            self._copy_attr.histoScaleValuemaxZ.setEnabled(True)
