"""Peak Finder 2 (click-to-fit): the press dispatch, the drag-to-refit, the shape
menus, the results table and the edit popup. The fitting maths is Qt-free in
``services/peak_finder.py``."""

import logging

import numpy as np

from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtWidgets import (QDialog, QFormLayout, QHBoxLayout, QInputDialog,
                             QLineEdit, QPushButton, QTableWidgetItem,
                             QVBoxLayout)

from services.peak_finder import (autocomponent_refit, find_duplicate_mu,
                                  fit_composite, fit_composite_auto,
                                  fix_peak_window, format_composite_fit_row,
                                  fwhm_to_sigma, nearest_component_index,
                                  nearest_window_edge, primary_component,
                                  sigma_to_fwhm, validate_gauss_edit)


class _NumericItem(QTableWidgetItem):
    """Peak Finder 2 results-table cell that sorts by a stored numeric value
    (Qt.UserRole) rather than its displayed '<value> ± <err>' string."""
    def __lt__(self, other):
        try:
            return float(self.data(Qt.UserRole)) < float(other.data(Qt.UserRole))
        except (TypeError, ValueError):
            return super().__lt__(other)


class PeakFit2Controller:

    _PEAK2_SIGNAL_BY_LABEL = {"Gaussian": "gaussian", "Crystal ball": "crystal_ball"}
    _PEAK2_BG_BY_LABEL = {"Linear": "poly1", "Quadratic": "poly2", "Cubic": "poly3"}

    def __init__(self, peak_tab, spectra, get_store_info, get_view_info,
                 plot_controller, tabs=None, get_current_plot=None,
                 get_gate_popup=None, get_sum_popup=None,
                 name_from_index=None, parent_widget=None, logger=None):
        self._peak            = peak_tab          # extraPopup.peak
        self._spectra         = spectra
        self._get_store_info  = get_store_info
        self._get_view_info   = get_view_info
        self._plot_controller = plot_controller
        self._tabs            = tabs
        self._get_current_plot = get_current_plot
        # late-bound: both popups are MainWindow attributes that can be
        # replaced, and a torn-down one must read as "not blocking"
        self._get_gate_popup  = get_gate_popup
        self._get_sum_popup   = get_sum_popup
        self._name_from_index = name_from_index
        self._parent_widget   = parent_widget
        self.logger = logger or logging.getLogger(__name__)
        # which canvases carry the press handler
        self.peak2_conns = {}
        # arming state and the live drag context: 8b set these through seams
        # while the press handlers were still on MainWindow; 8c brought those
        # handlers here, so the seams are gone and these are plain attributes
        self.peak2_armed = False
        self.peak2_fix_armed = False
        self.peak2_drag = None
        # 8d: the fit records and the running peak number came here with the
        # results table, which retired the last of the temporary seams
        self.peak2_fits = []
        self.peak2_count = 0
        self._peak2_edit_linking = False

    # ------------------------------------------------------------------
    # arming and canvas connections
    # ------------------------------------------------------------------

    def _current_canvas(self):
        return self._tabs.plot(self._tabs.currentIndex()).canvas

    def _peak2_connect(self, canvas):
        """Ensure the unified press handler is connected on `canvas`."""
        if canvas is None or canvas in self.peak2_conns:
            return
        self.peak2_conns[canvas] = canvas.mpl_connect(
            "button_press_event", self.onPeakFit2Press)

    def _peak2_disconnect(self, canvas):
        cid = self.peak2_conns.pop(canvas, None)
        if cid is not None:
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                self.logger.debug('_peak2_disconnect failed', exc_info=True)

    def _peak2_fit_canvases(self):
        """Canvases that currently hold at least one recorded fit's artists."""
        canvases = set()
        for rec in self.peak2_fits:
            for art in rec.get("artists") or ():
                try:
                    canvases.add(art.axes.figure.canvas)
                except Exception:
                    pass
        return canvases

    def _peak2_sync_connections(self):
        """Drop the handler from canvases that are neither armed nor holding
        fits; keep it wherever fits still live (so drag/edit will work after
        Stop)."""
        keep = self._peak2_fit_canvases()
        if self.peak2_armed or self.peak2_fix_armed:
            try:
                keep.add(self._current_canvas())
            except Exception:
                pass
        for canvas in list(self.peak2_conns):
            if canvas not in keep:
                self._peak2_disconnect(canvas)

    def peakFit2Toggle(self, checked):
        """Start/Stop toggle: arm (or disarm) the current tab's canvas so each
        left-click fits a gaussian+linear around the clicked position. The press
        handler stays connected wherever fits remain, so future drag/edit
        interactions survive Stop."""
        self.logger.info('peakFit2Toggle - checked: %s', checked)
        btn = self._peak.peak2_start
        self.peak2_armed = bool(checked)
        if checked:
            # Start and Fix Peak are mutually exclusive arming modes
            self._peak.peak2_fix.setChecked(False)
            self._peak2_connect(self._current_canvas())
            btn.setText("Stop")
            btn.setStyleSheet("background-color:#ff6b6b;")
            self._peak2_status(
                "[armed] Left-click a peak on the pad to fit it. "
                "Click Stop to disarm.")
        else:
            btn.setText("Start")
            btn.setStyleSheet("background-color:#bcee68;")
            self._peak2_sync_connections()

    def peakFit2FixToggle(self, checked):
        """Fix Peak toggle: arm fixed-μ fitting on the current tab's canvas,
        mutually exclusive with Start. While armed, each left-click fits
        gaussian+linear with μ pinned exactly at the clicked x (window centred
        on the click; see `_peak2_max_window_bins` for its size)."""
        self.logger.info('peakFit2FixToggle - checked: %s', checked)
        btn = self._peak.peak2_fix
        self.peak2_fix_armed = bool(checked)
        if checked:
            self._peak.peak2_start.setChecked(False)   # mutual exclusion
            self._peak2_connect(self._current_canvas())
            btn.setStyleSheet("background-color:#ff6b6b;")
            self._peak2_status(
                "[armed: fix μ] Left-click at a peak centre to fit with μ "
                "pinned there. Click Fix Peak again to disarm.")
        else:
            btn.setStyleSheet("background-color:#bcee68;")
            self._peak2_sync_connections()

    def _peak2_other_mode_active(self):
        """True while another pad interaction owns clicks: rubber-band
        zoom, gate create/edit, or summing-region create. An armed Peak
        Finder 2 must not also fit on those presses."""
        cp = self._get_current_plot()
        if cp.zoomPress or cp.toCreateGate or cp.toEditGate or cp.toCreateSumRegion:
            return True
        try:
            return self._get_gate_popup().isVisible() or self._get_sum_popup().isVisible()
        except Exception:
            return False

    def peakFit2Config(self):
        """Config dialog: max fit window in bins (empty = no cap)."""
        self.logger.info('peakFit2Config')
        current = self._peak2_max_window_bins()
        text, ok = QInputDialog.getText(
            self._parent_widget, "Peak Finder — Config",
            "Max fit window (in bins), empty = no cap:\n"
            "With a cap set, clicks that can't be fitted are skipped silently.",
            text="" if current is None else str(current))
        if not ok:
            return
        s = QSettings()
        text = text.strip()
        if text == "":
            s.setValue("PeakFinder2/max_window_bins", "")
            self._peak2_status("[config] Max window: no cap.")
            return
        try:
            n = int(text)
            if n <= 0:
                raise ValueError
        except ValueError:
            self._peak2_status("[config] Max window must be a positive integer or empty — unchanged.")
            return
        s.setValue("PeakFinder2/max_window_bins", str(n))
        self._peak2_status(f"[config] Max window: {n} bins.")

    # ------------------------------------------------------------------
    # results table, shape menus and the edit popup
    # ------------------------------------------------------------------

    def _peak2_add_row(self, peak_no, r, tag=None):
        """Insert one fitted peak as a row in the results table (compact
        columns + numeric sort keys; hover shows the full detail)."""
        row = format_composite_fit_row(peak_no, r, tag=tag)
        t = self._peak.peak2_table
        t.setSortingEnabled(False)     # don't re-sort mid-insert
        ri = t.rowCount()
        t.insertRow(ri)
        for ci, (text, sortval) in enumerate(row["cells"]):
            item = _NumericItem(text)
            # the # cell's sort value (== peak number) doubles as the row->fit
            # lookup key for delete/highlight
            item.setData(Qt.UserRole, float(sortval))
            item.setToolTip(row["tooltip"])
            t.setItem(ri, ci, item)
        t.setSortingEnabled(True)

    def _peak2_update_row(self, number, r, tag=None):
        """Refresh the table row for fit `number` in place after a refit."""
        row = format_composite_fit_row(number, r, tag=tag)
        t = self._peak.peak2_table
        for ri in range(t.rowCount()):
            it = t.item(ri, 0)
            if it is not None and int(round(float(it.data(Qt.UserRole)))) == number:
                for ci, (text, sortval) in enumerate(row["cells"]):
                    cell = t.item(ri, ci)
                    if cell is not None:
                        cell.setText(text)
                        cell.setData(Qt.UserRole, float(sortval))
                        cell.setToolTip(row["tooltip"])
                break

    def _peak2_selected_number(self):
        """Fit number of the currently-selected table row, or None."""
        t = self._peak.peak2_table
        ri = t.currentRow()
        if ri < 0 or t.item(ri, 0) is None:
            return None
        try:
            return int(round(float(t.item(ri, 0).data(Qt.UserRole))))
        except (TypeError, ValueError):
            return None

    def _peak2_row_selected(self):
        """Highlight the selected fit's curve on the pad (thicker + orange);
        restore the others to the default red. Also syncs the shape menus to the
        selected fit's model, so the menus reflect the fit you'd act on."""
        sel = self._peak2_selected_number()
        sel_rec = None
        canvases = set()
        for rec in self.peak2_fits:
            arts = rec.get("artists") or ()
            if not arts:
                continue
            curve = arts[0]
            try:
                if rec.get("number") == sel:
                    sel_rec = rec
                    curve.set_linewidth(3.2)
                    curve.set_color("tab:orange")
                else:
                    curve.set_linewidth(1.8)
                    curve.set_color("tab:red")
                canvases.add(curve.axes.figure.canvas)
            except Exception:
                self.logger.debug('_peak2_row_selected - restyle failed', exc_info=True)
        # sync the menus to the selected fit's spec (signals blocked inside, so
        # this never triggers _peak2_shape_changed's re-fit)
        if sel_rec is not None:
            self._peak2_sync_menus_to_spec(sel_rec["result"].get("spec", {}))
        for c in canvases:
            try:
                c.draw_idle()
            except Exception:
                self.logger.debug('_peak2_row_selected - redraw failed', exc_info=True)

    def _peak2_delete_selected(self):
        """Delete the selected fit: its table row, its record, and its artists
        on the pad."""
        num = self._peak2_selected_number()
        if num is None:
            self._peak2_status("[delete] Select a fit row first.")
            return
        rec = next((rc for rc in self.peak2_fits if rc.get("number") == num), None)
        canvases = set()
        if rec is not None:
            for art in rec.get("artists") or ():
                try:
                    canvases.add(art.axes.figure.canvas)
                    art.remove()
                except Exception:
                    self.logger.debug('_peak2_delete_selected - remove failed', exc_info=True)
            self.peak2_fits.remove(rec)
        t = self._peak.peak2_table
        for ri in range(t.rowCount()):
            it = t.item(ri, 0)
            if it is not None and int(round(float(it.data(Qt.UserRole)))) == num:
                t.removeRow(ri)
                break
        try:
            canvases.add(self._get_current_plot().canvas)
        except Exception:
            pass
        for c in canvases:
            try:
                c.draw_idle()
            except Exception:
                self.logger.debug('_peak2_delete_selected - redraw failed', exc_info=True)
        # a canvas that just lost its last fit and isn't armed can release the handler
        self._peak2_sync_connections()
        self._peak2_status(f"[delete] Removed Peak {num}.")

    def peakFit2Clear(self):
        """Remove every Peak Finder 2 artist and clear its output box."""
        self.logger.info('peakFit2Clear')
        canvases = set()
        for rec in self.peak2_fits:
            for art in rec.get("artists") or ():
                try:
                    canvases.add(art.axes.figure.canvas)
                except Exception:
                    pass
                try:
                    art.remove()
                except Exception:
                    self.logger.debug('peakFit2Clear - artist remove failed', exc_info=True)
        self.peak2_fits = []
        self.peak2_count = 0
        self._peak.peak2_table.setRowCount(0)
        self._peak2_status("")
        # no fits remain: drop the handler from canvases unless still armed
        self._peak2_sync_connections()
        # redraw every canvas that held a fit, including other tabs'
        try:
            canvases.add(self._get_current_plot().canvas)
        except Exception:
            pass
        for canvas in canvases:
            try:
                canvas.draw_idle()
            except Exception:
                self.logger.debug('peakFit2Clear - redraw failed', exc_info=True)

    def _peak2_load_shape_menus(self):
        """Restore the last-used shape-menu selections from QSettings (signals
        blocked so restoring does not trigger the persist/refit slot)."""
        s = QSettings()
        p = self._peak
        for widget, key in ((p.peak2_signal, "signal_shape"),
                            (p.peak2_bg, "background_shape"),
                            (p.peak2_cb_tail, "cb_tail_side")):
            val = s.value(f"PeakFinder2/{key}", "", type=str)
            if val:
                widget.blockSignals(True)
                widget.setCurrentText(val)
                widget.blockSignals(False)

    def _peak2_sync_menus_to_spec(self, spec):
        """Write `spec` back into the three shape menus WITH combo signals
        blocked, so syncing the menus to the selected fit never triggers
        `_peak2_shape_changed`'s re-fit (the re-entrancy guard)."""
        p = self._peak
        sig = next((k for k, v in self._PEAK2_SIGNAL_BY_LABEL.items()
                    if v == spec.get("signal")), "Gaussian")
        bg = next((k for k, v in self._PEAK2_BG_BY_LABEL.items()
                   if v == spec.get("background")), "Linear")
        for widget, text in ((p.peak2_signal, sig), (p.peak2_bg, bg),
                             (p.peak2_cb_tail, spec.get("tail_side", "low"))):
            widget.blockSignals(True)
            widget.setCurrentText(text)
            widget.blockSignals(False)

    def _peak2_shape_changed(self, *_):
        """A shape menu changed: persist the selection, then — if a fit row is
        selected — re-fit THAT fit in place with the new model. With no row
        selected the menu only sets the default for the next new fit."""
        s = QSettings()
        p = self._peak
        s.setValue("PeakFinder2/signal_shape", p.peak2_signal.currentText())
        s.setValue("PeakFinder2/background_shape", p.peak2_bg.currentText())
        s.setValue("PeakFinder2/cb_tail_side", p.peak2_cb_tail.currentText())
        num = self._peak2_selected_number()
        if num is None:
            return
        rec = next((rc for rc in self.peak2_fits if rc.get("number") == num), None)
        if rec is not None:
            self._peak2_refit_selected(rec)

    def _peak2_refit_selected(self, rec):
        """Re-fit `rec` in place over its stored window with the current menu
        model (keeping the fit's own component count; μ seeded from the fit so
        it stays on the same peak). Reuses the drag/edit redraw path; on failure
        the previous fit is kept untouched."""
        prev = rec["result"]
        xx = prev["xx"]
        lo, hi = float(xx[0]), float(xx[-1])
        spec = self._peak2_current_spec()
        spec["n_components"] = prev["spec"].get("n_components", 1)
        seeds = {f"mu{i + 1}": c["mu"] for i, c in enumerate(prev["components"])}
        ax = self._peak2_live_axes(rec)
        arrays = None if ax is None else self._peak2_spectrum_arrays(rec["name"])
        if arrays is None:
            self._peak2_status(f"[shape] Peak {rec['number']}: spectrum unavailable.")
            return
        xc, y = arrays
        r = fit_composite(xc, y, lo, hi, spec, seeds=seeds)
        if not r["ok"]:
            self._peak2_status(f"[shape] Peak {rec['number']}: "
                               f"{r.get('error', 'refit failed')} — unchanged.")
            return
        for art in rec.get("artists") or ():
            try:
                art.remove()
            except Exception:
                pass
        rec["result"] = r
        rec["artists"] = self._peak2_draw(ax, r)
        self._peak2_update_row(rec["number"], r, tag="edited")
        self._peak2_status(f"Peak {rec['number']} → {spec['signal']}/"
                           f"{spec['background']}: μ = {self._peak2_result_mu(r):.6g}")
        ax.figure.canvas.draw_idle()

    def _peak2_open_edit(self, rec, click_x):
        """Modal μ/σ/FWHM editor for one fit. On a multi-component fit the
        edited component is the one whose μ is nearest the right-clicked x
        (named in the dialog title)."""
        prev = rec["result"]
        comps = prev["components"]
        if click_x is None:                       # right-click without an x → first
            click_x = comps[0]["mu"]
        ci = nearest_component_index(comps, click_x) or 0
        c = comps[ci]
        mu_txt0 = f"{c['mu']:.6g}"
        sig_txt0 = f"{c['sigma']:.6g}"
        dlg = QDialog(self._parent_widget)
        title = f"Edit Peak {rec['number']}"
        if len(comps) > 1:
            title += f" — component {ci + 1}/{len(comps)} (μ≈{c['mu']:.4g})"
        dlg.setWindowTitle(title)
        mu_edit = QLineEdit(mu_txt0)
        sigma_edit = QLineEdit(sig_txt0)
        fwhm_edit = QLineEdit(f"{c['fwhm']:.6g}")
        form = QFormLayout()
        form.addRow("μ", mu_edit)
        form.addRow("σ", sigma_edit)
        form.addRow("FWHM", fwhm_edit)

        # link σ <-> FWHM live (guard against the re-entrant echo)
        self._peak2_edit_linking = False

        def _from_sigma(_):
            if self._peak2_edit_linking:
                return
            self._peak2_edit_linking = True
            try:
                fwhm_edit.setText(f"{sigma_to_fwhm(sigma_edit.text()):.6g}")
            except (ValueError, TypeError):
                pass
            finally:
                self._peak2_edit_linking = False

        def _from_fwhm(_):
            if self._peak2_edit_linking:
                return
            self._peak2_edit_linking = True
            try:
                sigma_edit.setText(f"{fwhm_to_sigma(fwhm_edit.text()):.6g}")
            except (ValueError, TypeError):
                pass
            finally:
                self._peak2_edit_linking = False

        sigma_edit.textEdited.connect(_from_sigma)
        fwhm_edit.textEdited.connect(_from_fwhm)

        apply_btn = QPushButton("Apply")
        cancel_btn = QPushButton("Cancel")
        apply_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        btns = QHBoxLayout()
        btns.addWidget(apply_btn)
        btns.addWidget(cancel_btn)
        lay = QVBoxLayout()
        lay.addLayout(form)
        lay.addLayout(btns)
        dlg.setLayout(lay)

        if dlg.exec_() != QDialog.Accepted:
            return

        # fields whose text changed become fixed (σ text also changes when the
        # user edits FWHM, via the link — so a width edit either way is caught)
        fixed = {}
        try:
            if mu_edit.text() != mu_txt0:
                fixed["mu"] = float(mu_edit.text())
            if sigma_edit.text() != sig_txt0:
                fixed["sigma"] = float(sigma_edit.text())
        except ValueError:
            self._peak2_status(f"[edit] Peak {rec['number']}: invalid number — unchanged.")
            return
        if not fixed:
            self._peak2_status(f"[edit] Peak {rec['number']}: nothing changed.")
            return
        xx = prev["xx"]
        bad = validate_gauss_edit(fixed, lo=float(xx[0]), hi=float(xx[-1]))
        if bad:
            self._peak2_status(f"[edit] Peak {rec['number']}: {bad} — unchanged.")
            return

        ax = self._peak2_live_axes(rec)
        arrays = None if ax is None else self._peak2_spectrum_arrays(rec["name"])
        if arrays is None:
            self._peak2_status(f"[edit] Peak {rec['number']}: spectrum unavailable.")
            return
        xc, y = arrays
        # validate uses the flat mu/sigma names; the fit core takes the
        # suffixed per-component names (mu{k}/sigma{k} of the edited component),
        # while the other components are seeded on their current centroids so
        # they stay put through the refit
        suffix = ci + 1
        fixed_c = {(f"mu{suffix}" if k == "mu" else f"sigma{suffix}"): v
                   for k, v in fixed.items()}
        seeds = {f"mu{i + 1}": comp["mu"] for i, comp in enumerate(comps)}
        r = fit_composite(xc, y, float(xx[0]), float(xx[-1]), prev["spec"],
                          fixed=fixed_c, seeds=seeds)
        if not r["ok"]:
            self._peak2_status(f"[failed] edit (Peak {rec['number']}): "
                               f"{r.get('error', 'fit failed')}")
            return
        for art in rec.get("artists") or ():
            try:
                art.remove()
            except Exception:
                pass
        rec["result"] = r
        rec["artists"] = self._peak2_draw(ax, r)
        self._peak2_update_row(rec["number"], r, tag="edited")
        self._peak2_status(f"Peak {rec['number']} (edited): "
                           f"μ = {self._peak2_result_mu(r):.6g}")
        ax.figure.canvas.draw_idle()

    # ------------------------------------------------------------------
    # press dispatch and drag
    # ------------------------------------------------------------------

    def _peak2_try_grab(self, event):
        """Drag-to-refit: if the left-press landed within the pick radius of a
        fit's end-handle, start dragging that window edge. Returns True when a
        drag starts."""
        if self.peak2_drag is not None:
            return False
        ax = event.inaxes
        tol_px = 8.0
        for rec in self.peak2_fits:
            arts = rec.get("artists") or ()
            if not arts or arts[0].axes is not ax:
                continue
            xx = rec["result"].get("xx")
            if xx is None or len(xx) < 2:
                continue
            try:
                lo_px = ax.transData.transform((xx[0], 0.0))[0]
                hi_px = ax.transData.transform((xx[-1], 0.0))[0]
            except Exception:
                continue
            edge = nearest_window_edge(event.x, lo_px, hi_px, tol_px)
            if edge is None:
                continue
            (guide,) = ax.plot([event.xdata, event.xdata], list(ax.get_ylim()),
                               color="tab:green", lw=1.0, ls="--", zorder=5)
            canvas = ax.figure.canvas
            self.peak2_drag = dict(
                rec=rec, edge=edge, ax=ax, guide=guide,
                cid_move=canvas.mpl_connect("motion_notify_event", self._peak2_on_drag_motion),
                cid_up=canvas.mpl_connect("button_release_event", self._peak2_on_drag_release))
            self._peak2_status(f"[drag] Peak {rec['number']}: drag the {edge} edge, "
                               "release to refit.")
            canvas.draw_idle()
            return True
        return False

    def _peak2_on_drag_motion(self, event):
        """Move the dashed guide line to follow the cursor during a drag."""
        d = self.peak2_drag
        if d is None or event.inaxes is not d["ax"] or event.xdata is None:
            return
        try:
            d["guide"].set_xdata([event.xdata, event.xdata])
            d["ax"].figure.canvas.draw_idle()
        except Exception:
            self.logger.debug('_peak2_on_drag_motion failed', exc_info=True)

    def _peak2_on_drag_release(self, event):
        """Release: refit the fit with the dragged edge moved (other edge + μ
        seed kept). A drag is explicit — its failures are always reported and
        never cap-silenced; on failure the previous fit is kept untouched."""
        d = self.peak2_drag
        if d is None:
            return
        ax = d["ax"]
        canvas = ax.figure.canvas
        # tear down the drag interaction first
        for cid in (d["cid_move"], d["cid_up"]):
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                pass
        try:
            d["guide"].remove()
        except Exception:
            pass
        self.peak2_drag = None

        rec, edge = d["rec"], d["edge"]
        new_x = event.xdata
        if new_x is None:
            self._peak2_status(f"[drag] Peak {rec['number']}: cancelled (released off the pad).")
            canvas.draw_idle()
            return
        prev = rec["result"]
        xx = prev["xx"]
        lo, hi = ((float(new_x), float(xx[-1])) if edge == "lo"
                  else (float(xx[0]), float(new_x)))
        arrays = self._peak2_spectrum_arrays(rec["name"])
        if arrays is None:
            self._peak2_status(f"[drag] Peak {rec['number']}: spectrum unavailable.")
            canvas.draw_idle()
            return
        xc, y = arrays
        # auto-components: the new window may now cover extra peaks (add) or
        # have dropped some (shrink) — refit the component set to match it
        r = autocomponent_refit(xc, y, lo, hi, prev)
        if not r["ok"]:
            self._peak2_status(f"[failed] window edit (Peak {rec['number']}): "
                               f"{r.get('error', 'fit failed')}")
            canvas.draw_idle()
            return
        for art in rec.get("artists") or ():
            try:
                art.remove()
            except Exception:
                pass
        rec["result"] = r
        rec["artists"] = self._peak2_draw(ax, r)
        self._peak2_update_row(rec["number"], r, tag="edited")
        ncomp = len(r["components"])
        extra = f" ({ncomp} components)" if ncomp > 1 else ""
        self._peak2_status(f"Peak {rec['number']} (window edited){extra}: "
                           f"μ = {self._peak2_result_mu(r):.6g}")
        canvas.draw_idle()

    def _peak2_try_edit(self, event):
        """Right-click inside a fit's blue fill opens the edit popup for that
        fit. Returns True when a popup opened."""
        ax = event.inaxes
        for rec in self.peak2_fits:
            arts = rec.get("artists") or ()
            if len(arts) < 3 or arts[0].axes is not ax:
                continue
            try:
                hit, _ = arts[2].contains(event)     # the fill (PolyCollection)
            except Exception:
                hit = False
            if hit:
                self._peak2_open_edit(rec, event.xdata)
                return True
        return False

    def onPeakFit2Press(self, event):
        """Unified Peak Finder 2 press handler. Priority: (1) an end-handle
        grab starts a drag; (2) a right-click inside a fit's fill opens the
        edit popup; (3) a left-click while armed (Start or Fix) fits."""
        if event.inaxes is None or event.xdata is None:
            return
        if self._peak2_other_mode_active():
            return
        if event.button == 1 and not event.dblclick and self._peak2_try_grab(event):
            return
        if event.button == 3:
            self._peak2_try_edit(event)
            return
        if event.button != 1 or event.dblclick:
            return
        if not (self.peak2_armed or self.peak2_fix_armed):
            return
        self._peak2_fit_at_press(event)

    def _peak2_fit_at_press(self, event):
        """Armed-mode fit: gaussian+linear around the click on the clicked pad
        (fix-μ when Fix Peak is armed, automatic window otherwise), draw the
        curve + dashed background + blue net-area fill, and add a results row.
        Guards are owned by the `onPeakFit2Press` dispatcher."""
        try:
            if "colorbar_" in event.inaxes.get_label():
                return
            # resolve the clicked pad (same rule as on_press)
            index = list(self._get_current_plot().figure.axes).index(event.inaxes)
            if self._get_current_plot().isEnlarged:
                index = self._tabs.selectedPad(self._tabs.currentIndex())

            name = self._name_from_index(index)
            if not name:
                self._peak2_status("[skip] Clicked pad holds no spectrum.")
                return
            if self._get_store_info("dim", index=index) != 1:
                self._peak2_status("[skip] Peak Finder works on 1D spectra only.")
                return

            binx     = self._get_store_info("binx", index=index)
            minxREST = self._get_store_info("minx", index=index)
            maxxREST = self._get_store_info("maxx", index=index)
            xtmp = self._plot_controller.createRange(binx, minxREST, maxxREST)
            ytmp = self._get_store_info("data", index=index)
            # bin centres to match the counts array; the fit window is chosen
            # automatically from the data around the click (plan A)
            xc = np.asarray(xtmp[:-1]) + 0.5 * np.diff(np.asarray(xtmp))

            # Config cap (bins -> x units); with a cap set, unfittable clicks
            # are skipped silently by design
            cap_bins = self._peak2_max_window_bins()
            bw = float(maxxREST - minxREST) / float(binx)
            max_hw = 0.5 * cap_bins * bw if cap_bins else None

            cx = float(event.xdata)
            spec = self._peak2_current_spec()
            if self.peak2_fix_armed:
                # Fix Peak: pin μ at the clicked x over a window centred on the
                # click (cap sizes it, else 50 bins) — never estimate_fit_window,
                # which would snap onto a bigger neighbour. Failures are always
                # reported here; the cap only sizes the window, it does not
                # silence the click.
                lo, hi = fix_peak_window(cx, bw, cap_bins=cap_bins)
                r = fit_composite(xc, np.asarray(ytmp)[1:], lo, hi, spec,
                                  fixed={"mu1": cx})
                tag = "fixed μ"
                if not r["ok"]:
                    self._peak2_status(f"[failed] {r.get('error', 'fit failed')}")
                    return
            else:
                r = fit_composite_auto(xc, np.asarray(ytmp)[1:], cx, spec,
                                       max_half_window=max_hw)
                tag = None
                if not r["ok"]:
                    if cap_bins:
                        self.logger.debug('_peak2_fit_at_press - capped fit skipped: %s',
                                          r.get('error'))
                        return
                    # failures don't consume a peak number
                    self._peak2_status(f"[failed] {r.get('error', 'fit failed')}")
                    return
                # duplicate suppression (auto mode only): an off-peak flank
                # click re-fits an already-fitted peak on the same spectrum;
                # skip it if the centroid lands within ~1 bin of an existing fit.
                # Both sides use the primary component, so the comparison is
                # between the peaks the two fits are reported by; with the
                # one-component default that is the only component there is.
                same = [rec for rec in self.peak2_fits if rec.get("name") == name]
                new_mu = self._peak2_result_mu(r)
                dup = find_duplicate_mu(
                    new_mu, [self._peak2_result_mu(rec["result"]) for rec in same], bw)
                if dup is not None:
                    self._peak2_status(f"[skip] already fitted near μ = {new_mu:.6g} "
                                       f"(Peak {same[dup]['number']}).")
                    return
            self.peak2_count += 1
            self._peak2_add_row(self.peak2_count, r, tag=tag)

            artists = self._peak2_draw(event.inaxes, r)
            # full record (data included) so the fit can be redrawn or refit
            # later without depending on artist survival
            self.peak2_fits.append(dict(number=self.peak2_count, index=index,
                                        name=name, result=r, artists=artists))
            # keep the handler alive on this canvas even after Stop, so future
            # drag/edit interactions on existing fits keep working
            self._peak2_connect(event.inaxes.figure.canvas)
            self._get_current_plot().canvas.draw_idle()
        except Exception:
            # a click must never crash the GUI; report instead
            self.logger.exception('_peak2_fit_at_press - fit failed')
            self._peak2_status("[error] Fit failed — see log.")

    # ------------------------------------------------------------------
    # spectrum data and pad liveness
    # ------------------------------------------------------------------

    def _peak2_spectrum_arrays(self, name):
        """(xc, y) — bin-centre x and counts for the spectrum called `name`,
        or None. Mirrors the fit handler's array setup; used by drag-refit,
        the edit popup and the shape-menu refit."""
        # Ask the store outright rather than inferring absence from the TypeError
        # a None binx would raise downstream: a removed spectrum is an expected
        # state, not an error, and the explicit test cannot be defeated by a
        # record that survives removal with some fields still readable.
        if not self._spectra.contains(name):
            self.logger.debug('_peak2_spectrum_arrays - %s is no longer in the store', name)
            return None
        try:
            binx = self._get_store_info("binx", name=name)
            minx = self._get_store_info("minx", name=name)
            maxx = self._get_store_info("maxx", name=name)
            xtmp = self._plot_controller.createRange(binx, minx, maxx)
            ytmp = self._get_store_info("data", name=name)
            xc = np.asarray(xtmp[:-1]) + 0.5 * np.diff(np.asarray(xtmp))
            return xc, np.asarray(ytmp)[1:]
        except Exception:
            self.logger.debug('_peak2_spectrum_arrays failed for %s', name, exc_info=True)
            return None

    @staticmethod
    def _peak2_live_axes(rec):
        """The axes this fit is still drawn on, or None once the pad went away
        under it. Two teardowns have to be caught and they leave different
        wreckage."""
        arts = rec.get("artists") or ()
        ax = arts[0].axes if arts else None
        if ax is None or ax.figure is None:
            return None
        return ax if ax in ax.figure.axes else None

    # ------------------------------------------------------------------
    # widget reads
    # ------------------------------------------------------------------

    def _peak2_status(self, msg):
        """Show the latest Peak Finder 2 status/feedback line (armed/config/
        skip/failed/error)."""
        self._peak.peak2_status.setText(msg)

    def _peak2_current_spec(self):
        """The fit spec selected in the shape menus. Single component here; a
        multi-component fit only arises from the auto-add-on-drag."""
        p = self._peak
        return {
            "signal": self._PEAK2_SIGNAL_BY_LABEL.get(p.peak2_signal.currentText(),
                                                      "gaussian"),
            "n_components": 1,
            "background": self._PEAK2_BG_BY_LABEL.get(p.peak2_bg.currentText(),
                                                      "poly1"),
            "tail_side": p.peak2_cb_tail.currentText(),
        }

    def _peak2_max_window_bins(self):
        """The Config cap (max fit window in bins), or None when unset."""
        try:
            raw = QSettings().value("PeakFinder2/max_window_bins", "", type=str)
            n = int(raw)
            return n if n > 0 else None
        except Exception:
            return None

    @staticmethod
    def _peak2_result_mu(r):
        """μ of a composite fit's primary component — the same one the results
        table quotes, so the status line and the row never name different peaks
        for one fit. Single-component fits are unaffected either way."""
        return primary_component(r)["mu"]

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------

    def _peak2_draw(self, ax, r):
        """Draw one fit result on `ax` (curve + dashed bg + blue net-area fill +
        square end-handles, plus thin dashed per-component curves when the fit
        has more than one component) and return the artist tuple; the curve is
        always index 0 and the fill index 2 (drag-grab / edit-hit rely on that).
        Factored out so a fit record can be redrawn from its stored curves at
        any time (lifecycle safety)."""
        (curve,) = ax.plot(r["xx"], r["y_fit"], color="tab:red", lw=1.8)
        (bgline,) = ax.plot(r["xx"], r["y_bg"], color="grey", lw=1.0, ls="--")
        fill = ax.fill_between(r["xx"], r["y_bg"], r["y_fit"],
                               where=r["y_fit"] >= r["y_bg"],
                               color="tab:blue", alpha=0.45)
        # square end-handles (grab targets for drag-to-refit)
        (handles,) = ax.plot([r["xx"][0], r["xx"][-1]],
                             [r["y_fit"][0], r["y_fit"][-1]],
                             marker="s", ms=6, ls="None",
                             color="tab:red", mec="black", mew=0.6, zorder=6)
        # per-component overlays (each drawn over the background); only when the
        # fit is a genuine multi-component one — a single component == the curve
        comps = []
        y_comp = r.get("y_comp") or []
        if len(y_comp) > 1:
            for yc in y_comp:
                (ln,) = ax.plot(r["xx"], yc, color="tab:red", lw=0.8, ls=":")
                comps.append(ln)
        # Nothing reads this gid. Every current path finds fit artists through
        # the fit records instead, each of which holds its own artist tuple,
        # so the tag is part of no lifecycle here and is not load-bearing
        # however much it looks it.
        for art in (curve, bgline, fill, handles, *comps):
            if hasattr(art, "set_gid"):
                art.set_gid("peakfit2")
        return (curve, bgline, fill, handles, *comps)

    def peakFit2RedrawAll(self):
        """Redraw every recorded fit from its stored curves onto its pad's
        current axis (recovers from axis rebuilds, e.g. enlarge/un-enlarge)."""
        fits = self.peak2_fits
        self.logger.info('peakFit2RedrawAll - %d record(s)', len(fits))
        canvases = set()
        for rec in fits:
            for art in rec.get("artists") or ():
                try:
                    art.remove()
                except Exception:
                    self.logger.debug('peakFit2RedrawAll - stale artist', exc_info=True)
            try:
                ax = self._get_view_info("axis", index=rec["index"])
            except Exception:
                ax = None
            if ax is None:
                rec["artists"] = ()
                continue
            rec["artists"] = self._peak2_draw(ax, rec["result"])
            canvases.add(ax.figure.canvas)
        for canvas in canvases:
            try:
                canvas.draw_idle()
            except Exception:
                self.logger.debug('peakFit2RedrawAll - redraw failed', exc_info=True)
