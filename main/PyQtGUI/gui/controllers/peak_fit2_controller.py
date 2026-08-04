"""Peak Finder 2 (click-to-fit): the shared floor.

Lifted out of MainWindow (ARCH.md §7 D8, FACTORIZATION.md stage 8a). This is
the first of four sub-stages: what lives here is what every other part of the
cluster calls — drawing a fit, fetching the spectrum a fit belongs to, deciding
whether the pad it was drawn on is still alive, reading the shape menus and the
Config cap, the status line, and the redraw sweep. The press dispatch, the drag
handling, the results table and the edit popup follow in 8b through 8d and land
on this same object.

Because the state does not partition cleanly — `peak2_fits` is read from the
groups still on MainWindow — the fit records stay on the window for now and are
reached through the `get_fits` seam. That seam disappears in 8d, when the last
group moves and the records come with it.

Two things here are load-bearing and easy to undo by accident.

`_peak2_spectrum_arrays` resolves by spectrum NAME, never by pad index. A fit
records the spectrum it was made on, and a refit can fire while a different tab
is up or after the geometry moved that spectrum, so resolving the index again
would hand back a different spectrum's counts.

`_peak2_live_axes` has to reject two different teardowns. Removing a spectrum
clears its pad, which nulls every cleared artist's `.axes`; applying a geometry
instead detaches the axes, which leaves both `artist.axes` and `axes.figure`
pointing at real objects and only drops the axes out of `figure.axes`. A plain
None test waves the second one through, and the refit then draws onto a pad
that is no longer part of the figure — invisible, and reported as success.
"""

import logging

import numpy as np

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QInputDialog

from services.peak_finder import (autocomponent_refit, find_duplicate_mu,
                                  fit_composite, fit_composite_auto,
                                  fix_peak_window, nearest_window_edge,
                                  primary_component)


class PeakFit2Controller:

    _PEAK2_SIGNAL_BY_LABEL = {"Gaussian": "gaussian", "Crystal ball": "crystal_ball"}
    _PEAK2_BG_BY_LABEL = {"Linear": "poly1", "Quadratic": "poly2", "Cubic": "poly3"}

    def __init__(self, peak_tab, spectra, get_store_info, get_view_info,
                 plot_controller, get_fits, tabs=None, get_current_plot=None,
                 get_gate_popup=None, get_sum_popup=None,
                 name_from_index=None, get_count=None, set_count=None,
                 add_row=None, update_row=None, open_edit=None,
                 parent_widget=None, logger=None):
        self._peak            = peak_tab          # extraPopup.peak
        self._spectra         = spectra
        self._get_store_info  = get_store_info
        self._get_view_info   = get_view_info
        self._plot_controller = plot_controller
        # the fit records still live on MainWindow until 8d; read them, never
        # rebind them
        self._get_fits        = get_fits
        self._tabs            = tabs
        self._get_current_plot = get_current_plot
        # late-bound: both popups are MainWindow attributes that can be
        # replaced, and a torn-down one must read as "not blocking"
        self._get_gate_popup  = get_gate_popup
        self._get_sum_popup   = get_sum_popup
        self._name_from_index = name_from_index
        # the running peak number is reset by peakFit2Clear, still 8d's; these
        # two seams retire when that group moves
        self._get_count       = get_count
        self._set_count       = set_count
        # 8d's results-table methods, late-bound for the same reason
        self._add_row         = add_row
        self._update_row      = update_row
        self._open_edit       = open_edit
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
        for rec in self._get_fits():
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
    # press dispatch and drag
    # ------------------------------------------------------------------

    def _peak2_try_grab(self, event):
        """Drag-to-refit: if the left-press landed within the pick radius of a
        fit's end-handle, start dragging that window edge. Returns True when a
        drag starts. Works whether or not Start is armed."""
        if self.peak2_drag is not None:
            return False
        ax = event.inaxes
        tol_px = 8.0
        for rec in self._get_fits():
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
        self._update_row(rec["number"], r, tag="edited")
        ncomp = len(r["components"])
        extra = f" ({ncomp} components)" if ncomp > 1 else ""
        self._peak2_status(f"Peak {rec['number']} (window edited){extra}: "
                           f"μ = {self._peak2_result_mu(r):.6g}")
        canvas.draw_idle()

    def _peak2_try_edit(self, event):
        """Right-click inside a fit's blue fill opens the edit popup for that
        fit. Returns True when a popup opened."""
        ax = event.inaxes
        for rec in self._get_fits():
            arts = rec.get("artists") or ()
            if len(arts) < 3 or arts[0].axes is not ax:
                continue
            try:
                hit, _ = arts[2].contains(event)     # the fill (PolyCollection)
            except Exception:
                hit = False
            if hit:
                self._open_edit(rec, event.xdata)
                return True
        return False

    def onPeakFit2Press(self, event):
        """Unified Peak Finder 2 press handler. Priority: (1) an end-handle grab
        starts a drag; (2) a right-click inside a fit's fill opens the edit
        popup; (3) a left-click while armed (Start or Fix) fits. Zoom / gate /
        summing-region presses are never treated as any of those."""
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
                same = [rec for rec in self._get_fits() if rec.get("name") == name]
                new_mu = self._peak2_result_mu(r)
                dup = find_duplicate_mu(
                    new_mu, [self._peak2_result_mu(rec["result"]) for rec in same], bw)
                if dup is not None:
                    self._peak2_status(f"[skip] already fitted near μ = {new_mu:.6g} "
                                       f"(Peak {same[dup]['number']}).")
                    return
            self._set_count(self._get_count() + 1)
            self._add_row(self._get_count(), r, tag=tag)

            artists = self._peak2_draw(event.inaxes, r)
            # full record (data included) so the fit can be redrawn or refit
            # later without depending on artist survival
            self._get_fits().append(dict(number=self._get_count(), index=index,
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
        """(xc, y) — bin-centre x and counts for the spectrum called `name`, or
        None. Mirrors the fit handler's array setup; used by drag-refit, the
        edit popup and the shape-menu refit.

        Keyed by NAME, never by pad index: a pad index only means anything
        against the tab that is currently showing (nameFromIndex reads
        currentPlot.h_dict_geo, and answers with the enlarged spectrum for any
        index while a pad is enlarged). A fit records the spectrum it was made
        on, and a refit triggered from the popup can happen while a different
        tab is up or after the geometry moved that spectrum, so resolving the
        index again would hand back a different spectrum's counts. The store is
        name-keyed and tab-independent, so this stays correct either way; an
        unknown name yields None and the callers report 'spectrum unavailable'."""
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
        under it.

        Two teardowns have to be caught and they leave different wreckage.
        Removing a spectrum clears its pad (`_on_spectrum_removed_rest` calls
        `ax.clear()`), which sets every cleared artist's `.axes` to None.
        Applying a new geometry instead DETACHES the old axes
        (`InitializeCanvas` runs `figure.delaxes`), which leaves both
        `artist.axes` and `axes.figure` pointing at real objects and only drops
        the axes out of `figure.axes`. So a plain None test waves the geometry
        case straight through, and the refit then draws onto a pad that is no
        longer part of the figure — invisible, and reported as success.
        Attachment is the test that catches both."""
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
        # the fit records instead, each of which holds its own artist tuple, so
        # the tag is part of no lifecycle here and is not load-bearing however
        # much it looks it. It is kept rather than deleted for the deferred
        # redraw-on-zoom work: that has to cope with artists orphaned on a pad
        # whose axes was rebuilt underneath us, and a tag on the artist is the
        # only handle on those once the records point at dead objects. If that
        # work lands without needing it, delete it then.
        for art in (curve, bgline, fill, handles, *comps):
            if hasattr(art, "set_gid"):
                art.set_gid("peakfit2")
        return (curve, bgline, fill, handles, *comps)

    def peakFit2RedrawAll(self):
        """Redraw every recorded fit from its stored curves onto its pad's
        current axis (recovers from axis rebuilds, e.g. enlarge/un-enlarge)."""
        fits = self._get_fits()
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
