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

from services.peak_finder import primary_component


class PeakFit2Controller:

    _PEAK2_SIGNAL_BY_LABEL = {"Gaussian": "gaussian", "Crystal ball": "crystal_ball"}
    _PEAK2_BG_BY_LABEL = {"Linear": "poly1", "Quadratic": "poly2", "Cubic": "poly3"}

    def __init__(self, peak_tab, spectra, get_store_info, get_view_info,
                 plot_controller, get_fits, logger=None):
        self._peak            = peak_tab          # extraPopup.peak
        self._spectra         = spectra
        self._get_store_info  = get_store_info
        self._get_view_info   = get_view_info
        self._plot_controller = plot_controller
        # the fit records still live on MainWindow until 8d; read them, never
        # rebind them
        self._get_fits        = get_fits
        self.logger = logger or logging.getLogger(__name__)

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
