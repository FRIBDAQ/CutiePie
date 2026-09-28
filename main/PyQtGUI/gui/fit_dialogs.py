"""The dialogs the fit path opens, kept out of the fit service.

One class, constructed with the window the dialogs are parented to. Each
method builds one dialog and returns a plain value; persisted settings and
the fit's own state stay with the caller.
"""

from types import SimpleNamespace

import numpy as np

from PyQt5.QtCore import Qt, QEventLoop, QTimer
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
    QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)


class FitDialogs:
    """Builds the fit path's dialogs on behalf of FitManager."""

    def __init__(self, parent_widget=None):
        self._parent = parent_widget
        self._cal = None                  # the active calibration session, or None

    def prompt_shape_flags(self, default_global=True, default_iso_scales=False,
                           default_chain=True):
        """Modal "Shape fitting options" dialog. Returns the three flags as a
        dict, or None when cancelled."""
        dlg = QDialog(self._parent); dlg.setWindowTitle("Shape fitting options")
        lbl = QLabel("Choose how to treat peak shapes:")

        cb_global = QCheckBox("Fit global shape (σ, τ₁, τ₂, η)")
        cb_global.setChecked(bool(default_global))

        cb_iso = QCheckBox("Enable per-isotope scale multipliers (requires global)")
        cb_iso.setChecked(bool(default_iso_scales and default_global))
        cb_iso.setEnabled(cb_global.isChecked())

        cb_chain = QCheckBox("Normalize intensities within decay chain")
        cb_chain.setChecked(default_chain)

        def on_global_toggled(on):
            if not on:
                cb_iso.setChecked(False)
            cb_iso.setEnabled(on)

        cb_global.toggled.connect(on_global_toggled)

        btn_ok = QPushButton("OK"); btn_cancel = QPushButton("Cancel")
        btn_ok.clicked.connect(dlg.accept); btn_cancel.clicked.connect(dlg.reject)

        v = QVBoxLayout(dlg)
        v.addWidget(lbl)
        v.addWidget(cb_global)
        v.addWidget(cb_iso)
        v.addWidget(cb_chain)

        h = QHBoxLayout()
        h.addStretch(1)
        h.addWidget(btn_ok)
        h.addWidget(btn_cancel)
        v.addLayout(h)

        if dlg.exec_() != QDialog.Accepted:
            return None

        return {
            "fit_global_shapes": cb_global.isChecked(),
            "fit_iso_shape_scales": (cb_global.isChecked() and cb_iso.isChecked()),
            "normalize_chains": cb_chain.isChecked()
        }

    def choose_shape_file(self, start_dir):
        """Open-file chooser for a shape file, starting in `start_dir`.
        Returns the path, or "" when cancelled."""
        path, _ = QFileDialog.getOpenFileName(
            self._parent, "Choose shape file", start_dir,
            "Text/CSV files (*.txt *.csv);;All files (*)"
        )
        return path or ""

    def choose_calibration_file(self, start_dir):
        """Open-file chooser for a calibration file, starting in `start_dir`.
        Returns the path, or "" when cancelled."""
        path, _ = QFileDialog.getOpenFileName(
            self._parent, "Calibration file", start_dir,
            "Text/CSV/JSON (*.txt *.csv *.json);;All files (*)"
        )
        return path or ""

    def ask_run_calibration(self):
        """Ask whether to run energy calibration before the fit. Returns
        ("yes" | "no" | "cancel", dont_ask_again)."""
        msg = QMessageBox(self._parent)
        msg.setWindowTitle("Energy calibration")
        msg.setText("Run energy calibration before fitting?")
        btn_yes    = msg.addButton("Yes",    QMessageBox.YesRole)
        btn_no     = msg.addButton("No",     QMessageBox.NoRole)
        btn_cancel = msg.addButton("Cancel", QMessageBox.RejectRole)
        dont_ask = QCheckBox("Don't ask me again")
        msg.setCheckBox(dont_ask)

        msg.exec_()
        clicked = msg.clickedButton()
        if clicked is btn_cancel:
            choice = "cancel"
        elif clicked is btn_yes:
            choice = "yes"
        else:
            choice = "no"
        try:
            ticked = bool(dont_ask.isChecked())
        except Exception:
            ticked = False
        return choice, ticked

    def open_loaded_fit_panel(self, ax, structure, group, logger):
        """Modeless panel bound to a loaded fit `group`: a chain → isotope sum →
        peaks tree whose checkboxes add/remove that component on the pad live.
        The caller owns the returned dialog and closes it."""
        dlg = QDialog(self._parent)
        dlg.setWindowTitle("Loaded fit components")
        dlg.setWindowModality(Qt.NonModal)
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Tick to add a component, untick to remove it:"))

        inner = QWidget()
        iv = QVBoxLayout(inner)
        checks = {}

        def _redraw():
            try:
                ax.figure.canvas.draw_idle()
            except Exception:
                logger.debug('loaded-fit panel - could not draw', exc_info=True)

        def _add(name, label, indent):
            cb = QCheckBox(label)
            cb.setChecked(True)                 # group is already fully drawn
            if indent:
                cb.setStyleSheet(f"margin-left: {indent}px;")
            # connect AFTER the initial setChecked so it doesn't fire on build
            cb.toggled.connect(lambda on, nm=name: (group.set(nm, on), _redraw()))
            checks[name] = cb
            iv.addWidget(cb)

        for d in structure:
            if d["kind"] == "total":
                _add(d["name"], "fit total", 0)

        chains = {}
        for d in structure:
            if d["kind"] in ("isotope", "peak", "component"):
                chains.setdefault(d["chain"] or "Unchained", []).append(d)
        for ch in sorted(chains):
            iv.addWidget(QLabel(f"<b>{ch}</b>"))
            for d in chains[ch]:
                if d["kind"] == "peak":
                    e = d["E"]
                    label = f"{d['isotope']} · {e:.0f} keV" if e is not None else d["name"]
                    _add(d["name"], label, 40)
                else:                              # isotope sum (or bare component)
                    _add(d["name"], f"{d['isotope']} (sum)", 16)
        iv.addStretch(1)

        # An isotope-sum checkbox is a parent: toggling it toggles all its peaks.
        for d in structure:
            if d["kind"] != "isotope":
                continue
            sum_cb = checks.get(d["name"])
            kids = [checks[p["name"]] for p in structure
                    if p["kind"] == "peak"
                    and p["chain"] == d["chain"] and p["isotope"] == d["isotope"]
                    and p["name"] in checks]
            if sum_cb is not None and kids:
                sum_cb.toggled.connect(
                    lambda on, kids=kids: [k.setChecked(on) for k in kids])

        area = QScrollArea()
        area.setWidget(inner)
        area.setWidgetResizable(True)
        v.addWidget(area)

        hb = QHBoxLayout()
        btn_all = QPushButton("All")
        btn_none = QPushButton("None")
        btn_all.clicked.connect(lambda: [c.setChecked(True) for c in checks.values()])
        btn_none.clicked.connect(lambda: [c.setChecked(False) for c in checks.values()])
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.close)
        hb.addWidget(btn_all)
        hb.addWidget(btn_none)
        hb.addStretch(1)
        hb.addWidget(btn_close)
        v.addLayout(hb)

        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        return dlg

    def prompt_energy_calibration(self, ax, x, y, *, min_pts=2, max_pts=4,
                                  snap_halfwin=150, logger, on_session):
        """Non-modal window that collects (μ, E) pairs by ctrl+click and fits
        μ = a·E + b. `on_session` receives the live session while points are
        being collected and None when the window closes, so the caller can
        route canvas clicks to it."""
        dlg = QDialog(self._parent)
        dlg.setWindowTitle("Energy calibration")
        info = QLabel(
            f"Ctrl + Left-click on the spectrum to pick μ.\n"
            f"Enter energy when prompted. Pick {min_pts}–{max_pts} points.\n"
            "Use Undo to remove last point; Done to finish."
        )
        status = QLabel("0 points")

        btn_undo = QPushButton("Undo")
        btn_done = QPushButton("Done")
        btn_cancel = QPushButton("Cancel")
        btn_done.setEnabled(False)

        hl = QHBoxLayout(); hl.addWidget(btn_undo); hl.addWidget(btn_done); hl.addWidget(btn_cancel)
        lay = QVBoxLayout(dlg); lay.addWidget(info); lay.addWidget(status); lay.addLayout(hl)

        self._cal = SimpleNamespace(
            ax=ax, x=np.asarray(x), y=np.asarray(y),
            min_pts=int(min_pts), max_pts=int(max_pts), halfwin=float(snap_halfwin),
            MU=[], E=[], artists=[], dlg=dlg, status=status, pending=False
        )
        cal = self._cal
        on_session(cal)

        def _snap_mu(x0):
            x = self._cal.x; y = self._cal.y
            m = (x >= x0 - self._cal.halfwin) & (x <= x0 + self._cal.halfwin)
            if np.any(m):
                idx = np.argmax(y[m])
                mu = x[m][idx]; yy = y[m][idx]
            else:
                mu = float(x0); yy = float(np.interp(mu, x, y))
            return mu, yy

        def _update_status():
            n = len(self._cal.MU)
            self._cal.status.setText(
                f"{n} point(s): " + ", ".join(f"μ={mu:.1f}@{E:.2f}keV"
                                            for mu, E in zip(self._cal.MU, self._cal.E))
            )
            btn_done.setEnabled(n >= self._cal.min_pts)

        def _prompt_point(xdata):
            # Runs from a zero-delay timer, never inside the canvas press
            # handler. A modal dialog created while the mouse button is still
            # held can stay unmapped on some platforms (seen on WSL), and an
            # application-modal dialog nobody can see is a frozen GUI.
            try:
                if self._cal is not cal:
                    return
                if len(cal.MU) >= cal.max_pts:
                    QMessageBox.information(self._parent, "Max points",
                                            f"Already have {cal.max_pts} points.")
                    return
                mu, yy = _snap_mu(xdata)
                val, ok = QInputDialog.getDouble(self._parent, "Energy (keV)", "Energy:",
                                                 0.0, -1e9, 1e9, 6)
                if not ok or self._cal is not cal:
                    return
                cal.MU.append(float(mu)); cal.E.append(float(val))
                m1, = ax.plot([mu], [yy], marker='o', ms=6)
                m2 = ax.axvline(mu, ls=':', lw=1.0)
                cal.artists.append((m1, m2))
                ax.figure.canvas.draw_idle()
                _update_status()
            finally:
                cal.pending = False

        def _add_point(xdata):
            if cal.pending:
                return
            cal.pending = True
            QTimer.singleShot(0, lambda: _prompt_point(xdata))
        cal.add_point = _add_point

        def _undo():
            if not self._cal.MU: return
            self._cal.MU.pop(); self._cal.E.pop()
            for a in self._cal.artists.pop():
                try: a.remove()
                except Exception: logger.debug('could not remove calibration artist', exc_info=True)
            ax.figure.canvas.draw_idle()
            _update_status()

        def _finish():
            n = len(self._cal.MU)
            if not (self._cal.min_pts <= n <= self._cal.max_pts):
                QMessageBox.information(dlg, "Need more points",
                                        f"Pick between {self._cal.min_pts} and {self._cal.max_pts} points.")
                return

            MU = np.asarray(self._cal.MU, float); E = np.asarray(self._cal.E, float)
            A = np.vstack([E, np.ones_like(E)]).T
            a, b = np.linalg.lstsq(A, MU, rcond=None)[0]
            mu_hat = a*E + b
            ss_res = float(np.sum((MU - mu_hat)**2))
            ss_tot = float(np.sum((MU - MU.mean())**2)) if MU.size else 1.0
            R2 = 1.0 - ss_res/max(ss_tot, 1e-16)
            self._cal.result = dict(a=float(a), b=float(b), R2=float(R2),
                                    E=E.tolist(), mu=MU.tolist())
            dlg.accept()

        def _cancel():
            self._cal.result = None
            dlg.reject()

        btn_undo.clicked.connect(_undo)
        btn_done.clicked.connect(_finish)
        btn_cancel.clicked.connect(_cancel)

        try:
            dlg.setWindowModality(Qt.NonModal)
            dlg.show()

            loop = QEventLoop()
            dlg.finished.connect(loop.quit)
            loop.exec_()
            return getattr(self._cal, "result", None)

        finally:
            for arts in list(self._cal.artists):
                for a in arts:
                    try: a.remove()
                    except Exception: logger.debug('could not remove fit artist', exc_info=True)
            ax.figure.canvas.draw_idle()
            self._cal = None
            on_session(None)
