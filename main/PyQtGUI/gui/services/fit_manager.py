import logging
import os
import re
import json
import matplotlib
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

from PyQt5.QtWidgets import (
    QMessageBox, QFileDialog, QDialog, QLabel, QPushButton, QCheckBox,
    QHBoxLayout, QVBoxLayout, QInputDialog, QTextEdit, QApplication,
)
from PyQt5.QtCore import Qt, QSettings, QEventLoop

from alpha_filter_dialog import AlphaChainIsoFilterDialog

FIT_PREFIX = "fit-_-"


class FitManager:
    """Owns all fitting operations: execute, CSV, manage artists, result popups."""

    FIT_PREFIX = FIT_PREFIX

    def __init__(self, fit_factory, spectra, extra_popup, parent_widget=None, logger=None):
        self._parent_widget = parent_widget  # Qt dialog parent only — no domain calls
        self._factory = fit_factory
        self._spectra = spectra
        self._popup   = extra_popup
        self.logger   = logger or logging.getLogger(__name__)
        # instance state
        self._abort_fit        = False
        self._use_csv_fit      = False
        self._csv_x            = None
        self._csv_y            = None
        self._csv_path         = None
        self._csv_ax           = None
        self._cal              = None
        self._alphaFilterDlg   = None
        self._lastFitResultsText = None

    @staticmethod
    def _create_range(bins, vmin, vmax):
        step = (float(vmax) - float(vmin)) / float(bins)
        return [float(vmin)] + list(np.arange(float(vmin), float(vmax), step) + step)

    # ------------------------------------------------------------------
    # Axis limits helper
    # ------------------------------------------------------------------

    def axisLimitsForFit(self, ax):
        left, right = ax.get_xlim()
        self.logger.info('axisLimitsForFit - left, right: %s, %s', left, right)
        if self._popup.fit_range_min.text():
            try:
                left = float(self._popup.fit_range_min.text())
            except ValueError:
                self.logger.warning('axisLimitsForFit - Invalid input for Min X. Please enter a valid number.')
        else:
            left = ax.get_xlim()[0]
        if self._popup.fit_range_max.text():
            try:
                right = float(self._popup.fit_range_max.text())
            except ValueError:
                self.logger.warning('axisLimitsForFit - Invalid input for Max X. Please enter a valid number.')
        else:
            right = ax.get_xlim()[1]
        if left > right:
            left, right = right, left
            QMessageBox.about(self._parent_widget, "Warning", "xmin > xmax, the provided limits will be swapped for the fit")
        return left, right

    # ------------------------------------------------------------------
    # Abort
    # ------------------------------------------------------------------

    def on_abort_clicked(self):
        self._abort_fit = True
        try: self._popup.abort_button.setEnabled(False)
        except Exception: pass
        try: self._popup.fit_results.append("[abort] Requested…")
        except Exception: pass

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def on_fit_csv_clicked(self):
        if not hasattr(self, "_csv_x") or self._csv_x is None or len(self._csv_x) == 0:
            QMessageBox.warning(self._parent_widget, "No CSV loaded", "Click 'Plot CSV' first.")
            return
        if not hasattr(self, "_csv_ax") or self._csv_ax is None:
            QMessageBox.warning(self._parent_widget, "No CSV plot", "Click 'Plot CSV' first.")
            return

        self._use_csv_fit = True
        try:
            self.fit()
        finally:
            self._use_csv_fit = False

    def on_plot_csv_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self._parent_widget, "Select CSV for plotting (x,y)", "",
            "CSV files (*.csv *.txt);;All files (*)"
        )
        if not path:
            return

        self._csv_path = path

        arr = np.genfromtxt(path, delimiter=",", comments="#")
        if arr.ndim == 1:
            x = np.arange(arr.size, dtype=float)
            y = arr.astype(float)
        else:
            x = arr[:, 0].astype(float)
            y = arr[:, 1].astype(float)

        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        o = np.argsort(x)
        x, y = x[o], y[o]

        self._csv_x, self._csv_y = x, y

        if not hasattr(self, "_csv_ax") or self._csv_ax is None:
            fig, ax = plt.subplots()
            try: fig.canvas.manager.set_window_title("CSV")
            except Exception: pass
            self._csv_ax = ax

        ax = self._csv_ax
        ax.clear()
        ax.plot(self._csv_x, self._csv_y, lw=1)
        ax.set_title(os.path.basename(path))
        ax.relim()
        ax.autoscale_view()
        ax.figure.canvas.draw_idle()
        ax.figure.show()

    # ------------------------------------------------------------------
    # Main fit entry point
    # ------------------------------------------------------------------

    def fit(self, index=None, name=None, ax=None):
        self.logger.info('fit')

        fit_funct = self._popup.fit_list.currentText().strip()

        self._close_alpha_filter_popup()

        model_name = fit_funct
        force_prompt = bool(True)

        try:
            config = self.prepare_fit_config(fit_funct, force_prompt=force_prompt)
        except ValueError as e:
            QMessageBox.warning(self._parent_widget, "Fit cancelled", str(e))
            return

        self._abort_fit = False
        self._popup.fit_button.setEnabled(False)
        self._popup.abort_button.setEnabled(True)

        use_csv = getattr(self, "_use_csv_fit", False)
        if use_csv:
            ax = self._csv_ax
            spectrumName = f"CSV: {os.path.basename(getattr(self, '_csv_path', 'data.csv'))}"
            index = None
        else:
            if name is None or ax is None:
                self.logger.warning('fit - called without name/ax context; cannot fit histogram')
                return
            spectrumName = name

        self.logger.debug('fit - spectrumName, fit_funct, index: %s, %s, %s', spectrumName, fit_funct, index)

        if not use_csv:
            self._clear_all_fit_artists_in_figure(ax.figure)

        before_ids = self._snap_ax_ids(ax)

        if use_csv:
            dim = 1
            x = np.asarray(self._csv_x, float)
            y = np.asarray(self._csv_y, float)
            binx = len(x)
            minxREST = float(np.nanmin(x))
            maxxREST = float(np.nanmax(x))
        else:
            dim      = self._spectra.get(spectrumName, "dim")
            binx     = self._spectra.get(spectrumName, "binx")
            minxREST = self._spectra.get(spectrumName, "minx")
            maxxREST = self._spectra.get(spectrumName, "maxx")

        try:
            if spectrumName != "":
                if dim == 1:
                    widgets = (
                        self._popup.fit_p0, self._popup.fit_p1, self._popup.fit_p2,
                        self._popup.fit_p3, self._popup.fit_p4, self._popup.fit_p5,
                        self._popup.fit_p6, self._popup.fit_p7, self._popup.fit_p8,
                        self._popup.fit_p9, self._popup.fit_p10, self._popup.fit_p11,
                        self._popup.fit_p12, self._popup.fit_p13, self._popup.fit_p14,
                        self._popup.fit_p15, self._popup.fit_p16, self._popup.fit_p17,
                        self._popup.fit_p18, self._popup.fit_p19
                    )

                    fitpar = []
                    for w in widgets:
                        t = w.text().strip() if (w is not None) else ""
                        try:
                            fitpar.append(float(t) if t != "" else None)
                        except Exception:
                            fitpar.append(None)

                    if use_csv:
                        xmin, xmax = self.axisLimitsForFit(ax)

                        xdata_min = float(np.nanmin(x))
                        xdata_max = float(np.nanmax(x))
                        lo, hi = sorted([float(xmin), float(xmax)])
                        xmin = max(lo, xdata_min)
                        xmax = min(hi, xdata_max)

                        if (not np.isfinite(xmin)) or (not np.isfinite(xmax)) or (xmin >= xmax):
                            xmin, xmax = xdata_min, xdata_max

                        m = (x >= xmin) & (x <= xmax)
                        if np.count_nonzero(m) < 3:
                            QMessageBox.warning(self._parent_widget, "Fit range too small",
                                                "Min X / Max X selects < 3 points. Widen the range.")
                            return
                        x = x[m]
                        y = y[m]

                    else:
                        x = []
                        y = []
                        xtmp = self._create_range(binx, minxREST, maxxREST)
                        ytmp = self._spectra.get(spectrumName, "data").tolist()
                        xmin, xmax = self.axisLimitsForFit(ax)
                        for i in range(1, len(xtmp)):
                            if (xtmp[i] > xmin and xtmp[i] <= xmax):
                                x.append(xtmp[i-1] + (xtmp[i] - xtmp[i-1]) / 2)
                                y.append(ytmp[i])
                        x = np.array(x); y = np.array(y)

                    run_cal = False

                    ask_pref = True
                    try:
                        s = QSettings("YourLab", "AlphaGUI")
                        ask_pref = s.value("calibration/ask", True, type=bool)
                    except Exception:
                        pass

                    if fit_funct in {"AlphaEMGMulti", "AlphaEMGMultiSigma"} and (force_prompt or ask_pref):
                        msg = QMessageBox(self._parent_widget)
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
                            return
                        run_cal = (clicked is btn_yes)

                        try:
                            if dont_ask.isChecked():
                                s.setValue("calibration/ask", False)
                        except Exception:
                            pass
                    else:
                        run_cal = False

                    cal = None
                    if run_cal and fit_funct in {"AlphaEMGMulti", "AlphaEMGMultiSigma"}:
                        cal = self._prompt_energy_calibration(ax, x, y, min_pts=2, max_pts=4, snap_halfwin=150)
                        if cal is not None:
                            config = dict(config)
                            config["calib_a"] = float(cal["a"])
                            config["calib_b"] = float(cal["b"])
                            try:
                                s.setValue("calibration/a", cal["a"])
                                s.setValue("calibration/b", cal["b"])
                                s.setValue("calibration/R2", cal["R2"])
                            except Exception:
                                pass

                    if fit_funct == "AlphaEMGMultiSigma":
                        flags = self._prompt_shape_flags(
                            default_global=config.get("fit_global_shapes", True),
                            default_iso_scales=config.get("fit_iso_shape_scales", False),
                        )
                        if flags is not None:
                            if flags["fit_iso_shape_scales"] and not flags["fit_global_shapes"]:
                                flags["fit_iso_shape_scales"] = False
                            config.update(flags)

                    fit = self._factory.create(fit_funct, **config)

                    setattr(fit, "_should_abort", lambda: getattr(self, "_abort_fit", False))

                    EMG_MODELS = {
                        "AlphaEMG1","AlphaEMG12","AlphaEMG2","AlphaEMG22",
                        "AlphaEMG3","AlphaEMG32","AlphaEMGMulti","AlphaEMGMultiSigma"
                    }

                    if model_name in EMG_MODELS:
                        if use_csv:
                            bw = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
                        else:
                            try:
                                bw = float(maxxREST - minxREST) / float(binx)
                                if not np.isfinite(bw) or bw <= 0:
                                    raise ValueError
                            except Exception:
                                bw = float(np.median(np.diff(x))) if len(x) > 1 else 1.0

                        if model_name in {"AlphaEMGMulti", "AlphaEMGMultiSigma"}:
                            wmode_ui = fitpar[12] if len(fitpar) > 12 else None

                            tail = [
                                bw,
                                wmode_ui,
                                fitpar[9],
                                fitpar[10],
                                fitpar[11],
                                fitpar[13],
                                fitpar[14],
                                fitpar[15],
                            ]
                            fitpar.extend(tail)

                        elif model_name in {"AlphaEMG22"}:
                            wmode = 1
                            bw_idx, wm_idx = 12, 13
                            need_len = wm_idx + 1
                            if len(fitpar) < need_len:
                                fitpar += [None] * (need_len - len(fitpar))
                            fitpar[bw_idx] = bw
                            fitpar[wm_idx] = wmode

                        elif model_name in {"AlphaEMG32"}:
                            wmode = 1
                            bw_idx, wm_idx = 18, 19
                            need_len = wm_idx + 1
                            if len(fitpar) < need_len:
                                fitpar += [None] * (need_len - len(fitpar))
                            fitpar[bw_idx] = bw
                            fitpar[wm_idx] = wmode

                        elif model_name in {"AlphaEMG12"}:
                            wmode = 2
                            need_len = 6
                            if len(fitpar) < need_len:
                                fitpar += [None] * (need_len - len(fitpar))
                            fitpar[4] = bw
                            fitpar[5] = wmode

                        else:
                            need_len = 6
                            if len(fitpar) < need_len:
                                fitpar += [None] * (need_len - len(fitpar))
                            fitpar[4] = bw
                            fitpar[5] = 1

                    else:
                        bw = None

                    fitResultsText = QTextEdit()
                    print(f"... Fitting {fit_funct} ...")

                    fitln = fit.start(x, y, xmin, xmax, fitpar, ax, fitResultsText)

                    self._maybe_show_alpha_filter_popup(fitln, fit_funct)

                    if fitln is None and self._abort_fit:
                        fitResultsText.append("[abort] Fit stopped by user.")
                        print("Fit aborted by user.")
                        return

                    print("Fitting is done.")

                    if cal is not None:
                        fitResultsText.insertPlainText(
                            f"[calibration] μ = {cal['a']:.8g} * E + {cal['b']:.8g}  "
                            f"(R²={cal['R2']:.6f}, N={len(cal['E'])})\n\n"
                        )
                    elif "calib_a" in config and "calib_b" in config:
                        src = config.get("calibration_file", "saved defaults")
                        fitResultsText.insertPlainText(
                            f"[calibration] μ = {config['calib_a']:.8g} * E + {config['calib_b']:.8g}  "
                            f"(source: {src})\n\n"
                        )

                    if fit_funct in EMG_MODELS:
                        def _shape_mode(f):
                            g = bool(getattr(f, "fit_global_shapes", False))
                            i = bool(getattr(f, "fit_iso_shape_scales", False))
                            if g and i:   return "global + per-isotope"
                            if g:         return "global"
                            if i:         return "per-isotope only (baseline from file)"
                            return "frozen (file σ,τ₁,τ₂,η)"

                        shape_src = config.get("shape_file", getattr(fit, "shape_file", None))
                        fitResultsText.insertPlainText(
                            f"[shapes] mode = {_shape_mode(fit)} ; source: {shape_src or 'unknown'}\n\n"
                        )

                    self.setFitLineLabel(ax, fitln, fitResultsText, spectrumName)
                    self._tag_new_fit_artists(ax, before_ids)

                    if model_name == "AlphaEMG22":
                        try:
                            s = QSettings("YourLab", "AlphaGUI")
                            s.setValue("AlphaEMG22/mu1", float(self._popup.fit_p1.text()))
                            s.setValue("AlphaEMG22/mu2", float(self._popup.fit_p7.text()))
                        except Exception:
                            pass

                    txt = fitResultsText.toPlainText()
                    print("\n----- FIT RESULTS -----\n" + txt)
                    self.logger.info("Fit results for %s [%s]:\n%s", spectrumName, fit_funct, txt)
                    fitResultsText.setReadOnly(True)
                    fitResultsText.setWindowTitle(f"Fit results — {fit_funct} : {spectrumName}")
                    fitResultsText.resize(900, 700)
                    fitResultsText.show()
                    self._lastFitResultsText = fitResultsText

                else:
                    QMessageBox.about(self._parent_widget, "Warning", "Sorry 2D fitting is not implemented yet")
            else:
                QMessageBox.about(self._parent_widget, "Warning", "Histogram not existing. Please load a histogram...")

            ax.figure.canvas.draw_idle()

        except NameError as err:
            print(err)
            pass

        finally:
            self._popup.abort_button.setEnabled(False)
            self._popup.fit_button.setEnabled(True)

    # ------------------------------------------------------------------
    # Energy calibration dialog
    # ------------------------------------------------------------------

    def _prompt_energy_calibration(self, ax, x, y, min_pts=2, max_pts=4, snap_halfwin=150):
        dlg = QDialog(self._parent_widget)
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
            active=True, ax=ax, x=np.asarray(x), y=np.asarray(y),
            min_pts=int(min_pts), max_pts=int(max_pts), halfwin=float(snap_halfwin),
            MU=[], E=[], artists=[], dlg=dlg, status=status
        )

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

        def _add_point(xdata):
            if len(self._cal.MU) >= self._cal.max_pts:
                QMessageBox.information(dlg, "Max points", f"Already have {self._cal.max_pts} points.")
                return
            mu, yy = _snap_mu(xdata)
            val, ok = QInputDialog.getDouble(dlg, "Energy (keV)", "Energy:", 0.0, -1e9, 1e9, 6)
            if not ok:
                return
            self._cal.MU.append(float(mu)); self._cal.E.append(float(val))
            m1, = ax.plot([mu], [yy], marker='o', ms=6)
            m2 = ax.axvline(mu, ls=':', lw=1.0)
            self._cal.artists.append((m1, m2))
            ax.figure.canvas.draw_idle()
            _update_status()
        self._cal.add_point = _add_point

        def _undo():
            if not self._cal.MU: return
            self._cal.MU.pop(); self._cal.E.pop()
            for a in self._cal.artists.pop():
                try: a.remove()
                except: pass
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
                    except: pass
            ax.figure.canvas.draw_idle()
            self._cal = None

    # ------------------------------------------------------------------
    # Shape flags dialog
    # ------------------------------------------------------------------

    def _prompt_shape_flags(self, default_global=True, default_iso_scales=False, default_chain=True):
        dlg = QDialog(self._parent_widget); dlg.setWindowTitle("Shape fitting options")
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

    # ------------------------------------------------------------------
    # File choosers
    # ------------------------------------------------------------------

    def _choose_shape_file(self, settings_key: str) -> str:
        s = QSettings("YourLab", "AlphaGUI")
        last = s.value(settings_key, "", type=str) or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self._parent_widget, "Choose shape file", last,
            "Text/CSV files (*.txt *.csv);;All files (*)"
        )
        if path:
            s.setValue(settings_key, path)
        return path or ""

    def _choose_calibration_file(self, settings_key: str) -> str:
        s = QSettings("YourLab", "AlphaGUI")
        last = s.value(settings_key, "", type=str) or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self._parent_widget, "Calibration file", last,
            "Text/CSV/JSON (*.txt *.csv *.json);;All files (*)"
        )
        if path:
            s.setValue(settings_key, path)
        return path or ""

    # ------------------------------------------------------------------
    # Calibration file loader
    # ------------------------------------------------------------------

    def load_calibration_any(self, path):
        with open(path, "r") as f:
            txt = f.read()

        try:
            obj = json.loads(txt)
            for key_a, key_b in (("a", "b"), ("calib_a", "calib_b")):
                if key_a in obj and key_b in obj:
                    return float(obj[key_a]), float(obj[key_b])
        except Exception:
            pass

        a = b = None
        for m in re.finditer(r'^\s*([ab]|calib_a|calib_b)\s*=\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)\s*$', txt, re.M):
            k = m.group(1).lower()
            v = float(m.group(2))
            if k in ("a", "calib_a"): a = v
            elif k in ("b", "calib_b"): b = v
        if a is not None and b is not None:
            return a, b

        nums = re.findall(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', txt)
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])

        raise ValueError(f"Could not parse calibration from: {path}")

    # ------------------------------------------------------------------
    # Fit config preparation
    # ------------------------------------------------------------------

    def prepare_fit_config(self, fit_funct: str, force_prompt: bool = False) -> dict:
        config = dict(self._factory._configs.get(fit_funct) or {})

        if fit_funct in {"AlphaEMGMulti", "AlphaEMGMultiSigma"}:
            key_shape = f"{fit_funct}/shape_file"
            s = QSettings("YourLab", "AlphaGUI")

            shape_path = config.get("shape_file", "")
            if force_prompt:
                shape_path = ""
            if not (isinstance(shape_path, str) and os.path.isfile(shape_path)):
                shape_path = self._choose_shape_file(key_shape)
                if not shape_path:
                    raise ValueError("No shape file selected.")
                if not os.path.isfile(shape_path):
                    raise ValueError(f"Shape file not found:\n{shape_path}")
                config["shape_file"] = shape_path
                s.setValue(key_shape, shape_path)

            per_model_key = f"{fit_funct}/calibration_file"

            cal_path = (
                config.get("calibration_file")
                or s.value(per_model_key, "", type=str)
                or s.value("calibration/file", "", type=str)
                or ""
            )

            have_numbers_already = ("calib_a" in config and "calib_b" in config)

            need_prompt = (
                force_prompt
                or (not os.path.isfile(cal_path) and not have_numbers_already)
            )
            if need_prompt:
                cal_path = self._choose_calibration_file(per_model_key)

            if cal_path and os.path.isfile(cal_path):
                try:
                    a, b = self.load_calibration_any(cal_path)
                    config["calib_a"] = float(a)
                    config["calib_b"] = float(b)
                    config["calibration_file"] = cal_path
                    s.setValue(per_model_key, cal_path)
                    s.setValue("calibration/file", cal_path)
                    s.setValue("calibration/a", a)
                    s.setValue("calibration/b", b)
                except Exception as e:
                    QMessageBox.warning(
                        self._parent_widget, "Calibration",
                        f"Could not parse calibration from:\n{cal_path}\n\n{e}\n\nUsing defaults."
                    )
                    if not have_numbers_already:
                        raise ValueError("No valid calibration available.")
            else:
                if not have_numbers_already:
                    raise ValueError("No calibration file selected.")

            self._factory._configs[fit_funct] = config

        return config

    # ------------------------------------------------------------------
    # Alpha filter popup helpers
    # ------------------------------------------------------------------

    def _close_alpha_filter_popup(self):
        dlg = getattr(self, "_alphaFilterDlg", None)
        if dlg is not None:
            try:
                dlg.close()
            except Exception:
                pass
        self._alphaFilterDlg = None

    def _maybe_show_alpha_filter_popup(self, fitln, fit_funct: str):
        if fit_funct.strip() != "AlphaEMGMultiSigma":
            self._close_alpha_filter_popup()
            return

        dlg = self._alphaFilterDlg or AlphaChainIsoFilterDialog(self._parent_widget)
        if not dlg.supports(fitln):
            self._close_alpha_filter_popup()
            return

        self._alphaFilterDlg = dlg
        dlg.bind(fitln)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    # ------------------------------------------------------------------
    # Artist tagging and clearing
    # ------------------------------------------------------------------

    def _snap_ax_ids(self, ax):
        ids = set()
        for l in getattr(ax, "lines", []): ids.add(id(l))
        for c in getattr(ax, "collections", []): ids.add(id(c))
        for t in getattr(ax, "texts", []): ids.add(id(t))
        for p in getattr(ax, "patches", []): ids.add(id(p))
        for a in getattr(ax, "artists", []): ids.add(id(a))
        return ids

    def _tag_new_fit_artists(self, ax, before_ids, fit_idx=None):
        def _tag(a):
            if hasattr(a, "set_gid"): a.set_gid("fit")
            if fit_idx is not None and hasattr(a, "set_label"):
                lab = getattr(a, "get_label", lambda: "")() or ""
                if not lab or lab == "_nolegend_":
                    a.set_label(f"{FIT_PREFIX}{fit_idx}-art")
        for l in getattr(ax, "lines", []):
            if id(l) not in before_ids: _tag(l)
        for c in getattr(ax, "collections", []):
            if id(c) not in before_ids: _tag(c)
        for t in getattr(ax, "texts", []):
            if id(t) not in before_ids: _tag(t)
        for p in getattr(ax, "patches", []):
            if id(p) not in before_ids: _tag(p)
        for a in getattr(ax, "artists", []):
            if id(a) not in before_ids: _tag(a)

    def _clear_all_fit_artists_in_figure(self, fig):
        total_removed = 0
        for ax in list(fig.axes):
            try:
                labels = self.listFitLineLabels(ax)
                if hasattr(self._popup, "delete_fitIdx_list"):
                    self._popup.delete_fitIdx_list.setText(" ".join(labels) if labels else "")
            except Exception:
                pass

            removed = 0
            for coll in (getattr(ax, "lines", []),
                        getattr(ax, "collections", []),
                        getattr(ax, "texts", []),
                        getattr(ax, "patches", []),
                        getattr(ax, "artists", []),
                        getattr(ax, "images", [])):
                for a in list(coll):
                    lab = getattr(a, "get_label", lambda: "")() or ""
                    gid = getattr(a, "get_gid",   lambda: None)()
                    if gid == "fit" or (isinstance(lab, str) and lab.startswith(FIT_PREFIX)):
                        try:
                            a.remove()
                            removed += 1
                        except Exception:
                            pass

            leg = ax.get_legend()
            if leg:
                try: leg.remove()
                except Exception: pass

            total_removed += removed

        try:
            fig.canvas.draw_idle()
        except Exception:
            pass
        self.logger.debug("clear_all_fit_artists_in_figure: removed %d artists", total_removed)

    # ------------------------------------------------------------------
    # Fit line labels
    # ------------------------------------------------------------------

    def setFitResultsLineLabel(self, fitLineLabelIdx, resultsText, spectrumName):
        self.logger.info('setFitResultsLineLabel - fitLineLabelIdx: %d', fitLineLabelIdx)
        title = 'Fit ' + str(fitLineLabelIdx) + ' (' + spectrumName + ') :'
        self._popup.fit_results.append(title)
        self._popup.fit_results.append(resultsText.toPlainText())
        self._popup.fit_results.append(' ')

    def setFitLineLabel(self, ax, line, resultsText, spectrumName):
        self.logger.info('setFitLineLabel')
        fitLineLabel = "fit-_-"
        fitLabels = self.listFitLineLabels(ax)
        fitIdxs = [int(label) for label in fitLabels]
        fitIdx = 0

        if line is None:
            self.logger.debug('setFitLineLabel - line is None')
            return

        if not fitIdxs:
            fitLineLabel += "0"
        else:
            i = 0
            while i in fitIdxs:
                i += 1
            fitIdx = i
            fitLineLabel += str(i)

        line.set_label(fitLineLabel)
        self.setFitResultsLineLabel(fitIdx, resultsText, spectrumName)
        self.logger.debug('setFitLineLabel - line label: %s', line.get_label())

    def deleteFit(self, index=None, name=None, ax=None):
        self.logger.info('deleteFit')

        self._close_alpha_filter_popup()

        if ax is None:
            self.logger.warning('deleteFit - called without ax context; cannot delete fit')
            return
        userFitIdxs = self._popup.delete_fitIdx_list.text().split()
        availableFitIdxs = self.listFitLineLabels(ax)
        notAvailableFitIdxs = [fitIdx for fitIdx in userFitIdxs if fitIdx not in availableFitIdxs]
        if notAvailableFitIdxs:
            self.logger.warning('deleteFit - fit line(s) index(es): %s cannot be deleted', notAvailableFitIdxs)
        else:
            for fitIdx in userFitIdxs:
                fitLineIdentifier = "fit-_-" + fitIdx
                for fitLine in ax.get_children():
                    if type(fitLine) == matplotlib.lines.Line2D and fitLineIdentifier in fitLine.get_label():
                        fitLine.remove()
                        self.logger.debug('deleteFit - removed fit line: %s', fitLineIdentifier)

    def listFitLineLabels(self, ax):
        self.logger.info('listFitLineLabels')
        fitLabels = []
        fitLineIdentifier = "fit-_-"
        fitLineLabels = [
            fitLine.get_label()
            for fitLine in ax.get_children()
            if type(fitLine) == matplotlib.lines.Line2D and fitLineIdentifier in fitLine.get_label()
        ]
        for label in fitLineLabels:
            labelSplit = label.split("-_-")
            fitLabels.append(labelSplit[1])
        return fitLabels

    def printFitLineLabels(self, index=None, name=None, ax=None):
        self.logger.info('printFitLineLabels')
        if ax is None:
            self.logger.warning('printFitLineLabels - called without ax context; cannot list labels')
            return
        fitLabels = self.listFitLineLabels(ax)
        textLabels = ''
        for label in fitLabels:
            textLabels += label + " "
        self._popup.delete_fitIdx_list.setText(textLabels)
