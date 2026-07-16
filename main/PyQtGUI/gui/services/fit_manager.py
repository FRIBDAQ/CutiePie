import logging
import os
import re
import json
from datetime import datetime
import matplotlib
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

from PyQt5.QtWidgets import (
    QMessageBox, QFileDialog, QDialog, QLabel, QPushButton, QCheckBox,
    QHBoxLayout, QVBoxLayout, QInputDialog, QTextEdit, QApplication,
)
from PyQt5.QtCore import Qt, QObject, QSettings, QEventLoop, pyqtSignal

from alpha_filter_dialog import AlphaChainIsoFilterDialog

FIT_PREFIX = "fit-_-"


class FitManager(QObject):
    """Owns all fitting operations: execute, CSV, manage artists, result popups."""

    FIT_PREFIX = FIT_PREFIX

    # extraPopup widget writes inverted into signals — MainWindow owns the
    # widgets (adapters _on_fit_busy / _on_abort_enabled / _append_fit_results
    # / _set_fit_labels_text).
    fitBusyChanged      = pyqtSignal(bool)  # True: fit running (fit off, abort on)
    abortEnabledChanged = pyqtSignal(bool)  # abort button enable state only
    fitResultsAppended  = pyqtSignal(str)   # one fit_results text-box line
    fitLabelsTextChanged = pyqtSignal(str)  # delete_fitIdx_list contents

    def __init__(self, fit_factory, spectra, parent_widget=None, logger=None):
        super().__init__()
        self._parent_widget = parent_widget  # Qt dialog parent only — no domain calls
        self._factory = fit_factory
        self._spectra = spectra
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
        self._lastFitCurve     = None   # {'x','y','model','name'} of the last drawn fit total

    @staticmethod
    def _create_range(bins, vmin, vmax):
        return np.linspace(float(vmin), float(vmax), int(bins) + 1)

    # ------------------------------------------------------------------
    # Axis limits helper
    # ------------------------------------------------------------------

    def axisLimitsForFit(self, ax, range_min_text="", range_max_text=""):
        """Fit x-range from the popup's Min/Max X fields, supplied as text by
        the MainWindow adapter; empty/invalid fields fall back to xlim."""
        left, right = ax.get_xlim()
        self.logger.info('axisLimitsForFit - left, right: %s, %s', left, right)
        range_min_text = range_min_text or ""
        range_max_text = range_max_text or ""
        if range_min_text:
            try:
                left = float(range_min_text)
            except ValueError:
                self.logger.warning('axisLimitsForFit - Invalid input for Min X. Please enter a valid number.')
        else:
            left = ax.get_xlim()[0]
        if range_max_text:
            try:
                right = float(range_max_text)
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
        self.abortEnabledChanged.emit(False)
        self.fitResultsAppended.emit("[abort] Requested…")

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def on_fit_csv_clicked(self, fit_funct="", fitpar_texts=None,
                           range_min_text="", range_max_text=""):
        if not hasattr(self, "_csv_x") or self._csv_x is None or len(self._csv_x) == 0:
            QMessageBox.warning(self._parent_widget, "No CSV loaded", "Click 'Plot CSV' first.")
            return
        if not hasattr(self, "_csv_ax") or self._csv_ax is None:
            QMessageBox.warning(self._parent_widget, "No CSV plot", "Click 'Plot CSV' first.")
            return

        self._use_csv_fit = True
        try:
            self.fit(fit_funct=fit_funct, fitpar_texts=fitpar_texts,
                     range_min_text=range_min_text, range_max_text=range_max_text)
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
            except Exception: self.logger.debug('could not set CSV window title', exc_info=True)
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
    # Save / load a fit curve (total + per-isotope components <-> CSV)
    # ------------------------------------------------------------------

    def _stash_fit_curve(self, fitln, fit_funct, spectrumName):
        """Remember the last drawn fit so it can be saved.

        Always captures the sampled total (``fitln.component_data`` for the
        AlphaEMG* creators, else the line's own x/y). When the creator also
        stashes per-isotope series (``component_series``) it captures those plus
        the isotope→chain map and per-component colors, enabling a full
        multi-component save."""
        if fitln is None:
            return
        try:
            cd = getattr(fitln, "component_data", None)
            if cd is not None and "x" in cd and "ytot" in cd:
                cx = np.asarray(cd["x"], dtype=float)
                cy = np.asarray(cd["ytot"], dtype=float)
            else:
                cx = np.asarray(fitln.get_xdata(), dtype=float)
                cy = np.asarray(fitln.get_ydata(), dtype=float)
            curve = {"x": cx, "y": cy, "model": fit_funct, "name": spectrumName}

            series = getattr(fitln, "component_series", None)
            if series:
                curve["components"] = [(str(n), np.asarray(y, dtype=float))
                                       for n, y in series]
                curve["chains"] = dict(getattr(fitln, "_iso_to_chain", {}) or {})
                curve["colors"] = dict(getattr(fitln, "component_colors", {}) or {})

            ps = getattr(fitln, "peak_series", None)
            if ps:
                curve["peaks"] = [
                    dict(chain=str(p.get("chain") or "Unchained"),
                         isotope=str(p.get("isotope")),
                         E=(float(p["E"]) if p.get("E") is not None else None),
                         y=np.asarray(p["y"], dtype=float),
                         params={k: float(v) for k, v in (p.get("params") or {}).items()})
                    for p in ps]
            self._lastFitCurve = curve
        except Exception:
            self.logger.debug('could not stash fit curve', exc_info=True)

    def save_fit_curve(self, path=None):
        """Write the last drawn fit to a CSV.

        AlphaEMGMultiSigma fits (which capture per-peak data) write a grouped
        per-peak file: ``x, fit total``, then per chain the isotope sum and each
        peak, plus a human-readable ``# per-peak parameters`` block and a real
        header row (Excel-friendly, self-describing via ``chain/isotope/E`` names).
        Other fits write a two-column (x, y_total) file.

        `path` is supplied by tests; in the GUI it is chosen via QFileDialog."""
        curve = getattr(self, "_lastFitCurve", None)
        if not curve or curve.get("x") is None or len(curve["x"]) == 0:
            QMessageBox.warning(self._parent_widget, "No fit to save",
                                "Run a fit first, then Save Fit.")
            return None

        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self._parent_widget, "Save fit curve", "",
                "CSV files (*.csv);;All files (*)")
            if not path:
                return None
            if not path.lower().endswith(".csv"):
                path += ".csv"

        peaks = curve.get("peaks")
        if peaks:
            self._write_fit_peaks_csv(
                path, curve["x"], curve["y"], peaks,
                model=curve.get("model", ""), name=curve.get("name", ""))
            self.logger.info('save_fit_curve - wrote %d peaks to %s', len(peaks), path)
            return path

        comps = curve.get("components")
        if comps and len(comps) > 1:
            self._write_fit_components_csv(
                path, curve["x"], comps,
                model=curve.get("model", ""), name=curve.get("name", ""),
                chains=curve.get("chains"), colors=curve.get("colors"))
            self.logger.info('save_fit_curve - wrote %d components to %s', len(comps), path)
            return path

        # total-only fallback (non-multi models, or a fit with a single component)
        x = np.asarray(curve["x"], dtype=float)
        y = np.asarray(curve["y"], dtype=float)
        header = (
            "CutiePie fit curve\n"
            f"model = {curve.get('model', '')}\n"
            f"spectrum = {curve.get('name', '')}\n"
            f"saved = {datetime.now().isoformat(timespec='seconds')}\n"
            f"npoints = {x.size}\n"
            "x,y_total"
        )
        np.savetxt(path, np.column_stack([x, y]), delimiter=",",
                   header=header, comments="# ")
        self.logger.info('save_fit_curve - wrote %d points to %s', x.size, path)
        return path

    def _write_fit_peaks_csv(self, path, x, total_y, peaks, model="", name=""):
        """Write the grouped per-peak / per-chain file (Version 2, self-describing
        via column names — no JSON). Columns: ``x, fit total``, then per chain
        (sorted): the isotope sum ``<chain>/<isotope>`` then each peak
        ``<chain>/<isotope>/<E>``. A ``# per-peak parameters`` comment block and a
        real (non-comment) header row make it readable and Excel-friendly."""
        x = np.asarray(x, dtype=float)

        # group peaks: chain (sorted) → isotope (first-seen order) → peaks (by E)
        by_chain = {}
        iso_order = {}
        for p in peaks:
            ch = p.get("chain") or "Unchained"
            iso = str(p.get("isotope"))
            by_chain.setdefault(ch, {}).setdefault(iso, []).append(p)
            iso_order.setdefault((ch, iso), len(iso_order))

        columns = ["x", "fit total"]
        col_arrays = [x, np.asarray(total_y, dtype=float)]
        param_rows = []
        seen = set(columns)

        def _uniq(base):
            nm, k = base, 2
            while nm in seen:
                nm = f"{base}#{k}"; k += 1
            seen.add(nm)
            return nm

        for ch in sorted(by_chain):
            isos = sorted(by_chain[ch], key=lambda iso: iso_order[(ch, iso)])
            for iso in isos:
                plist = sorted(by_chain[ch][iso],
                               key=lambda p: (p.get("E") if p.get("E") is not None else 0.0))
                iso_sum = np.sum([np.asarray(p["y"], dtype=float) for p in plist], axis=0)
                columns.append(_uniq(f"{ch}/{iso}"))
                col_arrays.append(iso_sum)
                for p in plist:
                    ename = f"{p['E']:.0f}" if p.get("E") is not None else "NA"
                    columns.append(_uniq(f"{ch}/{iso}/{ename}"))
                    col_arrays.append(np.asarray(p["y"], dtype=float))
                    param_rows.append((ch, iso, ename, p.get("params") or {}))

        data = np.column_stack(col_arrays)
        with open(path, "w", newline="") as f:
            f.write(f"# CutiePie fit curve — {model} : {name}   "
                    f"(saved {datetime.now().isoformat(timespec='seconds')})\n")
            f.write("#\n# per-peak parameters:\n")
            f.write("#   {:<8} {:<10} {:>7}   {:>11} {:>11} {:>7} {:>6} {:>6} {:>5}\n".format(
                "chain", "isotope", "E_keV", "A", "mu", "sigma", "tau1", "tau2", "eta"))
            for ch, iso, ename, pr in param_rows:
                def g(k):
                    v = pr.get(k)
                    return float(v) if v is not None else float("nan")
                f.write("#   {:<8} {:<10} {:>7}   {:>11.4e} {:>11.4f} {:>7.3f} "
                        "{:>6.3f} {:>6.3f} {:>5.3f}\n".format(
                            ch, iso, ename, g("A"), g("mu"), g("sigma"),
                            g("tau1"), g("tau2"), g("eta")))
            f.write("#\n")
            f.write(",".join(columns) + "\n")
            np.savetxt(f, data, delimiter=",", fmt="%.8g")
        return path

    def _write_fit_components_csv(self, path, x, comps, model="", name="",
                                  chains=None, colors=None):
        """Write x + one column per component, with a `# meta` JSON header that
        records column names, the isotope→chain map, and per-component colors."""
        x = np.asarray(x, dtype=float)
        names = [n for n, _ in comps]
        data = np.column_stack([x] + [np.asarray(y, dtype=float) for _, y in comps])
        chains = chains or {}
        colors = colors or {}
        meta = {
            "model": model,
            "spectrum": name,
            "saved": datetime.now().isoformat(timespec="seconds"),
            "columns": ["x"] + names,
            "chains": {n: chains.get(n, "") for n in names if n != "fit total"},
            "colors": {n: colors[n] for n in names if colors.get(n) is not None},
        }
        header = ("CutiePie fit curve (multi-component)\n"
                  "meta = " + json.dumps(meta) + "\n"
                  + ",".join(["x"] + names))
        np.savetxt(path, data, delimiter=",", header=header, comments="# ")
        return path

    @staticmethod
    def _parse_component_name(name):
        """Derive {kind, chain, isotope, E} from a column name. Version 2 names
        are ``chain/isotope`` (isotope sum) or ``chain/isotope/E`` (a peak);
        ``fit total`` is the total; anything else is a bare component."""
        if name == "fit total":
            return dict(name=name, kind="total", chain=None, isotope=None, E=None)
        parts = name.split("/")
        if len(parts) == 3:
            try:
                E = float(parts[2])
            except ValueError:
                E = None
            return dict(name=name, kind="peak", chain=parts[0], isotope=parts[1], E=E)
        if len(parts) == 2:
            return dict(name=name, kind="isotope", chain=parts[0], isotope=parts[1], E=None)
        return dict(name=name, kind="component", chain=None, isotope=name, E=None)

    def _read_fit_curve_file(self, path):
        """Parse a saved fit file. Returns
        ``{x, components:[(name,y)…], structure:[…], chains, colors, multi}`` or
        None if the file has no usable numeric (x, y[, …]) block. Handles the
        Version 2 per-peak files (real header row, self-describing names), the
        legacy ``# meta`` JSON files, and plain two-column total-only files."""
        meta = None
        header_names = None
        try:
            with open(path) as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("#"):
                        body = s.lstrip("#").strip()
                        if body.lower().startswith("meta"):
                            eq = body.find("=")
                            if eq != -1:
                                try:
                                    meta = json.loads(body[eq + 1:].strip())
                                except Exception:
                                    meta = None
                        continue
                    if not s:
                        continue
                    # first non-comment line: a header row iff its first field
                    # isn't numeric (Version 2); otherwise it is data.
                    fields = [c.strip() for c in s.split(",")]
                    try:
                        float(fields[0])
                    except ValueError:
                        header_names = fields
                    break
        except Exception:
            self.logger.debug('read_fit_curve_file - header scan failed', exc_info=True)

        try:
            arr = np.genfromtxt(path, delimiter=",", comments="#")
        except Exception:
            return None
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        arr = arr[np.isfinite(arr[:, 0])]      # drops the Version 2 header row (NaN)
        if arr.shape[0] == 0:
            return dict(x=np.array([]), components=[], structure=[],
                        chains={}, colors={}, multi=False)

        x = arr[:, 0].astype(float)
        ncols = arr.shape[1]
        colors = {}
        if header_names is not None and len(header_names) == ncols:
            names = [str(n) for n in header_names[1:]]        # Version 2
        elif isinstance(meta, dict) and isinstance(meta.get("columns"), list) \
                and len(meta["columns"]) == ncols:
            names = [str(n) for n in meta["columns"][1:]]      # legacy meta
            colors = {str(k): v for k, v in (meta.get("colors") or {}).items()}
        else:
            names = ["fit total"] + [f"col{j}" for j in range(2, ncols)]  # total-only

        components = [(names[j], arr[:, j + 1].astype(float)) for j in range(len(names))]
        structure = [self._parse_component_name(n) for n in names]

        # legacy meta carried the isotope→chain map explicitly; fold it in
        meta_chains = {}
        if isinstance(meta, dict):
            meta_chains = {str(k): str(v) for k, v in (meta.get("chains") or {}).items()}
        for d in structure:
            if d["chain"] is None and d["isotope"] in meta_chains:
                d["chain"] = meta_chains[d["isotope"]]
                if d["kind"] == "component":
                    d["kind"] = "isotope"

        chains = {d["name"]: d["chain"] for d in structure if d.get("chain")}
        multi = len(names) > 1
        return dict(x=x, components=components, structure=structure,
                    chains=chains, colors=colors, multi=multi)

    def _plot_fit_components(self, ax, x, comps, colors=None):
        """Draw the given components on `ax` as ONE fit group. All lines share a
        per-index gid ``fit-<N>``; the carrier (the total if present, else the
        first) also gets the ``fit-_-<N>`` label so the group has an index that
        Sel. All / Delete / clear-on-next-fit recognise. Styling: total solid,
        isotope sums dashed, individual peaks dotted; a component and its peaks
        share a colour. Returns (index, lines)."""
        colors = colors or {}
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color") \
            or ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        iso_color = {}

        def _color(name):
            if colors.get(name) is not None:
                return colors[name]
            if name == "fit total":
                return "tab:orange"
            key = "/".join(name.split("/")[:2])     # chain/isotope groups together
            if key not in iso_color:
                iso_color[key] = cycle[len(iso_color) % len(cycle)]
            return iso_color[key]

        idxs = [int(l) for l in self.listFitLineLabels(ax)]
        i = 0
        while i in idxs:
            i += 1
        gid = f"fit-{i}"
        carrier = "fit total" if any(n == "fit total" for n, _ in comps) \
            else (comps[0][0] if comps else None)
        lines = []
        for name, y in comps:
            if name == "fit total":
                (ln,) = ax.plot(x, y, lw=2, color=_color(name))
            elif name.count("/") >= 2:              # an individual peak
                (ln,) = ax.plot(x, y, lw=1.1, ls=":", alpha=0.85, color=_color(name))
            else:                                    # an isotope sum (or bare component)
                (ln,) = ax.plot(x, y, lw=1.6, ls="--", alpha=0.9, color=_color(name))
            if hasattr(ln, "set_gid"):
                ln.set_gid(gid)
            if name == carrier:
                ln.set_label(f"{FIT_PREFIX}{i}")
                carrier = None            # label only the first match
            lines.append(ln)
        return i, lines

    def load_fit_curve(self, index=None, name=None, ax=None, path=None):
        """Draw a saved fit curve onto the currently selected pad's axis.

        Multi-component files open a chain-grouped picker so you can choose which
        components to draw; total-only files draw the single curve. Everything is
        tagged as one deletable fit group (``fit-<N>`` gid + ``fit-_-<N>`` label)."""
        if ax is None:
            self.logger.warning('load_fit_curve - called without ax context; cannot draw')
            QMessageBox.warning(self._parent_widget, "No plot selected",
                                "Select a plot pad first, then Load Fit.")
            return None

        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self._parent_widget, "Load fit curve", "",
                "CSV files (*.csv *.txt);;All files (*)")
            if not path:
                return None

        struct = self._read_fit_curve_file(path)
        if struct is None:
            QMessageBox.warning(self._parent_widget, "Bad fit file",
                                "Expected a numeric (x, y[, …]) CSV.")
            return None
        if struct["x"].size == 0 or not struct["components"]:
            QMessageBox.warning(self._parent_widget, "Bad fit file",
                                "No finite rows in file.")
            return None

        comps = struct["components"]
        if struct["multi"]:
            chosen = self._prompt_component_selection(struct["structure"])
            if chosen is None:
                return None                     # cancelled
            comps = [(n, y) for n, y in comps if n in chosen]
            if not comps:
                QMessageBox.warning(self._parent_widget, "Nothing selected",
                                    "No components chosen to plot.")
                return None

        i, lines = self._plot_fit_components(ax, struct["x"], comps, struct["colors"])
        self.fitResultsAppended.emit(
            f"Loaded fit {i} ({len(lines)} component(s)) from {os.path.basename(path)}")
        try:
            ax.figure.canvas.draw_idle()
        except Exception:
            self.logger.debug('load_fit_curve - could not draw', exc_info=True)
        self.logger.info('load_fit_curve - drew %d component(s) from %s', len(lines), path)
        return lines[0] if lines else None

    def _prompt_component_selection(self, structure):
        """Modal checkable tree of components (chain → isotope sum → peaks).
        `structure` is the list of column descriptors from _read_fit_curve_file.
        Returns the list of selected column names, or None if cancelled."""
        from PyQt5.QtWidgets import QScrollArea, QWidget  # live-only widgets
        dlg = QDialog(self._parent_widget)
        dlg.setWindowTitle("Choose components to plot")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Select the fit components to draw:"))

        inner = QWidget()
        iv = QVBoxLayout(inner)
        checks = {}

        def _add(name, label, indent):
            cb = QCheckBox(label)
            cb.setChecked(True)
            if indent:
                cb.setStyleSheet(f"margin-left: {indent}px;")
            checks[name] = cb
            iv.addWidget(cb)

        for d in structure:
            if d["kind"] == "total":
                _add(d["name"], "fit total", 0)

        # chain → isotope sum → peaks, preserving structure order within a chain
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

        # An isotope-sum checkbox is a parent: toggling it checks/unchecks all of
        # that isotope's individual peak checkboxes at once.
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
        hb.addWidget(btn_all)
        hb.addWidget(btn_none)
        hb.addStretch(1)
        btn_ok = QPushButton("OK")
        btn_cancel = QPushButton("Cancel")
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        hb.addWidget(btn_ok)
        hb.addWidget(btn_cancel)
        v.addLayout(hb)

        if dlg.exec_() != QDialog.Accepted:
            return None
        return [n for n, c in checks.items() if c.isChecked()]

    # ------------------------------------------------------------------
    # Main fit entry point
    # ------------------------------------------------------------------

    def fit(self, index=None, name=None, ax=None, fit_funct="",
            fitpar_texts=None, range_min_text="", range_max_text=""):
        """Run a fit. All popup-field values (model name, the 20 parameter
        fields, the Min/Max X range) arrive as arguments gathered by the
        MainWindow adapter."""
        self.logger.info('fit')

        fit_funct = (fit_funct or "").strip()

        self._close_alpha_filter_popup()

        model_name = fit_funct
        force_prompt = bool(True)

        try:
            config = self.prepare_fit_config(fit_funct, force_prompt=force_prompt)
        except ValueError as e:
            QMessageBox.warning(self._parent_widget, "Fit cancelled", str(e))
            return

        # resolve the fit context BEFORE going busy — the early return
        # below sits outside the try/finally, so a busy state entered first
        # would never be cleared and the fit button stayed disabled.
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

        self._abort_fit = False
        self.fitBusyChanged.emit(True)

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
                    texts = list(fitpar_texts or [])
                    texts += [""] * (20 - len(texts))
                    texts = texts[:20]

                    fitpar = []
                    for raw in texts:
                        t = raw.strip() if raw is not None else ""
                        try:
                            fitpar.append(float(t) if t != "" else None)
                        except Exception:
                            fitpar.append(None)

                    if use_csv:
                        xmin, xmax = self.axisLimitsForFit(ax, range_min_text, range_max_text)

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
                        xmin, xmax = self.axisLimitsForFit(ax, range_min_text, range_max_text)
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
                    self.logger.info('fit - fitting %s ...', fit_funct)

                    fitln = fit.start(x, y, xmin, xmax, fitpar, ax, fitResultsText)

                    self._maybe_show_alpha_filter_popup(fitln, fit_funct)

                    if fitln is None and self._abort_fit:
                        fitResultsText.append("[abort] Fit stopped by user.")
                        self.logger.info('fit - aborted by user')
                        return

                    self.logger.info('fit - fitting done')

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

                    fitIdx = self.setFitLineLabel(ax, fitln, fitResultsText, spectrumName)
                    self._tag_new_fit_artists(ax, before_ids, fit_idx=fitIdx)

                    if model_name == "AlphaEMG22":
                        try:
                            s = QSettings("YourLab", "AlphaGUI")
                            s.setValue("AlphaEMG22/mu1", float(texts[1]))
                            s.setValue("AlphaEMG22/mu2", float(texts[7]))
                        except Exception:
                            pass

                    txt = fitResultsText.toPlainText()
                    self.logger.info("Fit results for %s [%s]:\n%s", spectrumName, fit_funct, txt)
                    fitResultsText.setReadOnly(True)
                    fitResultsText.setWindowTitle(f"Fit results — {fit_funct} : {spectrumName}")
                    fitResultsText.resize(900, 700)
                    fitResultsText.show()
                    self._lastFitResultsText = fitResultsText
                    self._stash_fit_curve(fitln, fit_funct, spectrumName)

                else:
                    QMessageBox.about(self._parent_widget, "Warning", "Sorry 2D fitting is not implemented yet")
            else:
                QMessageBox.about(self._parent_widget, "Warning", "Histogram not existing. Please load a histogram...")

            ax.figure.canvas.draw_idle()

        except NameError as err:
            self.logger.exception('fit - NameError')

        finally:
            self.fitBusyChanged.emit(False)

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
                except Exception: self.logger.debug('could not remove calibration artist', exc_info=True)
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
                    except Exception: self.logger.debug('could not remove fit artist', exc_info=True)
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
        # Group every artist a fit adds under a per-index gid ("fit-<N>") so the
        # WHOLE fit — total line, subpeak curves, text labels — can be deleted as
        # a unit (deleteFit). fit_idx=None keeps the generic "fit" gid, used where
        # no index is known (e.g. the tag-and-clear characterization path).
        gid = "fit" if fit_idx is None else f"fit-{fit_idx}"
        def _tag(a):
            if hasattr(a, "set_gid"): a.set_gid(gid)
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
                self.fitLabelsTextChanged.emit(" ".join(labels) if labels else "")
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
                    is_fit_gid = gid == "fit" or (isinstance(gid, str) and gid.startswith("fit-"))
                    if is_fit_gid or (isinstance(lab, str) and lab.startswith(FIT_PREFIX)):
                        try:
                            a.remove()
                            removed += 1
                        except Exception:
                            pass

            leg = ax.get_legend()
            if leg:
                try: leg.remove()
                except Exception: self.logger.debug('could not remove legend', exc_info=True)

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
        self.fitResultsAppended.emit(title)
        self.fitResultsAppended.emit(resultsText.toPlainText())
        self.fitResultsAppended.emit(' ')

    def setFitLineLabel(self, ax, line, resultsText, spectrumName):
        self.logger.info('setFitLineLabel')
        fitLineLabel = "fit-_-"
        fitLabels = self.listFitLineLabels(ax)
        fitIdxs = [int(label) for label in fitLabels]
        fitIdx = 0

        if line is None:
            self.logger.debug('setFitLineLabel - line is None')
            return None

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
        return fitIdx

    def deleteFit(self, index=None, name=None, ax=None, fit_idx_text=""):
        """Delete the fit lines whose indices appear in `fit_idx_text`, the
        popup field contents supplied by the MainWindow adapter."""
        self.logger.info('deleteFit')

        self._close_alpha_filter_popup()

        if ax is None:
            self.logger.warning('deleteFit - called without ax context; cannot delete fit')
            return
        userFitIdxs = (fit_idx_text or "").split()
        availableFitIdxs = self.listFitLineLabels(ax)
        notAvailableFitIdxs = [fitIdx for fitIdx in userFitIdxs if fitIdx not in availableFitIdxs]
        if notAvailableFitIdxs:
            self.logger.warning('deleteFit - fit line(s) index(es): %s cannot be deleted', notAvailableFitIdxs)
            return

        # Remove the WHOLE fit, not just its labelled total line: every artist a
        # fit drew shares the per-index gid "fit-<N>" (subpeak curves, text
        # labels), while the total line also carries the exact "fit-_-<N>" label.
        removed_any = False
        for fitIdx in userFitIdxs:
            group_gid = "fit-" + fitIdx
            label = FIT_PREFIX + fitIdx
            for artist in list(ax.get_children()):
                a_gid = getattr(artist, "get_gid", lambda: None)()
                a_lab = getattr(artist, "get_label", lambda: "")() or ""
                if a_gid == group_gid or a_lab == label:
                    try:
                        artist.remove()
                        removed_any = True
                    except Exception:
                        self.logger.debug('deleteFit - could not remove artist', exc_info=True)
            self.logger.debug('deleteFit - removed fit group: %s', fitIdx)

        if removed_any:
            leg = ax.get_legend()   # its entries for the deleted curves are now stale
            if leg is not None:
                try: leg.remove()
                except Exception: self.logger.debug('deleteFit - could not remove legend', exc_info=True)
            try:
                ax.figure.canvas.draw_idle()
            except Exception:
                self.logger.debug('deleteFit - could not draw', exc_info=True)

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
        self.fitLabelsTextChanged.emit(textLabels)
