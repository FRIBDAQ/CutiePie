# alpha_filter_dialog.py
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
)
from PyQt5.QtCore import Qt


class AlphaChainIsoFilterDialog(QDialog):
    """
    Modeless popup to filter AlphaEMGMultiSigma components by chain/isotope.

    Expects `fitln` (Line2D) to have the stash you added in step (1) of the fitter:
      _chains, _chain_to_isos, _isotopes, _iso_lines, _iso_texts
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Alpha components filter")
        self.setWindowModality(Qt.NonModal)

        self._fitln = None

        self.chainCombo = QComboBox()
        self.isoCombo = QComboBox()

        self.chainCombo.currentTextChanged.connect(self._on_chain_changed)
        self.isoCombo.currentTextChanged.connect(self.apply)

        row = QHBoxLayout()
        row.addWidget(QLabel("Chain:"))
        row.addWidget(self.chainCombo, 1)
        row.addWidget(QLabel("Isotope:"))
        row.addWidget(self.isoCombo, 1)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)

        lay = QVBoxLayout(self)
        lay.addLayout(row)
        lay.addWidget(btn_close, alignment=Qt.AlignRight)

    def supports(self, fitln) -> bool:
        need = ("_chains", "_chain_to_isos", "_isotopes", "_iso_lines", "_iso_texts")
        return (fitln is not None) and all(hasattr(fitln, k) for k in need)

    def bind(self, fitln):
        """Bind to a new fit result (fit total line) and populate dropdowns."""
        self._fitln = fitln

        chains = list(getattr(fitln, "_chains", []))
        isotopes = list(getattr(fitln, "_isotopes", []))

        self.chainCombo.blockSignals(True)
        self.isoCombo.blockSignals(True)

        self.chainCombo.clear()
        self.chainCombo.addItem("All chains")
        for ch in chains:
            self.chainCombo.addItem(ch)
        self.chainCombo.setCurrentText("All chains")

        self.isoCombo.clear()
        self.isoCombo.addItem("All isotopes")
        self.isoCombo.addItem("Total only")
        for iso in isotopes:
            self.isoCombo.addItem(iso)
        self.isoCombo.setCurrentText("All isotopes")

        self.chainCombo.blockSignals(False)
        self.isoCombo.blockSignals(False)

        self.apply()

    def _on_chain_changed(self, chain_choice: str):
        fitln = self._fitln
        if fitln is None:
            return

        chain_to_isos = getattr(fitln, "_chain_to_isos", {}) or {}
        all_isos = list(getattr(fitln, "_isotopes", []) or [])

        keep_iso = self.isoCombo.currentText()

        self.isoCombo.blockSignals(True)
        self.isoCombo.clear()
        self.isoCombo.addItem("All isotopes")
        self.isoCombo.addItem("Total only")

        if chain_choice == "All chains":
            for iso in all_isos:
                self.isoCombo.addItem(iso)
        else:
            for iso in chain_to_isos.get(chain_choice, []):
                self.isoCombo.addItem(iso)

        idx = self.isoCombo.findText(keep_iso)
        if idx >= 0:
            self.isoCombo.setCurrentIndex(idx)
        else:
            self.isoCombo.setCurrentText("All isotopes")

        self.isoCombo.blockSignals(False)
        self.apply()

    def apply(self):
        """Apply current dropdown selections to the bound plot artists."""
        fitln = self._fitln
        if fitln is None:
            return

        ax = getattr(fitln, "axes", None)
        if ax is None:
            return

        chain_choice = self.chainCombo.currentText()
        iso_choice = self.isoCombo.currentText()

        all_isos = list(getattr(fitln, "_isotopes", []) or [])
        chain_to_isos = getattr(fitln, "_chain_to_isos", {}) or {}
        iso_lines = getattr(fitln, "_iso_lines", {}) or {}
        iso_texts = getattr(fitln, "_iso_texts", {}) or {}

        # chain filter set
        if chain_choice == "All chains":
            allowed = set(all_isos)
        else:
            allowed = set(chain_to_isos.get(chain_choice, []) or [])

        # isotope selection set
        if iso_choice == "Total only":
            show = set()
        elif iso_choice == "All isotopes":
            show = allowed
        else:
            show = {iso_choice} if iso_choice in allowed else set()

        # total fit always visible
        try:
            fitln.set_visible(True)
        except Exception:
            pass

        # components + labels
        for iso, lines in iso_lines.items():
            vis = (iso in show)
            for ln in lines:
                try:
                    ln.set_visible(vis)
                except Exception:
                    pass

        for iso, texts in iso_texts.items():
            vis = (iso in show)
            for t in texts:
                try:
                    t.set_visible(vis)
                except Exception:
                    pass

        # legend rebuild (keeps it consistent with what’s visible)
        leg = ax.get_legend()
        if leg is not None:
            try:
                leg.remove()
            except Exception:
                pass

        handles = [fitln]
        labels = ["fit total"]
        for iso in all_isos:
            if iso in show and iso_lines.get(iso):
                handles.append(iso_lines[iso][0])
                labels.append(iso)

        try:
            ax.legend(handles, labels, loc="best", frameon=False)
        except Exception:
            pass

        try:
            ax.figure.canvas.draw_idle()
        except Exception:
            pass
