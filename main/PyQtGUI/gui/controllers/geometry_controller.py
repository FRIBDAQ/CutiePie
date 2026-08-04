"""Geometry save/load and session apply. The parsing and serialising live in
``services/geometry_io.py``; what is here is the orchestration around it."""

import logging

from PyQt5.QtWidgets import QMessageBox

from services import geometry_io


class GeometryController:

    def __init__(self, tabs, conf, spectra,
                 get_current_plot, set_current_plot,
                 get_store_info, get_view_info, set_view_info,
                 get_geo, set_geo,
                 plot_controller, gate_manager, sum_region_manager,
                 connection_manager, gate_popup, sum_region_popup,
                 set_canvas_layout, add_plot, auto_update_start,
                 bind_dynamic_signal, tab_geo_widget_and_flags,
                 open_file_dialog, save_file_dialog,
                 parent_widget=None, logger=None):
        self._tabs                = tabs                # wTab
        self._conf                = conf                # wConf (the geo combos)
        self._spectra             = spectra
        self._get_current_plot    = get_current_plot
        self._set_current_plot    = set_current_plot    # applySession reassigns it
        self._get_store_info      = get_store_info
        self._get_view_info       = get_view_info
        self._set_view_info       = set_view_info
        self._get_geo             = get_geo
        self._set_geo             = set_geo
        self._plot_controller     = plot_controller
        self._gate_manager        = gate_manager
        self._sum_region_manager  = sum_region_manager
        self._connection_manager  = connection_manager
        self._gate_popup          = gate_popup
        self._sum_region_popup    = sum_region_popup
        self._set_canvas_layout   = set_canvas_layout
        self._add_plot            = add_plot
        self._auto_update_start   = auto_update_start
        self._bind_dynamic_signal = bind_dynamic_signal
        self._tab_geo_widget_and_flags = tab_geo_widget_and_flags
        # the two file pickers stay in MainWindow: they are generic, they need
        # a parent widget, and the next controller that needs one will want the
        # same pair rather than its own copy
        self._open_file_dialog    = open_file_dialog
        self._save_file_dialog    = save_file_dialog
        self._parent_widget       = parent_widget
        self.logger               = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def saveGeo(self):
        fileName = self._save_file_dialog()
        self.logger.info('saveGeo - fileName: %s', fileName)
        if not fileName:
            return
        try:
            properties = {}
            geo = self._get_geo()
            for index in range(len(geo)):
                try:
                    h_name = geo[index]
                    x_range, y_range = self._plot_controller.getAxisProperties(index)
                    scale = True if self._get_view_info("log", index=index) else False
                    properties[index] = {"name": h_name, "x": x_range, "y": y_range, "scale": scale}
                except Exception:
                    self.logger.debug('saveGeo - pad %s skipped', index, exc_info=True)
                    properties[index] = {"name": '', "x": None, "y": None, "scale": None}
            tmp_text = geometry_io.serialize_geometry(
                self._conf.histo_geo_row.currentText(),
                self._conf.histo_geo_col.currentText(),
                properties)
            with open(fileName, "w") as f:
                f.write(tmp_text)
        except Exception:
            # was a bare `except:` that logged at debug and still showed
            # the success dialog (shown before the write, at that)
            self.logger.exception('saveGeo - failed to save %s', fileName)
            QMessageBox.warning(self._parent_widget, "Saving...", "Could not save the window configuration — see the log.")
            return
        QMessageBox.about(self._parent_widget, "Saving...", "Window configuration saved!")

    def saveGeoAll(self):
        """Save EVERY tab's geometry as one v2 session file (design
        2026-07-10). Reads the per-tab VIEW tier (slots), not live axes —
        background tabs' axes aren't reliably current, and the view tier is
        exactly what load consumes."""
        fileName = self._save_file_dialog()
        self.logger.info('saveGeoAll - fileName: %s', fileName)
        if not fileName:
            return
        try:
            tabs = []
            for tabIdx in sorted(self._tabs.sessions.indices()):
                nRow, nCol = self._tabs.tabLayout(tabIdx)
                plotW = self._tabs.plot(tabIdx)
                slots = self._tabs.tabSlots(tabIdx) if tabIdx in self._tabs.sessions else {}
                properties = {}
                for index in range(nRow * nCol):
                    try:
                        h_name = plotW.h_dict_geo.get(index, "")
                        slot = slots.get(index)
                        x_range = y_range = None
                        scale = False
                        if slot is not None:
                            # slot fields default to [] (DisplaySlot); 0.0 is a
                            # legitimate limit, so test emptiness, not truthiness
                            if slot.minx not in ("", [], None) and slot.maxx not in ("", [], None):
                                x_range = [float(slot.minx), float(slot.maxx)]
                            if slot.miny not in ("", [], None) and slot.maxy not in ("", [], None):
                                y_range = [float(slot.miny), float(slot.maxy)]
                            scale = bool(slot.log) if slot.log not in ([], None) else False
                        properties[index] = {"name": h_name if h_name != "empty" else "",
                                             "x": x_range, "y": y_range, "scale": scale}
                    except Exception:
                        self.logger.debug('saveGeoAll - tab %s pad %s skipped',
                                          tabIdx, index, exc_info=True)
                        properties[index] = {"name": '', "x": None, "y": None, "scale": None}
                tabs.append({"name": self._tabs.tabText(tabIdx), "row": nRow,
                             "col": nCol, "geo": properties})
            tmp_text = geometry_io.serialize_session(tabs)
            with open(fileName, "w") as f:
                f.write(tmp_text)
        except Exception:
            self.logger.exception('saveGeoAll - failed to save %s', fileName)
            QMessageBox.warning(self._parent_widget, "Saving...", "Could not save the session — see the log.")
            return
        QMessageBox.about(self._parent_widget, "Saving...", "All tabs saved!")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _resolveSpectrumName(self, name):
        """Return a spectrum name present in the store that matches `name`, tolerating
        case differences (legacy .win files often store names upper-cased). Returns the
        exact name if it exists, a unique case-insensitive match otherwise, or None."""
        if self._get_store_info("dim", name=name) is not None:
            return name
        lowered = name.lower()
        matches = [n for n in self._spectra.all_names() if n.lower() == lowered]
        return matches[0] if len(matches) == 1 else None

    def applyGeometryToCurrentTab(self, infoGeo):
        """Apply one tab's geometry payload ({"row","col","geo"}) to the
        CURRENT tab. Shared by the single-tab load and the session load, so
        both run the same code."""
        # the file stores counts, the combos are 0-indexed
        row = infoGeo["row"] - 1
        col = infoGeo["col"] - 1
        index_row = row
        index_col = col

        notFound = []
        if index_row >= 0 and index_col >= 0:
            self._conf.histo_geo_row.setCurrentIndex(index_row)
            self._conf.histo_geo_col.setCurrentIndex(index_col)
            self._set_canvas_layout()
            for index, val_dict in infoGeo["geo"].items():
                if not val_dict["name"]:
                    continue
                resolved = self._resolveSpectrumName(val_dict["name"])
                if resolved is None:
                    notFound.append(val_dict["name"])
                    continue

                self._set_geo(index, resolved)
                self._set_view_info(log=val_dict["scale"], index=index)
                # Old .win files may omit the view range (no "Expanded"); when it
                # is absent the spectrum keeps its natural full range from the store.
                if val_dict.get("x") is not None:
                    self._set_view_info(minx=val_dict["x"][0], index=index)
                    self._set_view_info(maxx=val_dict["x"][1], index=index)
                if val_dict.get("y") is not None:
                    self._set_view_info(miny=val_dict["y"][0], index=index)
                    self._set_view_info(maxy=val_dict["y"][1], index=index)

            if len(notFound) > 0:
                self.logger.warning('loadGeo - definition not found for: %s', notFound)

            cp = self._get_current_plot()
            cp.isLoaded = True
            self._tabs.setSelectedPad(self._tabs.currentIndex(), None)
            cp.selected_plot_index = None
            cp.next_plot_index = -1

        self._add_plot()
        self._plot_controller.updatePlot()
        self._get_current_plot().isLoaded = False
        return notFound

    def loadGeo(self):
        fileName = self._open_file_dialog()
        self.logger.info('loadGeo - fileName: %s', fileName)
        if not fileName:
            return
        # Detect the format instead of assuming a single-tab file: a multi-tab
        # session dropped here used to crash with KeyError 'row'.
        tagged = geometry_io.read_geometry_any(fileName, self.logger)
        if tagged is None:
            QMessageBox.warning(self._parent_widget, "Load Geometry",
                                "Not a readable geometry file — nothing was changed.")
            return
        kind, payload = tagged
        if kind == "session":
            nTabs = len(payload.get("tabs") or [])
            reply = QMessageBox.question(
                self._parent_widget, "Load Geometry",
                f"This file is a multi-tab session ({nTabs} tabs). "
                "Load all tabs? This replaces your current tabs.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.applySession(payload["tabs"])
            return
        self.applyGeometryToCurrentTab(payload)

    def loadGeoAll(self):
        """Load a session file, REPLACING all tabs (user-approved semantics,
        design 2026-07-10). The file is parsed and validated COMPLETELY before
        any tab is touched, so a bad file can never half-destroy the
        workspace."""
        fileName = self._open_file_dialog()
        self.logger.info('loadGeoAll - fileName: %s', fileName)
        if not fileName:
            return
        tagged = geometry_io.read_geometry_any(fileName, self.logger)
        if tagged is None:
            QMessageBox.warning(self._parent_widget, "Load All Tabs",
                                "Not a readable geometry/session file — nothing was changed.")
            return
        kind, payload = tagged
        try:
            if kind == "single":
                tabsInfo = [{"name": "Tab 1", "row": int(payload["row"]),
                             "col": int(payload["col"]), "geo": payload["geo"]}]
            else:
                tabsInfo = payload["tabs"]
        except (KeyError, TypeError, ValueError):
            QMessageBox.warning(self._parent_widget, "Load All Tabs",
                                "Geometry file is missing required fields — nothing was changed.")
            return

        self.applySession(tabsInfo)

    def applySession(self, tabsInfo):
        """Replace ALL tabs with the parsed session `tabsInfo` (a list of
        {"name","row","col","geo"}). Shared by loadGeoAll and loadGeo's
        session branch."""
        # quiesce: same guards clickedTab uses, then stop the auto-update tick
        cp = self._get_current_plot()
        if cp.toCreateGate or cp.toEditGate or self._gate_popup.isVisible():
            self._gate_manager.cancelGate()
        if cp.toCreateSumRegion or self._sum_region_popup.isVisible():
            self._sum_region_manager.cancelSumRegion()
        self._connection_manager._stop_auto_thread()

        # rebuild the tab set with existing primitives only (danger
        # zone: deleteTab reindexes the parallel dicts and plt.closes figures)
        self._tabs.setCurrentIndex(0)
        self._set_current_plot(self._tabs.plot(0))
        while len(self._tabs.sessions) > 1:
            self._tabs.deleteTab(len(self._tabs.sessions) - 1)
        for k in range(1, len(tabsInfo)):
            self._tabs.addTab(k)

        notFoundByTab = {}
        for k, tabInfo in enumerate(tabsInfo):
            self._tabs.setCurrentIndex(k)
            self._tab_geo_widget_and_flags(k)
            infoGeo = {"row": tabInfo["row"], "col": tabInfo["col"], "geo": tabInfo["geo"]}
            try:
                notFound = self.applyGeometryToCurrentTab(infoGeo)
            except TypeError:
                self.logger.debug('applySession - TypeError applying tab %s', k, exc_info=True)
                notFound = []
            if notFound:
                notFoundByTab[str(tabInfo.get("name") or f"Tab {k+1}")] = notFound
            self._tabs.setTabText(k, str(tabInfo.get("name") or f"Tab {k+1}"))

        self._tabs.setCurrentIndex(0)
        self._tab_geo_widget_and_flags(0)
        self._bind_dynamic_signal()
        if notFoundByTab:
            lines = "\n".join(f"{tab}: {', '.join(names)}"
                              for tab, names in notFoundByTab.items())
            QMessageBox.warning(self._parent_widget, "Load All Tabs",
                                "Some spectra were not found on the connected SpecTcl; "
                                "their pads were left empty:\n\n" + lines)
        self._auto_update_start()
