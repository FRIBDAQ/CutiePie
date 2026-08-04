"""The two metadata tiers and the accessors every controller is built against.
The store is the canonical axis definition; the per-tab slot holds the current
view range, and data-coordinate math must read the store."""

import logging
import threading
import time

from services.display_slot import DisplaySlot, SLOT_KEYS


class ViewState:

    _GATE_NAME_TTL = 2.0  # seconds — max staleness of gate-name label during mouse hover

    def __init__(self, spectra, tabs, get_current_plot, applylistgate,
                 gate_name_fetched, logger=None):
        self.spectra   = spectra
        self.wTab      = tabs
        self._get_current_plot = get_current_plot
        # the REST lookup and the Qt signal that carries its result back stay
        # outside: this object is Qt-free apart from that one emit, and the
        # label it eventually corrects belongs to the hover cluster
        self._applylistgate    = applylistgate
        self._gate_name_fetched = gate_name_fetched
        self.logger = logger or logging.getLogger(__name__)
        self._gate_name_cache: dict = {}   # spectrum_name -> (gate_or_None, monotonic_ts)
        self._gate_name_inflight: set = set()  # names with a background fetch running

    def getSpectrumStoreInfo(self, *info, **identifier):
        # self.logger.info('getSpectrumStoreInfo - info, identifier: %s, %s',info, identifier)
        name = None
        if not identifier and self.getEnlargedSpectrum():
            name = self.getEnlargedSpectrum()[1]
        elif "index" in identifier:
            name = self.nameFromIndex(identifier["index"])
        elif "name" in identifier:
            name = identifier["name"]
        else:
            self.logger.debug('getSpectrumStoreInfo - wrong identifier - expects name=histo_name or index=histo_index or shoud be in zoomed mode')
            # print("getSpectrumViewInfo - wrong identifier - expects name=histo_name or index=histo_index or shoud be in zoomed mode")
            return
        if name is not None:
            return self.spectra.get(name, info[0])

    def setSpectrumViewInfo(self, **info):
        self.logger.debug('setSpectrumViewInfo - info: %s',info)
        name = None
        index = None
        if self.getEnlargedSpectrum():
            index = self.getEnlargedSpectrum()[0]
            name = self.getEnlargedSpectrum()[1]
        elif "index" in info:
            index = info["index"]
            name = self.nameFromIndex(info["index"])
        # commented the following option because can have several versions of the same plot in a window (name not a unique id)
        # elif "name" in info:
        #     name = info["name"]
        else:
            self.logger.debug('setSpectrumViewInfo - wrong identifier - expects index=histo_index or shoud be in zoomed mode')
            # print("setSpectrumViewInfo - wrong identifier - expects index=histo_index or shoud be in zoomed mode")
            return
        # print("Simon - setSpectrumViewInfo - ", index,name,info["index"])
        for key, value in info.items():
            if key in SLOT_KEYS and index is not None:
                if index not in self.wTab.tabSlots(self.wTab.currentIndex()):
                    self.logger.debug('setSpectrumViewInfo - %s not in tabSlots', name)
                    return
                slot = self.wTab.tabSlots(self.wTab.currentIndex())[index]
                setattr(slot, key, value)          # typed DisplaySlot field (key is whitelisted above)
                #set axes info at the same time than spectrum
                if key == "spectrum":
                    slot.axis = value.axes

    def getSpectrumViewInfo(self, *info, **identifier):
        self.logger.debug('getSpectrumViewInfo - info, identifier: %s, %s', info, identifier)
        name = None
        index = None
        if self.getEnlargedSpectrum():
            index = self.getEnlargedSpectrum()[0]
            # name = self.getEnlargedSpectrum()[1]
        elif "index" in identifier:
            index = identifier["index"]
            # name = self.nameFromIndex(identifier["index"])
        # commented the following option because can have several versions of the same plot in a window (name not a unique id)
        # elif "name" in identifier:
        #     name = identifier["name"]
        else:
            self.logger.debug('getSpectrumViewInfo - wrong identifier - expects index=histo_index or shoud be in zoomed mode')
            # print("getSpectrumViewInfo - wrong identifier - expects index=histo_index or shoud be in zoomed mode")
            return
        if index is not None and index in self.wTab.tabSlots(self.wTab.currentIndex()) and info[0] in SLOT_KEYS:
            return getattr(self.wTab.tabSlots(self.wTab.currentIndex())[index], info[0])   # typed access (info[0] whitelisted above)

    def getSpectrumViewDict(self):
        return self.wTab.tabSlots(self.wTab.currentIndex())

    def getSpectrumStoreDict(self):
        return self.spectra.as_dict()

    def nameFromIndex(self, index):
        # self.logger.info('nameFromIndex - index: %s', index)
        #Can call getSpectrumViewInfo and setSpectrumViewInfo with an identifier but still check if in zoom mode,
        #which is important for autoScaleAxis/setAxisScale
        if self.getEnlargedSpectrum():
            return self.getEnlargedSpectrum()[1]
        elif index in self._get_current_plot().h_dict_geo:
            return self._get_current_plot().h_dict_geo[index]

    def setGeo(self, index, name):
        self.logger.info('setGeo - index, name: %s, %s', index, name)
        self._get_current_plot().h_dict_geo[index] = name
        #Set also here the per-tab slots with only the spectra defined in the geo
        if index not in self.wTab.tabSlots(self.wTab.currentIndex()):
            self.wTab.tabSlots(self.wTab.currentIndex())[index] = DisplaySlot()
        slot = self.wTab.tabSlots(self.wTab.currentIndex())[index]
        slot.name = name
        # Initialize with the same info as in self.spectra. "data" is
        # intentionally NOT copied: the canonical array lives solely in the
        # SpectrumStore and is derived (with cutoff) on demand by the plot
        # controller, so it is never duplicated into the per-tab display tier.
        record = self.spectra.get_record(name)
        if record is None:
            self.logger.warning('setGeo - %s not in SpectrumStore; slot left with name only', name)
            return
        for key, value in record.items():
            if key == "data":
                continue
            setattr(slot, key, value)          # typed DisplaySlot field

    def getGeo(self):
        return self._get_current_plot().h_dict_geo

    def setEnlargedSpectrum(self, index, name):
        self.logger.info('setEnlargedSpectrum')
        self.wTab.setZoomInfo(self.wTab.currentIndex(), None)
        if index is not None and name is not None:
            self.wTab.setZoomInfo(self.wTab.currentIndex(), [index, name])

    def getEnlargedSpectrum(self):
        # self.logger.info('getEnlargedSpectrum')
        result = None
        if self.wTab.zoomInfo(self.wTab.currentIndex()):
            result = self.wTab.zoomInfo(self.wTab.currentIndex())
        return result

    def getAppliedGateName(self, **identifier):
        """Return the gate name applied to a spectrum — cache only, never
        blocking. Stale-while-revalidate: a fresh cache entry is returned
        as-is; a cold/expired one returns the stale value (or None)
        immediately and triggers a background REST fetch."""
        spectrumName = None
        if "index" in identifier:
            spectrumName = self.nameFromIndex(identifier["index"])
        elif "name" in identifier:
            spectrumName = identifier["name"]
        else:
            self.logger.debug('getAppliedGateName - wrong identifier')
            return None
        now = time.monotonic()
        cached = self._gate_name_cache.get(spectrumName)
        if cached is not None and (now - cached[1]) < self._GATE_NAME_TTL:
            return cached[0]
        self._refreshGateNameAsync(spectrumName)
        return cached[0] if cached is not None else None

    def _refreshGateNameAsync(self, spectrumName):
        """Fetch applylistgate on a worker thread; result lands via _gateNameFetched."""
        if spectrumName in self._gate_name_inflight:
            return
        self._gate_name_inflight.add(spectrumName)

        def fetch():
            gate = self._applylistgate(spectrumName)
            try:
                self._gate_name_fetched.emit(spectrumName, gate)
            except RuntimeError:
                pass  # window destroyed during shutdown

        threading.Thread(target=fetch, daemon=True,
                         name=f"gate-name-fetch-{spectrumName}").start()

    def storeGateName(self, spectrumName, result):
        """Record a fetched gate name and clear its in-flight mark. Called by
        MainWindow's `_on_gate_name_fetched` slot once the worker thread's
        result has crossed back onto the GUI thread."""
        self._gate_name_inflight.discard(spectrumName)
        self._gate_name_cache[spectrumName] = (result, time.monotonic())
