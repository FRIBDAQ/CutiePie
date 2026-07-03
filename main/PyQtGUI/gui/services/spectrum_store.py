import logging

import numpy as np

logger = logging.getLogger(__name__)

_VALID_KEYS = frozenset((
    "dim", "binx", "minx", "maxx",
    "biny", "miny", "maxy",
    "data", "parameters", "type",
))


class SpectrumStore:
    """Owns the canonical REST-side spectrum registry (replaces spectrum_dict_rest)."""

    def __init__(self):
        self._store: dict = {}

    def set(self, name: str, allow_data_replacement: bool = False, **info) -> None:
        """Upsert valid-key fields for the named spectrum. Invalid keys are silently ignored.

        `data` arrays are live shared-memory views (BUGS.md B4): replacing one
        with an array that does not alias it silently freezes the spectrum
        (the C1 regression class), so such writes are refused and logged.
        Callers installing views from a NEW mirror — connect/reconnect and
        trace adds after a CPyConverter.Update() — must pass
        allow_data_replacement=True.
        """
        valid = {k: v for k, v in info.items() if k in _VALID_KEYS}
        if not valid:
            return
        if name not in self._store:
            self._store[name] = dict(valid)
            return
        if "data" in valid and not allow_data_replacement:
            old = self._store[name].get("data")
            new = valid["data"]
            if isinstance(old, np.ndarray) and not (
                    isinstance(new, np.ndarray) and np.shares_memory(old, new)):
                logger.error(
                    "SpectrumStore.set(%r): refusing to replace the live shm "
                    "data view with a non-aliasing array (B4/C1 regression "
                    "class — derive on demand instead; pass "
                    "allow_data_replacement=True only for new-mirror views)",
                    name)
                valid.pop("data")
                if not valid:
                    return
        self._store[name].update(valid)

    def get(self, name: str, key: str):
        """Return store[name][key], or None if name or key is absent."""
        record = self._store.get(name)
        if record is None:
            return None
        return record.get(key)

    def remove(self, name: str) -> None:
        """Delete entry. No-op if absent — callers always guard with `in` first."""
        self._store.pop(name, None)

    def get_record(self, name: str):
        """Return the live record dict for `name`, or None if absent.

        Field values are shared by reference and meant to be read; use set() to
        change them. Prefer this over as_dict() when you only need one spectrum."""
        return self._store.get(name)

    def contains(self, name: str) -> bool:
        return name in self._store

    def all_names(self) -> list:
        return sorted(self._store)

    def as_dict(self) -> dict:
        """Return a shallow snapshot copy of the registry: {name: record}.

        The top-level mapping is copied so callers cannot add or remove spectra by
        mutating the result. Per-spectrum record objects are shared by reference and
        must not be mutated in place — use set()/remove() to change the store."""
        return dict(self._store)
