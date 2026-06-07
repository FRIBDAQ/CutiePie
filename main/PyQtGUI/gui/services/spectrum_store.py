_VALID_KEYS = frozenset((
    "dim", "binx", "minx", "maxx",
    "biny", "miny", "maxy",
    "data", "parameters", "type",
))


class SpectrumStore:
    """Owns the canonical REST-side spectrum registry (replaces spectrum_dict_rest)."""

    def __init__(self):
        self._store: dict = {}

    def set(self, name: str, **info) -> None:
        """Upsert valid-key fields for the named spectrum. Invalid keys are silently ignored."""
        valid = {k: v for k, v in info.items() if k in _VALID_KEYS}
        if not valid:
            return
        if name not in self._store:
            self._store[name] = {k: [] for k in _VALID_KEYS}
        self._store[name].update(valid)

    def get(self, name: str, key: str):
        """Return store[name][key], or None if name or key is absent."""
        record = self._store.get(name)
        if record is None:
            return None
        return record.get(key)

    def remove(self, name: str) -> None:
        """Delete entry. Raises KeyError if absent (mirrors prior `del` behaviour)."""
        del self._store[name]

    def contains(self, name: str) -> bool:
        return name in self._store

    def all_names(self) -> list:
        return sorted(self._store)

    def as_dict(self) -> dict:
        """Return the live internal dict. Callers that iterate it see mutations immediately."""
        return self._store
