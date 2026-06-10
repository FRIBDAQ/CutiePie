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
            self._store[name] = dict(valid)
        else:
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
