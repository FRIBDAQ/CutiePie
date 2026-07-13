"""Per-pad display state.

A ``DisplaySlot`` owns the *display-tier* state for one pad in one tab: the
matplotlib axis/artist, log flag, cutoff, and a cached copy of the spectrum's
axis-definition fields (dim/binx/minx/... ) as they were when the pad was
populated. It is the typed replacement for the untyped
``wTab.spectrum_dict[tab][padIdx]`` dict that ``getSpectrumViewInfo`` /
``setSpectrumViewInfo`` read and write.

Two invariants carried over from the dict it replaces:

* **No counts.** ``data`` is a permanent empty placeholder — the canonical
  counts array lives ONLY in ``SpectrumStore`` and is derived on demand
  Nothing may store counts here.
* **Empty-list default.** Every field defaults to ``[]`` to match the old dict
  template byte-for-byte: callers test truthiness (an un-set ``log`` reads as
  ``[]`` = falsy = linear), so the default must stay falsy-and-equal-to-``[]``.

Strangler migration:

* **C1** introduced this class with the mapping protocol (``__getitem__`` /
  ``__setitem__`` / ``__contains__``) so it could drop into the existing
  ``spectrum_dict`` slot with every call site unchanged; **C2** made ``setGeo``
  build it instead of a raw dict.
* **C3 (done)** migrated all production access to *typed* form — the accessors
  and ``setGeo`` use ``getattr``/``setattr``/``slot.attr`` (dynamic keys stay
  dynamic via ``getattr``, guarded by the same whitelist as before). No
  production code subscripts a slot anymore.

The mapping shim below is therefore no longer on any production path; it is
retained as a small, tested compatibility surface (and a safety net for any
dynamically-built access). This module imports no Qt — it is headless-testable
(the axis/spectrum artists are held as opaque refs).
"""

from dataclasses import dataclass, field


# The 17 per-pad keys, in the historical template order (GUI.setGeo). This is
# also the whitelist enforced by get/setSpectrumViewInfo.
SLOT_KEYS = (
    "name", "dim", "binx", "minx", "maxx", "biny", "miny", "maxy", "data",
    "parameters", "type", "log", "minz", "maxz", "spectrum", "axis", "cutoff",
)


@dataclass
class DisplaySlot:
    name: object = field(default_factory=list)
    dim: object = field(default_factory=list)
    binx: object = field(default_factory=list)
    minx: object = field(default_factory=list)
    maxx: object = field(default_factory=list)
    biny: object = field(default_factory=list)
    miny: object = field(default_factory=list)
    maxy: object = field(default_factory=list)
    data: object = field(default_factory=list)          # placeholder — never counts
    parameters: object = field(default_factory=list)
    type: object = field(default_factory=list)
    log: object = field(default_factory=list)
    minz: object = field(default_factory=list)
    maxz: object = field(default_factory=list)
    spectrum: object = field(default_factory=list)      # matplotlib artist (opaque)
    axis: object = field(default_factory=list)          # matplotlib Axes (opaque)
    cutoff: object = field(default_factory=list)

    # --- dict-compatibility shim (strangler) ---------------------------------
    # Mirrors exactly what the old dict allowed: item get/set for the 17 keys,
    # KeyError for anything else (the template only ever held these keys).

    def __getitem__(self, key):
        if key not in SLOT_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key, value):
        if key not in SLOT_KEYS:
            raise KeyError(key)
        setattr(self, key, value)

    def __contains__(self, key):
        return key in SLOT_KEYS

    def keys(self):
        return iter(SLOT_KEYS)

    def values(self):
        return (getattr(self, k) for k in SLOT_KEYS)

    def items(self):
        return ((k, getattr(self, k)) for k in SLOT_KEYS)

    def as_dict(self):
        """Plain-dict snapshot in template order (for debugging / equivalence)."""
        return {k: getattr(self, k) for k in SLOT_KEYS}
