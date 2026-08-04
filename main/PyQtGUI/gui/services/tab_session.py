"""Per-tab session state — the Qt-free core of the tab registry. ``Tabs`` used
to keep six parallel containers keyed by tab index (``wPlot``,
``spectrum_dict``, ``zoomPlotInfo``, ``countClickTab``,
``selected_plot_index_bak``, ``layout``), re-synchronized by hand on every
add, delete and swap."""

from collections.abc import MutableMapping, MutableSequence
from dataclasses import dataclass, field


@dataclass
class TabSession:
    """Everything one tab owns.

    ``widget`` is the Plot; ``slots`` maps pad index to DisplaySlot;
    ``zoom_info`` is ``[padIndex, name]`` while a pad is enlarged, and ``None``
    or ``[]`` otherwise — callers test truthiness, so both empty forms are kept
    rather than normalized to one.
    """
    widget: object = None
    slots: dict = field(default_factory=dict)
    zoom_info: object = field(default_factory=list)
    layout: list = field(default_factory=lambda: [1, 1])
    click_bound: bool = False
    selected_bak: object = None


class TabSessionRegistry:
    """Ordered ``{tab_index: TabSession}`` with contiguous integer keys.

    Delete renumbers so keys stay ``0..n-1``, matching the hand-rolled
    ``deleteDictEntry`` it replaces: tab indices are positions, and Qt has
    already re-numbered its own tabs by the time we are called.
    """

    def __init__(self):
        self._sessions = {}

    def add(self, index, session):
        self._sessions[index] = session
        return session

    def __getitem__(self, index):
        return self._sessions[index]

    def __len__(self):
        return len(self._sessions)

    def __contains__(self, index):
        return index in self._sessions

    def indices(self):
        return list(self._sessions.keys())

    def delete(self, index):
        """Remove one tab and re-close the numbering. Returns the removed
        session so the caller can tear down its widget before the indices
        shift under it."""
        removed = self._sessions.pop(index)   # KeyError on an unknown index
        self._sessions = {i: s for i, s in enumerate(self._sessions.values())}
        return removed

    def swap(self, indexFrom, indexTo):
        a, b = self._sessions[indexFrom], self._sessions[indexTo]
        self._sessions[indexFrom], self._sessions[indexTo] = b, a

    def mapping_view(self, fieldName):
        return _MappingView(self, fieldName)

    def sequence_view(self, fieldName):
        return _SequenceView(self, fieldName)


class _MappingView(MutableMapping):
    """Live ``{tab_index: <one field>}`` view. Writes land on the session."""

    def __init__(self, registry, fieldName):
        self._registry = registry
        self._field = fieldName

    def __getitem__(self, index):
        return getattr(self._registry[index], self._field)

    def __setitem__(self, index, value):
        setattr(self._registry[index], self._field, value)

    def __delitem__(self, index):
        raise TypeError("delete the whole tab session, not one of its fields")

    def __iter__(self):
        return iter(self._registry.indices())

    def __len__(self):
        return len(self._registry)


class _SequenceView(MutableSequence):
    """Live list-shaped view over one field, indexed by tab index.
    Out-of-range raises ``IndexError``, not the registry's ``KeyError``: the
    sequence protocol iterates by walking indices until ``IndexError``, so a
    ``KeyError`` here would escape out of any plain ``for``/``list()`` over
    the view instead of ending it."""

    def __init__(self, registry, fieldName):
        self._registry = registry
        self._field = fieldName

    def _session(self, index):
        if index < 0:
            index += len(self._registry)
        if index not in self._registry:
            raise IndexError("tab index out of range")
        return self._registry[index]

    def __getitem__(self, index):
        return getattr(self._session(index), self._field)

    def __setitem__(self, index, value):
        setattr(self._session(index), self._field, value)

    def __delitem__(self, index):
        raise TypeError("delete the whole tab session, not one of its fields")

    def __len__(self):
        return len(self._registry)

    def insert(self, index, value):
        raise TypeError("add a tab session, not one field of one")
