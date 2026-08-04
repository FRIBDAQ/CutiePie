"""The composition root builds nothing before the thing it is built from.
`MainWindow.__init__` is a sequence of phases, and `_build_services` inside it
assigns ~15 collaborators in a fixed order, each constructed against the ones
before it."""

import ast
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))
sys.path.insert(0, os.path.dirname(__file__))

import gui_stubs

# the phases __init__ calls, in the order it calls them
PHASES = ["_setup_logging", "_build_widgets", "_build_services",
          "_init_runtime_state", "_wire_signals"]


def _mainwindow_class():
    gui = gui_stubs.import_gui()
    with open(gui.__file__.replace(".pyc", ".py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == "MainWindow")


def _own_attributes(cls):
    """Attributes MainWindow assigns to itself, anywhere in the class. Only
    these can be ordered wrongly."""
    own = set()
    for node in ast.walk(cls):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                a = _self_attr(t)
                if a:
                    own.add(a)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            a = _self_attr(node.target)
            if a:
                own.add(a)
    return own


def _outside_lambdas(stmt):
    """Every node in `stmt` except those inside a lambda body.

    A lambda is resolved when it is CALLED, not where it is written — the
    composition root passes plenty of `lambda: self.currentPlot` seams for
    things built later, and that is the whole point of them.
    """
    skip = set()
    for node in ast.walk(stmt):
        if isinstance(node, ast.Lambda):
            for inner in ast.walk(node):
                skip.add(id(inner))
    return [n for n in ast.walk(stmt) if id(n) not in skip]


def _self_attr(node):
    """`self.x` -> "x", anything else -> None."""
    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "self"):
        return node.attr
    return None


def test_no_phase_reads_an_attribute_before_it_is_assigned():
    cls = _mainwindow_class()
    methods = {m.name: m for m in cls.body if isinstance(m, ast.FunctionDef)}
    own = _own_attributes(cls)
    # methods and class attributes exist before __init__ runs
    known = set(methods)
    for node in cls.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    known.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            known.add(node.target.id)

    problems = []
    for phase in PHASES:
        body = methods.get(phase)
        if body is None:
            continue
        for stmt in body.body:
            # what this statement READS, before recording what it assigns
            targets = []
            if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                tlist = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                targets = [a for a in (_self_attr(t) for t in tlist) if a]
            for node in _outside_lambdas(stmt):
                attr = _self_attr(node)
                if (attr and attr in own and attr not in known
                        and attr not in targets):
                    problems.append(f"{phase}: reads self.{attr} before it is assigned")
            known.update(targets)

    assert not problems, (
        "the composition root uses something it has not built yet:\n  "
        + "\n  ".join(dict.fromkeys(problems)))


def test_view_state_is_built_before_every_service_that_takes_it():
    """The specific ordering stage 9 got wrong, pinned by name so a future
    reshuffle of `_build_services` cannot quietly reintroduce it."""
    cls = _mainwindow_class()
    build = next(m for m in cls.body
                 if isinstance(m, ast.FunctionDef) and m.name == "_build_services")
    assigned_at = None
    for stmt in build.body:
        if isinstance(stmt, ast.Assign) and any(
                _self_attr(t) == "view_state" for t in stmt.targets):
            assigned_at = stmt.lineno
            break
    assert assigned_at is not None, "_build_services no longer builds view_state"

    first_use = None
    for node in _outside_lambdas(build):
        if (isinstance(node, ast.Attribute)
                and _self_attr(node.value) == "view_state"):
            if first_use is None or node.lineno < first_use:
                first_use = node.lineno
    if first_use is not None:
        assert assigned_at < first_use, (
            f"view_state is used at line {first_use} but only assigned at "
            f"{assigned_at} — this is the stage-9 startup crash")
