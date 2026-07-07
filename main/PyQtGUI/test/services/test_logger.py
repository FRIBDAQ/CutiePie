import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gui'))

import logging

import logger as logger_mod


def test_setup_logging_does_not_configure_root(monkeypatch):
    # L2: root-logger config is single-sourced at startup; setup_logging must
    # NOT call basicConfig (its call was a dead no-op and is now removed).
    calls = []
    monkeypatch.setattr(logging, "basicConfig",
                        lambda *a, **k: calls.append((a, k)))
    logger_mod.setup_logging("/tmp/whatever.log")
    assert calls == []                      # root config untouched


def test_setup_logging_routes_log_to_logging_debug(monkeypatch):
    # After setup_logging, logger.log() forwards to logging.debug (the sink).
    captured = []
    monkeypatch.setattr(logging, "debug", lambda msg: captured.append(msg))
    logger_mod.setup_logging()
    logger_mod.log("hello")
    assert captured == ["hello"]


def test_set_logger_overrides_the_sink():
    # The jupyter path swaps the sink to the docked console via set_logger.
    captured = []
    logger_mod.set_logger(lambda msg: captured.append(msg))
    logger_mod.log("  world  ")             # log() strips before forwarding
    assert captured == ["world"]


def test_log_is_safe_when_no_sink_set():
    logger_mod.set_logger(None)
    logger_mod.log("noop")                  # must not raise
