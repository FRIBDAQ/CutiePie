import logging
_logger = None

def log(message):
    if _logger is not None:
        try:
            _logger(str(message).strip())
        except Exception as e:
            logging.debug(e)

def set_logger(logger):
    global _logger
    _logger = logger

def setup_logging(logfile=None):
    # Sink management only (AUDIT L2). Root-logger configuration is
    # single-sourced at MainWindow startup (GUI.py, one logging.basicConfig).
    # This function used to call basicConfig again (filename/DEBUG/asctime),
    # but root already has a handler by the time it runs, so that call was
    # ALWAYS a no-op — the file/DEBUG config never took effect. Dropped the
    # dead call; behaviour is unchanged (log() still routes to logging.debug).
    # `logfile` is retained for call-site compatibility and marks where real
    # file logging WOULD attach if it were deliberately enabled (a separate,
    # behaviour-changing decision — not done here).
    global _logger
    _logger = lambda message: logging.debug(message)
