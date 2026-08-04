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
    # Sink management only. Root-logger configuration is single-sourced at
    # MainWindow startup (GUI.py, one logging.basicConfig).
    global _logger
    _logger = lambda message: logging.debug(message)
