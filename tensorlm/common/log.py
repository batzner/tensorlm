import logging
import os

# All code should be called from the root directory
from logging.handlers import RotatingFileHandler

_log_path = 'out/log/log.log'
_log_dir = os.path.dirname(_log_path)
if not os.path.isdir(_log_dir):
    os.makedirs(_log_dir)

# Make all logs look equal
logFormatter = logging.Formatter(
    "%(asctime)s [%(levelname)-5.5s] %(message)s")

# Log to a file from each module
_fileHandler = RotatingFileHandler(_log_path, maxBytes=(1048576 * 5), backupCount=7)
_fileHandler.setFormatter(logFormatter)

# Log to the console from each module
_consoleHandler = logging.StreamHandler()
_consoleHandler.setFormatter(logFormatter)


def get_logger(name):
    # Log to stderr with the module name
    logger = logging.getLogger(name)
    logger.addHandler(_fileHandler)
    logger.addHandler(_consoleHandler)
    logger.setLevel(logging.DEBUG)
    return logger
