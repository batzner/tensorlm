# Copyright (c) 2017 Kilian Batzner All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
"""Module for defining a logger that modules can obtain using get_logger()."""

import logging
import os

# All code should be called from the root directory
from logging.handlers import RotatingFileHandler

_log_path = 'out/log/log.log'
_log_dir = os.path.dirname(_log_path)
if not os.path.isdir(_log_dir):
    os.makedirs(_log_dir)

# Make all logs look equal
_log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)-5.5s] %(message)s")

# Log to a file from each module
_fileHandler = RotatingFileHandler(_log_path, maxBytes=(1048576 * 5), backupCount=7)
_fileHandler.setFormatter(_log_formatter)

# Log to the console from each module
_consoleHandler = logging.StreamHandler()
_consoleHandler.setFormatter(_log_formatter)


def get_logger(name):
    # Log to stderr with the module name
    logger = logging.getLogger(name)
    logger.addHandler(_fileHandler)
    logger.addHandler(_consoleHandler)
    logger.setLevel(logging.DEBUG)
    return logger
