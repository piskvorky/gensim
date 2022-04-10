"""
This package contains functionality to transform documents (strings) into vectors, and calculate
similarities between documents.

"""

__version__ = "4.1.3.dev0"

import logging

from gensim import (
    corpora,
    interfaces,
    matutils,
    models,  # noqa:F401
    parsing,
    similarities,
    utils,
)

logger = logging.getLogger("gensim")
if not logger.handlers:  # To ensure reload() doesn't add another one
    logger.addHandler(logging.NullHandler())
