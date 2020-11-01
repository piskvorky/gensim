"""
This package contains functionality to transform documents (strings) into vectors, and calculate
similarities between documents.

"""

import logging

from gensim import parsing, corpora, matutils, interfaces, models, similarities, utils  # noqa:F401

__version__ = '4.0.0beta'

logger = logging.getLogger('gensim')
if not logger.handlers:  # To ensure reload() doesn't add another one
    logger.addHandler(logging.NullHandler())
