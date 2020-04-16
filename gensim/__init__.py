"""This package contains interfaces and functionality to compute pair-wise document similarities within a corpus
of documents.
"""

from gensim import parsing, corpora, matutils, interfaces, models, similarities, summarization, utils  # noqa:F401
import logging

__version__ = '3.8.3'


logger = logging.getLogger('gensim')
if len(logger.handlers) == 0:  # To ensure reload() doesn't add another one
    logger.addHandler(logging.NullHandler())
