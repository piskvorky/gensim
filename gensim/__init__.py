"""
This package contains interfaces and functionality to compute pair-wise document
similarities within a corpus of documents.
"""

from gensim import parsing, matutils, interfaces, corpora, models, similarities, summarization
import logging

try:
    __version__ = __import__('pkg_resources').get_distribution('gensim').version
except:
    __version__ = '?'


class NullHandler(logging.Handler):
    """For python versions <= 2.6; same as `logging.NullHandler` in 2.7."""
    def emit(self, record):
        pass

logger = logging.getLogger('gensim')
if len(logger.handlers) == 0:	# To ensure reload() doesn't add another one
    logger.addHandler(NullHandler())


def get_includes():
    """
        get folder paths that contain .h files
    """
    import os.path
    # voidptr.h under models
    model_path = os.path.dirname(os.path.abspath(models.__file__))
    return [
        model_path,
    ]
    