"""
This package contains interfaces and functionality to compute pair-wise document
similarities within a corpus of documents.
"""

from gensim import parsing, matutils, interfaces, corpora, models, similarities, summarization, utils  # noqa:F401
import logging

import platform
import sys
import numpy
import scipy
import gensim
from gensim.models.word2vec import FAST_VERSION


__version__ = '3.3.0'


class NullHandler(logging.Handler):
    """For python versions <= 2.6; same as `logging.NullHandler` in 2.7."""

    def emit(self, record):
        pass


logger = logging.getLogger('gensim')
if len(logger.handlers) == 0:  # To ensure reload() doesn't add another one
    logger.addHandler(NullHandler())


def package_info():
    """Get the versions of Gensim and its dependencies, the location where Gensim is\
 installed and platform on which the system is running.

    Returns
    -------
    dict
        Returns a dictionary containing the versions pof Gensim, Python, NumPy, SciPy and platform information.

    .. program-output:: python -m gensim --info
    """
    dict = {}
    dict["platform"] = platform.platform()
    dict["python"] = sys.version
    dict["NumPy"] = numpy.__version__
    dict["SciPy"] = scipy.__version__
    dict["gensim"] = gensim.__version__, gensim.__file__
#    dict["smart_open"] = smart_open.__version__
    dict["fast_version"] = FAST_VERSION

    return dict
