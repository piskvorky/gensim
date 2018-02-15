"""This package contains interfaces and functionality to compute pair-wise document similarities within a corpus
of documents.

Also provide CLI interface

Command line arguments
----------------------
.. program-output:: python -m gensim --help
   :ellipsis: 0, -4

"""

from gensim import parsing, matutils, interfaces, corpora, models, similarities, summarization, utils  # noqa:F401
import logging

__version__ = '3.3.0'


class NullHandler(logging.Handler):
    """For python versions <= 2.6; same as `logging.NullHandler` in 2.7."""

    def emit(self, record):
        pass


logger = logging.getLogger('gensim')
if len(logger.handlers) == 0:  # To ensure reload() doesn't add another one
    logger.addHandler(NullHandler())


def package_info():
    """Get the versions of Gensim and its dependencies,
    the location where Gensim is installed and platform on which the system is running.

    Returns
    -------
    dict of (str, str)
        Dictionary containing the versions of Gensim, Python, NumPy, SciPy and platform information.

    """
    import platform
    import sys
    import os
    import numpy
    import scipy
    from gensim.models.word2vec import FAST_VERSION

    return {
        "Platform": platform.platform(),
        "Python": sys.version.replace("\n", ', '),
        "NumPy": numpy.__version__,
        "SciPy": scipy.__version__,
        "Gensim": __version__,
        "Location": os.path.abspath(__file__),
        "FAST_VERSION": FAST_VERSION
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__[:-112], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--info", help="Information about Gensim package", action="store_true")
    args = parser.parse_args()

    if args.info:
        print("Gensim installation information\n")
        for (k, v) in sorted(package_info().items()):
            print("{}: {}".format(k, v))
