"""Get basic information about gensim & dependencies (useful for bug-reporting).

Examples
--------
You can use it through python

>>> from gensim.scripts.package_info import package_info
>>>
>>> info = package_info()

or using CLI interface

::

    python -m gensim.scripts.package_info --info


.. program-output:: python -m gensim.scripts.package_info --help
   :ellipsis: 0, -4
"""
import argparse
import platform
import sys
import os
import numpy
import scipy
import gensim
from gensim.models.word2vec import FAST_VERSION


def package_info():
    """Get the versions of Gensim and its dependencies,
    the location where Gensim is installed and platform on which the system is running.

    Returns
    -------
    dict of (str, str)
        Dictionary containing the versions of Gensim, Python, NumPy, SciPy and platform information.

    """
    return {
        "Platform": platform.platform(),
        "Python": sys.version.replace("\n", ', '),
        "NumPy": numpy.__version__,
        "SciPy": scipy.__version__,
        "Gensim": gensim.__version__,
        "Location": os.path.abspath(__file__),
        "FAST_VERSION": FAST_VERSION
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__[:-65], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--info", help="Information about Gensim package", action="store_true")
    args = parser.parse_args()

    if args.info:
        print("Gensim installation information\n")
        for (k, v) in sorted(package_info().items()):
            print("{}: {}".format(k, v))
