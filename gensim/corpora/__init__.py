"""
This package contains implementations of various streaming corpus I/O format.
"""

# bring corpus classes directly into package namespace, to save some typing
from .indexedcorpus import IndexedCorpus  # noqa:F401 must appear before the other classes


from .bleicorpus import BleiCorpus  # noqa:F401
from .svmlightcorpus import SvmLightCorpus  # noqa:F401
from .lowcorpus import LowCorpus  # noqa:F401
from .dictionary import Dictionary  # noqa:F401
from .hashdictionary import HashDictionary  # noqa:F401
from .wikicorpus import WikiCorpus  # noqa:F401
from .textcorpus import TextCorpus, TextDirectoryCorpus  # noqa:F401
from .malletcorpus import MalletCorpus  # noqa:F401

# following try/except block is only necessary if we need to check FAST_VERSION
# before c-extensions have been built.  There is no non-c version of MmReader
# so imports of MmCorpus and UciCorpus will fail if c-extensions have not been
# built yet
try:
    from .mmcorpus import MmCorpus  # noqa:F401
    from .ucicorpus import UciCorpus  # noqa:F401
except ImportError as e:
    # this means something went wrong with the c-extension compilation
    import logging
    logger = logging.getLogger('gensim.corpora')
    logger.warning('No valid MmReader found, compilation must have failed')
    print("here")
    # create a dummy classes that will prevent other imports from failing.
    # this class will not work if actually used!
    class MmCorpus(object):
        pass

    class UciCorpus(object):
        pass

