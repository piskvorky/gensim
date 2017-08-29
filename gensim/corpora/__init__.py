"""
This package contains implementations of various streaming corpus I/O format.
"""

from .indexedcorpus import IndexedCorpus  # must appear before the other classes
# bring corpus classes directly into package namespace, to save some typing
from .bleicorpus import BleiCorpus
from .dictionary import Dictionary
from .hashdictionary import HashDictionary
from .lowcorpus import LowCorpus
from .malletcorpus import MalletCorpus
from .mmcorpus import MmCorpus
from .svmlightcorpus import SvmLightCorpus
from .textcorpus import TextCorpus, TextDirectoryCorpus
from .ucicorpus import UciCorpus
from .wikicorpus import WikiCorpus
