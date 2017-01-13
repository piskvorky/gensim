"""
This package contains implementations of various streaming corpus I/O format.
"""

# bring corpus classes directly into package namespace, to save some typing
from .indexedcorpus import IndexedCorpus  # must appear before the other classes

from .mmcorpus import MmCorpus
from .bleicorpus import BleiCorpus
from .svmlightcorpus import SvmLightCorpus
from .lowcorpus import LowCorpus
from .dictionary import Dictionary
from .hashdictionary import HashDictionary
from .wikicorpus import WikiCorpus
from .textcorpus import TextCorpus
from .ucicorpus import UciCorpus
from .malletcorpus import MalletCorpus
