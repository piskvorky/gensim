"""
This package contains implementations of various streaming corpus I/O format.
"""

# bring corpus classes directly into package namespace, to save some typing
from mmcorpus import MmCorpus
from bleicorpus import BleiCorpus
from svmlightcorpus import SvmLightCorpus
from lowcorpus import LowCorpus
from dictionary import Dictionary
