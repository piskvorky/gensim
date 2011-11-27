"""
This package contains algorithms for extracting document representations from their raw
bag-of-word counts.
"""

# bring model classes directly into package namespace, to save some typing
from hdpmodel import HdpModel
from ldamodel import LdaModel
from lsimodel import LsiModel
from tfidfmodel import TfidfModel
from rpmodel import RpModel
from logentropy_model import LogEntropyModel
