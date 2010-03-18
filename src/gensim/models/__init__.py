"""
This package contains algorithms for extracting document representations from their raw 
bag-of-word counts.
"""

# bring model classes directly into package namespace, to save some typing
from ldamodel import LdaModel
from lsimodel import LsiModel
from tfidfmodel import TfidfModel
#from rpmodel import RpModel
