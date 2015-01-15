"""
This package contains algorithms for extracting document representations from their raw
bag-of-word counts.
"""

# bring model classes directly into package namespace, to save some typing
from .hdpmodel import HdpModel
from .ldamodel import LdaModel
from .ldamallet import LdaMallet
from .lsimodel import LsiModel
from .tfidfmodel import TfidfModel
from .rpmodel import RpModel
from .logentropy_model import LogEntropyModel
from .word2vec import Word2Vec
from .doc2vec import Doc2Vec
from .ldamulticore import LdaMulticore
from .dtmmodel import DtmModel
from .phrases import Phrases

from gensim import interfaces, utils


class VocabTransform(interfaces.TransformationABC):
    """
    Remap feature ids to new values.

    Given a mapping between old ids and new ids (some old ids may be missing = these
    features are to be discarded), this will wrap a corpus so that iterating over
    `VocabTransform[corpus]` returns the same vectors but with the new ids.

    Old features that have no counterpart in the new ids are discarded. This
    can be used to filter vocabulary of a corpus "online"::

    >>> old2new = dict((oldid, newid) for newid, oldid in enumerate(ids_you_want_to_keep))
    >>> vt = VocabTransform(old2new)
    >>> for vec_with_new_ids in vt[corpus_with_old_ids]:
    >>>     ...

    """
    def __init__(self, old2new, id2token=None):
        # id2word = dict((newid, oldid2word[oldid]) for oldid, newid in old2new.iteritems())
        self.old2new = old2new
        self.id2token = id2token


    def __getitem__(self, bow):
        """
        Return representation with the ids transformed.
        """
        # if the input vector is in fact a corpus, return a transformed corpus as a result
        is_corpus, bow = utils.is_corpus(bow)
        if is_corpus:
            return self._apply(bow)

        return sorted((self.old2new[oldid], weight) for oldid, weight in bow if oldid in self.old2new)
#endclass VocabTransform
