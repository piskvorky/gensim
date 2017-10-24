"""
This package contains algorithms for extracting document representations from their raw
bag-of-word counts.
"""

# bring model classes directly into package namespace, to save some typing
from .coherencemodel import CoherenceModel  # noqa:F401
from .hdpmodel import HdpModel  # noqa:F401
from .ldamodel import LdaModel  # noqa:F401
from .lsimodel import LsiModel  # noqa:F401
from .tfidfmodel import TfidfModel  # noqa:F401
from .rpmodel import RpModel  # noqa:F401
from .logentropy_model import LogEntropyModel  # noqa:F401
from .word2vec import Word2Vec  # noqa:F401
from .doc2vec import Doc2Vec  # noqa:F401
from .keyedvectors import KeyedVectors  # noqa:F401
from .ldamulticore import LdaMulticore  # noqa:F401
from .phrases import Phrases  # noqa:F401
from .normmodel import NormModel  # noqa:F401
from .atmodel import AuthorTopicModel  # noqa:F401
from .ldaseqmodel import LdaSeqModel  # noqa:F401
from .summarization import keywords, summarize, summarize_corpus  # noqa:F401

from . import wrappers  # noqa:F401

from gensim import interfaces, utils


class VocabTransform(interfaces.TransformationABC):
    """
    Remap feature ids to new values.

    Given a mapping between old ids and new ids (some old ids may be missing = these
    features are to be discarded), this will wrap a corpus so that iterating over
    `VocabTransform[corpus]` returns the same vectors but with the new ids.

    Old features that have no counterpart in the new ids are discarded. This
    can be used to filter vocabulary of a corpus "online"::

    >>> old2new = {oldid: newid for newid, oldid in enumerate(ids_you_want_to_keep)}
    >>> vt = VocabTransform(old2new)
    >>> for vec_with_new_ids in vt[corpus_with_old_ids]:
    >>>     ...

    """

    def __init__(self, old2new, id2token=None):
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
