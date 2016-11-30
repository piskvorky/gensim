#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2016 Olavur Mortensen <olavurmortensen@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Author-topic model.
"""

# TODO: write proper docstrings.

import pdb
from pdb import set_trace as st
from pprint import pprint

import logging
import np # for arrays, array broadcasting etc.
import numbers

from gensim import interfaces, utils, matutils
from gensim.models import LdaModel
from gensim.models.ldamodel import dirichlet_expectation, get_random_seed, LdaState
from itertools import chain
from scipy.special import gammaln, psi  # gamma function utils
from scipy.special import polygamma
from six.moves import xrange
import six

# log(sum(exp(x))) that tries to avoid overflow
try:
    # try importing from here if older scipy is installed
    from scipy.maxentropy import logsumexp
except ImportError:
    # maxentropy has been removed in recent releases, logsumexp now in misc
    from scipy.misc import logsumexp

logger = logging.getLogger('gensim.models.atmodel')

# TODO: should there be an AuthorTopicState, instead of just using the LdaState?
#class AutorTopicState(utils.SaveLoad):
#    """
#    Encapsulate information for distributed computation of AuthorTopicModel objects.
#
#    Objects of this class are sent over the network, so try to keep them lean to
#    reduce traffic.
#    """
#
#    def __init__(self, eta, shape):
#        self.eta = eta
#        self.sstats = np.zeros(shape)
#        self.numdocs = 0
#        self.lda_state = LdaState(self.eta, shape)
#
#    def reset(self):
#        self.lda_state.reset()

class AuthorTopicModel(LdaModel):
    """
    """
    def __init__(self, corpus=None, num_topics=100, id2word=None,
                author2doc=None, doc2author=None, id2author=None, var_lambda=None,
                 distributed=False, chunksize=2000, passes=1, update_every=1,
                 alpha='symmetric', eta='symmetric', decay=0.5, offset=1.0,
                 eval_every=10, iterations=50, gamma_threshold=0.001,
                 minimum_probability=0.01, random_state=None, ns_conf={},
                 minimum_phi_value=0.01, per_word_topics=False):
        """
        """
        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError('at least one of corpus/id2word must be specified, to establish input space dimensionality')

        # NOTE: Why would id2word not be none, but have length 0? (From LDA code)
        if self.id2word is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        elif len(self.id2word) > 0:
            self.num_terms = 1 + max(self.id2word.keys())
        else:
            self.num_terms = 0

        if self.num_terms == 0:
            raise ValueError("cannot compute LDA over an empty collection (no terms)")

        logger.info('Vocabulary consists of %d words.', self.num_terms)

        if doc2author is None and author2doc is None:
            raise ValueError('at least one of author2doc/doc2author must be specified, to establish input space dimensionality')

        # TODO: consider whether there is a more elegant way of doing this (more importantly, a more efficient way).
        # If either doc2author or author2doc is missing, construct them from the other.
        if doc2author is None:
            # Make a mapping from document IDs to author IDs.
            doc2author = {}
            for d, _ in enumerate(corpus):
                author_ids = []
                for a, a_doc_ids in author2doc.items():
                    if d in a_doc_ids:
                        author_ids.append(a)
                doc2author[d] = author_ids
        elif author2doc is None:
            # Make a mapping from author IDs to document IDs.

            # First get a set of all authors.
            authors_ids = set()
            for d, a_doc_ids in doc2author.items():
                for a in a_doc_ids:
                    authors_ids.add(a)

            # Now construct the dictionary.
            author2doc = {}
            for a in range(len(authors_ids)):
                author2doc[a] = []
                for d, a_ids in doc2author.items():
                    if a in a_ids:
                        author2doc[a].append(d)

        self.author2doc = author2doc
        self.doc2author = doc2author

        self.num_authors = len(self.author2doc)
        logger.info('Number of authors: %d.', self.num_authors)

        self.id2author = id2author
        if self.id2author is None:
            logger.warning("no author id mapping provided; initializing from corpus, assuming identity")
            author_integer_ids = [str(i) for i in range(len(author2doc))]
            self.id2author = dict(zip(range(len(author2doc)), author_integer_ids))

        # Make the reverse mapping, from author names to author IDs.
        self.author2id = dict(zip(self.id2author.values(), self.id2author.keys()))

        self.distributed = False  # NOTE: distributed not yet implemented.
        self.num_topics = num_topics
        self.chunksize = chunksize
        self.decay = decay
        self.offset = offset
        self.minimum_probability = minimum_probability 
        self.num_updates = 0

        self.passes = passes
        self.update_every = update_every
        self.eval_every = eval_every
        self.minimum_phi_value = minimum_phi_value
        self.per_word_topics = per_word_topics

        self.corpus = corpus
        self.iterations = iterations
        self.threshold = threshold
        self.num_authors = len(author2doc)
        self.random_state = random_state

        # NOTE: this is not necessarily a good way to initialize the topics.
        self.alpha = numpy.asarray([1.0 / self.num_topics for i in xrange(self.num_topics)])
        self.eta = numpy.asarray([1.0 / self.num_terms for i in xrange(self.num_terms)])

        self.random_state = get_random_state(random_state)

        if corpus is not None:
            self.update(corpus)

    def init_dir_prior(self, prior, name):
        # TODO: all of this
        init_prior = None
        is_auto = None
        return init_prior, is_auto

    def __str__(self):
        return "AuthorTopicModel(num_terms=%s, num_topics=%s, num_authors=%s, decay=%s)" % \
            (self.num_terms, self.num_topics, self.num_authors, self.decay)

    def sync_state(self):
        """sync_state not implemented for AuthorTopicModel."""
        pass

    def clear(self):
        """clear not implemented for AuthorTopicModel."""
        pass

    def inference(self, chunk, collect_sstats=False):
        """
        """
        return gamma, sstats

    def do_estep(self, chunk, state=None):
        """
        """
        return gamma

    # TODO: probably just use LdaModel's update_alpha and update_eta (once my PR fixing eta is merged).
    def update_alpha(self, gammat, rho):
        """
        """
        return self.alpha

    def update_eta(self, lambdat, rho):
        """
        """
        return self.eta

    # NOTE: this method can be used directly, but self.bound needs to be updated slightly.
    # def log_perplexity(self, chunk, total_docs=None):

    def update(self, corpus, chunksize=None, decay=None, offset=None,
               passes=None, update_every=None, eval_every=None, iterations=None,
               gamma_threshold=None, chunks_as_numpy=False):
        """
        """
        # TODO: this
        pass

    def do_mstep(self, rho, other, extra_pass=False):
        """
        """
        # TODO: this
        pass

    def bound(self, corpus, gamma=None, subsample_ratio=1.0):
        """
        """
        # TODO: this
        pass

    def print_topics(self, num_topics=10, num_words=10):
        # TODO: this
        pass

    def show_topics(self, num_topics=10, num_words=10, log=False, formatted=True):
        """
        """
        # TODO: this
        pass

    def show_topic(self, topicid, topn=10):
        """
        """
        # TODO: this
        pass

    def get_topic_terms(self, topicid, topn=10):
        """
        """
        # TODO: this
        pass

    def print_topic(self, topicid, topn=10):
        # TODO: this
        pass

    def top_topics(self, corpus, num_words=20):
        # TODO: this
        pass

    def get_term_topics(self, word_id, minimum_probability=None):
        # TODO: this
        pass

    def __getitem__(self, bow, eps=None):
        """
        """
        # TODO: this
        pass

    def save(self, fname, ignore=['state', 'dispatcher'], *args, **kwargs):
        """
        Save the model to file.

        Large internal arrays may be stored into separate files, with `fname` as prefix.

        `separately` can be used to define which arrays should be stored in separate files.

        `ignore` parameter can be used to define which variables should be ignored, i.e. left
        out from the pickled lda model. By default the internal `state` is ignored as it uses
        its own serialisation not the one provided by `LdaModel`. The `state` and `dispatcher`
        will be added to any ignore parameter defined.


        Note: do not save as a compressed file if you intend to load the file back with `mmap`.

        Note: If you intend to use models across Python 2/3 versions there are a few things to
        keep in mind:

          1. The pickled Python dictionaries will not work across Python versions
          2. The `save` method does not automatically save all NumPy arrays using NumPy, only
             those ones that exceed `sep_limit` set in `gensim.utils.SaveLoad.save`. The main
             concern here is the `alpha` array if for instance using `alpha='auto'`.

        Please refer to the wiki recipes section (https://github.com/piskvorky/gensim/wiki/Recipes-&-FAQ#q9-how-do-i-load-a-model-in-python-3-that-was-trained-and-saved-using-python-2)
        for an example on how to work around these issues.
        """
        # TODO: this
        if self.state is not None:
            self.state.save(utils.smart_extension(fname, '.state'), *args, **kwargs)

        # make sure 'state' and 'dispatcher' are ignored from the pickled object, even if
        # someone sets the ignore list themselves
        if ignore is not None and ignore:
            if isinstance(ignore, six.string_types):
                ignore = [ignore]
            ignore = [e for e in ignore if e] # make sure None and '' are not in the list
            ignore = list(set(['state', 'dispatcher']) | set(ignore))
        else:
            ignore = ['state', 'dispatcher']
        super(LdaModel, self).save(fname, *args, ignore=ignore, **kwargs)

    @classmethod
    def load(cls, fname, *args, **kwargs):
        """
        Load a previously saved object from file (also see `save`).

        Large arrays can be memmap'ed back as read-only (shared memory) by setting `mmap='r'`:

            >>> LdaModel.load(fname, mmap='r')

        """
        # TODO: this
        kwargs['mmap'] = kwargs.get('mmap', None)
        result = super(LdaModel, cls).load(fname, *args, **kwargs)
        state_fname = utils.smart_extension(fname, '.state')
        try:
            result.state = super(LdaModel, cls).load(state_fname, *args, **kwargs)
        except Exception as e:
            logging.warning("failed to load state from %s: %s", state_fname, e)
        return result
# endclass LdaModel
