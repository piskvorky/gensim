#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2016 Olavur Mortensen <olavurmortensen@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""Author-topic model.

This module trains the author-topic model on documents and corresponding author-document dictionaries.
The training is online and is constant in memory w.r.t. the number of documents.
The model is *not* constant in memory w.r.t. the number of authors.

The model can be updated with additional documents after training has been completed. It is
also possible to continue training on the existing data.

The model is closely related to :class:`~gensim.models.ldamodel.LdaModel`.
The :class:`~gensim.models.atmodel.AuthorTopicModel` class inherits  :class:`~gensim.models.ldamodel.LdaModel`,
and its usage is thus similar.

The model was introduced by  `Rosen-Zvi and co-authors: "The Author-Topic Model for Authors and Documents"
<https://arxiv.org/abs/1207.4169>`_. The model correlates the authorship information with the topics to give a better
insight on the subject knowledge of an author.

Example
-------
>>> from gensim.models import AuthorTopicModel
>>> from gensim.corpora import mmcorpus
>>> from gensim.test.utils import common_dictionary, datapath, temporary_file

>>> author2doc = {
...     'john': [0, 1, 2, 3, 4, 5, 6],
...     'jane': [2, 3, 4, 5, 6, 7, 8],
...     'jack': [0, 2, 4, 6, 8]
... }
>>>
>>> corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
>>>
>>> with temporary_file("serialized") as s_path:
...     model = AuthorTopicModel(
...          corpus, author2doc=author2doc, id2word=common_dictionary, num_topics=4,
...          serialized=True, serialization_path=s_path
...     )
...
...     model.update(corpus, author2doc)  # update the author-topic model with additional documents
>>>
>>> # construct vectors for authors
>>> author_vecs = [model.get_author_topics(author) for author in model.id2author.values()]

"""
# TODO: this class inherits LdaModel and overwrites some methods. There is some code
# duplication still, and a refactor could be made to avoid this. Comments with "TODOs"
# are included in the code where this is the case, for example in the log_perplexity
# and do_estep methods.

import logging
import numpy as np  # for arrays, array broadcasting etc.
from copy import deepcopy
from shutil import copyfile
from os.path import isfile
from os import remove

from gensim import utils
from gensim.models import LdaModel
from gensim.models.ldamodel import LdaState
from gensim.matutils import dirichlet_expectation
from gensim.corpora import MmCorpus
from itertools import chain
from scipy.special import gammaln  # gamma function utils
from six.moves import xrange
import six

logger = logging.getLogger(__name__)


class AuthorTopicState(LdaState):
    """Encapsulate information for computation of :class:`~gensim.models.atmodel.AuthorTopicModel`."""

    def __init__(self, eta, lambda_shape, gamma_shape):
        """

        Parameters
        ----------
        eta: numpy.ndarray
            Dirichlet topic parameter for sparsity.
        lambda_shape: (int, int)
            Initialize topic parameters.
        gamma_shape: int
            Initialize topic parameters.

        """
        self.eta = eta
        self.sstats = np.zeros(lambda_shape)
        self.gamma = np.zeros(gamma_shape)
        self.numdocs = 0
        self.dtype = np.float64  # To be compatible with LdaState


def construct_doc2author(corpus, author2doc):
    """Create a mapping from document IDs to author IDs.

    Parameters
    ----------
    corpus: iterable of list of (int, float)
        Corpus in BoW format.
    author2doc: dict of (str, list of int)
        Mapping of authors to documents.

    Returns
    -------
    dict of (int, list of str)
        Document to Author mapping.

    """
    doc2author = {}
    for d, _ in enumerate(corpus):
        author_ids = []
        for a, a_doc_ids in author2doc.items():
            if d in a_doc_ids:
                author_ids.append(a)
        doc2author[d] = author_ids
    return doc2author


def construct_author2doc(doc2author):
    """Make a mapping from author IDs to document IDs.

    Parameters
    ----------
    doc2author: dict of (int, list of str)
        Mapping of document id to authors.

    Returns
    -------
    dict of (str, list of int)
        Mapping of authors to document ids.

    """
    # First get a set of all authors.
    authors_ids = set()
    for d, a_doc_ids in doc2author.items():
        for a in a_doc_ids:
            authors_ids.add(a)

    # Now construct the dictionary.
    author2doc = {}
    for a in authors_ids:
        author2doc[a] = []
        for d, a_ids in doc2author.items():
            if a in a_ids:
                author2doc[a].append(d)
    return author2doc


class AuthorTopicModel(LdaModel):
    """The constructor estimates the author-topic model parameters based on a training corpus."""

    def __init__(self, corpus=None, num_topics=100, id2word=None, author2doc=None, doc2author=None,
                 chunksize=2000, passes=1, iterations=50, decay=0.5, offset=1.0,
                 alpha='symmetric', eta='symmetric', update_every=1, eval_every=10,
                 gamma_threshold=0.001, serialized=False, serialization_path=None,
                 minimum_probability=0.01, random_state=None):
        """

        Parameters
        ----------
        corpus : iterable of list of str
            Corpus of documents in appropriate format(BoW, UCI etc).
        num_topics : int, optional
            Number of topics to be extracted from the training corpus.
        id2word : dict of (int, str), optional
            A mapping from word ids (integers) to words (strings).
        author2doc : dict of (str, list of int)
            A dictionary where keys are the names of authors and values are lists of
            documents that the author contributes to.
        doc2author : dict of (int, list of str)
            A dictionary where the keys are document IDs and the values are lists of author names.
        passes : int
            Number of times the model makes a pass over the entire training data.
        iterations : int
            Maximum number of times the model loops over each document
        chunksize : int
            Controls the size of the mini-batches.
        alpha : float
            Hyperparameters for author-topic model.Supports special values of 'asymmetric'
            and 'auto': the former uses a fixed normalized asymmetric 1.0/topicno prior,
            the latter learns an asymmetric prior directly from your data.
        eta : float
            Hyperparameters for author-topic model.
        update_every : int
            Make updates in topic probaility for latest mini-batch.
        eval_every : int
            Calculate and estimate log perplexity for latest mini-batch.
        gamma_threshold : float
            Threshold value of gamma(topic difference between consecutive two topics)
            until which the iterations continue.
        decay : float
            Controls how old documents are forgotten.
        offset : float
            Controls down-weighting of iterations.
        minimum_probability : float
            Controls filtering the topics returned for a document (bow).
        random_state : int or a numpy.random.RandomState object.
            Set the state of the random number generator inside the author-topic model.
        serialized : bool
            Indicates whether the input corpora to the model are simple lists
            or saved to the hard-drive.
        serialization_path : str
            Must be set to a filepath, if `serialized = True` is used.


        """
        # NOTE: this doesn't call constructor of a base class, but duplicates most of this code
        # so we have to set dtype to float64 default here
        self.dtype = np.float64

        # NOTE: as distributed version of this model is not implemented, "distributed" is set to false. Some of the
        # infrastructure to implement a distributed author-topic model is already in place,
        # such as the AuthorTopicState.
        distributed = False
        self.dispatcher = None
        self.numworkers = 1

        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError(
                "at least one of corpus/id2word must be specified, to establish input space dimensionality"
            )

        if self.id2word is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        elif len(self.id2word) > 0:
            self.num_terms = 1 + max(self.id2word.keys())
        else:
            self.num_terms = 0

        if self.num_terms == 0:
            raise ValueError("cannot compute the author-topic model over an empty collection (no terms)")

        logger.info('Vocabulary consists of %d words.', self.num_terms)

        self.author2doc = {}
        self.doc2author = {}

        self.distributed = distributed
        self.num_topics = num_topics
        self.num_authors = 0
        self.chunksize = chunksize
        self.decay = decay
        self.offset = offset
        self.minimum_probability = minimum_probability
        self.num_updates = 0
        self.total_docs = 0

        self.passes = passes
        self.update_every = update_every
        self.eval_every = eval_every

        self.author2id = {}
        self.id2author = {}

        self.serialized = serialized
        if serialized and not serialization_path:
            raise ValueError(
                "If serialized corpora are used, a the path to a folder "
                "where the corpus should be saved must be provided (serialized_path)."
            )
        if serialized and serialization_path:
            assert not isfile(serialization_path), \
                "A file already exists at the serialization_path path; " \
                "choose a different serialization_path, or delete the file."
        self.serialization_path = serialization_path

        # Initialize an empty self.corpus.
        self.init_empty_corpus()

        self.alpha, self.optimize_alpha = self.init_dir_prior(alpha, 'alpha')

        assert self.alpha.shape == (self.num_topics,), \
            "Invalid alpha shape. Got shape %s, but expected (%d, )" % (str(self.alpha.shape), self.num_topics)

        if isinstance(eta, six.string_types):
            if eta == 'asymmetric':
                raise ValueError("The 'asymmetric' option cannot be used for eta")

        self.eta, self.optimize_eta = self.init_dir_prior(eta, 'eta')

        self.random_state = utils.get_random_state(random_state)

        assert (self.eta.shape == (self.num_terms,) or self.eta.shape == (self.num_topics, self.num_terms)), (
            "Invalid eta shape. Got shape %s, but expected (%d, 1) or (%d, %d)" %
            (str(self.eta.shape), self.num_terms, self.num_topics, self.num_terms)
        )

        # VB constants
        self.iterations = iterations
        self.gamma_threshold = gamma_threshold

        # Initialize the variational distributions q(beta|lambda) and q(theta|gamma)
        self.state = AuthorTopicState(self.eta, (self.num_topics, self.num_terms), (self.num_authors, self.num_topics))
        self.state.sstats = self.random_state.gamma(100., 1. / 100., (self.num_topics, self.num_terms))
        self.expElogbeta = np.exp(dirichlet_expectation(self.state.sstats))

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None and (author2doc is not None or doc2author is not None):
            use_numpy = self.dispatcher is not None
            self.update(corpus, author2doc, doc2author, chunks_as_numpy=use_numpy)

    def __str__(self):
        """Return a string representation of AuthorTopicModel class.

        Returns
        -------
        str
            String representation of Author-Topic model class.
        """
        return "AuthorTopicModel(num_terms=%s, num_topics=%s, num_authors=%s, decay=%s, chunksize=%s)" % \
            (self.num_terms, self.num_topics, self.num_authors, self.decay, self.chunksize)

    def init_empty_corpus(self):
        """
        Initialize an empty corpus. If the corpora are to be treated as lists, simply
        initialize an empty list. If serialization is used, initialize an empty corpus
        of the class `gensim.corpora.MmCorpus`.

        """
        if self.serialized:
            # Initialize the corpus as a serialized empty list.
            # This corpus will be extended in self.update.
            MmCorpus.serialize(self.serialization_path, [])  # Serialize empty corpus.
            self.corpus = MmCorpus(self.serialization_path)  # Store serialized corpus object in self.corpus.
        else:
            # All input corpora are assumed to just be lists.
            self.corpus = []

    def extend_corpus(self, corpus):
        """
        Add new documents in `corpus` to `self.corpus`. If serialization is used,
        then the entire corpus (`self.corpus`) is re-serialized and the new documents
        are added in the process. If serialization is not used, the corpus, as a list
        of documents, is simply extended.

        Parameters
        ----------
        corpus : iterable of list of str
            Corpus of documents.
        """
        if self.serialized:
            # Re-serialize the entire corpus while appending the new documents.
            if isinstance(corpus, MmCorpus):
                # Check that we are not attempting to overwrite the serialized corpus.
                assert self.corpus.input != corpus.input, \
                    'Input corpus cannot have the same file path as the model corpus (serialization_path).'
            corpus_chain = chain(self.corpus, corpus)  # A generator with the old and new documents.
            # Make a temporary copy of the file where the corpus is serialized.
            copyfile(self.serialization_path, self.serialization_path + '.tmp')
            self.corpus.input = self.serialization_path + '.tmp'  # Point the old corpus at this temporary file.
            # Re-serialize the old corpus, and extend it with the new corpus.
            MmCorpus.serialize(self.serialization_path, corpus_chain)
            self.corpus = MmCorpus(self.serialization_path)  # Store the new serialized corpus object in self.corpus.
            remove(self.serialization_path + '.tmp')  # Remove the temporary file again.
        else:
            # self.corpus and corpus are just lists, just extend the list.
            # First check that corpus is actually a list.
            assert isinstance(corpus, list), "If serialized == False, all input corpora must be lists."
            self.corpus.extend(corpus)

    def compute_phinorm(self, expElogthetad, expElogbetad):
        """Efficiently computes the normalizing factor in phi.

        Parameters
        ----------
        expElogthetad: numpy.ndarray
            Value of variational distribution :math: q(\theta|\gamma).
        expElogbetad: numpy.ndarray
            Value of variational distribution :math: q(\beta|\lambda).

        Returns
        -------
        float
            Value of normalizing factor.

        """
        expElogtheta_sum = expElogthetad.sum(axis=0)
        phinorm = expElogtheta_sum.dot(expElogbetad) + 1e-100

        return phinorm

    def inference(self, chunk, author2doc, doc2author, rhot, collect_sstats=False, chunk_doc_idx=None):
        """
        Given a chunk of sparse document vectors, update gamma (parameters
        controlling the topic weights) for each author corresponding to the
        documents in the chunk.

        The whole input chunk of document is assumed to fit in RAM; chunking of
        a large corpus must be done earlier in the pipeline.

        If `collect_sstats` is True, also collect sufficient statistics needed
        to update the model's topic-word distributions, and return a 2-tuple
        `(gamma_chunk, sstats)`. Otherwise, return `(gamma_chunk, None)`.
        `gamma_cunk` is of shape `len(chunk_authors) x self.num_topics`, where
        `chunk_authors` is the number of authors in the documents in the
        current chunk.

        Avoids computing the `phi` variational parameter directly using the
        optimization presented in **Lee, Seung: Algorithms for non-negative matrix factorization, NIPS 2001**.

        Parameters
        ----------
        chunk : int
            The chunk numer of the sparse document vector on which inference needs to be done.
        author2doc : dict of (str, list of int)
            A dictionary where keys are the names of authors and values are lists of
            documents that the author contributes to.
        doc2author : dict of (int, list of str)
            A dictionary where the keys are document IDs and the values are lists of author names.
        rhot : float
            Value of rho for conducting inference on documents.
        collect_sstats : boolean, optional
            If True,  collect sufficient statistics needed to update the model's topic-word
            distributions, and return a 2-tuple `(gamma_chunk, sstats)`.
            Otherwise, returns `(gamma_chunk, None)`.`gamma_chunk` is of shape
            `len(chunk_authors) x self.num_topics`,where `chunk_authors` is the
            number of authors in the documents in the current chunk.
        chunk_doc_idx : numpy.ndarray
            Assigns the value for document index.

        """
        try:
            len(chunk)
        except TypeError:
            # convert iterators/generators to plain list, so we have len() etc.
            chunk = list(chunk)
        if len(chunk) > 1:
            logger.debug("performing inference on a chunk of %i documents", len(chunk))

        # Initialize the variational distribution q(theta|gamma) for the chunk
        if collect_sstats:
            sstats = np.zeros_like(self.expElogbeta)
        else:
            sstats = None
        converged = 0

        # Stack all the computed gammas into this output array.
        gamma_chunk = np.zeros((0, self.num_topics))

        # Now, for each document d update gamma and phi w.r.t. all authors in those documents.
        for d, doc in enumerate(chunk):
            if chunk_doc_idx is not None:
                doc_no = chunk_doc_idx[d]
            else:
                doc_no = d
            # Get the IDs and counts of all the words in the current document.
            # TODO: this is duplication of code in LdaModel. Refactor.
            if doc and not isinstance(doc[0][0], six.integer_types + (np.integer,)):
                # make sure the term IDs are ints, otherwise np will get upset
                ids = [int(idx) for idx, _ in doc]
            else:
                ids = [idx for idx, _ in doc]
            cts = np.array([cnt for _, cnt in doc])

            # Get all authors in current document, and convert the author names to integer IDs.
            authors_d = [self.author2id[a] for a in self.doc2author[doc_no]]

            gammad = self.state.gamma[authors_d, :]  # gamma of document d before update.
            tilde_gamma = gammad.copy()  # gamma that will be updated.

            # Compute the expectation of the log of the Dirichlet parameters theta and beta.
            Elogthetad = dirichlet_expectation(tilde_gamma)
            expElogthetad = np.exp(Elogthetad)
            expElogbetad = self.expElogbeta[:, ids]

            # Compute the normalizing constant of phi for the current document.
            phinorm = self.compute_phinorm(expElogthetad, expElogbetad)

            # Iterate between gamma and phi until convergence
            for _ in xrange(self.iterations):
                lastgamma = tilde_gamma.copy()

                # Update gamma.
                # phi is computed implicitly below,
                for ai, a in enumerate(authors_d):
                    tilde_gamma[ai, :] = self.alpha + len(self.author2doc[self.id2author[a]])\
                        * expElogthetad[ai, :] * np.dot(cts / phinorm, expElogbetad.T)

                # Update gamma.
                # Interpolation between document d's "local" gamma (tilde_gamma),
                # and "global" gamma (gammad).
                tilde_gamma = (1 - rhot) * gammad + rhot * tilde_gamma

                # Update Elogtheta and Elogbeta, since gamma and lambda have been updated.
                Elogthetad = dirichlet_expectation(tilde_gamma)
                expElogthetad = np.exp(Elogthetad)

                # Update the normalizing constant in phi.
                phinorm = self.compute_phinorm(expElogthetad, expElogbetad)

                # Check for convergence.
                # Criterion is mean change in "local" gamma.
                meanchange_gamma = np.mean(abs(tilde_gamma - lastgamma))
                gamma_condition = meanchange_gamma < self.gamma_threshold
                if gamma_condition:
                    converged += 1
                    break
            # End of iterations loop.

            # Store the updated gammas in the model state.
            self.state.gamma[authors_d, :] = tilde_gamma

            # Stack the new gammas into the output array.
            gamma_chunk = np.vstack([gamma_chunk, tilde_gamma])

            if collect_sstats:
                # Contribution of document d to the expected sufficient
                # statistics for the M step.
                expElogtheta_sum_a = expElogthetad.sum(axis=0)
                sstats[:, ids] += np.outer(expElogtheta_sum_a.T, cts / phinorm)

        if len(chunk) > 1:
            logger.debug(
                "%i/%i documents converged within %i iterations",
                converged, len(chunk), self.iterations
            )

        if collect_sstats:
            # This step finishes computing the sufficient statistics for the
            # M step, so that
            # sstats[k, w] = \sum_d n_{dw} * \sum_a phi_{dwak}
            # = \sum_d n_{dw} * exp{Elogtheta_{ak} + Elogbeta_{kw}} / phinorm_{dw}.
            sstats *= self.expElogbeta
        return gamma_chunk, sstats

    def do_estep(self, chunk, author2doc, doc2author, rhot, state=None, chunk_doc_idx=None):
        """
        Performs inference on a chunk of documents, and accumulate the collected
        sufficient statistics in `state` (or `self.state` if None).

        Parameters
        ----------
        chunk : int
            The chunk numer of the sparse document vector on which inference needs to be done.
        author2doc : dict of (str, list of int)
            A dictionary where keys are the names of authors and values are lists of
            documents that the author contributes to.
        doc2author : dict of (intm list of str)
            A dictionary where the keys are document IDs and the values are lists of author names.
        rhot : float
            Value of rho for conducting inference on documents.
        state : int, optional
            Initializes the state for a new E-M iteration.
        chunk_doc_idx : numpy.ndarray
            Assigns the value for document index.

        Returns
        -------
        float
            Value of gamma for training of model.
        """

        # TODO: this method is somewhat similar to the one in LdaModel. Refactor if possible.
        if state is None:
            state = self.state
        gamma, sstats = self.inference(
            chunk, author2doc, doc2author, rhot,
            collect_sstats=True, chunk_doc_idx=chunk_doc_idx
        )
        state.sstats += sstats
        state.numdocs += len(chunk)
        return gamma

    def log_perplexity(self, chunk, chunk_doc_idx=None, total_docs=None):
        """Calculate per-word likelihood bound, using the `chunk` of documents as evaluation corpus.

        Parameters
        ----------
        chunk : iterable of list of (int, float)
            Corpus in BoW format.
        chunk_doc_idx : numpy.ndarray, optional
            Assigns the value for document index.
        total_docs : int, optional
            Initializes the value for total number of documents.

        Returns
        -------
        float
            Value of per-word likelihood bound.

        """
        # TODO: This method is very similar to the one in LdaModel. Refactor.
        if total_docs is None:
            total_docs = len(chunk)
        corpus_words = sum(cnt for document in chunk for _, cnt in document)
        subsample_ratio = 1.0 * total_docs / len(chunk)
        perwordbound = self.bound(chunk, chunk_doc_idx, subsample_ratio=subsample_ratio) / \
            (subsample_ratio * corpus_words)
        logger.info(
            "%.3f per-word bound, %.1f perplexity estimate based on a corpus of %i documents with %i words",
            perwordbound, np.exp2(-perwordbound), len(chunk), corpus_words
        )
        return perwordbound

    def update(self, corpus=None, author2doc=None, doc2author=None, chunksize=None, decay=None, offset=None,
               passes=None, update_every=None, eval_every=None, iterations=None,
               gamma_threshold=None, chunks_as_numpy=False):
        """Train the model with new documents, by EM-iterating over `corpus` until the topics converge (or until the
        maximum number of allowed iterations is reached).

        Notes
        -----
        This update also supports updating an already trained model (self)
        with new documents from `corpus`: the two models are then merged in proportion to the number of old vs. new
        documents. This feature is still experimental for non-stationary input streams.

        For stationary input (no topic drift in new documents), on the other hand, this equals the online update of
        `Hoffman et al. Stochastic Variational Inference
        <http://www.jmlr.org/papers/volume14/hoffman13a/hoffman13a.pdf>`_ and is guaranteed to converge for any `decay`
        in (0.5, 1.0>. Additionally, for smaller `corpus` sizes, an increasing `offset` may be beneficial (see
        Table 1 in Hoffman et al.)

        If update is called with authors that already exist in the model, it will resume training on not only new
        documents for that author, but also the previously seen documents. This is necessary for those authors' topic
        distributions to converge.

        Every time `update(corpus, author2doc)` is called, the new documents are to appended to all the previously seen
        documents, and author2doc is combined with the previously seen authors.

        To resume training on all the data seen by the model, simply call
        :meth:`~gensim.models.atmodel.AuthorTopicModel.update`.

        It is not possible to add new authors to existing documents, as all documents in `corpus` are assumed to be
        new documents.

        Parameters
        ----------
        corpus : iterable of list of (int, float)
            The corpus in BoW format.
        author2doc : dict of (str, list of int), optional
            A dictionary where keys are the names of authors and values are lists of document IDs that the author
            contributes to.
        doc2author : dict of (int, list of str), optional
            A dictionary where the keys are document IDs and the values are lists of author names.
        chunksize : int, optional
            Controls the size of the mini-batches.
        decay : float, optional
            Controls how old documents are forgotten.
        offset : float, optional
            Controls down-weighting of iterations.
        passes : int, optional
            Number of times the model makes a pass over the entire training data.
        update_every : int, optional
            Make updates in topic probability for latest mini-batch.
        eval_every : int, optional
            Calculate and estimate log perplexity for latest mini-batch.
        iterations : int, optional
            Maximum number of times the model loops over each document
        gamma_threshold : float, optional
            Threshold value of gamma(topic difference between consecutive two topics)
            until which the iterations continue.
        chunks_as_numpy : bool, optional
            Whether each chunk passed to :meth:`~gensim.models.atmodel.AuthorTopicModel.inference` should be a numpy
            array of not. Numpy can in some settings turn the term IDs into floats, these will be converted back into
            integers in inference, which incurs a performance hit. For distributed computing (not supported now)
            it may be desirable to keep the chunks as numpy arrays.

        """
        # use parameters given in constructor, unless user explicitly overrode them
        if decay is None:
            decay = self.decay
        if offset is None:
            offset = self.offset
        if passes is None:
            passes = self.passes
        if update_every is None:
            update_every = self.update_every
        if eval_every is None:
            eval_every = self.eval_every
        if iterations is None:
            iterations = self.iterations
        if gamma_threshold is None:
            gamma_threshold = self.gamma_threshold

        # TODO: if deepcopy is not used here, something goes wrong. When unit tests are run (specifically "testPasses"),
        # the process simply gets killed.
        author2doc = deepcopy(author2doc)
        doc2author = deepcopy(doc2author)

        # TODO: it is not possible to add new authors to an existing document (all input documents are treated
        # as completely new documents). Perhaps this functionality could be implemented.
        # If it's absolutely necessary, the user can delete the documents that have new authors, and call update
        # on them with the new and old authors.

        if corpus is None:
            # Just keep training on the already available data.
            # Assumes self.update() has been called before with input documents and corresponding authors.
            assert self.total_docs > 0, 'update() was called with no documents to train on.'
            train_corpus_idx = [d for d in xrange(self.total_docs)]
            num_input_authors = len(self.author2doc)
        else:
            if doc2author is None and author2doc is None:
                raise ValueError(
                    'at least one of author2doc/doc2author must be specified, to establish input space dimensionality'
                )

            # If either doc2author or author2doc is missing, construct them from the other.
            if doc2author is None:
                doc2author = construct_doc2author(corpus, author2doc)
            elif author2doc is None:
                author2doc = construct_author2doc(doc2author)

            # Number of authors that need to be updated.
            num_input_authors = len(author2doc)

            try:
                len_input_corpus = len(corpus)
            except TypeError:
                logger.warning("input corpus stream has no len(); counting documents")
                len_input_corpus = sum(1 for _ in corpus)
            if len_input_corpus == 0:
                logger.warning("AuthorTopicModel.update() called with an empty corpus")
                return

            self.total_docs += len_input_corpus

            # Add new documents in corpus to self.corpus.
            self.extend_corpus(corpus)

            # Obtain a list of new authors.
            new_authors = []
            # Sorting the author names makes the model more reproducible.
            for a in sorted(author2doc.keys()):
                if not self.author2doc.get(a):
                    new_authors.append(a)

            num_new_authors = len(new_authors)

            # Add new authors do author2id/id2author dictionaries.
            for a_id, a_name in enumerate(new_authors):
                self.author2id[a_name] = a_id + self.num_authors
                self.id2author[a_id + self.num_authors] = a_name

            # Increment the number of total authors seen.
            self.num_authors += num_new_authors

            # Initialize the variational distributions q(theta|gamma)
            gamma_new = self.random_state.gamma(100., 1. / 100., (num_new_authors, self.num_topics))
            self.state.gamma = np.vstack([self.state.gamma, gamma_new])

            # Combine author2doc with self.author2doc.
            # First, increment the document IDs by the number of previously seen documents.
            for a, doc_ids in author2doc.items():
                doc_ids = [d + self.total_docs - len_input_corpus for d in doc_ids]

            # For all authors in the input corpus, add the new documents.
            for a, doc_ids in author2doc.items():
                if self.author2doc.get(a):
                    # This is not a new author, append new documents.
                    self.author2doc[a].extend(doc_ids)
                else:
                    # This is a new author, create index.
                    self.author2doc[a] = doc_ids

            # Add all new documents to self.doc2author.
            for d, a_list in doc2author.items():
                self.doc2author[d] = a_list

            # Train on all documents of authors in input_corpus.
            train_corpus_idx = []
            for _ in author2doc.keys():  # For all authors in input corpus.
                for doc_ids in self.author2doc.values():  # For all documents in total corpus.
                    train_corpus_idx.extend(doc_ids)

            # Make the list of training documents unique.
            train_corpus_idx = list(set(train_corpus_idx))

        # train_corpus_idx is only a list of indexes, so "len" is valid.
        lencorpus = len(train_corpus_idx)

        if chunksize is None:
            chunksize = min(lencorpus, self.chunksize)

        self.state.numdocs += lencorpus

        if update_every:
            updatetype = "online"
            updateafter = min(lencorpus, update_every * self.numworkers * chunksize)
        else:
            updatetype = "batch"
            updateafter = lencorpus
        evalafter = min(lencorpus, (eval_every or 0) * self.numworkers * chunksize)

        updates_per_pass = max(1, lencorpus / updateafter)
        logger.info(
            "running %s author-topic training, %s topics, %s authors, "
            "%i passes over the supplied corpus of %i documents, updating model once "
            "every %i documents, evaluating perplexity every %i documents, "
            "iterating %ix with a convergence threshold of %f",
            updatetype, self.num_topics, num_input_authors, passes, lencorpus, updateafter,
            evalafter, iterations, gamma_threshold
        )

        if updates_per_pass * passes < 10:
            logger.warning(
                "too few updates, training might not converge; "
                "consider increasing the number of passes or iterations to improve accuracy"
            )

        # rho is the "speed" of updating; TODO try other fncs
        # pass_ + num_updates handles increasing the starting t for each pass,
        # while allowing it to "reset" on the first pass of each update
        def rho():
            return pow(offset + pass_ + (self.num_updates / chunksize), -decay)

        for pass_ in xrange(passes):
            if self.dispatcher:
                logger.info('initializing %s workers', self.numworkers)
                self.dispatcher.reset(self.state)
            else:
                # gamma is not needed in "other", thus its shape is (0, 0).
                other = AuthorTopicState(self.eta, self.state.sstats.shape, (0, 0))
            dirty = False

            reallen = 0
            for chunk_no, chunk_doc_idx in enumerate(
                    utils.grouper(train_corpus_idx, chunksize, as_numpy=chunks_as_numpy)):
                chunk = [self.corpus[d] for d in chunk_doc_idx]
                reallen += len(chunk)  # keep track of how many documents we've processed so far

                if eval_every and ((reallen == lencorpus) or ((chunk_no + 1) % (eval_every * self.numworkers) == 0)):
                    # log_perplexity requires the indexes of the documents being evaluated, to know what authors
                    # correspond to the documents.
                    self.log_perplexity(chunk, chunk_doc_idx, total_docs=lencorpus)

                if self.dispatcher:
                    # add the chunk to dispatcher's job queue, so workers can munch on it
                    logger.info(
                        "PROGRESS: pass %i, dispatching documents up to #%i/%i",
                        pass_, chunk_no * chunksize + len(chunk), lencorpus
                    )
                    # this will eventually block until some jobs finish, because the queue has a small finite length
                    self.dispatcher.putjob(chunk)
                else:
                    logger.info(
                        "PROGRESS: pass %i, at document #%i/%i",
                        pass_, chunk_no * chunksize + len(chunk), lencorpus
                    )
                    # do_estep requires the indexes of the documents being trained on, to know what authors
                    # correspond to the documents.
                    gammat = self.do_estep(chunk, self.author2doc, self.doc2author, rho(), other, chunk_doc_idx)

                    if self.optimize_alpha:
                        self.update_alpha(gammat, rho())

                dirty = True
                del chunk

                # perform an M step. determine when based on update_every, don't do this after every chunk
                if update_every and (chunk_no + 1) % (update_every * self.numworkers) == 0:
                    if self.dispatcher:
                        # distributed mode: wait for all workers to finish
                        logger.info("reached the end of input; now waiting for all remaining jobs to finish")
                        other = self.dispatcher.getstate()
                    self.do_mstep(rho(), other, pass_ > 0)
                    del other  # frees up memory

                    if self.dispatcher:
                        logger.info('initializing workers')
                        self.dispatcher.reset(self.state)
                    else:
                        other = AuthorTopicState(self.eta, self.state.sstats.shape, (0, 0))
                    dirty = False
            # endfor single corpus iteration
            if reallen != lencorpus:
                raise RuntimeError("input corpus size changed during training (don't use generators as input)")

            if dirty:
                # finish any remaining updates
                if self.dispatcher:
                    # distributed mode: wait for all workers to finish
                    logger.info("reached the end of input; now waiting for all remaining jobs to finish")
                    other = self.dispatcher.getstate()
                self.do_mstep(rho(), other, pass_ > 0)
                del other

    def bound(self, chunk, chunk_doc_idx=None, subsample_ratio=1.0, author2doc=None, doc2author=None):
        """Estimate the variational bound of documents from `corpus`.

        :math:`\mathbb{E_{q}}[\log p(corpus)] - \mathbb{E_{q}}[\log q(corpus)]`

        Notes
        -----
        There are basically two use cases of this method:

        #. `chunk` is a subset of the training corpus, and `chunk_doc_idx` is provided,
           indicating the indexes of the documents in the training corpus.
        #. `chunk` is a test set (held-out data), and `author2doc` and `doc2author` corresponding to this test set
           are provided. There must not be any new authors passed to this method, `chunk_doc_idx` is not needed
           in this case.

        Parameters
        ----------
        chunk : iterable of list of (int, float)
            Corpus in BoW format.
        chunk_doc_idx : numpy.ndarray, optional
            Assigns the value for document index.
        subsample_ratio : float, optional
            Used for calculation of word score for estimation of variational bound.
        author2doc : dict of (str, list of int), optinal
            A dictionary where keys are the names of authors and values are lists of documents that the author
            contributes to.
        doc2author : dict of (int, list of str), optional
            A dictionary where the keys are document IDs and the values are lists of author names.

        Returns
        -------
        float
            Value of variational bound score.

        """
        # TODO: enable evaluation of documents with new authors. One could, for example, make it
        # possible to pass a list of documents to self.inference with no author dictionaries,
        # assuming all the documents correspond to one (unseen) author, learn the author's
        # gamma, and return gamma (without adding it to self.state.gamma). Of course,
        # collect_sstats should be set to false, so that the model is not updated w.r.t. these
        # new documents.

        _lambda = self.state.get_lambda()
        Elogbeta = dirichlet_expectation(_lambda)
        expElogbeta = np.exp(Elogbeta)

        gamma = self.state.gamma

        if author2doc is None and doc2author is None:
            # Evaluating on training documents (chunk of self.corpus).
            author2doc = self.author2doc
            doc2author = self.doc2author

            if not chunk_doc_idx:
                # If author2doc and doc2author are not provided, chunk is assumed to be a subset of
                # self.corpus, and chunk_doc_idx is thus required.
                raise ValueError(
                    'Either author dictionaries or chunk_doc_idx must be provided. '
                    'Consult documentation of bound method.'
                )
        elif author2doc is not None and doc2author is not None:
            # Training on held-out documents (documents not seen during training).
            # All authors in dictionaries must still be seen during training.
            for a in author2doc.keys():
                if not self.author2doc.get(a):
                    raise ValueError('bound cannot be called with authors not seen during training.')

            if chunk_doc_idx:
                raise ValueError(
                    'Either author dictionaries or chunk_doc_idx must be provided, not both. '
                    'Consult documentation of bound method.'
                )
        else:
            raise ValueError(
                'Either both author2doc and doc2author should be provided, or neither. '
                'Consult documentation of bound method.'
            )

        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)

        word_score = 0.0
        theta_score = 0.0
        for d, doc in enumerate(chunk):
            if chunk_doc_idx:
                doc_no = chunk_doc_idx[d]
            else:
                doc_no = d
            # Get all authors in current document, and convert the author names to integer IDs.
            authors_d = [self.author2id[a] for a in self.doc2author[doc_no]]
            ids = np.array([id for id, _ in doc])  # Word IDs in doc.
            cts = np.array([cnt for _, cnt in doc])  # Word counts.

            if d % self.chunksize == 0:
                logger.debug("bound: at document #%i in chunk", d)

            # Computing the bound requires summing over expElogtheta[a, k] * expElogbeta[k, v], which
            # is the same computation as in normalizing phi.
            phinorm = self.compute_phinorm(expElogtheta[authors_d, :], expElogbeta[:, ids])
            word_score += np.log(1.0 / len(authors_d)) * sum(cts) + cts.dot(np.log(phinorm))

        # Compensate likelihood for when `chunk` above is only a sample of the whole corpus. This ensures
        # that the likelihood is always roughly on the same scale.
        word_score *= subsample_ratio

        # E[log p(theta | alpha) - log q(theta | gamma)]
        for a in author2doc.keys():
            a = self.author2id[a]
            theta_score += np.sum((self.alpha - gamma[a, :]) * Elogtheta[a, :])
            theta_score += np.sum(gammaln(gamma[a, :]) - gammaln(self.alpha))
            theta_score += gammaln(np.sum(self.alpha)) - gammaln(np.sum(gamma[a, :]))

        # theta_score is rescaled in a similar fashion.
        # TODO: treat this in a more general way, similar to how it is done with word_score.
        theta_score *= self.num_authors / len(author2doc)

        # E[log p(beta | eta) - log q (beta | lambda)]
        beta_score = 0.0
        beta_score += np.sum((self.eta - _lambda) * Elogbeta)
        beta_score += np.sum(gammaln(_lambda) - gammaln(self.eta))
        sum_eta = np.sum(self.eta)
        beta_score += np.sum(gammaln(sum_eta) - gammaln(np.sum(_lambda, 1)))

        total_score = word_score + theta_score + beta_score

        return total_score

    def get_document_topics(self, word_id, minimum_probability=None):
        """Override :meth:`~gensim.models.ldamodel.LdaModel.get_document_topics` and simply raises an exception.

        Warnings
        --------
        This method invalid for model, use :meth:`~gensim.models.atmodel.AuthorTopicModel.get_author_topics` or
        :meth:`~gensim.models.atmodel.AuthorTopicModel.get_new_author_topics` instead.

        Raises
        ------
        NotImplementedError
            Always.

        """
        raise NotImplementedError(
            'Method "get_document_topics" is not valid for the author-topic model. '
            'Use the "get_author_topics" method.'
        )

    def get_new_author_topics(self, corpus, minimum_probability=None):
        """Infers topics for new author.

        Infers a topic distribution for a new author over the passed corpus of docs,
        assuming that all documents are from this single new author.

        Parameters
        ----------
        corpus : iterable of list of (int, float)
            Corpus in BoW format.
        minimum_probability : float, optional
            Ignore topics with probability below this value, if None - 1e-8 is used.

        Returns
        -------
        list of (int, float)
            Topic distribution for the given `corpus`.

        """
        def rho():
            return pow(self.offset + 1 + 1, -self.decay)

        def rollback_new_author_chages():
            self.state.gamma = self.state.gamma[0:-1]

            del self.author2doc[new_author_name]
            a_id = self.author2id[new_author_name]
            del self.id2author[a_id]
            del self.author2id[new_author_name]

            for new_doc_id in corpus_doc_idx:
                del self.doc2author[new_doc_id]

        try:
            len_input_corpus = len(corpus)
        except TypeError:
            logger.warning("input corpus stream has no len(); counting documents")
            len_input_corpus = sum(1 for _ in corpus)
        if len_input_corpus == 0:
            raise ValueError("AuthorTopicModel.get_new_author_topics() called with an empty corpus")

        new_author_name = "placeholder_name"
        # indexes representing the documents in the input corpus
        corpus_doc_idx = list(range(self.total_docs, self.total_docs + len_input_corpus))

        # Add the new placeholder author to author2id/id2author dictionaries.
        num_new_authors = 1
        author_id = self.num_authors
        if new_author_name in self.author2id:
            raise ValueError("self.author2id already has 'placeholder_name' author")
        self.author2id[new_author_name] = author_id
        self.id2author[author_id] = new_author_name

        # Add new author in author2doc and doc into doc2author.
        self.author2doc[new_author_name] = corpus_doc_idx
        for new_doc_id in corpus_doc_idx:
            self.doc2author[new_doc_id] = [new_author_name]

        gamma_new = self.random_state.gamma(100., 1. / 100., (num_new_authors, self.num_topics))
        self.state.gamma = np.vstack([self.state.gamma, gamma_new])

        # Should not record the sstats, as we are goint to delete the new author after calculated.
        try:
            gammat, _ = self.inference(
                corpus, self.author2doc, self.doc2author, rho(),
                collect_sstats=False, chunk_doc_idx=corpus_doc_idx
            )
            new_author_topics = self.get_author_topics(new_author_name, minimum_probability)
        finally:
            rollback_new_author_chages()
        return new_author_topics

    def get_author_topics(self, author_name, minimum_probability=None):
        """Get topic distribution the given author.

        Parameters
        ----------
        author_name : str
            Name of the author for which the topic distribution needs to be estimated.
        minimum_probability : float, optional
            Sets the minimum probability value for showing the topics of a given author, topics with probability <
            `minimum_probability` will be ignored.

        Returns
        -------
        list of (int, float)
            Topic distribution of an author.

        Example
        -------
        >>> from gensim.models import AuthorTopicModel
        >>> from gensim.corpora import mmcorpus
        >>> from gensim.test.utils import common_dictionary, datapath, temporary_file

        >>> author2doc = {
        ...     'john': [0, 1, 2, 3, 4, 5, 6],
        ...     'jane': [2, 3, 4, 5, 6, 7, 8],
        ...     'jack': [0, 2, 4, 6, 8]
        ... }
        >>>
        >>> corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        >>>
        >>> with temporary_file("serialized") as s_path:
        ...     model = AuthorTopicModel(
        ...          corpus, author2doc=author2doc, id2word=common_dictionary, num_topics=4,
        ...          serialized=True, serialization_path=s_path
        ...     )
        ...
        ...     model.update(corpus, author2doc)  # update the author-topic model with additional documents
        >>>
        >>> # construct vectors for authors
        >>> author_vecs = [model.get_author_topics(author) for author in model.id2author.values()]

        """
        author_id = self.author2id[author_name]

        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        minimum_probability = max(minimum_probability, 1e-8)  # never allow zero values in sparse output

        topic_dist = self.state.gamma[author_id, :] / sum(self.state.gamma[author_id, :])

        author_topics = [
            (topicid, topicvalue) for topicid, topicvalue in enumerate(topic_dist)
            if topicvalue >= minimum_probability
        ]

        return author_topics

    def __getitem__(self, author_names, eps=None):
        """Get topic distribution for input `author_names`.

        Parameters
        ----------
        author_names : {str, list of str}
            Name(s) of the author for which the topic distribution needs to be estimated.
        eps : float, optional
            The minimum probability value for showing the topics of a given author, topics with probability < `eps`
            will be ignored.

        Returns
        -------
        list of (int, float) **or** list of list of (int, float)
            Topic distribution for the author(s), type depends on type of `author_names`.

        """
        if isinstance(author_names, list):
            items = []
            for a in author_names:
                items.append(self.get_author_topics(a, minimum_probability=eps))
        else:
            items = self.get_author_topics(author_names, minimum_probability=eps)

        return items
