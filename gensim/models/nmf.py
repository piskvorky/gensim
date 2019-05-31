"""Online Non-Negative Matrix Factorization.
Implementation of the efficient incremental algorithm of Renbo Zhao, Vincent Y. F. Tan et al.
`[PDF] <https://arxiv.org/abs/1604.02634>`_.

This NMF implementation updates in a streaming fashion and works best with sparse corpora.

- W is a word-topic matrix
- h is a topic-document matrix
- v is an input corpus batch, word-document matrix
- A, B - matrices that accumulate information from every consecutive chunk. A = h.dot(ht), B = v.dot(ht).

The idea of the algorithm is as follows:

.. code-block:: text

    Initialize W, A and B matrices

    Input the corpus
    Split the corpus into batches

    for v in batches:
        infer h:
            do coordinate gradient descent step to find h that minimizes (v - Wh) l2 norm

            bound h so that it is non-negative

        update A and B:
            A = h.dot(ht)
            B = v.dot(ht)

        update W:
            do gradient descent step to find W that minimizes 0.5*trace(WtWA) - trace(WtB) l2 norm

Examples
--------

Train an NMF model using a Gensim corpus

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_texts
    >>> from gensim.corpora.dictionary import Dictionary
    >>>
    >>> # Create a corpus from a list of texts
    >>> common_dictionary = Dictionary(common_texts)
    >>> common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
    >>>
    >>> # Train the model on the corpus.
    >>> nmf = Nmf(common_corpus, num_topics=10)

Save a model to disk, or reload a pre-trained model

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>>
    >>> # Save model to disk.
    >>> temp_file = datapath("model")
    >>> nmf.save(temp_file)
    >>>
    >>> # Load a potentially pretrained model from disk.
    >>> nmf = Nmf.load(temp_file)

Infer vectors for new documents

.. sourcecode:: pycon

    >>> # Create a new corpus, made of previously unseen documents.
    >>> other_texts = [
    ...     ['computer', 'time', 'graph'],
    ...     ['survey', 'response', 'eps'],
    ...     ['human', 'system', 'computer']
    ... ]
    >>> other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]
    >>>
    >>> unseen_doc = other_corpus[0]
    >>> vector = Nmf[unseen_doc]  # get topic probability distribution for a document

Update the model by incrementally training on the new corpus

.. sourcecode:: pycon

    >>> nmf.update(other_corpus)
    >>> vector = nmf[unseen_doc]

A lot of parameters can be tuned to optimize training for your specific case

.. sourcecode:: pycon

    >>> nmf = Nmf(common_corpus, num_topics=50, kappa=0.1, eval_every=5)  # decrease training step size

The NMF should be used whenever one needs extremely fast and memory optimized topic model.

"""
import collections

import logging
import numpy as np
import scipy.sparse
from gensim.models.nmf_pgd import solve_h
from scipy.stats import halfnorm

from gensim import interfaces
from gensim import matutils
from gensim import utils
from gensim.interfaces import TransformedCorpus
from gensim.models import basemodel, CoherenceModel

logger = logging.getLogger(__name__)

OLD_SCIPY = int(scipy.__version__.split('.')[1]) <= 18


class Nmf(interfaces.TransformationABC, basemodel.BaseTopicModel):
    """Online Non-Negative Matrix Factorization.

    `Renbo Zhao et al :"Online Nonnegative Matrix Factorization with Outliers" <https://arxiv.org/abs/1604.02634>`_

    """

    def __init__(
        self,
        corpus=None,
        num_topics=100,
        id2word=None,
        chunksize=2000,
        passes=1,
        kappa=1.0,
        minimum_probability=0.01,
        w_max_iter=200,
        w_stop_condition=1e-4,
        h_max_iter=50,
        h_stop_condition=1e-3,
        eval_every=10,
        normalize=True,
        random_state=None,
    ):
        r"""

        Parameters
        ----------
        corpus : iterable of list of (int, float) or `csc_matrix` with the shape (n_tokens, n_documents), optional
            Training corpus.
            Can be either iterable of documents, which are lists of `(word_id, word_count)`,
            or a sparse csc matrix of BOWs for each document.
            If not specified, the model is left uninitialized (presumably, to be trained later with `self.train()`).
        num_topics : int, optional
            Number of topics to extract.
        id2word: {dict of (int, str), :class:`gensim.corpora.dictionary.Dictionary`}
            Mapping from word IDs to words. It is used to determine the vocabulary size, as well as for
            debugging and topic printing.
        chunksize: int, optional
            Number of documents to be used in each training chunk.
        passes: int, optional
            Number of full passes over the training corpus.
            Leave at default `passes=1` if your input is an iterator.
        kappa : float, optional
            Gradient descent step size.
            Larger value makes the model train faster, but could lead to non-convergence if set too large.
        minimum_probability:
            If `normalize` is True, topics with smaller probabilities are filtered out.
            If `normalize` is False, topics with smaller factors are filtered out.
            If set to None, a value of 1e-8 is used to prevent 0s.
        w_max_iter: int, optional
            Maximum number of iterations to train W per each batch.
        w_stop_condition: float, optional
            If error difference gets less than that, training of ``W`` stops for the current batch.
        h_max_iter: int, optional
            Maximum number of iterations to train h per each batch.
        h_stop_condition: float
            If error difference gets less than that, training of ``h`` stops for the current batch.
        eval_every: int, optional
            Number of batches after which l2 norm of (v - Wh) is computed. Decreases performance if set too low.
        normalize: bool or None, optional
            Whether to normalize the result. Allows for estimation of perplexity, coherence, e.t.c.
        random_state: {np.random.RandomState, int}, optional
            Seed for random generator. Needed for reproducibility.

        """
        self.num_topics = num_topics
        self.id2word = id2word
        self.chunksize = chunksize
        self.passes = passes
        self._kappa = kappa
        self.minimum_probability = minimum_probability
        self._w_max_iter = w_max_iter
        self._w_stop_condition = w_stop_condition
        self._h_max_iter = h_max_iter
        self._h_stop_condition = h_stop_condition
        self.eval_every = eval_every
        self.normalize = normalize
        self.random_state = utils.get_random_state(random_state)

        self.v_max = None

        if self.id2word is None:
            self.id2word = utils.dict_from_corpus(corpus)

        self.num_tokens = len(self.id2word)

        self.A = None
        self.B = None

        self._W = None
        self.w_std = None
        self._w_error = np.inf

        self._h = None

        if corpus is not None:
            self.update(corpus)

    def get_topics(self, normalize=None):
        """Get the term-topic matrix learned during inference.

        Parameters
        ----------
        normalize: bool or None, optional
            Whether to normalize the result. Allows for estimation of perplexity, coherence, e.t.c.

        Returns
        -------
        numpy.ndarray
            The probability for each word in each topic, shape (`num_topics`, `vocabulary_size`).

        """
        dense_topics = self._W.T
        if normalize is None:
            normalize = self.normalize
        if normalize:
            return dense_topics / dense_topics.sum(axis=1).reshape(-1, 1)

        return dense_topics

    def __getitem__(self, bow, eps=None):
        return self.get_document_topics(bow, eps)

    def show_topics(self, num_topics=10, num_words=10, log=False, formatted=True, normalize=None):
        """Get the topics sorted by sparsity.

        Parameters
        ----------
        num_topics : int, optional
            Number of topics to be returned. Unlike LSA, there is no natural ordering between the topics in NMF.
            The returned topics subset of all topics is therefore arbitrary and may change between two NMF
            training runs.
        num_words : int, optional
            Number of words to be presented for each topic. These will be the most relevant words (assigned the highest
            probability for each topic).
        log : bool, optional
            Whether the result is also logged, besides being returned.
        formatted : bool, optional
            Whether the topic representations should be formatted as strings. If False, they are returned as
            2 tuples of (word, probability).
        normalize: bool or None, optional
            Whether to normalize the result. Allows for estimation of perplexity, coherence, e.t.c.

        Returns
        -------
        list of {str, tuple of (str, float)}
            a list of topics, each represented either as a string (when `formatted` == True) or word-probability
            pairs.

        """
        if normalize is None:
            normalize = self.normalize

        # Compute fraction of zero elements in each column

        sparsity = np.zeros(self._W.shape[1])

        for row in self._W:
            sparsity += (row == 0)

        sparsity /= self._W.shape[0]

        if num_topics < 0 or num_topics >= self.num_topics:
            num_topics = self.num_topics
            chosen_topics = range(num_topics)
        else:
            num_topics = min(num_topics, self.num_topics)

            sorted_topics = list(matutils.argsort(sparsity))
            chosen_topics = (
                sorted_topics[: num_topics // 2] + sorted_topics[-num_topics // 2:]
            )

        shown = []

        topics = self.get_topics(normalize=normalize)

        for i in chosen_topics:
            topic = topics[i]
            bestn = matutils.argsort(topic, num_words, reverse=True).ravel()
            topic = [(self.id2word[id], topic[id]) for id in bestn]
            if formatted:
                topic = " + ".join(['%.3f*"%s"' % (v, k) for k, v in topic])

            shown.append((i, topic))
            if log:
                logger.info("topic #%i (%.3f): %s", i, sparsity[i], topic)

        return shown

    def show_topic(self, topicid, topn=10, normalize=None):
        """Get the representation for a single topic. Words here are the actual strings, in constrast to
        :meth:`~gensim.models.nmf.Nmf.get_topic_terms` that represents words by their vocabulary ID.

        Parameters
        ----------
        topicid : int
            The ID of the topic to be returned
        topn : int, optional
            Number of the most significant words that are associated with the topic.
        normalize: bool or None, optional
            Whether to normalize the result. Allows for estimation of perplexity, coherence, e.t.c.

        Returns
        -------
        list of (str, float)
            Word - probability pairs for the most relevant words generated by the topic.

        """
        if normalize is None:
            normalize = self.normalize

        return [
            (self.id2word[id], value)
            for id, value in self.get_topic_terms(topicid, topn,
                                                  normalize=normalize)
        ]

    def get_topic_terms(self, topicid, topn=10, normalize=None):
        """Get the representation for a single topic. Words the integer IDs, in constrast to
        :meth:`~gensim.models.nmf.Nmf.show_topic` that represents words by the actual strings.

        Parameters
        ----------
        topicid : int
            The ID of the topic to be returned
        topn : int, optional
            Number of the most significant words that are associated with the topic.
        normalize: bool or None, optional
            Whether to normalize the result. Allows for estimation of perplexity, coherence, e.t.c.

        Returns
        -------
        list of (int, float)
            Word ID - probability pairs for the most relevant words generated by the topic.

        """
        topic = self._W[:, topicid]

        if normalize is None:
            normalize = self.normalize
        if normalize:
            topic /= topic.sum()

        bestn = matutils.argsort(topic, topn, reverse=True)
        return [(idx, topic[idx]) for idx in bestn]

    def top_topics(self, corpus, texts=None, dictionary=None, window_size=None,
                   coherence='u_mass', topn=20, processes=-1):
        """Get the topics sorted by coherence.

        Parameters
        ----------
        corpus : iterable of list of (int, float) or `csc_matrix` with the shape (n_tokens, n_documents)
            Training corpus.
            Can be either iterable of documents, which are lists of `(word_id, word_count)`,
            or a sparse csc matrix of BOWs for each document.
            If not specified, the model is left uninitialized (presumably, to be trained later with `self.train()`).
        texts : list of list of str, optional
            Tokenized texts, needed for coherence models that use sliding window based (i.e. coherence=`c_something`)
            probability estimator .
        dictionary : {dict of (int, str), :class:`gensim.corpora.dictionary.Dictionary`}, optional
            Dictionary mapping of id word to create corpus.
            If `model.id2word` is present, this is not needed. If both are provided, passed `dictionary` will be used.
        window_size : int, optional
            Is the size of the window to be used for coherence measures using boolean sliding window as their
            probability estimator. For 'u_mass' this doesn't matter.
            If None - the default window sizes are used which are: 'c_v' - 110, 'c_uci' - 10, 'c_npmi' - 10.
        coherence : {'u_mass', 'c_v', 'c_uci', 'c_npmi'}, optional
            Coherence measure to be used.
            Fastest method - 'u_mass', 'c_uci' also known as `c_pmi`.
            For 'u_mass' corpus should be provided, if texts is provided, it will be converted to corpus
            using the dictionary. For 'c_v', 'c_uci' and 'c_npmi' `texts` should be provided (`corpus` isn't needed)
        topn : int, optional
            Integer corresponding to the number of top words to be extracted from each topic.
        processes : int, optional
            Number of processes to use for probability estimation phase, any value less than 1 will be interpreted as
            num_cpus - 1.

        Returns
        -------
        list of (list of (int, str), float)
            Each element in the list is a pair of a topic representation and its coherence score. Topic representations
            are distributions of words, represented as a list of pairs of word IDs and their probabilities.

        """
        cm = CoherenceModel(
            model=self, corpus=corpus, texts=texts, dictionary=dictionary,
            window_size=window_size, coherence=coherence, topn=topn,
            processes=processes
        )
        coherence_scores = cm.get_coherence_per_topic()

        str_topics = []
        for topic in self.get_topics():  # topic = array of vocab_size floats, one per term
            bestn = matutils.argsort(topic, topn=topn, reverse=True)  # top terms for topic
            beststr = [(topic[_id], self.id2word[_id]) for _id in bestn]  # membership, token
            str_topics.append(beststr)  # list of topn (float membership, token) tuples

        scored_topics = zip(str_topics, coherence_scores)
        return sorted(scored_topics, key=lambda tup: tup[1], reverse=True)

    def get_term_topics(self, word_id, minimum_probability=None, normalize=None):
        """Get the most relevant topics to the given word.

        Parameters
        ----------
        word_id : int
            The word for which the topic distribution will be computed.
        minimum_probability : float, optional
            If `normalize` is True, topics with smaller probabilities are filtered out.
            If `normalize` is False, topics with smaller factors are filtered out.
            If set to None, a value of 1e-8 is used to prevent 0s.
        normalize: bool or None, optional
            Whether to normalize the result. Allows for estimation of perplexity, coherence, e.t.c.

        Returns
        -------
        list of (int, float)
            The relevant topics represented as pairs of their ID and their assigned probability, sorted
            by relevance to the given word.

        """
        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        minimum_probability = max(minimum_probability, 1e-8)

        # if user enters word instead of id in vocab, change to get id
        if isinstance(word_id, str):
            word_id = self.id2word.doc2bow([word_id])[0][0]

        values = []

        word_topics = self._W[word_id]

        if normalize is None:
            normalize = self.normalize
        if normalize and word_topics.sum() > 0:
            word_topics /= word_topics.sum()

        for topic_id in range(0, self.num_topics):
            word_coef = word_topics[topic_id]

            if word_coef >= minimum_probability:
                values.append((topic_id, word_coef))

        return values

    def get_document_topics(self, bow, minimum_probability=None,
                            normalize=None):
        """Get the topic distribution for the given document.

        Parameters
        ----------
        bow : list of (int, float)
            The document in BOW format.
        minimum_probability : float
            If `normalize` is True, topics with smaller probabilities are filtered out.
            If `normalize` is False, topics with smaller factors are filtered out.
            If set to None, a value of 1e-8 is used to prevent 0s.
        normalize: bool or None, optional
            Whether to normalize the result. Allows for estimation of perplexity, coherence, e.t.c.

        Returns
        -------
        list of (int, float)
            Topic distribution for the whole document. Each element in the list is a pair of a topic's id, and
            the probability that was assigned to it.

        """
        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        minimum_probability = max(minimum_probability, 1e-8)

        # if the input vector is a corpus, return a transformed corpus
        is_corpus, corpus = utils.is_corpus(bow)

        if is_corpus:
            kwargs = dict(minimum_probability=minimum_probability)
            return self._apply(corpus, **kwargs)

        v = matutils.corpus2csc([bow], self.num_tokens)
        h = self._solveproj(v, self._W, v_max=np.inf)

        if normalize is None:
            normalize = self.normalize
        if normalize:
            the_sum = h.sum()
            if the_sum:
                h /= the_sum

        return [
            (idx, proba)
            for idx, proba in enumerate(h[:, 0])
            if not minimum_probability or proba > minimum_probability
        ]

    def _setup(self, v):
        """Infer info from the first batch and initialize the matrices.

        Parameters
        ----------
        v : `csc_matrix` with the shape (n_tokens, chunksize)
            Batch of bows.

        """
        self.w_std = np.sqrt(v.mean() / (self.num_tokens * self.num_topics))

        self._W = np.abs(
            self.w_std
            * halfnorm.rvs(
                size=(self.num_tokens, self.num_topics), random_state=self.random_state
            )
        )

        self.A = np.zeros((self.num_topics, self.num_topics))
        self.B = np.zeros((self.num_tokens, self.num_topics))

    def l2_norm(self, v):
        Wt = self._W.T

        l2 = 0

        for doc, doc_topics in zip(v.T, self._h.T):
            l2 += np.sum(np.square((doc - doc_topics.dot(Wt))))

        return np.sqrt(l2)

    def update(self, corpus, chunksize=None, passes=None, eval_every=None):
        """Train the model with new documents.

        Parameters
        ----------
        corpus : iterable of list of (int, float) or `csc_matrix` with the shape (n_tokens, n_documents)
            Training corpus.
            Can be either iterable of documents, which are lists of `(word_id, word_count)`,
            or a sparse csc matrix of BOWs for each document.
            If not specified, the model is left uninitialized (presumably, to be trained later with `self.train()`).
        chunksize: int, optional
            Number of documents to be used in each training chunk.
        passes: int, optional
            Number of full passes over the training corpus.
            Leave at default `passes=1` if your input is an iterator.
        eval_every: int, optional
            Number of batches after which l2 norm of (v - Wh) is computed. Decreases performance if set too low.

        """

        # use parameters given in constructor, unless user explicitly overrode them
        if passes is None:
            passes = self.passes
        if eval_every is None:
            eval_every = self.eval_every

        lencorpus = np.inf

        if isinstance(corpus, scipy.sparse.csc.csc_matrix):
            lencorpus = corpus.shape[1]
        else:
            try:
                lencorpus = len(corpus)
            except TypeError:
                logger.info("input corpus stream has no len()")

        if chunksize is None:
            chunksize = min(lencorpus, self.chunksize)

        evalafter = min(lencorpus, (eval_every or 0) * chunksize)

        if lencorpus == 0:
            logger.warning("Nmf.update() called with an empty corpus")
            return

        if isinstance(corpus, collections.Iterator) and self.passes > 1:
            raise ValueError("Corpus is an iterator, only `passes=1` is valid.")

        logger.info(
            "running NMF training, %s topics, %i passes over the supplied corpus of %s documents, evaluating l2 norm "
            "every %i documents",
            self.num_topics, passes, lencorpus, evalafter,
        )

        chunk_overall_idx = 1

        for pass_ in range(passes):
            if isinstance(corpus, scipy.sparse.csc.csc_matrix):
                grouper = (
                    # Older scipy (0.19 etc) throw an error when slicing beyond the actual sparse array dimensions, so
                    # we clip manually with min() here.

                    corpus[:, col_idx:min(corpus.shape[1], col_idx + self.chunksize)]
                    for col_idx
                    in range(0, corpus.shape[1], self.chunksize)
                )
            else:
                grouper = utils.grouper(corpus, self.chunksize)

            for chunk_idx, chunk in enumerate(grouper):
                if isinstance(corpus, scipy.sparse.csc.csc_matrix):
                    v = chunk[:, self.random_state.permutation(chunk.shape[1])]

                    chunk_len = v.shape[1]
                else:
                    self.random_state.shuffle(chunk)

                    v = matutils.corpus2csc(
                        chunk,
                        num_terms=self.num_tokens,
                    )

                    chunk_len = len(chunk)

                logger.info(
                    "PROGRESS: pass %i, at document #%i/%s",
                    pass_, chunk_idx * chunksize + chunk_len, lencorpus
                )

                if self._W is None:
                    # If `self._W` is not set (i.e. the first batch being handled), compute the initial matrix using the
                    # batch mean.

                    self._setup(v)

                self._h = self._solveproj(v, self._W, h=self._h, v_max=self.v_max)
                h = self._h

                if eval_every and (((chunk_idx + 1) * chunksize >= lencorpus) or (chunk_idx + 1) % eval_every == 0):
                    logger.info("L2 norm: {}".format(self.l2_norm(v)))
                    self.print_topics(5)

                self.A *= chunk_overall_idx - 1
                self.A += h.dot(h.T)
                self.A /= chunk_overall_idx

                self.B *= chunk_overall_idx - 1
                self.B += v.dot(h.T)
                self.B /= chunk_overall_idx

                previous_w_error = self._w_error

                self._solve_w()

                chunk_overall_idx += 1

                logger.info("W error diff: {}".format((self._w_error - previous_w_error)))

    def _solve_w(self):
        """Update W."""

        def error(WA):
            """An optimized version of 0.5 * trace(WtWA) - trace(WtB)."""
            return 0.5 * np.einsum('ij,ij', WA, self._W) - np.einsum('ij,ij', self._W, self.B)

        eta = self._kappa / np.linalg.norm(self.A)

        for iter_number in range(self._w_max_iter):
            logger.debug("w_error: {}".format(self._w_error))

            WA = self._W.dot(self.A)

            error_ = error(WA)

            if (
                self._w_error < np.inf
                and np.abs((error_ - self._w_error) / self._w_error) < self._w_stop_condition
            ):
                break

            self._w_error = error_

            self._W -= eta * (WA - self.B)
            self._transform()

    def _apply(self, corpus, chunksize=None, **kwargs):
        """Apply the transformation to a whole corpus and get the result as another corpus.

        Parameters
        ----------
        corpus : iterable of list of (int, float) or `csc_matrix` with the shape (n_tokens, n_documents)
            Training corpus.
            Can be either iterable of documents, which are lists of `(word_id, word_count)`,
            or a sparse csc matrix of BOWs for each document.
            If not specified, the model is left uninitialized (presumably, to be trained later with `self.train()`).
        chunksize : int, optional
            If provided, a more effective processing will performed.

        Returns
        -------
        :class:`~gensim.interfaces.TransformedCorpus`
            Transformed corpus.

        """
        return TransformedCorpus(self, corpus, chunksize, **kwargs)

    def _transform(self):
        """Apply boundaries on W."""
        np.clip(self._W, 0, self.v_max, out=self._W)
        sumsq = np.sqrt(np.einsum('ij,ij->j', self._W, self._W))
        np.maximum(sumsq, 1, out=sumsq)
        self._W /= sumsq

    @staticmethod
    def _dense_dot_csc(dense, csc):
        if OLD_SCIPY:
            return (csc.T.dot(dense.T)).T
        else:
            return scipy.sparse.csc_matrix.dot(dense, csc)

    def _solveproj(self, v, W, h=None, v_max=None):
        """Update residuals and representation(h) matrices.

        Parameters
        ----------
        v : scipy.sparse.csc_matrix
            Subset of training corpus.
        W : ndarray
            Dictionary matrix.
        h : ndarray
            Representation matrix.
        v_max : float
            Maximum possible value in matrices.

        """
        m, n = W.shape
        if v_max is not None:
            self.v_max = v_max
        elif self.v_max is None:
            self.v_max = v.max()

        batch_size = v.shape[1]
        hshape = (n, batch_size)

        if h is None or h.shape != hshape:
            h = np.zeros(hshape)

        Wt = W.T
        WtW = Wt.dot(W)

        h_error = None

        for iter_number in range(self._h_max_iter):
            logger.debug("h_error: {}".format(h_error))

            Wtv = self._dense_dot_csc(Wt, v)

            permutation = self.random_state.permutation(self.num_topics).astype(np.int32)

            error_ = solve_h(h, Wtv, WtW, permutation, self._kappa)

            error_ /= m

            if h_error and np.abs(h_error - error_) < self._h_stop_condition:
                break

            h_error = error_

        return h
