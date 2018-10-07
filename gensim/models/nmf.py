import numpy as np
import scipy.sparse
import logging
from scipy.stats import halfnorm
from gensim import utils
from gensim import matutils
from gensim import interfaces
from gensim.models import basemodel
from gensim.models.nmf_pgd import solve_h, solve_r
import itertools

logger = logging.getLogger(__name__)


class Nmf(interfaces.TransformationABC, basemodel.BaseTopicModel):
    """Online Non-Negative Matrix Factorization.
    """

    def __init__(
        self,
        corpus=None,
        num_topics=100,
        id2word=None,
        chunksize=2000,
        passes=1,
        lambda_=1.,
        kappa=1.,
        minimum_probability=0.01,
        use_r=False,
        store_r=False,
        w_max_iter=200,
        w_stop_condition=1e-4,
        h_r_max_iter=50,
        h_r_stop_condition=1e-3,
        eval_every=10,
        v_max=None,
        normalize=True,
        sparse_coef=3
    ):
        """

        Parameters
        ----------
        corpus : Corpus
            Training corpus
        num_topics : int
            Number of components in resulting matrices.
        id2word: Dict[int, str]
            Token id to word mapping
        chunksize: int
            Number of documents in a chunk
        passes: int
            Number of full passes over the training corpus
        lambda_ : float
            Weight of the residuals regularizer
        kappa : float
            Optimization step size
        store_r : bool
            Whether to save residuals during training
        normalize
        """
        self._w_error = None
        self.n_features = None
        self.num_topics = num_topics
        self.id2word = id2word
        self.chunksize = chunksize
        self.passes = passes
        self._lambda_ = lambda_
        self._kappa = kappa
        self.minimum_probability = minimum_probability
        self.use_r = use_r
        self._w_max_iter = w_max_iter
        self._w_stop_condition = w_stop_condition
        self._h_r_max_iter = h_r_max_iter
        self._h_r_stop_condition = h_r_stop_condition
        self._H = []
        self.v_max = v_max
        self.eval_every = eval_every
        self.normalize = normalize
        self.sparse_coef = sparse_coef

        self.A = None
        self.B = None

        self.w_std = None

        if store_r:
            self._R = []
        else:
            self._R = None

        if corpus is not None:
            self.update(corpus)

    def get_topics(self):
        if self.normalize:
            return self._W.T.toarray() / self._W.T.toarray().sum(axis=1).reshape(-1, 1)

        return self._W.T.toarray()

    def __getitem__(self, bow, eps=None):
        return self.get_document_topics(bow, eps)

    def show_topics(self, num_topics=10, num_words=10, log=False, formatted=True):
        """
        Args:
            num_topics (int): show results for first `num_topics` topics.
                Unlike LSA, there is no natural ordering between the topics in LDA.
                The returned `num_topics <= self.num_topics` subset of all topics is
                therefore arbitrary and may change between two LDA training runs.
            num_words (int): include top `num_words` with highest probabilities in topic.
            log (bool): If True, log output in addition to returning it.
            formatted (bool): If True, format topics as strings, otherwise return them as
                `(word, probability)` 2-tuples.
        Returns:
            list: `num_words` most significant words for `num_topics` number of topics
            (10 words for top 10 topics, by default).
        """
        # TODO: maybe count sparsity in some other way
        sparsity = self._W.getnnz(axis=0)

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

        topics = self.get_topics()

        # print(topics)

        for i in chosen_topics:
            topic = topics[i]
            # print(topic)
            # print(topic.shape)
            bestn = matutils.argsort(topic, num_words, reverse=True).ravel()
            # print(type(bestn))
            # print(bestn.shape)
            topic = [(self.id2word[id], topic[id]) for id in bestn]
            if formatted:
                topic = " + ".join(['%.3f*"%s"' % (v, k) for k, v in topic])

            shown.append((i, topic))
            if log:
                logger.info("topic #%i (%.3f): %s", i, sparsity[i], topic)

        return shown

    def show_topic(self, topicid, topn=10):
        """
        Args:
            topn (int): Only return 2-tuples for the topn most probable words
                (ignore the rest).

        Returns:
            list: of `(word, probability)` 2-tuples for the most probable
            words in topic `topicid`.
        """
        return [
            (self.id2word[id], value)
            for id, value in self.get_topic_terms(topicid, topn)
        ]

    def get_topic_terms(self, topicid, topn=10):
        """
        Args:
            topn (int): Only return 2-tuples for the topn most probable words
                (ignore the rest).

        Returns:
            list: `(word_id, probability)` 2-tuples for the most probable words
            in topic with id `topicid`.
        """
        topic = self.get_topics()[topicid]
        bestn = matutils.argsort(topic, topn, reverse=True)
        return [(idx, topic[idx]) for idx in bestn]

    def get_term_topics(self, word_id, minimum_probability=None):
        """
        Args:
            word_id (int): ID of the word to get topic probabilities for.
            minimum_probability (float): Only include topic probabilities above this
                value (None by default). If set to None, use 1e-8 to prevent including 0s.
        Returns:
            list: The most likely topics for the given word. Each topic is represented
            as a tuple of `(topic_id, term_probability)`.
        """
        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        minimum_probability = max(minimum_probability, 1e-8)

        # if user enters word instead of id in vocab, change to get id
        if isinstance(word_id, str):
            word_id = self.id2word.doc2bow([word_id])[0][0]

        values = []

        for topic_id in range(0, self.num_topics):
            word_coef = self._W[word_id, topic_id]

            if word_coef >= minimum_probability:
                values.append((topic_id, word_coef))

        return values

    def get_document_topics(self, bow, minimum_probability=None):
        v = matutils.corpus2csc([bow], len(self.id2word)).tocsr()
        h, _ = self._solveproj(v, self._W, v_max=np.inf)

        if self.normalize:
            h /= h.sum(axis=0)

        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        minimum_probability = max(minimum_probability, 1e-8)

        return [
            (idx, proba)
            for idx, proba
            in enumerate(h[:, 0])
            if not minimum_probability or proba > minimum_probability
        ]

    def _setup(self, corpus):
        self._h, self._r = None, None
        first_doc_it = itertools.tee(corpus, 1)
        first_doc = next(first_doc_it[0])
        first_doc = matutils.corpus2csc([first_doc], len(self.id2word))[:, 0]
        self.n_features = first_doc.shape[0]
        self.w_std = np.sqrt(
            first_doc.mean()
            / (self.n_features * self.num_topics)
        )

        self._W = np.abs(
            self.w_std
            * halfnorm.rvs(size=(self.n_features, self.num_topics))
        )

        is_great_enough = self._W > self.w_std * self.sparse_coef

        self._W *= is_great_enough | ~is_great_enough.all(axis=0)

        self._W = scipy.sparse.csc_matrix(self._W)

        self.A = scipy.sparse.csr_matrix((self.num_topics, self.num_topics))
        self.B = scipy.sparse.csc_matrix((self.n_features, self.num_topics))

    def update(self, corpus, chunks_as_numpy=False):
        """

        Parameters
        ----------
        corpus : matrix or iterator
            Matrix to factorize.
        chunks_as_numpy: bool
        """

        if self.n_features is None:
            self._setup(corpus)

        chunk_idx = 1

        for _ in range(self.passes):
            for chunk in utils.grouper(
                corpus, self.chunksize, as_numpy=chunks_as_numpy
            ):
                v = matutils.corpus2csc(chunk, len(self.id2word)).tocsr()
                self._h, self._r = self._solveproj(v, self._W, r=self._r, h=self._h, v_max=self.v_max)
                h, r = self._h, self._r
                self._H.append(h)
                if self._R is not None:
                    self._R.append(r)

                self.A += h.dot(h.T)
                self.A *= (max(len(self._H) - 1, 1)) / len(self._H)

                self.B += (v - r).dot(h.T)
                self.B *= (max(len(self._H) - 1, 1)) / len(self._H)

                self._solve_w()

                if chunk_idx % self.eval_every == 0:
                    logger.info(
                        "Loss (no outliers): {}\tLoss (with outliers): {}".format(
                            scipy.sparse.linalg.norm(v - self._W.dot(h)),
                            scipy.sparse.linalg.norm(v - self._W.dot(h) - r),
                        )
                    )

                chunk_idx += 1

        logger.info(
            "Loss (no outliers): {}\tLoss (with outliers): {}".format(
                scipy.sparse.linalg.norm(v - self._W.dot(h)),
                scipy.sparse.linalg.norm(v - self._W.dot(h) - r),
            )
        )

    def _solve_w(self):
        def error():
            # print(type(self._W))
            # print(self._W[:5, :5])
            # print(type(self.A))
            # print(self.A[:5, :5])
            # print(type(self.B))
            # print(self.B[:5, :5])
            return (
                0.5 * self._W.T.dot(self._W).dot(self.A).diagonal().sum()
                - self._W.T.dot(self.B).diagonal().sum()
            )

        eta = self._kappa / scipy.sparse.linalg.norm(self.A)

        if not self._w_error:
            self._w_error = error()

        for iter_number in range(self._w_max_iter):
            logger.debug("w_error: %s" % self._w_error)

            self._W -= eta * (self._W.dot(self.A) - self.B)
            self.__transform()

            error_ = error()

            if np.abs((error_ - self._w_error) / self._w_error) < self._w_stop_condition:
                break

            self._w_error = error_

    @staticmethod
    def __solve_r(r, r_actual, lambda_, v_max):
        r_actual_sign = np.sign(r_actual.data)

        np.abs(r_actual.data, out=r_actual.data)
        r_actual.data -= lambda_
        np.maximum(r_actual.data, 0.0, out=r_actual.data)

        r_actual.data *= r_actual_sign
        r_actual.eliminate_zeros()

        np.clip(r_actual.data, -v_max, v_max, out=r_actual.data)

        violation = scipy.sparse.linalg.norm(r - r_actual)

        r.indices = r_actual.indices
        r.indptr = r_actual.indptr
        r.data = r_actual.data

        return violation

    @staticmethod
    def __solve_h(h, Wt_v_minus_r, WtW, eta):
        grad = (WtW.dot(h) - Wt_v_minus_r) * eta
        grad = scipy.sparse.csr_matrix(grad)
        new_h = h - grad

        np.maximum(new_h.data, 0.0, out=new_h.data)
        new_h.eliminate_zeros()

        return new_h, scipy.sparse.linalg.norm(grad)

    def __transform(self):
        np.clip(self._W.data, 0, self.v_max, out=self._W.data)
        self._W.eliminate_zeros()
        sumsq = scipy.sparse.linalg.norm(self._W, axis=0)
        np.maximum(sumsq, 1, out=sumsq)
        sumsq = np.repeat(sumsq, self._W.getnnz(axis=0))
        self._W.data /= sumsq

        is_great_enough_data = self._W.data > self.w_std * self.sparse_coef
        is_great_enough = self._W.toarray() > self.w_std * self.sparse_coef
        is_all_too_small = is_great_enough.sum(axis=0) == 0
        is_all_too_small = np.repeat(
            is_all_too_small,
            self._W.getnnz(axis=0)
        )

        is_great_enough_data |= is_all_too_small

        self._W.data *= is_great_enough_data
        self._W.eliminate_zeros()

    def _solveproj(self, v, W, h=None, r=None, v_max=None):
        m, n = W.shape
        if v_max is not None:
            self.v_max = v_max
        elif self.v_max is None:
            self.v_max = v.max()

        batch_size = v.shape[1]
        rshape = (m, batch_size)
        hshape = (n, batch_size)

        if h is None or h.shape != hshape:
            h = scipy.sparse.csr_matrix(hshape)

        if r is None or r.shape != rshape:
            r = scipy.sparse.csr_matrix(rshape)

        WtW = W.T.dot(W)

        eta = self._kappa / scipy.sparse.linalg.norm(W) ** 2

        _h_r_error = None

        for iter_number in range(self._h_r_max_iter):
            logger.debug("h_r_error: %s" % _h_r_error)

            error_ = 0

            Wt_v_minus_r = W.T.dot(v - r)

            # h_ = h.toarray()
            # error_ += solve_h(h_, Wt_v_minus_r.toarray(), WtW.toarray(), self._kappa)
            # h = scipy.sparse.csr_matrix(h_)
            h, error_h = self.__solve_h(h, Wt_v_minus_r, WtW, eta)
            error_ += error_h

            if self.use_r:
                r_actual = v - W.dot(h)
                # error_ += solve_r(
                #     r.indptr, r.indices, r.data,
                #     r_actual.indptr, r_actual.indices, r_actual.data,
                #     self._lambda_,
                #     self.v_max
                # )
                # r = r_actual
                error_ += self.__solve_r(r, r_actual, self._lambda_, self.v_max)

            error_ /= m

            if not _h_r_error:
                _h_r_error = error_
                continue

            if np.abs(_h_r_error - error_) < self._h_r_stop_condition:
                break

            _h_r_error = error_

        return h, r
