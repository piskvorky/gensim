import numpy as np
import logging
from scipy.stats import halfnorm
from gensim import utils
from gensim import matutils
from gensim import interfaces
from gensim.models import basemodel

logger = logging.getLogger('gensim.models.nmf')


class Nmf(interfaces.TransformationABC, basemodel.BaseTopicModel):
    """Online Non-Negative Matrix Factorization.

    Attributes
    ----------
    _W : dictionary matrix
    _H : loadings matrix
    _lambda: weight of outliers regularizer
    _kappa: step size coefficient

    """

    def __init__(self, corpus=None, num_topics=100, id2word=None,
                 chunksize=2000, passes=1, lambda_=1.,
                 kappa=1., store_r=False, max_iter=1e9):
        """

        Parameters
        ----------
        num_topics : int
            Number of components in resulting matrices.
        lambda_ : float
        kappa : float
        """
        self.n_features = None
        self.num_topics = num_topics
        self.id2word = id2word
        self.chunksize = chunksize
        self.passes = passes
        self._lambda_ = lambda_
        self._kappa = kappa
        self._H = []
        self.v_max = None
        self.max_iter = max_iter
        if store_r:
            self._R = []
        else:
            self._R = None

        if corpus is not None:
            self.update(corpus, chunksize)

    @property
    def A(self):
        return self._A / len(self._H)

    @A.setter
    def A(self, value):
        self._A = value

    @property
    def B(self):
        return self._B / len(self._H)

    @B.setter
    def B(self, value):
        self._B = value

    def get_topics(self):
        raise NotImplementedError

    def __getitem__(self, vec):
        raise NotImplementedError

    def _setup(self, X):
        self._h, self._r = None, None
        x = next(iter(X))
        x_asarray = matutils.corpus2dense([x], len(self.id2word), 1)[:, 0]
        m = len(x_asarray)
        avg = np.sqrt(x_asarray.mean() / m)

        self.n_features = len(x_asarray)

        self._W = np.abs(avg * halfnorm.rvs(size=(self.n_features, self.num_topics)) /
                         np.sqrt(self.num_topics))

        self.A = np.zeros((self.num_topics, self.num_topics))
        self.B = np.zeros((self.n_features, self.num_topics))
        return X

    def update(self, corpus, chunks_as_numpy=False):
        """

        Parameters
        ----------
        corpus : matrix or iterator
            Matrix to factorize.
        """

        if self.n_features is None:
            corpus = self._setup(corpus)

        r, h = self._r, self._h

        for _ in range(self.passes):
            for chunk in utils.grouper(corpus, self.chunksize, as_numpy=chunks_as_numpy):
                v = matutils.corpus2dense(chunk, len(self.id2word), len(chunk)).T
                h, r = self._solveproj(v, self._W, r=r, h=h, v_max=self.v_max)
                self._H.append(h)
                if self._R is not None:
                    self._R.append(r)

                self.A += np.dot(h, h.T)
                self.B += np.dot((v.T - r), h.T)
                self._solve_w()
                logger.info(
                    'Loss (no outliers): {}\tLoss (with outliers): {}'
                        .format(
                        np.linalg.norm(v.T - self._W.dot(h)),
                        np.linalg.norm(v.T - self._W.dot(h) - r)
                    )
                )

        self._r = r
        self._h = h

    def _solve_w(self):
        eta = self._kappa / np.linalg.norm(self.A, 'fro')
        n = 0
        lasttwo = np.zeros(2)
        while n <= 2 or (np.abs(
                (lasttwo[1] - lasttwo[0]) / lasttwo[0]) > 1e-5 and n < 1e9):
            self._W -= eta * (np.dot(self._W, self.A) - self.B)
            self._W = self.__transform(self._W)
            n += 1
            lasttwo[0] = lasttwo[1]
            lasttwo[1] = 0.5 * np.trace(self._W.T.dot(self._W).dot(self.A)) - \
                         np.trace(self._W.T.dot(self.B))

    def transform(self, corpus, return_r=False):
        H = []
        if return_r:
            R = []

        W = self._W
        for chunk in corpus:
            v = matutils.corpus2dense([chunk], len(self.id2word), 1).T[0]
            h, r = self._solveproj(v, W, v_max=np.inf)
            H.append(h.copy())
            if return_r:
                R.append(r.copy())

        H = np.stack(H, axis=-1)
        if return_r:
            return H, np.stack(R, axis=-1)
        else:
            return H

    def get_factor_matrices(self):
        if len(self._H) > 0:
            if len(self._H[0].shape) == 1:
                H = np.stack(self._H, axis=-1)
            else:
                H = np.concatenate(self._H, axis=1)

            return self._W, H
        else:
            return self._W, 0

    @staticmethod
    def __thresh(X, lambda1, vmax):
        res = np.abs(X) - lambda1
        np.maximum(res, 0.0, out=res)
        res *= np.sign(X)
        np.clip(res, -vmax, vmax, out=res)
        return res

    def __transform(self, W):
        W_ = W.copy()
        np.clip(W_, 0, self.v_max, out=W_)
        sumsq = np.linalg.norm(W_, axis=0)
        np.maximum(sumsq, 1, out=sumsq)
        return W_ / sumsq

    def _solveproj(self, v, W, h=None, r=None, v_max=None):
        m, n = W.shape
        v = v.T
        if v_max is not None:
            self.v_max = v_max
        elif self.v_max is None:
            self.v_max = v.max()
        # else:
        #     self.v_max = np.max((self.v_max, v.max()))
        if len(v.shape) == 2:
            batch_size = v.shape[1]
            rshape = (m, batch_size)
            hshape = (n, batch_size)
        else:
            rshape = m,
            hshape = n,
        if h is None or h.shape != hshape:
            h = np.zeros(hshape)

        if r is None or r.shape != rshape:
            r = np.zeros(rshape)

        eta = self._kappa / np.linalg.norm(W, 'fro') ** 2

        for _ in range(int(self.max_iter)):
            error = v - np.dot(W, h)

            # Solve for h
            h_ = h
            h = h - eta * np.dot(-W.T, error - r)
            np.maximum(h, 0.0, out=h)

            # Solve for r
            r_ = r
            r = self.__thresh(error, self._lambda_, self.v_max)

            # Stop conditions
            stoph = np.linalg.norm(h - h_, 2)
            # stopr = np.linalg.norm(r - r_, 2)
            # stop = max(stoph, stopr) / m
            stop = stoph / m
            if stop < 1e-5:
                break

        return h, r
