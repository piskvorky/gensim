from itertools import chain

import numpy as np
from scipy.stats import halfnorm


class NMF(object):
    """Online Non-Negative Matrix Factorization.

    Attributes
    ----------
    _W : dictionary matrix
    _H : loadings matrix
    _lambda: weight of outliers regularizer
    _kappa: step size coefficient

    """

    def __init__(self, corpus=None, n_components=100, lambda_=1.,
                 kappa=1., store_r=False, v_max=None):
        """

        Parameters
        ----------
        n_components : int
            Number of components in resulting matrices.
        lambda_ : float
        kappa : float
        """
        self.n_features = None
        self.n_components = n_components
        self._lambda_ = lambda_
        self._kappa = kappa
        self._H = []
        self.v_max = v_max
        if store_r:
            self._R = []
        else:
            self._R = None

    @property
    def A(self):
        return self._A / len(self._H)
        # return self._A

    @A.setter
    def A(self, value):
        self._A = value

    @property
    def B(self):
        return self._B / len(self._H)
        # return self._B

    @B.setter
    def B(self, value):
        self._B = value

    def _setup(self, X):
        self.h, self.r = None, None
        X_ = iter(X)
        x = next(X_)
        m = len(x)
        avg = np.sqrt(x.mean() / m)
        X = chain([x], X_)

        self.n_features = len(x)

        self._W = np.abs(avg * halfnorm.rvs(size=(self.n_features, self.n_components)) /
                        np.sqrt(self.n_components))

        self.A = np.zeros((self.n_components, self.n_components))
        self.B = np.zeros((self.n_features, self.n_components))
        return X

    def fit(self, X, batch_size=None):
        """

        Parameters
        ----------
        X : matrix or iterator
            Matrix to factorize.
        batch_size : int or None
            If None than batch_size equals 1 sample.
        """
        if self.n_features is None:
            X = self._setup(X)

        prod = np.outer

        if batch_size is not None:
            from itertools import zip_longest

            def grouper(iterable, n):
                args = [iter(iterable)] * n
                return zip_longest(*args)

            prod = np.dot
            X = (np.array([e for e in batch if e is not None]) for batch in grouper(X, batch_size))

        r, h = self.r, self.h
        for v in X:
            h, r = self._solveproj(v, self._W, r=r, h=h)
            self._H.append(h)
            if self._R is not None:
                self._R.append(r)

            self.A += prod(h, h.T)
            self.B += prod((v.T - r), h.T)
            self._solve_w()
            print(
                'Loss (no outliers): {}\tLoss (with outliers): {}'
                .format(
                    np.linalg.norm(v.T - self._W.dot(h)),
                    np.linalg.norm(v.T - self._W.dot(h) - r)
                )
            )
        self.r = r
        self.h = h

    def _solve_w(self):
        eta = self._kappa / np.linalg.norm(self.A, 'fro')
        n = 0
        lasttwo = np.zeros(2)
        while n <= 2 or (np.abs(
                (lasttwo[1] - lasttwo[0]) / lasttwo[0]) > 1e-5 and n < 1e9):
            self._W -= eta * (np.dot(self._W, self.A) - self.B)
            self._W = self._transform(self._W)
            n += 1
            lasttwo[0] = lasttwo[1]
            lasttwo[1] = 0.5 * np.trace(self._W.T.dot(self._W).dot(self.A)) - \
                         np.trace(self._W.T.dot(self.B))

    def transform(self, X, return_r=False, max_iter=1e9):
        H = []
        if return_r:
            R = []

        num = None
        W = self._W
        for v in X:
            h, r = self._solveproj(v, W, v_max=np.inf, max_iter=max_iter)
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
    def _thresh(X, lambda1, vmax):
        res = np.abs(X) - lambda1
        np.maximum(res, 0.0, out=res)
        res *= np.sign(X)
        np.clip(res, -vmax, vmax, out=res)
        return res

        return X

    def _transform(self, W):
        W_ = W.copy()
        np.clip(W_, 0, self.v_max, out=W_)
        sumsq = np.linalg.norm(W_, axis=0)
        np.maximum(sumsq, 1, out=sumsq)
        return W_ / sumsq

    def _solveproj(self, v, W, h=None, r=None, max_iter=1e9, v_max=None):
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

        for _ in range(int(max_iter)):
            # Solve for h
            h_ = h
            h = h - eta * np.dot(-W.T, v - np.dot(W, h) - r)
            np.maximum(h, 0.0, out=h)

            # Solve for r
            r_ = r
            r = self._thresh(v - np.dot(W, h), self._lambda_, self.v_max)

            # Stop conditions
            stoph = np.linalg.norm(h - h_, 2)
            stopr = np.linalg.norm(r - r_, 2)
            stop = max(stoph, stopr) / m
            if stop < 1e-5:
                break

        return h, r
