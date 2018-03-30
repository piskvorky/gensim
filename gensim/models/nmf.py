from itertools import chain

import numpy as np
from scipy.stats import halfnorm


def _thresh(X, lambda_, v_max):
    res = np.abs(X) - lambda_
    np.maximum(res, 0.0, out=res)
    res *= np.sign(X)
    np.clip(res, -v_max, v_max, out=res)
    return res


def _mrdivide(B, A):
    """Solve xB = A
    """
    if len(B.shape) == 2 and B.shape[0] == B.shape[1]:
        return np.linalg.solve(B.T, A.T).T
    else:
        return np.linalg.lstsq(A.T, B.T, rcond=None)[0].T


def _transform(W):
    newW = W.copy()
    np.maximum(newW, 0, out=newW)
    sumsq = np.sqrt(np.sum(W**2, axis=0))
    np.maximum(sumsq, 1, out=sumsq)
    return _mrdivide(newW, np.diag(sumsq))


def _solveproj(v, W, lambda_, kappa=1, h=None, r=None, v_max=None, max_iter=1e9):
    m, n = W.shape
    v = v.T
    if v_max is None:
        v_max = v.max()
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

    eta = kappa / np.linalg.norm(W, 'fro')**2

    iters = 0

    while True:
        iters += 1
        # Solve for h
        htmp = h
        h = h - eta * np.dot(W.T, np.dot(W, h) + r - v)
        np.maximum(h, 0.0, out=h)

        # Solve for r
        rtmp = r
        r = _thresh(v - np.dot(W, h), lambda_, v_max)

        # Stop conditions
        stoph = np.linalg.norm(h - htmp, 2)
        stopr = np.linalg.norm(r - rtmp, 2)
        stop = max(stoph, stopr) / m
        if stop < 1e-5 or iters > max_iter:
            break

    return h, r


class NMF:
    """Online Non-Negative Matrix Factorization.

    Attributes
    ----------
    _W : matrix

    """
    def __init__(self, n_components, lambda_=1., kappa=1.):
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
        self.lambda_ = lambda_
        self.kappa = kappa
        self._H = []
        self.R = None
        self.is_fitted = False

    def _setup(self, X):
        self.h, self.r = None, None
        if isinstance(X, np.ndarray):
            n_samples, n_features = X.shape
            avg = np.sqrt(X.mean() / n_features)
        else:
            x = next(X)
            n_features = len(x)
            avg = np.sqrt(x.mean() / n_features)
            X = chain([x], X)

        self.n_features = n_features

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
            if isinstance(X, np.ndarray):
                raise ValueError
            else:
                prod = np.dot
                length = X.shape[0]
                n_batches = max(length // batch_size, 1)
                X = np.array_split(X, n_batches, axis=0)
        if isinstance(X, np.ndarray):
            X = iter(X)
        r, h = self.r, self.h
        for v in X:
            h, r = _solveproj(v, self._W, self.lambda_, self.kappa, r=r, h=h)
            self._H.append(h)
            if self.R is not None:
                self.R.append(r)

            self.A += prod(h, h.T)
            self.B += prod((v.T - r), h.T)
            self._solve_W()
        self.r = r
        self.h = h

        self.is_fitted = True

    def _solve_W(self):
        eta = self.kappa / np.linalg.norm(self.A, 'fro')
        n = 0
        lasttwo = np.zeros(2)
        while n <= 2 or (np.abs(
                (lasttwo[1] - lasttwo[0]) / lasttwo[0]) > 1e-5 and n < 1e9):
            self._W -= eta * (np.dot(self._W, self.A) - self.B)
            self._W = _transform(self._W)
            n += 1
            lasttwo[0] = lasttwo[1]
            lasttwo[1] = 0.5 * np.trace(self._W.T.dot(self._W).dot(self.A)) - \
                         np.trace(self._W.T.dot(self.B))

    def transform(self, X, return_R=False):
        H = []
        if return_R:
            R = []

        num = None
        W = self._W
        lambda_ = self.lambda_
        kappa = self.kappa
        if isinstance(X, np.ndarray):
            num = X.shape[0]
            X = iter(X)
        for v in X:
            h, r = _solveproj(v, W, lambda_, kappa, v_max=np.inf)
            H.append(h.copy())
            if return_R:
                R.append(r.copy())

        H = np.stack(H, axis=-1)
        if return_R:
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
