"""Schur decomposition functions."""
from __future__ import division, print_function, absolute_import

import numpy
from numpy import asarray_chkfinite, single, asarray

from scipy._lib.six import callable

# Local imports.
from . import misc
from .misc import LinAlgError, _datacopied
from .lapack import get_lapack_funcs
from .decomp import eigvals

__all__ = ['schur', 'rsf2csf']

_double_precision = ['i','l','d']


def schur(a, output='real', lwork=None, overwrite_a=False, sort=None,
          check_finite=True):
    """
    Compute Schur decomposition of a matrix.

    The Schur decomposition is::

        A = Z T Z^H

    where Z is unitary and T is either upper-triangular, or for real
    Schur decomposition (output='real'), quasi-upper triangular.  In
    the quasi-triangular form, 2x2 blocks describing complex-valued
    eigenvalue pairs may extrude from the diagonal.

    Parameters
    ----------
    a : (M, M) array_like
        Matrix to decompose
    output : {'real', 'complex'}, optional
        Construct the real or complex Schur decomposition (for real matrices).
    lwork : int, optional
        Work array size. If None or -1, it is automatically computed.
    overwrite_a : bool, optional
        Whether to overwrite data in a (may improve performance).
    sort : {None, callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
        Specifies whether the upper eigenvalues should be sorted.  A callable
        may be passed that, given a eigenvalue, returns a boolean denoting
        whether the eigenvalue should be sorted to the top-left (True).
        Alternatively, string parameters may be used::

            'lhp'   Left-hand plane (x.real < 0.0)
            'rhp'   Right-hand plane (x.real > 0.0)
            'iuc'   Inside the unit circle (x*x.conjugate() <= 1.0)
            'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)

        Defaults to None (no sorting).
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    T : (M, M) ndarray
        Schur form of A. It is real-valued for the real Schur decomposition.
    Z : (M, M) ndarray
        An unitary Schur transformation matrix for A.
        It is real-valued for the real Schur decomposition.
    sdim : int
        If and only if sorting was requested, a third return value will
        contain the number of eigenvalues satisfying the sort condition.

    Raises
    ------
    LinAlgError
        Error raised under three conditions:

        1. The algorithm failed due to a failure of the QR algorithm to
           compute all eigenvalues
        2. If eigenvalue sorting was requested, the eigenvalues could not be
           reordered due to a failure to separate eigenvalues, usually because
           of poor conditioning
        3. If eigenvalue sorting was requested, roundoff errors caused the
           leading eigenvalues to no longer satisfy the sorting condition

    See also
    --------
    rsf2csf : Convert real Schur form to complex Schur form

    """
    if output not in ['real','complex','r','c']:
        raise ValueError("argument must be 'real', or 'complex'")
    if check_finite:
        a1 = asarray_chkfinite(a)
    else:
        a1 = asarray(a)
    if len(a1.shape) != 2 or (a1.shape[0] != a1.shape[1]):
        raise ValueError('expected square matrix')
    typ = a1.dtype.char
    if output in ['complex','c'] and typ not in ['F','D']:
        if typ in _double_precision:
            a1 = a1.astype('D')
            typ = 'D'
        else:
            a1 = a1.astype('F')
            typ = 'F'
    overwrite_a = overwrite_a or (_datacopied(a1, a))
    gees, = get_lapack_funcs(('gees',), (a1,))
    if lwork is None or lwork == -1:
        # get optimal work array
        result = gees(lambda x: None, a1, lwork=-1)
        lwork = result[-2][0].real.astype(numpy.int)

    if sort is None:
        sort_t = 0
        sfunction = lambda x: None
    else:
        sort_t = 1
        if callable(sort):
            sfunction = sort
        elif sort == 'lhp':
            sfunction = lambda x: (numpy.real(x) < 0.0)
        elif sort == 'rhp':
            sfunction = lambda x: (numpy.real(x) >= 0.0)
        elif sort == 'iuc':
            sfunction = lambda x: (abs(x) <= 1.0)
        elif sort == 'ouc':
            sfunction = lambda x: (abs(x) > 1.0)
        else:
            raise ValueError("sort parameter must be None, a callable, or " +
                "one of ('lhp','rhp','iuc','ouc')")

    result = gees(sfunction, a1, lwork=lwork, overwrite_a=overwrite_a,
        sort_t=sort_t)

    info = result[-1]
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal gees'
                                                                    % -info)
    elif info == a1.shape[0] + 1:
        raise LinAlgError('Eigenvalues could not be separated for reordering.')
    elif info == a1.shape[0] + 2:
        raise LinAlgError('Leading eigenvalues do not satisfy sort condition.')
    elif info > 0:
        raise LinAlgError("Schur form not found.  Possibly ill-conditioned.")

    if sort_t == 0:
        return result[0], result[-3]
    else:
        return result[0], result[-3], result[1]


eps = numpy.finfo(float).eps
feps = numpy.finfo(single).eps

_array_kind = {'b':0, 'h':0, 'B': 0, 'i':0, 'l': 0, 'f': 0, 'd': 0, 'F': 1, 'D': 1}
_array_precision = {'i': 1, 'l': 1, 'f': 0, 'd': 1, 'F': 0, 'D': 1}
_array_type = [['f', 'd'], ['F', 'D']]


def _commonType(*arrays):
    kind = 0
    precision = 0
    for a in arrays:
        t = a.dtype.char
        kind = max(kind, _array_kind[t])
        precision = max(precision, _array_precision[t])
    return _array_type[kind][precision]


def _castCopy(type, *arrays):
    cast_arrays = ()
    for a in arrays:
        if a.dtype.char == type:
            cast_arrays = cast_arrays + (a.copy(),)
        else:
            cast_arrays = cast_arrays + (a.astype(type),)
    if len(cast_arrays) == 1:
        return cast_arrays[0]
    else:
        return cast_arrays


def rsf2csf(T, Z, check_finite=True):
    """
    Convert real Schur form to complex Schur form.

    Convert a quasi-diagonal real-valued Schur form to the upper triangular
    complex-valued Schur form.

    Parameters
    ----------
    T : (M, M) array_like
        Real Schur form of the original matrix
    Z : (M, M) array_like
        Schur transformation matrix
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    T : (M, M) ndarray
        Complex Schur form of the original matrix
    Z : (M, M) ndarray
        Schur transformation matrix corresponding to the complex form

    See also
    --------
    schur : Schur decompose a matrix

    """
    if check_finite:
        Z, T = map(asarray_chkfinite, (Z, T))
    else:
        Z,T = map(asarray, (Z,T))
    if len(Z.shape) != 2 or Z.shape[0] != Z.shape[1]:
        raise ValueError("matrix must be square.")
    if len(T.shape) != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("matrix must be square.")
    if T.shape[0] != Z.shape[0]:
        raise ValueError("matrices must be same dimension.")
    N = T.shape[0]
    arr = numpy.array
    t = _commonType(Z, T, arr([3.0],'F'))
    Z, T = _castCopy(t, Z, T)
    conj = numpy.conj
    dot = numpy.dot
    r_ = numpy.r_
    transp = numpy.transpose
    for m in range(N-1, 0, -1):
        if abs(T[m,m-1]) > eps*(abs(T[m-1,m-1]) + abs(T[m,m])):
            k = slice(m-1, m+1)
            mu = eigvals(T[k,k]) - T[m,m]
            r = misc.norm([mu[0], T[m,m-1]])
            c = mu[0] / r
            s = T[m,m-1] / r
            G = r_[arr([[conj(c), s]], dtype=t), arr([[-s, c]], dtype=t)]
            Gc = conj(transp(G))
            j = slice(m-1, N)
            T[k,j] = dot(G, T[k,j])
            i = slice(0, m+1)
            T[i,k] = dot(T[i,k], Gc)
            i = slice(0, N)
            Z[i,k] = dot(Z[i,k], Gc)
        T[m,m-1] = 0.0
    return T, Z
