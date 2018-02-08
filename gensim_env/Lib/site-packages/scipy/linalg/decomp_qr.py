"""QR decomposition functions."""
from __future__ import division, print_function, absolute_import

import numpy

# Local imports
from .lapack import get_lapack_funcs
from .misc import _datacopied

__all__ = ['qr', 'qr_multiply', 'rq']


def safecall(f, name, *args, **kwargs):
    """Call a LAPACK routine, determining lwork automatically and handling
    error return values"""
    lwork = kwargs.get("lwork", None)
    if lwork in (None, -1):
        kwargs['lwork'] = -1
        ret = f(*args, **kwargs)
        kwargs['lwork'] = ret[-2][0].real.astype(numpy.int)
    ret = f(*args, **kwargs)
    if ret[-1] < 0:
        raise ValueError("illegal value in %d-th argument of internal %s"
                         % (-ret[-1], name))
    return ret[:-2]


def qr(a, overwrite_a=False, lwork=None, mode='full', pivoting=False,
       check_finite=True):
    """
    Compute QR decomposition of a matrix.

    Calculate the decomposition ``A = Q R`` where Q is unitary/orthogonal
    and R upper triangular.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to be decomposed
    overwrite_a : bool, optional
        Whether data in a is overwritten (may improve performance)
    lwork : int, optional
        Work array size, lwork >= a.shape[1]. If None or -1, an optimal size
        is computed.
    mode : {'full', 'r', 'economic', 'raw'}, optional
        Determines what information is to be returned: either both Q and R
        ('full', default), only R ('r') or both Q and R but computed in
        economy-size ('economic', see Notes). The final option 'raw'
        (added in Scipy 0.11) makes the function return two matrices
        (Q, TAU) in the internal format used by LAPACK.
    pivoting : bool, optional
        Whether or not factorization should include pivoting for rank-revealing
        qr decomposition. If pivoting, compute the decomposition
        ``A P = Q R`` as above, but where P is chosen such that the diagonal
        of R is non-increasing.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    Q : float or complex ndarray
        Of shape (M, M), or (M, K) for ``mode='economic'``.  Not returned
        if ``mode='r'``.
    R : float or complex ndarray
        Of shape (M, N), or (K, N) for ``mode='economic'``.  ``K = min(M, N)``.
    P : int ndarray
        Of shape (N,) for ``pivoting=True``. Not returned if
        ``pivoting=False``.

    Raises
    ------
    LinAlgError
        Raised if decomposition fails

    Notes
    -----
    This is an interface to the LAPACK routines dgeqrf, zgeqrf,
    dorgqr, zungqr, dgeqp3, and zgeqp3.

    If ``mode=economic``, the shapes of Q and R are (M, K) and (K, N) instead
    of (M,M) and (M,N), with ``K=min(M,N)``.

    Examples
    --------
    >>> from scipy import random, linalg, dot, diag, all, allclose
    >>> a = random.randn(9, 6)

    >>> q, r = linalg.qr(a)
    >>> allclose(a, np.dot(q, r))
    True
    >>> q.shape, r.shape
    ((9, 9), (9, 6))

    >>> r2 = linalg.qr(a, mode='r')
    >>> allclose(r, r2)
    True

    >>> q3, r3 = linalg.qr(a, mode='economic')
    >>> q3.shape, r3.shape
    ((9, 6), (6, 6))

    >>> q4, r4, p4 = linalg.qr(a, pivoting=True)
    >>> d = abs(diag(r4))
    >>> all(d[1:] <= d[:-1])
    True
    >>> allclose(a[:, p4], dot(q4, r4))
    True
    >>> q4.shape, r4.shape, p4.shape
    ((9, 9), (9, 6), (6,))

    >>> q5, r5, p5 = linalg.qr(a, mode='economic', pivoting=True)
    >>> q5.shape, r5.shape, p5.shape
    ((9, 6), (6, 6), (6,))

    """
    # 'qr' was the old default, equivalent to 'full'. Neither 'full' nor
    # 'qr' are used below.
    # 'raw' is used internally by qr_multiply
    if mode not in ['full', 'qr', 'r', 'economic', 'raw']:
        raise ValueError(
                 "Mode argument should be one of ['full', 'r', 'economic', 'raw']")

    if check_finite:
        a1 = numpy.asarray_chkfinite(a)
    else:
        a1 = numpy.asarray(a)
    if len(a1.shape) != 2:
        raise ValueError("expected 2D array")
    M, N = a1.shape
    overwrite_a = overwrite_a or (_datacopied(a1, a))

    if pivoting:
        geqp3, = get_lapack_funcs(('geqp3',), (a1,))
        qr, jpvt, tau = safecall(geqp3, "geqp3", a1, overwrite_a=overwrite_a)
        jpvt -= 1  # geqp3 returns a 1-based index array, so subtract 1
    else:
        geqrf, = get_lapack_funcs(('geqrf',), (a1,))
        qr, tau = safecall(geqrf, "geqrf", a1, lwork=lwork,
            overwrite_a=overwrite_a)

    if mode not in ['economic', 'raw'] or M < N:
        R = numpy.triu(qr)
    else:
        R = numpy.triu(qr[:N, :])

    if pivoting:
        Rj = R, jpvt
    else:
        Rj = R,

    if mode == 'r':
        return Rj
    elif mode == 'raw':
        return ((qr, tau),) + Rj

    gor_un_gqr, = get_lapack_funcs(('orgqr',), (qr,))

    if M < N:
        Q, = safecall(gor_un_gqr, "gorgqr/gungqr", qr[:, :M], tau,
            lwork=lwork, overwrite_a=1)
    elif mode == 'economic':
        Q, = safecall(gor_un_gqr, "gorgqr/gungqr", qr, tau, lwork=lwork,
            overwrite_a=1)
    else:
        t = qr.dtype.char
        qqr = numpy.empty((M, M), dtype=t)
        qqr[:, :N] = qr
        Q, = safecall(gor_un_gqr, "gorgqr/gungqr", qqr, tau, lwork=lwork,
            overwrite_a=1)

    return (Q,) + Rj


def qr_multiply(a, c, mode='right', pivoting=False, conjugate=False,
    overwrite_a=False, overwrite_c=False):
    """
    Calculate the QR decomposition and multiply Q with a matrix.

    Calculate the decomposition ``A = Q R`` where Q is unitary/orthogonal
    and R upper triangular. Multiply Q with a vector or a matrix c.

    Parameters
    ----------
    a : array_like, shape (M, N)
        Matrix to be decomposed
    c : array_like, one- or two-dimensional
        calculate the product of c and q, depending on the mode:
    mode : {'left', 'right'}, optional
        ``dot(Q, c)`` is returned if mode is 'left',
        ``dot(c, Q)`` is returned if mode is 'right'.
        The shape of c must be appropriate for the matrix multiplications,
        if mode is 'left', ``min(a.shape) == c.shape[0]``,
        if mode is 'right', ``a.shape[0] == c.shape[1]``.
    pivoting : bool, optional
        Whether or not factorization should include pivoting for rank-revealing
        qr decomposition, see the documentation of qr.
    conjugate : bool, optional
        Whether Q should be complex-conjugated. This might be faster
        than explicit conjugation.
    overwrite_a : bool, optional
        Whether data in a is overwritten (may improve performance)
    overwrite_c : bool, optional
        Whether data in c is overwritten (may improve performance).
        If this is used, c must be big enough to keep the result,
        i.e. c.shape[0] = a.shape[0] if mode is 'left'.


    Returns
    -------
    CQ : float or complex ndarray
        the product of Q and c, as defined in mode
    R : float or complex ndarray
        Of shape (K, N), ``K = min(M, N)``.
    P : ndarray of ints
        Of shape (N,) for ``pivoting=True``.
        Not returned if ``pivoting=False``.

    Raises
    ------
    LinAlgError
        Raised if decomposition fails

    Notes
    -----
    This is an interface to the LAPACK routines dgeqrf, zgeqrf,
    dormqr, zunmqr, dgeqp3, and zgeqp3.

    .. versionadded:: 0.11.0

    """
    if mode not in ['left', 'right']:
        raise ValueError("Mode argument should be one of ['left', 'right']")
    c = numpy.asarray_chkfinite(c)
    onedim = c.ndim == 1
    if onedim:
        c = c.reshape(1, len(c))
        if mode == "left":
            c = c.T

    a = numpy.asarray(a)  # chkfinite done in qr
    M, N = a.shape
    if not (mode == "left" and
                (not overwrite_c and min(M, N) == c.shape[0] or
                     overwrite_c and M == c.shape[0]) or
            mode == "right" and M == c.shape[1]):
        raise ValueError("objects are not aligned")

    raw = qr(a, overwrite_a, None, "raw", pivoting)
    Q, tau = raw[0]

    gor_un_mqr, = get_lapack_funcs(('ormqr',), (Q,))
    if gor_un_mqr.typecode in ('s', 'd'):
        trans = "T"
    else:
        trans = "C"

    Q = Q[:, :min(M, N)]
    if M > N and mode == "left" and not overwrite_c:
        if conjugate:
            cc = numpy.zeros((c.shape[1], M), dtype=c.dtype, order="F")
            cc[:, :N] = c.T
        else:
            cc = numpy.zeros((M, c.shape[1]), dtype=c.dtype, order="F")
            cc[:N, :] = c
            trans = "N"
        if conjugate:
            lr = "R"
        else:
            lr = "L"
        overwrite_c = True
    elif c.flags["C_CONTIGUOUS"] and trans == "T" or conjugate:
        cc = c.T
        if mode == "left":
            lr = "R"
        else:
            lr = "L"
    else:
        trans = "N"
        cc = c
        if mode == "left":
            lr = "L"
        else:
            lr = "R"
    cQ, = safecall(gor_un_mqr, "gormqr/gunmqr", lr, trans, Q, tau, cc,
            overwrite_c=overwrite_c)
    if trans != "N":
        cQ = cQ.T
    if mode == "right":
        cQ = cQ[:, :min(M, N)]
    if onedim:
        cQ = cQ.ravel()

    return (cQ,) + raw[1:]


def rq(a, overwrite_a=False, lwork=None, mode='full', check_finite=True):
    """
    Compute RQ decomposition of a matrix.

    Calculate the decomposition ``A = R Q`` where Q is unitary/orthogonal
    and R upper triangular.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to be decomposed
    overwrite_a : bool, optional
        Whether data in a is overwritten (may improve performance)
    lwork : int, optional
        Work array size, lwork >= a.shape[1]. If None or -1, an optimal size
        is computed.
    mode : {'full', 'r', 'economic'}, optional
        Determines what information is to be returned: either both Q and R
        ('full', default), only R ('r') or both Q and R but computed in
        economy-size ('economic', see Notes).
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    R : float or complex ndarray
        Of shape (M, N) or (M, K) for ``mode='economic'``.  ``K = min(M, N)``.
    Q : float or complex ndarray
        Of shape (N, N) or (K, N) for ``mode='economic'``.  Not returned
        if ``mode='r'``.

    Raises
    ------
    LinAlgError
        If decomposition fails.

    Notes
    -----
    This is an interface to the LAPACK routines sgerqf, dgerqf, cgerqf, zgerqf,
    sorgrq, dorgrq, cungrq and zungrq.

    If ``mode=economic``, the shapes of Q and R are (K, N) and (M, K) instead
    of (N,N) and (M,N), with ``K=min(M,N)``.

    Examples
    --------
    >>> from scipy import linalg
    >>> from numpy import random, dot, allclose
    >>> a = random.randn(6, 9)
    >>> r, q = linalg.rq(a)
    >>> allclose(a, dot(r, q))
    True
    >>> r.shape, q.shape
    ((6, 9), (9, 9))
    >>> r2 = linalg.rq(a, mode='r')
    >>> allclose(r, r2)
    True
    >>> r3, q3 = linalg.rq(a, mode='economic')
    >>> r3.shape, q3.shape
    ((6, 6), (6, 9))

    """
    if mode not in ['full', 'r', 'economic']:
        raise ValueError(
                 "Mode argument should be one of ['full', 'r', 'economic']")

    if check_finite:
        a1 = numpy.asarray_chkfinite(a)
    else:
        a1 = numpy.asarray(a)
    if len(a1.shape) != 2:
        raise ValueError('expected matrix')
    M, N = a1.shape
    overwrite_a = overwrite_a or (_datacopied(a1, a))

    gerqf, = get_lapack_funcs(('gerqf',), (a1,))
    rq, tau = safecall(gerqf, 'gerqf', a1, lwork=lwork,
                       overwrite_a=overwrite_a)
    if not mode == 'economic' or N < M:
        R = numpy.triu(rq, N-M)
    else:
        R = numpy.triu(rq[-M:, -M:])

    if mode == 'r':
        return R

    gor_un_grq, = get_lapack_funcs(('orgrq',), (rq,))

    if N < M:
        Q, = safecall(gor_un_grq, "gorgrq/gungrq", rq[-N:], tau, lwork=lwork,
                      overwrite_a=1)
    elif mode == 'economic':
        Q, = safecall(gor_un_grq, "gorgrq/gungrq", rq, tau, lwork=lwork,
                      overwrite_a=1)
    else:
        rq1 = numpy.empty((N, N), dtype=rq.dtype)
        rq1[-M:] = rq
        Q, = safecall(gor_un_grq, "gorgrq/gungrq", rq1, tau, lwork=lwork,
                      overwrite_a=1)

    return R, Q
