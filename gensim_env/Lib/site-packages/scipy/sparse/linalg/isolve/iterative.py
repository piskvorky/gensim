"""Iterative methods for solving linear systems"""

from __future__ import division, print_function, absolute_import

__all__ = ['bicg','bicgstab','cg','cgs','gmres','qmr']

from . import _iterative

from scipy.sparse.linalg.interface import LinearOperator
from scipy._lib.decorator import decorator
from .utils import make_system
from scipy._lib._util import _aligned_zeros
from scipy._lib._threadsafety import non_reentrant

_type_conv = {'f':'s', 'd':'d', 'F':'c', 'D':'z'}


# Part of the docstring common to all iterative solvers
common_doc1 = \
"""
Parameters
----------
A : {sparse matrix, dense matrix, LinearOperator}"""

common_doc2 = \
"""b : {array, matrix}
    Right hand side of the linear system. Has shape (N,) or (N,1).

Returns
-------
x : {array, matrix}
    The converged solution.
info : integer
    Provides convergence information:
        0  : successful exit
        >0 : convergence to tolerance not achieved, number of iterations
        <0 : illegal input or breakdown

Other Parameters
----------------
x0  : {array, matrix}
    Starting guess for the solution.
tol : float
    Tolerance to achieve. The algorithm terminates when either the relative
    or the absolute residual is below `tol`.
maxiter : integer
    Maximum number of iterations.  Iteration will stop after maxiter
    steps even if the specified tolerance has not been achieved.
M : {sparse matrix, dense matrix, LinearOperator}
    Preconditioner for A.  The preconditioner should approximate the
    inverse of A.  Effective preconditioning dramatically improves the
    rate of convergence, which implies that fewer iterations are needed
    to reach a given error tolerance.
callback : function
    User-supplied function to call after each iteration.  It is called
    as callback(xk), where xk is the current solution vector.

"""


def set_docstring(header, Ainfo, footer=''):
    def combine(fn):
        fn.__doc__ = '\n'.join((header, common_doc1,
                               '    ' + Ainfo.replace('\n', '\n    '),
                               common_doc2, footer))
        return fn
    return combine


@set_docstring('Use BIConjugate Gradient iteration to solve ``Ax = b``.',
               'The real or complex N-by-N matrix of the linear system.\n'
               'It is required that the linear operator can produce\n'
               '``Ax`` and ``A^T x``.')
@non_reentrant()
def bicg(A, b, x0=None, tol=1e-5, maxiter=None, M=None, callback=None):
    A,M,x,b,postprocess = make_system(A, M, x0, b)

    n = len(b)
    if maxiter is None:
        maxiter = n*10

    matvec, rmatvec = A.matvec, A.rmatvec
    psolve, rpsolve = M.matvec, M.rmatvec
    ltr = _type_conv[x.dtype.char]
    revcom = getattr(_iterative, ltr + 'bicgrevcom')
    stoptest = getattr(_iterative, ltr + 'stoptest2')

    resid = tol
    ndx1 = 1
    ndx2 = -1
    # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
    work = _aligned_zeros(6*n,dtype=x.dtype)
    ijob = 1
    info = 0
    ftflag = True
    bnrm2 = -1.0
    iter_ = maxiter
    while True:
        olditer = iter_
        x, iter_, resid, info, ndx1, ndx2, sclr1, sclr2, ijob = \
           revcom(b, x, work, iter_, resid, info, ndx1, ndx2, ijob)
        if callback is not None and iter_ > olditer:
            callback(x)
        slice1 = slice(ndx1-1, ndx1-1+n)
        slice2 = slice(ndx2-1, ndx2-1+n)
        if (ijob == -1):
            if callback is not None:
                callback(x)
            break
        elif (ijob == 1):
            work[slice2] *= sclr2
            work[slice2] += sclr1*matvec(work[slice1])
        elif (ijob == 2):
            work[slice2] *= sclr2
            work[slice2] += sclr1*rmatvec(work[slice1])
        elif (ijob == 3):
            work[slice1] = psolve(work[slice2])
        elif (ijob == 4):
            work[slice1] = rpsolve(work[slice2])
        elif (ijob == 5):
            work[slice2] *= sclr2
            work[slice2] += sclr1*matvec(x)
        elif (ijob == 6):
            if ftflag:
                info = -1
                ftflag = False
            bnrm2, resid, info = stoptest(work[slice1], b, bnrm2, tol, info)
        ijob = 2

    if info > 0 and iter_ == maxiter and resid > tol:
        # info isn't set appropriately otherwise
        info = iter_

    return postprocess(x), info


@set_docstring('Use BIConjugate Gradient STABilized iteration to solve '
               '``Ax = b``.',
               'The real or complex N-by-N matrix of the linear system.')
@non_reentrant()
def bicgstab(A, b, x0=None, tol=1e-5, maxiter=None, M=None, callback=None):
    A, M, x, b, postprocess = make_system(A, M, x0, b)

    n = len(b)
    if maxiter is None:
        maxiter = n*10

    matvec = A.matvec
    psolve = M.matvec
    ltr = _type_conv[x.dtype.char]
    revcom = getattr(_iterative, ltr + 'bicgstabrevcom')
    stoptest = getattr(_iterative, ltr + 'stoptest2')

    resid = tol
    ndx1 = 1
    ndx2 = -1
    # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
    work = _aligned_zeros(7*n,dtype=x.dtype)
    ijob = 1
    info = 0
    ftflag = True
    bnrm2 = -1.0
    iter_ = maxiter
    while True:
        olditer = iter_
        x, iter_, resid, info, ndx1, ndx2, sclr1, sclr2, ijob = \
           revcom(b, x, work, iter_, resid, info, ndx1, ndx2, ijob)
        if callback is not None and iter_ > olditer:
            callback(x)
        slice1 = slice(ndx1-1, ndx1-1+n)
        slice2 = slice(ndx2-1, ndx2-1+n)
        if (ijob == -1):
            if callback is not None:
                callback(x)
            break
        elif (ijob == 1):
            work[slice2] *= sclr2
            work[slice2] += sclr1*matvec(work[slice1])
        elif (ijob == 2):
            work[slice1] = psolve(work[slice2])
        elif (ijob == 3):
            work[slice2] *= sclr2
            work[slice2] += sclr1*matvec(x)
        elif (ijob == 4):
            if ftflag:
                info = -1
                ftflag = False
            bnrm2, resid, info = stoptest(work[slice1], b, bnrm2, tol, info)
        ijob = 2

    if info > 0 and iter_ == maxiter and resid > tol:
        # info isn't set appropriately otherwise
        info = iter_

    return postprocess(x), info


@set_docstring('Use Conjugate Gradient iteration to solve ``Ax = b``.',
               'The real or complex N-by-N matrix of the linear system.\n'
               '``A`` must represent a hermitian, positive definite matrix.')
@non_reentrant()
def cg(A, b, x0=None, tol=1e-5, maxiter=None, M=None, callback=None):
    A, M, x, b, postprocess = make_system(A, M, x0, b)

    n = len(b)
    if maxiter is None:
        maxiter = n*10

    matvec = A.matvec
    psolve = M.matvec
    ltr = _type_conv[x.dtype.char]
    revcom = getattr(_iterative, ltr + 'cgrevcom')
    stoptest = getattr(_iterative, ltr + 'stoptest2')

    resid = tol
    ndx1 = 1
    ndx2 = -1
    # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
    work = _aligned_zeros(4*n,dtype=x.dtype)
    ijob = 1
    info = 0
    ftflag = True
    bnrm2 = -1.0
    iter_ = maxiter
    while True:
        olditer = iter_
        x, iter_, resid, info, ndx1, ndx2, sclr1, sclr2, ijob = \
           revcom(b, x, work, iter_, resid, info, ndx1, ndx2, ijob)
        if callback is not None and iter_ > olditer:
            callback(x)
        slice1 = slice(ndx1-1, ndx1-1+n)
        slice2 = slice(ndx2-1, ndx2-1+n)
        if (ijob == -1):
            if callback is not None:
                callback(x)
            break
        elif (ijob == 1):
            work[slice2] *= sclr2
            work[slice2] += sclr1*matvec(work[slice1])
        elif (ijob == 2):
            work[slice1] = psolve(work[slice2])
        elif (ijob == 3):
            work[slice2] *= sclr2
            work[slice2] += sclr1*matvec(x)
        elif (ijob == 4):
            if ftflag:
                info = -1
                ftflag = False
            bnrm2, resid, info = stoptest(work[slice1], b, bnrm2, tol, info)
        ijob = 2

    if info > 0 and iter_ == maxiter and resid > tol:
        # info isn't set appropriately otherwise
        info = iter_

    return postprocess(x), info


@set_docstring('Use Conjugate Gradient Squared iteration to solve ``Ax = b``.',
               'The real-valued N-by-N matrix of the linear system.')
@non_reentrant()
def cgs(A, b, x0=None, tol=1e-5, maxiter=None, M=None, callback=None):
    A, M, x, b, postprocess = make_system(A, M, x0, b)

    n = len(b)
    if maxiter is None:
        maxiter = n*10

    matvec = A.matvec
    psolve = M.matvec
    ltr = _type_conv[x.dtype.char]
    revcom = getattr(_iterative, ltr + 'cgsrevcom')
    stoptest = getattr(_iterative, ltr + 'stoptest2')

    resid = tol
    ndx1 = 1
    ndx2 = -1
    # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
    work = _aligned_zeros(7*n,dtype=x.dtype)
    ijob = 1
    info = 0
    ftflag = True
    bnrm2 = -1.0
    iter_ = maxiter
    while True:
        olditer = iter_
        x, iter_, resid, info, ndx1, ndx2, sclr1, sclr2, ijob = \
           revcom(b, x, work, iter_, resid, info, ndx1, ndx2, ijob)
        if callback is not None and iter_ > olditer:
            callback(x)
        slice1 = slice(ndx1-1, ndx1-1+n)
        slice2 = slice(ndx2-1, ndx2-1+n)
        if (ijob == -1):
            if callback is not None:
                callback(x)
            break
        elif (ijob == 1):
            work[slice2] *= sclr2
            work[slice2] += sclr1*matvec(work[slice1])
        elif (ijob == 2):
            work[slice1] = psolve(work[slice2])
        elif (ijob == 3):
            work[slice2] *= sclr2
            work[slice2] += sclr1*matvec(x)
        elif (ijob == 4):
            if ftflag:
                info = -1
                ftflag = False
            bnrm2, resid, info = stoptest(work[slice1], b, bnrm2, tol, info)
        ijob = 2

    if info > 0 and iter_ == maxiter and resid > tol:
        # info isn't set appropriately otherwise
        info = iter_

    return postprocess(x), info


@non_reentrant()
def gmres(A, b, x0=None, tol=1e-5, restart=None, maxiter=None, M=None, callback=None, restrt=None):
    """
    Use Generalized Minimal RESidual iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse matrix, dense matrix, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
    b : {array, matrix}
        Right hand side of the linear system. Has shape (N,) or (N,1).

    Returns
    -------
    x : {array, matrix}
        The converged solution.
    info : int
        Provides convergence information:
          * 0  : successful exit
          * >0 : convergence to tolerance not achieved, number of iterations
          * <0 : illegal input or breakdown

    Other parameters
    ----------------
    x0 : {array, matrix}
        Starting guess for the solution (a vector of zeros by default).
    tol : float
        Tolerance to achieve. The algorithm terminates when either the relative
        or the absolute residual is below `tol`.
    restart : int, optional
        Number of iterations between restarts. Larger values increase
        iteration cost, but may be necessary for convergence.
        Default is 20.
    maxiter : int, optional
        Maximum number of iterations (restart cycles).  Iteration will stop
        after maxiter steps even if the specified tolerance has not been
        achieved.
    M : {sparse matrix, dense matrix, LinearOperator}
        Inverse of the preconditioner of A.  M should approximate the
        inverse of A and be easy to solve for (see Notes).  Effective
        preconditioning dramatically improves the rate of convergence,
        which implies that fewer iterations are needed to reach a given
        error tolerance.  By default, no preconditioner is used.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(rk), where rk is the current residual vector.
    restrt : int, optional
        DEPRECATED - use `restart` instead.

    See Also
    --------
    LinearOperator

    Notes
    -----
    A preconditioner, P, is chosen such that P is close to A but easy to solve
    for. The preconditioner parameter required by this routine is
    ``M = P^-1``. The inverse should preferably not be calculated
    explicitly.  Rather, use the following template to produce M::

      # Construct a linear operator that computes P^-1 * x.
      import scipy.sparse.linalg as spla
      M_x = lambda x: spla.spsolve(P, x)
      M = spla.LinearOperator((n, n), M_x)

    Examples
    --------
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import gmres
    >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
    >>> b = np.array([2, 4, -1], dtype=float)
    >>> x, exitCode = gmres(A, b)
    >>> print(exitCode)            # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True
    """

    # Change 'restrt' keyword to 'restart'
    if restrt is None:
        restrt = restart
    elif restart is not None:
        raise ValueError("Cannot specify both restart and restrt keywords. "
                         "Preferably use 'restart' only.")

    A, M, x, b,postprocess = make_system(A, M, x0, b)

    n = len(b)
    if maxiter is None:
        maxiter = n*10

    if restrt is None:
        restrt = 20
    restrt = min(restrt, n)

    matvec = A.matvec
    psolve = M.matvec
    ltr = _type_conv[x.dtype.char]
    revcom = getattr(_iterative, ltr + 'gmresrevcom')
    stoptest = getattr(_iterative, ltr + 'stoptest2')

    resid = tol
    ndx1 = 1
    ndx2 = -1
    # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
    work = _aligned_zeros((6+restrt)*n,dtype=x.dtype)
    work2 = _aligned_zeros((restrt+1)*(2*restrt+2),dtype=x.dtype)
    ijob = 1
    info = 0
    ftflag = True
    bnrm2 = -1.0
    iter_ = maxiter
    old_ijob = ijob
    first_pass = True
    resid_ready = False
    iter_num = 1
    while True:
        olditer = iter_
        x, iter_, resid, info, ndx1, ndx2, sclr1, sclr2, ijob = \
           revcom(b, x, restrt, work, work2, iter_, resid, info, ndx1, ndx2, ijob)
        # if callback is not None and iter_ > olditer:
        #    callback(x)
        slice1 = slice(ndx1-1, ndx1-1+n)
        slice2 = slice(ndx2-1, ndx2-1+n)
        if (ijob == -1):  # gmres success, update last residual
            if resid_ready and callback is not None:
                callback(resid)
                resid_ready = False

            break
        elif (ijob == 1):
            work[slice2] *= sclr2
            work[slice2] += sclr1*matvec(x)
        elif (ijob == 2):
            work[slice1] = psolve(work[slice2])
            if not first_pass and old_ijob == 3:
                resid_ready = True

            first_pass = False
        elif (ijob == 3):
            work[slice2] *= sclr2
            work[slice2] += sclr1*matvec(work[slice1])
            if resid_ready and callback is not None:
                callback(resid)
                resid_ready = False
                iter_num = iter_num+1

        elif (ijob == 4):
            if ftflag:
                info = -1
                ftflag = False
            bnrm2, resid, info = stoptest(work[slice1], b, bnrm2, tol, info)

        old_ijob = ijob
        ijob = 2

        if iter_num > maxiter:
            break

    if info >= 0 and resid > tol:
        # info isn't set appropriately otherwise
        info = maxiter

    return postprocess(x), info


@non_reentrant()
def qmr(A, b, x0=None, tol=1e-5, maxiter=None, M1=None, M2=None, callback=None):
    """Use Quasi-Minimal Residual iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse matrix, dense matrix, LinearOperator}
        The real-valued N-by-N matrix of the linear system.
        It is required that the linear operator can produce
        ``Ax`` and ``A^T x``.
    b : {array, matrix}
        Right hand side of the linear system. Has shape (N,) or (N,1).

    Returns
    -------
    x : {array, matrix}
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : illegal input or breakdown

    Other Parameters
    ----------------
    x0  : {array, matrix}
        Starting guess for the solution.
    tol : float
        Tolerance to achieve. The algorithm terminates when either the relative
        or the absolute residual is below `tol`.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M1 : {sparse matrix, dense matrix, LinearOperator}
        Left preconditioner for A.
    M2 : {sparse matrix, dense matrix, LinearOperator}
        Right preconditioner for A. Used together with the left
        preconditioner M1.  The matrix M1*A*M2 should have better
        conditioned than A alone.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.

    See Also
    --------
    LinearOperator

    Examples
    --------
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import qmr
    >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
    >>> b = np.array([2, 4, -1], dtype=float)
    >>> x, exitCode = qmr(A, b)
    >>> print(exitCode)            # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True
    """
    A_ = A
    A, M, x, b, postprocess = make_system(A, None, x0, b)

    if M1 is None and M2 is None:
        if hasattr(A_,'psolve'):
            def left_psolve(b):
                return A_.psolve(b,'left')

            def right_psolve(b):
                return A_.psolve(b,'right')

            def left_rpsolve(b):
                return A_.rpsolve(b,'left')

            def right_rpsolve(b):
                return A_.rpsolve(b,'right')
            M1 = LinearOperator(A.shape, matvec=left_psolve, rmatvec=left_rpsolve)
            M2 = LinearOperator(A.shape, matvec=right_psolve, rmatvec=right_rpsolve)
        else:
            def id(b):
                return b
            M1 = LinearOperator(A.shape, matvec=id, rmatvec=id)
            M2 = LinearOperator(A.shape, matvec=id, rmatvec=id)

    n = len(b)
    if maxiter is None:
        maxiter = n*10

    ltr = _type_conv[x.dtype.char]
    revcom = getattr(_iterative, ltr + 'qmrrevcom')
    stoptest = getattr(_iterative, ltr + 'stoptest2')

    resid = tol
    ndx1 = 1
    ndx2 = -1
    # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
    work = _aligned_zeros(11*n,x.dtype)
    ijob = 1
    info = 0
    ftflag = True
    bnrm2 = -1.0
    iter_ = maxiter
    while True:
        olditer = iter_
        x, iter_, resid, info, ndx1, ndx2, sclr1, sclr2, ijob = \
           revcom(b, x, work, iter_, resid, info, ndx1, ndx2, ijob)
        if callback is not None and iter_ > olditer:
            callback(x)
        slice1 = slice(ndx1-1, ndx1-1+n)
        slice2 = slice(ndx2-1, ndx2-1+n)
        if (ijob == -1):
            if callback is not None:
                callback(x)
            break
        elif (ijob == 1):
            work[slice2] *= sclr2
            work[slice2] += sclr1*A.matvec(work[slice1])
        elif (ijob == 2):
            work[slice2] *= sclr2
            work[slice2] += sclr1*A.rmatvec(work[slice1])
        elif (ijob == 3):
            work[slice1] = M1.matvec(work[slice2])
        elif (ijob == 4):
            work[slice1] = M2.matvec(work[slice2])
        elif (ijob == 5):
            work[slice1] = M1.rmatvec(work[slice2])
        elif (ijob == 6):
            work[slice1] = M2.rmatvec(work[slice2])
        elif (ijob == 7):
            work[slice2] *= sclr2
            work[slice2] += sclr1*A.matvec(x)
        elif (ijob == 8):
            if ftflag:
                info = -1
                ftflag = False
            bnrm2, resid, info = stoptest(work[slice1], b, bnrm2, tol, info)
        ijob = 2

    if info > 0 and iter_ == maxiter and resid > tol:
        # info isn't set appropriately otherwise
        info = iter_

    return postprocess(x), info
