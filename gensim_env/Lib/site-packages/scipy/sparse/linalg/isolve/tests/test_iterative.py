""" Test functions for the sparse.linalg.isolve module
"""

from __future__ import division, print_function, absolute_import

import numpy as np

from numpy.testing import (assert_equal, assert_array_equal,
     assert_, assert_allclose)
from pytest import raises as assert_raises
from scipy._lib._numpy_compat import suppress_warnings

from numpy import zeros, arange, array, abs, max, ones, eye, iscomplexobj
from scipy.linalg import norm
from scipy.sparse import spdiags, csr_matrix, SparseEfficiencyWarning

from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg.isolve import cg, cgs, bicg, bicgstab, gmres, qmr, minres, lgmres, gcrotmk

# TODO check that method preserve shape and type
# TODO test both preconditioner methods


class Case(object):
    def __init__(self, name, A, skip=None):
        self.name = name
        self.A = A
        if skip is None:
            self.skip = []
        else:
            self.skip = skip

    def __repr__(self):
        return "<%s>" % self.name


class IterativeParams(object):
    def __init__(self):
        # list of tuples (solver, symmetric, positive_definite )
        solvers = [cg, cgs, bicg, bicgstab, gmres, qmr, minres, lgmres, gcrotmk]
        sym_solvers = [minres, cg]
        posdef_solvers = [cg]
        real_solvers = [minres]

        self.solvers = solvers

        # list of tuples (A, symmetric, positive_definite )
        self.cases = []

        # Symmetric and Positive Definite
        N = 40
        data = ones((3,N))
        data[0,:] = 2
        data[1,:] = -1
        data[2,:] = -1
        Poisson1D = spdiags(data, [0,-1,1], N, N, format='csr')
        self.Poisson1D = Case("poisson1d", Poisson1D)
        self.cases.append(Case("poisson1d", Poisson1D))
        # note: minres fails for single precision
        self.cases.append(Case("poisson1d", Poisson1D.astype('f'),
                               skip=[minres]))

        # Symmetric and Negative Definite
        self.cases.append(Case("neg-poisson1d", -Poisson1D,
                               skip=posdef_solvers))
        # note: minres fails for single precision
        self.cases.append(Case("neg-poisson1d", (-Poisson1D).astype('f'),
                               skip=posdef_solvers + [minres]))

        # Symmetric and Indefinite
        data = array([[6, -5, 2, 7, -1, 10, 4, -3, -8, 9]],dtype='d')
        RandDiag = spdiags(data, [0], 10, 10, format='csr')
        self.cases.append(Case("rand-diag", RandDiag, skip=posdef_solvers))
        self.cases.append(Case("rand-diag", RandDiag.astype('f'),
                               skip=posdef_solvers))

        # Random real-valued
        np.random.seed(1234)
        data = np.random.rand(4, 4)
        self.cases.append(Case("rand", data, skip=posdef_solvers+sym_solvers))
        self.cases.append(Case("rand", data.astype('f'),
                               skip=posdef_solvers+sym_solvers))

        # Random symmetric real-valued
        np.random.seed(1234)
        data = np.random.rand(4, 4)
        data = data + data.T
        self.cases.append(Case("rand-sym", data, skip=posdef_solvers))
        self.cases.append(Case("rand-sym", data.astype('f'),
                               skip=posdef_solvers))

        # Random pos-def symmetric real
        np.random.seed(1234)
        data = np.random.rand(9, 9)
        data = np.dot(data.conj(), data.T)
        self.cases.append(Case("rand-sym-pd", data))
        # note: minres fails for single precision
        self.cases.append(Case("rand-sym-pd", data.astype('f'),
                               skip=[minres]))

        # Random complex-valued
        np.random.seed(1234)
        data = np.random.rand(4, 4) + 1j*np.random.rand(4, 4)
        self.cases.append(Case("rand-cmplx", data,
                               skip=posdef_solvers+sym_solvers+real_solvers))
        self.cases.append(Case("rand-cmplx", data.astype('F'),
                               skip=posdef_solvers+sym_solvers+real_solvers))

        # Random hermitian complex-valued
        np.random.seed(1234)
        data = np.random.rand(4, 4) + 1j*np.random.rand(4, 4)
        data = data + data.T.conj()
        self.cases.append(Case("rand-cmplx-herm", data,
                               skip=posdef_solvers+real_solvers))
        self.cases.append(Case("rand-cmplx-herm", data.astype('F'),
                               skip=posdef_solvers+real_solvers))

        # Random pos-def hermitian complex-valued
        np.random.seed(1234)
        data = np.random.rand(9, 9) + 1j*np.random.rand(9, 9)
        data = np.dot(data.conj(), data.T)
        self.cases.append(Case("rand-cmplx-sym-pd", data, skip=real_solvers))
        self.cases.append(Case("rand-cmplx-sym-pd", data.astype('F'),
                               skip=real_solvers))

        # Non-symmetric and Positive Definite
        #
        # cgs, qmr, and bicg fail to converge on this one
        #   -- algorithmic limitation apparently
        data = ones((2,10))
        data[0,:] = 2
        data[1,:] = -1
        A = spdiags(data, [0,-1], 10, 10, format='csr')
        self.cases.append(Case("nonsymposdef", A,
                               skip=sym_solvers+[cgs, qmr, bicg]))
        self.cases.append(Case("nonsymposdef", A.astype('F'),
                               skip=sym_solvers+[cgs, qmr, bicg]))


params = IterativeParams()


def check_maxiter(solver, case):
    A = case.A
    tol = 1e-12

    b = arange(A.shape[0], dtype=float)
    x0 = 0*b

    residuals = []

    def callback(x):
        residuals.append(norm(b - case.A*x))

    x, info = solver(A, b, x0=x0, tol=tol, maxiter=1, callback=callback)

    assert_equal(len(residuals), 1)
    assert_equal(info, 1)


def test_maxiter():
    case = params.Poisson1D
    for solver in params.solvers:
        if solver in case.skip:
            continue
        check_maxiter(solver, case)


def assert_normclose(a, b, tol=1e-8):
    residual = norm(a - b)
    tolerance = tol*norm(b)
    msg = "residual (%g) not smaller than tolerance %g" % (residual, tolerance)
    assert_(residual < tolerance, msg=msg)


def check_convergence(solver, case):
    A = case.A

    if A.dtype.char in "dD":
        tol = 1e-8
    else:
        tol = 1e-2

    b = arange(A.shape[0], dtype=A.dtype)
    x0 = 0*b

    x, info = solver(A, b, x0=x0, tol=tol)

    assert_array_equal(x0, 0*b)  # ensure that x0 is not overwritten
    assert_equal(info,0)
    assert_normclose(A.dot(x), b, tol=tol)


def test_convergence():
    for solver in params.solvers:
        for case in params.cases:
            if solver in case.skip:
                continue
            check_convergence(solver, case)


def check_precond_dummy(solver, case):
    tol = 1e-8

    def identity(b,which=None):
        """trivial preconditioner"""
        return b

    A = case.A

    M,N = A.shape
    D = spdiags([1.0/A.diagonal()], [0], M, N)

    b = arange(A.shape[0], dtype=float)
    x0 = 0*b

    precond = LinearOperator(A.shape, identity, rmatvec=identity)

    if solver is qmr:
        x, info = solver(A, b, M1=precond, M2=precond, x0=x0, tol=tol)
    else:
        x, info = solver(A, b, M=precond, x0=x0, tol=tol)
    assert_equal(info,0)
    assert_normclose(A.dot(x), b, tol)

    A = aslinearoperator(A)
    A.psolve = identity
    A.rpsolve = identity

    x, info = solver(A, b, x0=x0, tol=tol)
    assert_equal(info,0)
    assert_normclose(A*x, b, tol=tol)


def test_precond_dummy():
    case = params.Poisson1D
    for solver in params.solvers:
        if solver in case.skip:
            continue
        check_precond_dummy(solver, case)


def check_precond_inverse(solver, case):
    tol = 1e-8

    def inverse(b,which=None):
        """inverse preconditioner"""
        A = case.A
        if not isinstance(A, np.ndarray):
            A = A.todense()
        return np.linalg.solve(A, b)

    def rinverse(b,which=None):
        """inverse preconditioner"""
        A = case.A
        if not isinstance(A, np.ndarray):
            A = A.todense()
        return np.linalg.solve(A.T, b)

    matvec_count = [0]

    def matvec(b):
        matvec_count[0] += 1
        return case.A.dot(b)

    def rmatvec(b):
        matvec_count[0] += 1
        return case.A.T.dot(b)

    b = arange(case.A.shape[0], dtype=float)
    x0 = 0*b

    A = LinearOperator(case.A.shape, matvec, rmatvec=rmatvec)
    precond = LinearOperator(case.A.shape, inverse, rmatvec=rinverse)

    # Solve with preconditioner
    matvec_count = [0]
    x, info = solver(A, b, M=precond, x0=x0, tol=tol)

    assert_equal(info, 0)
    assert_normclose(case.A.dot(x), b, tol)

    # Solution should be nearly instant
    assert_(matvec_count[0] <= 3, repr(matvec_count))


def test_precond_inverse():
    case = params.Poisson1D
    for solver in params.solvers:
        if solver in case.skip:
            continue
        if solver is qmr:
            continue
        check_precond_inverse(solver, case)


def test_gmres_basic():
    A = np.vander(np.arange(10) + 1)[:, ::-1]
    b = np.zeros(10)
    b[0] = 1
    x = np.linalg.solve(A, b)

    x_gm, err = gmres(A, b, restart=5, maxiter=1)

    assert_allclose(x_gm[0], 0.359, rtol=1e-2)


def test_reentrancy():
    non_reentrant = [cg, cgs, bicg, bicgstab, gmres, qmr]
    reentrant = [lgmres, minres, gcrotmk]
    for solver in reentrant + non_reentrant:
        _check_reentrancy(solver, solver in reentrant)


def _check_reentrancy(solver, is_reentrant):
    def matvec(x):
        A = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]])
        y, info = solver(A, x)
        assert_equal(info, 0)
        return y
    b = np.array([1, 1./2, 1./3])
    op = LinearOperator((3, 3), matvec=matvec, rmatvec=matvec,
                        dtype=b.dtype)

    if not is_reentrant:
        assert_raises(RuntimeError, solver, op, b)
    else:
        y, info = solver(op, b)
        assert_equal(info, 0)
        assert_allclose(y, [1, 1, 1])


#------------------------------------------------------------------------------

class TestQMR(object):
    def test_leftright_precond(self):
        """Check that QMR works with left and right preconditioners"""

        from scipy.sparse.linalg.dsolve import splu
        from scipy.sparse.linalg.interface import LinearOperator

        n = 100

        dat = ones(n)
        A = spdiags([-2*dat, 4*dat, -dat], [-1,0,1],n,n)
        b = arange(n,dtype='d')

        L = spdiags([-dat/2, dat], [-1,0], n, n)
        U = spdiags([4*dat, -dat], [0,1], n, n)

        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, "splu requires CSC matrix format")
            L_solver = splu(L)
            U_solver = splu(U)

        def L_solve(b):
            return L_solver.solve(b)

        def U_solve(b):
            return U_solver.solve(b)

        def LT_solve(b):
            return L_solver.solve(b,'T')

        def UT_solve(b):
            return U_solver.solve(b,'T')

        M1 = LinearOperator((n,n), matvec=L_solve, rmatvec=LT_solve)
        M2 = LinearOperator((n,n), matvec=U_solve, rmatvec=UT_solve)

        x,info = qmr(A, b, tol=1e-8, maxiter=15, M1=M1, M2=M2)

        assert_equal(info,0)
        assert_normclose(A*x, b, tol=1e-8)


class TestGMRES(object):
    def test_callback(self):

        def store_residual(r, rvec):
            rvec[rvec.nonzero()[0].max()+1] = r

        # Define, A,b
        A = csr_matrix(array([[-2,1,0,0,0,0],[1,-2,1,0,0,0],[0,1,-2,1,0,0],[0,0,1,-2,1,0],[0,0,0,1,-2,1],[0,0,0,0,1,-2]]))
        b = ones((A.shape[0],))
        maxiter = 1
        rvec = zeros(maxiter+1)
        rvec[0] = 1.0
        callback = lambda r:store_residual(r, rvec)
        x,flag = gmres(A, b, x0=zeros(A.shape[0]), tol=1e-16, maxiter=maxiter, callback=callback)
        diff = max(abs((rvec - array([1.0, 0.81649658092772603]))))
        assert_(diff < 1e-5)

    def test_abi(self):
        # Check we don't segfault on gmres with complex argument
        A = eye(2)
        b = ones(2)
        r_x, r_info = gmres(A, b)
        r_x = r_x.astype(complex)

        x, info = gmres(A.astype(complex), b.astype(complex))

        assert_(iscomplexobj(x))
        assert_allclose(r_x, x)
        assert_(r_info == info)

