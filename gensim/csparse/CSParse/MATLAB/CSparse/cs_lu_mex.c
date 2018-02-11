#include "cs_mex.h"
/* cs_lu: sparse LU factorization, with optional fill-reducing ordering */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    css *S ;
    csn *N ;
    cs Amatrix, *A, *D ;
    csi n, order, *p ;
    double tol ;
    if (nargout > 4 || nargin > 3 || nargin < 1)
    {
        mexErrMsgTxt ("Usage: [L,U,p,q] = cs_lu (A,tol)") ;
    }
    A = cs_mex_get_sparse (&Amatrix, 1, 1, pargin [0]) ;        /* get A */
    n = A->n ;
    if (nargin == 2)                        /* determine tol and ordering */
    {
        tol = mxGetScalar (pargin [1]) ;
        order = (nargout == 4) ? 1 : 0 ;    /* amd (A+A'), or natural */
    }
    else
    {
        tol = 1 ;
        order = (nargout == 4) ? 2 : 0 ;    /* amd(S'*S) w/dense rows or I */
    }
    S = cs_sqr (order, A, 0) ;              /* symbolic ordering, no QR bound */
    N = cs_lu (A, S, tol) ;                 /* numeric factorization */
    if (!N) mexErrMsgTxt ("cs_lu failed (singular, or out of memory)") ;
    cs_dropzeros (N->L) ;                   /* drop zeros from L and sort it */
    D = cs_transpose (N->L, 1) ;
    cs_spfree (N->L) ;
    N->L = cs_transpose (D, 1) ;
    cs_spfree (D) ;
    cs_dropzeros (N->U) ;                   /* drop zeros from U and sort it */
    D = cs_transpose (N->U, 1) ;
    cs_spfree (N->U) ;
    N->U = cs_transpose (D, 1) ;
    cs_spfree (D) ;
    p = cs_pinv (N->pinv, n) ;              /* p=pinv' */
    pargout [0] = cs_mex_put_sparse (&(N->L)) ;     /* return L */
    pargout [1] = cs_mex_put_sparse (&(N->U)) ;     /* return U */
    pargout [2] = cs_mex_put_int (p, n, 1, 1) ;     /* return p */
    /* return Q */
    if (nargout == 4) pargout [3] = cs_mex_put_int (S->q, n, 1, 0) ;
    cs_nfree (N) ;
    cs_sfree (S) ;
}
