#include "cs_mex.h"
/* cs_chol: sparse Cholesky factorization */
void mexFunction (int nargout, mxArray *pargout [ ], int nargin,
    const mxArray *pargin [ ])
{
    cs Amatrix, *A ;
    csi order, n, drop, *p ;
    css *S ;
    csn *N ;
    if (nargout > 2 || nargin < 1 || nargin > 2)
        mexErrMsgTxt ("Usage: [L,p] = cs_chol(A,drop)") ;
    A = cs_mex_get_sparse (&Amatrix, 1, 1, pargin [0]) ;        /* get A */
    n = A->n ;
    order = (nargout > 1) ? 1 : 0 ;                 /* determine ordering */
    S = cs_schol (order, A) ;                       /* symbolic Cholesky */
    N = cs_chol (A, S) ;                            /* numeric Cholesky */
    if (!N) mexErrMsgTxt ("cs_chol failed: not positive definite\n") ;
    drop = (nargin == 1) ? 1 : mxGetScalar (pargin [1]) ;
    if (drop) cs_dropzeros (N->L) ;                 /* drop zeros if requested*/
    pargout [0] = cs_mex_put_sparse (&(N->L)) ;     /* return L */
    if (nargout > 1)
    {
        p = cs_pinv (S->pinv, n) ;                  /* p=pinv' */
        pargout [1] = cs_mex_put_int (p, n, 1, 1) ; /* return p */
    }
    cs_nfree (N) ;
    cs_sfree (S) ;
}
