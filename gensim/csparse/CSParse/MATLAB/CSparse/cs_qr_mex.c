#include "cs_mex.h"
/* cs_qr: sparse QR factorization */
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
    csi m, n, order, *p ;
    if (nargout > 5 || nargin != 1)
    {
        mexErrMsgTxt ("Usage: [V,beta,p,R,q] = cs_qr(A)") ;
    }
    A = cs_mex_get_sparse (&Amatrix, 0, 1, pargin [0]) ;        /* get A */
    m = A->m ;
    n = A->n ;
    if (m < n) mexErrMsgTxt ("A must have # rows >= # columns") ;
    order = (nargout == 5) ? 3 : 0 ;        /* determine ordering */
    S = cs_sqr (order, A, 1) ;              /* symbolic QR ordering & analysis*/
    N = cs_qr (A, S) ;                      /* numeric QR factorization */
    if (!N) mexErrMsgTxt ("qr failed") ;
    cs_dropzeros (N->L) ;                   /* drop zeros from V and sort */
    D = cs_transpose (N->L, 1) ;
    cs_spfree (N->L) ;
    N->L = cs_transpose (D, 1) ;
    cs_spfree (D) ;
    cs_dropzeros (N->U) ;                   /* drop zeros from R and sort */
    D = cs_transpose (N->U, 1) ;
    cs_spfree (N->U) ;
    N->U = cs_transpose (D, 1) ;
    cs_spfree (D) ;
    m = N->L->m ;                                   /* m may be larger now */
    p = cs_pinv (S->pinv, m) ;                      /* p = pinv' */
    pargout [0] = cs_mex_put_sparse (&(N->L)) ;     /* return V */
    cs_mex_put_double (n, N->B, &(pargout [1])) ;   /* return beta */
    pargout [2] = cs_mex_put_int (p, m, 1, 1) ;     /* return p */
    pargout [3] = cs_mex_put_sparse (&(N->U)) ;     /* return R */
    pargout [4] = cs_mex_put_int (S->q, n, 1, 0) ;  /* return q */
    cs_nfree (N) ;
    cs_sfree (S) ;
}
