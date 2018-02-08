#include "cs_mex.h"
/* cs_sqr: symbolic sparse QR factorization */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    double s ;
    css *S ;
    cs Amatrix, *A ;
    csi m, n, order, *p ;
    if (nargout > 7 || nargin != 1)
    {
        mexErrMsgTxt ("Usage: [vnz,rnz,parent,c,leftmost,p,q] = cs_sqr(A)") ;
    }
    A = cs_mex_get_sparse (&Amatrix, 0, 1, pargin [0]) ;        /* get A */
    m = A->m ;
    n = A->n ;
    if (m < n) mexErrMsgTxt ("A must have # rows >= # columns") ;
    order = (nargout == 7) ? 3 : 0 ;        /* determine ordering */
    S = cs_sqr (order, A, 1) ;              /* symbolic QR ordering & analysis*/
    if (!S) mexErrMsgTxt ("cs_sqr failed") ;
    s = S->lnz ;
    cs_mex_put_double (1, &s, &(pargout [0])) ;             /* return nnz(V) */
    s = S->unz ;
    cs_mex_put_double (1, &s, &(pargout [1])) ;             /* return nnz(R) */
    pargout [2] = cs_mex_put_int (S->parent, n, 1, 0) ;     /* return parent */
    pargout [3] = cs_mex_put_int (S->cp, n, 0, 0) ;         /* return c */
    pargout [4] = cs_mex_put_int (S->leftmost, m, 1, 0) ;   /* return leftmost*/
    p = cs_pinv (S->pinv, S->m2) ;                          /* p = pinv' */
    pargout [5] = cs_mex_put_int (p, S->m2, 1, 1) ;         /* return p */
    if (nargout > 6)
    {
        pargout [6] = cs_mex_put_int (S->q, n, 1, 0) ;      /* return q */
    }
    cs_sfree (S) ;
}
