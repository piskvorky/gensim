#include "cs_mex.h"
/* cs_symperm: symmetric permutation of a symmetric sparse matrix. */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs Amatrix, *A, *C, *D ;
    csi ignore, n, *P, *Pinv ;
    if (nargout > 1 || nargin != 2)
    {
        mexErrMsgTxt ("Usage: C = cs_symperm(A,p)") ;
    }
    A = cs_mex_get_sparse (&Amatrix, 1, 1, pargin [0]) ;
    n = A->n ;
    P = cs_mex_get_int (n, pargin [1], &ignore, 1) ;    /* get P */
    Pinv = cs_pinv (P, n) ;                             /* P=Pinv' */
    C = cs_symperm (A, Pinv, 1) ;                       /* C = A(p,p) */
    D = cs_transpose (C, 1) ;                           /* sort C */
    cs_spfree (C) ;
    C = cs_transpose (D, 1) ;
    cs_spfree (D) ;
    pargout [0] = cs_mex_put_sparse (&C) ;              /* return C */
    cs_free (P) ;
    cs_free (Pinv) ;
}
