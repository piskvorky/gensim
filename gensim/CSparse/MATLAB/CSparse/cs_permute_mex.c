#include "cs_mex.h"
/* cs_permute: permute a sparse matrix */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs Amatrix, *A, *C, *D ;
    csi ignore, *P, *Q, *Pinv ;
    if (nargout > 1 || nargin != 3)
    {
        mexErrMsgTxt ("Usage: C = cs_permute(A,p,q)") ;
    }
    A = cs_mex_get_sparse (&Amatrix, 0, 1, pargin [0]) ;    /* get A */
    P = cs_mex_get_int (A->m, pargin [1], &ignore, 1) ;     /* get P */
    Q = cs_mex_get_int (A->n, pargin [2], &ignore, 1) ;     /* get Q */
    Pinv = cs_pinv (P, A->m) ;              /* P = Pinv' */
    C = cs_permute (A, Pinv, Q, 1) ;        /* C = A(p,q) */
    D = cs_transpose (C, 1) ;               /* sort C via double transpose */
    cs_spfree (C) ;
    C = cs_transpose (D, 1) ;
    cs_spfree (D) ;
    pargout [0] = cs_mex_put_sparse (&C) ;          /* return C */
    cs_free (Pinv) ;
    cs_free (P) ;
    cs_free (Q) ;
}
