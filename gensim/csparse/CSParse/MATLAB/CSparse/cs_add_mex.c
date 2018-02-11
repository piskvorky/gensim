#include "cs_mex.h"
/* cs_add: sparse matrix addition */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    double alpha, beta ;
    cs Amatrix, Bmatrix, *A, *B, *C, *D ;
    if (nargout > 1 || nargin < 2 || nargin > 4)
    {
        mexErrMsgTxt ("Usage: C = cs_add(A,B,alpha,beta)") ;
    }
    A = cs_mex_get_sparse (&Amatrix, 0, 1, pargin [0]) ;    /* get A */
    B = cs_mex_get_sparse (&Bmatrix, 0, 1, pargin [1]) ;    /* get B */
    alpha = (nargin < 3) ? 1 : mxGetScalar (pargin [2]) ;   /* get alpha */
    beta  = (nargin < 4) ? 1 : mxGetScalar (pargin [3]) ;   /* get beta */
    C = cs_add (A,B,alpha,beta) ;       /* C = alpha*A + beta *B */
    cs_dropzeros (C) ;                  /* drop zeros */
    D = cs_transpose (C, 1) ;           /* sort result via double transpose */
    cs_spfree (C) ;
    C = cs_transpose (D, 1) ;
    cs_spfree (D) ;
    pargout [0] = cs_mex_put_sparse (&C) ;      /* return C */
}
