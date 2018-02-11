#include "cs_mex.h"
/* cs_multiply: sparse matrix multiply */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs Amatrix, Bmatrix, *A, *B, *C, *D ;
    if (nargout > 1 || nargin != 2)
    {
        mexErrMsgTxt ("Usage: C = cs_multiply(A,B)") ;
    }
    A = cs_transpose (cs_mex_get_sparse (&Amatrix, 0, 1, pargin [0]), 1) ;
    B = cs_transpose (cs_mex_get_sparse (&Bmatrix, 0, 1, pargin [1]), 1) ;
    D = cs_multiply (B,A) ;             /* D = B'*A' */
    cs_spfree (A) ;
    cs_spfree (B) ;
    cs_dropzeros (D) ;                  /* drop zeros from D */
    C = cs_transpose (D, 1) ;           /* C = D', so that C is sorted */
    cs_spfree (D) ;
    pargout [0] = cs_mex_put_sparse (&C) ;      /* return C */
}
