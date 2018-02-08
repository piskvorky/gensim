#include "cs_mex.h"
/* C = cs_transpose (A), computes C=A', where A must be sparse and real */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs Amatrix, *A, *C ;
    if (nargout > 1 || nargin != 1)
    {
        mexErrMsgTxt ("Usage: C = cs_transpose(A)") ;
    }
    A = cs_mex_get_sparse (&Amatrix, 0, 1, pargin [0]) ;    /* get A */
    C = cs_transpose (A, 1) ;                               /* C = A' */
    pargout [0] = cs_mex_put_sparse (&C) ;                  /* return C */
}
