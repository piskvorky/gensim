#include "cs_mex.h"
/* cs_utsolve: solve a lower triangular system U'*x=b */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs Umatrix, *U ;
    double *x, *b ;
    if (nargout > 1 || nargin != 2)
    {
        mexErrMsgTxt ("Usage: x = cs_utsolve(U,b)") ;
    }
    U = cs_mex_get_sparse (&Umatrix, 1, 1, pargin [0]) ;    /* get U */
    b = cs_mex_get_double (U->n, pargin [1]) ;              /* get b */
    x = cs_mex_put_double (U->n, b, &(pargout [0])) ;       /* x = b */
    cs_utsolve (U, x) ;                                     /* x = U'\x */
}
