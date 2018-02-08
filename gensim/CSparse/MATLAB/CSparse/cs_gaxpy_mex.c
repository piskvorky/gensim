#include "cs_mex.h"
/* z = cs_gaxpy (A,x,y) computes z = A*x+y */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs Amatrix, *A ;
    double *x, *y, *z ;
    if (nargout > 1 || nargin != 3)
    {
        mexErrMsgTxt ("Usage: z = cs_gaxpy(A,x,y)") ;
    }
    A = cs_mex_get_sparse (&Amatrix, 0, 1, pargin [0]) ;    /* get A */
    x = cs_mex_get_double (A->n, pargin [1]) ;              /* get x */
    y = cs_mex_get_double (A->m, pargin [2]) ;              /* get y */
    z = cs_mex_put_double (A->m, y, &(pargout [0])) ;       /* z = y */
    cs_gaxpy (A, x, z) ;                                    /* z = z + A*x */
}
