#include "cs_mex.h"
/* cs_lusol: solve A*x=b using a sparse LU factorization */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs *A, Amatrix ;
    csi order ;
    double *x, *b, tol ;
    if (nargout > 1 || nargin < 2 || nargin > 4)
    {
        mexErrMsgTxt ("Usage: x = cs_lusol(A,b,order,tol)") ;
    }
    A = cs_mex_get_sparse (&Amatrix, 1, 1, pargin [0]) ;    /* get A */
    b = cs_mex_get_double (A->n, pargin [1]) ;              /* get b */
    x = cs_mex_put_double (A->n, b, &(pargout [0])) ;       /* x = b */
    order = (nargin < 3) ? 2 : mxGetScalar (pargin [2]) ;
    order = CS_MAX (order, 0) ;
    order = CS_MIN (order, 3) ;
    if (nargin == 2)
    {
        tol = 1 ;                           /* normal partial pivoting */
    }
    else if (nargin == 3)
    {
        tol = (order == 1) ? 0.001 : 1 ;    /* tol = 0.001 for amd(A+A') */
    }
    else
    {
        tol = mxGetScalar (pargin [3]) ;
    }
    if (!cs_lusol (order, A, x, tol))                       /* x = A\x */
    {
        mexErrMsgTxt ("LU factorization failed (singular or out of memory)") ;
    }
}
