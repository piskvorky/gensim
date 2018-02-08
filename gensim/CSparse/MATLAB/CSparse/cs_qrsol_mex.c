#include "cs_mex.h"
/* cs_qrsol: solve least squares or underdetermined problem */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs *A, Amatrix ;
    double *x, *b ;
    csi k, order ;
    if (nargout > 1 || nargin < 2 || nargin > 3)
    {
        mexErrMsgTxt ("Usage: x = cs_qrsol(A,b,order)") ;
    }
    A = cs_mex_get_sparse (&Amatrix, 0, 1, pargin [0]) ;    /* get A */
    b = cs_mex_get_double (A->m, pargin [1]) ;              /* get b */
    x = cs_calloc (CS_MAX (A->m, A->n), sizeof (double)) ;  /* x = b */
    for (k = 0 ; k < A->m ; k++) x [k] = b [k] ;
    order = (nargin < 3) ? 3 : mxGetScalar (pargin [2]) ;
    order = CS_MAX (order, 0) ;
    order = CS_MIN (order, 3) ;
    if (!cs_qrsol (order, A, x))                            /* x = A\x */
    {
        mexErrMsgTxt ("QR solve failed") ;
    }
    cs_mex_put_double (A->n, x, &(pargout [0])) ;           /* return x */
    cs_free (x) ;
}
