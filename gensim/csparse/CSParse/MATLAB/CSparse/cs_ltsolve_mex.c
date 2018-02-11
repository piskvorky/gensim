#include "cs_mex.h"
/* cs_ltsolve: solve an upper triangular system L'*x=b */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs Lmatrix, *L ;
    double *x, *b ;
    if (nargout > 1 || nargin != 2)
    {
        mexErrMsgTxt ("Usage: x = cs_ltsolve(L,b)") ;
    }
    L = cs_mex_get_sparse (&Lmatrix, 1, 1, pargin [0]) ;    /* get L */
    b = cs_mex_get_double (L->n, pargin [1]) ;              /* get b */
    x = cs_mex_put_double (L->n, b, &(pargout [0])) ;       /* x = b */
    cs_ltsolve (L, x) ;                                     /* x = L'\x */
}
