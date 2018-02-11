#include "cs_mex.h"
/* cs_print: print the contents of a sparse matrix. */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs Amatrix, *A ;
    csi brief ;
    if (nargout > 0 || nargin < 1 || nargin > 2)
    {
        mexErrMsgTxt ("Usage: cs_print(A,brief)") ;
    }
    A = cs_mex_get_sparse (&Amatrix, 0, 1, pargin [0]) ;    /* get A */
    brief = (nargin < 2) ? 0 : mxGetScalar (pargin [1]) ;   /* get brief */
    cs_print (A, brief) ;                                   /* print A */
}
