#include "cs_mex.h"
/* cs_counts: column counts for sparse Cholesky factor L. */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs Amatrix, *A ;
    csi n, ata, *parent, *post, *c ;
    char mode [20] ;
    if (nargout > 2 || nargin < 1 || nargin > 2)
    {
        mexErrMsgTxt ("Usage: c = cs_counts(A,mode)") ;
    }
    ata = 0 ;                                           /* get mode */
    if (nargin > 1 && mxIsChar (pargin [1]))
    {
        mxGetString (pargin [1], mode, 8) ;
        ata = (mode [0] == 'c') ;
    }
    A = cs_mex_get_sparse (&Amatrix, !ata, 0, pargin [0]) ;     /* get A */
    n = A->n ;
    parent = cs_etree (A, ata) ;                        /* compute etree */
    post = cs_post (parent, n) ;                        /* postorder the etree*/
    c = cs_counts (A, parent, post, ata) ;              /* get column counts */
    pargout [0] = cs_mex_put_int (c, n, 0, 1) ;         /* return counts */
    cs_free (parent) ;
    cs_free (post) ;
}
