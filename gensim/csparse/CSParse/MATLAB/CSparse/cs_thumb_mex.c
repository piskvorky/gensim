#include "cs_mex.h"
/* cs_thumb: convert a sparse matrix to a dense 2D thumbnail matrix of size
 * at most k-by-k.  k defaults to 256.  A helper mexFunction for cspy. */

#define INDEX(i,j,lda) ((i)+(j)*(lda))
#define ISNAN(x) ((x) != (x))
#ifdef DBL_MAX
#define BIG_VALUE DBL_MAX
#else
#define BIG_VALUE 1.7e308
#endif

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs Amatrix, *A ;
    csi m, n, mn, m2, n2, k, s, j, ij, sj, si, p, *Ap, *Ai ;
    double aij, *S, *Ax ;
    if (nargout > 1 || nargin < 1 || nargin > 2)
    {
        mexErrMsgTxt ("Usage: S = cs_thumb(A,k)") ;
    }
    A = cs_mex_get_sparse (&Amatrix, 0, 1, pargin [0]) ;    /* get A */
    m = A->m ;
    n = A->n ;
    mn = CS_MAX (m,n) ;
    k = (nargin == 1) ? 256 : mxGetScalar (pargin [1]) ;    /* get k */
    /* s = size of each submatrix; A(1:s,1:s) maps to S(1,1) */
    s = (mn < k) ? 1 : (csi) ceil ((double) mn / (double) k) ;
    m2 = (csi) ceil ((double) m / (double) s) ;
    n2 = (csi) ceil ((double) n / (double) s) ;
    /* create S */
    pargout [0] = mxCreateDoubleMatrix (m2, n2, mxREAL) ;
    S = mxGetPr (pargout [0]) ;
    Ap = A->p ;
    Ai = A->i ;
    Ax = A->x ;
    for (j = 0 ; j < n ; j++)
    {
        sj = j/s ;
        for (p = Ap [j] ; p < Ap [j+1] ; p++)
        {
            si = Ai [p] / s ;
            ij = INDEX (si,sj,m2) ;
            aij = fabs (Ax [p]) ;
            if (ISNAN (aij)) aij = BIG_VALUE ;
            aij = CS_MIN (BIG_VALUE, aij) ;
            S [ij] = CS_MAX (S [ij], aij) ;
        }
    }
}
