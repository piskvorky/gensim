#include "cs_mex.h"
/* cs_dmperm: maximum matching or Dulmage-Mendelsohn permutation. */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    double seed ;
    cs *A, Amatrix ;
    csd *D ;
    csi m, n, *jmatch, iseed ;
    if (nargin < 1 || nargin > 2 || nargout > 6)
    {
        mexErrMsgTxt ("Usage: [p,q,r,s,cc,rr] = cs_dmperm (A,seed)") ;
    }
    seed = (nargin > 1) ? mxGetScalar (pargin [1]) : 0 ;        /* get seed */
    iseed = (seed > 0 && seed < 1) ? (seed * RAND_MAX) : seed ;
    A = cs_mex_get_sparse (&Amatrix, 0, 0, pargin [0]) ;        /* get A */
    n = A->n ;
    m = A->m ;
    if (nargout <= 1)
    {
        jmatch = cs_maxtrans (A, iseed) ;                   /* max. matching */
        pargout [0] = cs_mex_put_int (jmatch+m, n, 1, 0) ;  /* return imatch */
        cs_free (jmatch) ;
    }
    else
    {
        D = cs_dmperm (A, iseed) ;      /* Dulmage-Mendelsohn decomposition */
        pargout [0] = cs_mex_put_int (D->p, m, 1, 0) ;
        pargout [1] = cs_mex_put_int (D->q, n, 1, 0) ;
        pargout [2] = cs_mex_put_int (D->r, D->nb+1, 1, 0) ;
        pargout [3] = cs_mex_put_int (D->s, D->nb+1, 1, 0) ;
        pargout [4] = cs_mex_put_int (D->cc, 5, 1, 0) ;
        pargout [5] = cs_mex_put_int (D->rr, 5, 1, 0) ;
        cs_dfree (D) ;
    }
}
