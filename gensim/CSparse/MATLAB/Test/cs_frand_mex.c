#include "cs_mex.h"
/* A = cs_frand (n,nel,s) creates an n-by-n sparse matrix consisting of nel
 * finite elements, each of which are of size s-by-s with random symmetric
 * nonzero pattern, plus the identity matrix.
 * See also MATLAB/Demo/private/frand.m */

static
cs *cs_frand (csi n, csi nel, csi s)
{
    csi ss = s*s, nz = nel*ss, e, i, j, *P ;
    cs *A, *T = cs_spalloc (n, n, nz, 1, 1) ;
    if (!T) return (NULL) ;
    P = cs_malloc (s, sizeof (csi)) ;
    if (!P) return (cs_spfree (T)) ;
    for (e = 0 ; e < nel ; e++)
    {
        for (i = 0 ; i < s ; i++) P [i] = rand () % n ;
        for (j = 0 ; j < s ; j++)
        {
            for (i = 0 ; i < s ; i++)
            {
                cs_entry (T, P [i], P [j], rand () / (double) RAND_MAX) ;
            }
        }
    }
    for (i = 0 ; i < n ; i++) cs_entry (T, i, i, 1) ;
    A = cs_compress (T) ;
    cs_spfree (T) ;
    return (cs_dupl (A) ? A : cs_spfree (A)) ;
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    csi n, nel, s ;
    cs *A, *AT ;
    if (nargout > 1 || nargin != 3)
    {
        mexErrMsgTxt ("Usage: C = cs_frand(n,nel,s)") ;
    }
    n = mxGetScalar (pargin [0]) ;
    nel = mxGetScalar (pargin [1]) ;
    s = mxGetScalar (pargin [2]) ;

    n = CS_MAX (1,n) ;
    nel = CS_MAX (1,nel) ;
    s = CS_MAX (1,s) ;

    AT = cs_frand (n, nel, s) ;
    A = cs_transpose (AT, 1) ;
    cs_spfree (AT) ;
    cs_dropzeros (A) ;

    pargout [0] = cs_mex_put_sparse (&A) ;
}
