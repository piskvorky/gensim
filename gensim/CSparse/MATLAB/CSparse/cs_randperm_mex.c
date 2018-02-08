#include "cs_mex.h"
/* cs_randperm: random permutation.  p=cs_randperm(n,0) is 1:n,
 * p=cs_randperm(n,-1) is n:-1:1.  p = cs_randperm (n,seed) is a random
 * permutation using the given seed (where seed is not 0 or -1).
 * seed defaults to 1.  A single seed always gives a repeatable permutation.
 * Use p = cs_randperm(n,rand) to get a permutation that varies with each use.
 */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    double seed ;
    csi iseed, n, *p ;
    if (nargout > 1 || nargin < 1 || nargin > 2)
    {
        mexErrMsgTxt ("Usage: p = cs_randperm(n,seed)") ;
    }
    seed = (nargin > 1) ? mxGetScalar (pargin [1]) : 1 ;
    iseed = (seed > 0 && seed < 1) ? (seed * RAND_MAX) : seed ;
    n = mxGetScalar (pargin [0]) ;
    n = CS_MAX (n, 0) ;
    p = cs_randperm (n, iseed) ;
    pargout [0] = cs_mex_put_int (p, n, 1, 1) ;     /* return p */
}
