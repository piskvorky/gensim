#include "cs_mex.h"
/* A = cs_sparse2 (i,j,x), removing duplicates and numerically zero entries,
 * and returning A sorted (test cs_entry) */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    double *Tx ;
    cs *A, *C, *T ;
    csi k, m, n, nz, *Ti, *Tj ;
    if (nargout > 1 || nargin != 3)
    {
        mexErrMsgTxt ("Usage: A = cs_sparse2(i,j,x)") ;
    }
    nz = mxGetNumberOfElements (pargin [0]) ;
    Ti = cs_mex_get_int (nz, pargin [0], &m, 1) ;
    Tj = cs_mex_get_int (nz, pargin [1], &n, 1) ;
    cs_mex_check (1, nz, 1, 0, 0, 1, pargin [2]) ;
    Tx = mxGetPr (pargin [2]) ;

    T = cs_spalloc (n, m, 1, 1, 1) ;
    for (k = 0 ; k < nz ; k++)
    {
        cs_entry (T, Tj [k], Ti [k], Tx [k]) ;
    }

    C = cs_compress (T) ;
    cs_spfree (T) ;

    cs_dupl (C) ;
    cs_dropzeros (C) ;
    A = cs_transpose (C, 1) ;
    cs_spfree (C) ;
    pargout [0] = cs_mex_put_sparse (&A) ;
    cs_free (Ti) ;
    cs_free (Tj) ;
}
