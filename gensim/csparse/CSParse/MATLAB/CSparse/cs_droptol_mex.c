#include "cs_mex.h"
/* cs_droptol: remove small entries from A */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs Amatrix, *C, *A ;
    csi j, k ;
    double tol ;
    if (nargout > 1 || nargin != 2)
    {
        mexErrMsgTxt ("Usage: C = cs_droptol(A,tol)") ;
    }
    A = cs_mex_get_sparse (&Amatrix, 0, 1, pargin [0]) ;    /* get A */
    tol = mxGetScalar (pargin [1]) ;                        /* get tol */
    C = cs_spalloc (A->m, A->n, A->nzmax, 1, 0) ;           /* C = A */
    for (j = 0 ; j <= A->n ; j++) C->p [j] = A->p [j] ;
    for (k = 0 ; k < A->nzmax ; k++) C->i [k] = A->i [k] ;
    for (k = 0 ; k < A->nzmax ; k++) C->x [k] = A->x [k] ;
    cs_droptol (C, tol) ;                                   /* drop from C */
    pargout [0] = cs_mex_put_sparse (&C) ;                  /* return C */
}
