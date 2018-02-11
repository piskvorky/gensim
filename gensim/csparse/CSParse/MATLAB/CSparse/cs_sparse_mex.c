#include "cs_mex.h"
/* cs_sparse: convert triplet form into compress-column form sparse matrix */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs *A, *C, *T, Tmatrix ;
    if (nargout > 1 || nargin != 3)
    {
        mexErrMsgTxt ("Usage: A = cs_sparse(i,j,x)") ;
    }
    T = &Tmatrix ;                      /* get i,j,x and copy to triplet form */
    T->nz = mxGetNumberOfElements (pargin [0]) ;
    T->p = cs_mex_get_int (T->nz, pargin [0], &(T->n), 1) ;
    T->i = cs_mex_get_int (T->nz, pargin [1], &(T->m), 1) ;
    cs_mex_check (1, T->nz, 1, 0, 0, 1, pargin [2]) ;
    T->x = mxGetPr (pargin [2]) ;
    T->nzmax = T->nz ;
    C = cs_compress (T) ;               /* create sparse matrix C */
    cs_dupl (C) ;                       /* remove duplicates from C */
    cs_dropzeros (C) ;                  /* remove zeros from C */
    A = cs_transpose (C, 1) ;           /* A=C' */
    cs_spfree (C) ;
    pargout [0] = cs_mex_put_sparse (&A) ;      /* return A */
    cs_free (T->p) ;
    cs_free (T->i) ;
}
