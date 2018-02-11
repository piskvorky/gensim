#include "cs_mex.h"
/* cs_updown: sparse Cholesky update/downdate (rank-1 or multiple rank) */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs Lmatrix, *Lin, Cmatrix, *C, *L, Cvector, *Cvec ;
    csi ignore, j, k, n, lnz, *parent, sigma = 1, cp [2] ;
    char sigma_string [20] ;

    if (nargout > 1 || nargin < 3 || nargin > 4)
    {
        mexErrMsgTxt ("Usage: L = cs_updown(L,C,parent,sigma)") ;
    }
    Lin = cs_mex_get_sparse (&Lmatrix, 1, 1, pargin [0]) ;  /* get input L */
    n = Lin->n ;
    if (nargin > 3 && mxIsChar (pargin [3]))
    {
        mxGetString (pargin [3], sigma_string, 8) ;
        sigma = (sigma_string [0] == '-') ? (-1) : 1 ;
    }
    /* make a copy of L (this can take more work than updating L itself) */
    lnz = Lin->p [n] ;
    L = cs_spalloc (n, n, lnz, 1, 0) ;
    for (j = 0 ; j <= n ; j++) L->p [j] = Lin->p [j] ;
    for (k = 0 ; k < lnz ; k++) L->i [k] = Lin->i [k] ;
    for (k = 0 ; k < lnz ; k++) L->x [k] = Lin->x [k] ;
    cs_mex_check (0, n, -1, 0, 1, 1, pargin [1]) ;         /* get C */
    C = cs_mex_get_sparse (&Cmatrix, 0, 1, pargin [1]) ;
    parent = cs_mex_get_int (n, pargin [2], &ignore, 0) ;  /* get parent */

    /* do the update one column at a time */
    Cvec = &Cvector ;
    Cvec->m = n ;
    Cvec->n = 1 ;
    Cvec->p = cp ;
    Cvec->nz = -1 ;
    cp [0] = 0 ;
    for (k = 0 ; k < C->n ; k++)
    {
        /* extract C(:,k) */ 
        cp [1] = C->p [k+1] - C->p [k] ;
        Cvec->nzmax = cp [1] ;
        Cvec->i = C->i + C->p [k] ;
        Cvec->x = C->x + C->p [k] ;
        cs_updown (L, sigma, Cvec, parent) ;               /* update/downdate */
    }
    pargout [0] = cs_mex_put_sparse (&L) ;                 /* return new L */ 
}
