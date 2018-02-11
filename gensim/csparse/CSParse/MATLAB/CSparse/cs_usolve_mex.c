#include "cs_mex.h"
/* cs_usolve: x=U\b.  U must be sparse, real, and upper triangular.  b must be a
 * real full or sparse vector.  x is full or sparse, depending on b.
 *
 * Time taken is O(flop count), which may be less than n if b is sparse,
 * depending on U and b.
 *
 * This function works with MATLAB 7.2, but is not perfectly compatible with
 * the requirements of a MATLAB mexFunction when b is sparse.  X is returned
 * as an unsorted sparse vector.  Also, this mexFunction temporarily modifies
 * its input, U, by modifying U->p (in the cs_dfs function) and then restoring
 * it.  This could be corrected by creating a copy of U->p (see
 * cs_dmperm_mex.c), but this would take O(n) time, destroying the
 * O(flop count) time complexity of this function.
 */

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs Umatrix, Bmatrix, *U, *B, *X ;
    double *x, *b ;
    csi top, nz, p, *xi ;
    if (nargout > 1 || nargin != 2)
    {
        mexErrMsgTxt ("Usage: x = cs_usolve(U,b)") ;
    }
    U = cs_mex_get_sparse (&Umatrix, 1, 1, pargin [0]) ;    /* get U */
    if (mxIsSparse (pargin [1]))
    {
        B = cs_mex_get_sparse (&Bmatrix, 0, 1, pargin [1]) ;/* get sparse b */
        cs_mex_check (0, U->n, 1, 0, 1, 1, pargin [1]) ;
        xi = cs_malloc (2*U->n, sizeof (csi)) ;             /* get workspace */
        x  = cs_malloc (U->n, sizeof (double)) ;
        top = cs_spsolve (U, B, 0, xi, x, NULL, 0) ;        /* x = U\b */
        X = cs_spalloc (U->n, 1, U->n-top, 1, 0) ;          /* create sparse x*/
        X->p [0] = 0 ;
        nz = 0 ;
        for (p = top ; p < U->n ; p++)
        {
            X->i [nz] = xi [p] ;
            X->x [nz++] = x [xi [p]] ;
        }
        X->p [1] = nz ;
        pargout [0] = cs_mex_put_sparse (&X) ;
        cs_free (x) ;
        cs_free (xi) ;
    }
    else
    {
        b = cs_mex_get_double (U->n, pargin [1]) ;          /* get full b */
        x = cs_mex_put_double (U->n, b, &(pargout [0])) ;   /* x = b */
        cs_usolve (U, x) ;                                  /* x = U\x */
    }
}
