#include "cs_mex.h"
/* cs_lsolve: x=L\b.  L must be sparse, real, and lower triangular.  b must be a
 * real full or sparse vector.  x is full or sparse, depending on b.
 *
 * Time taken is O(flop count), which may be less than n if b is sparse,
 * depending on L and b.
 *
 * This function works with MATLAB 7.2, but is not perfectly compatible with
 * the requirements of a MATLAB mexFunction when b is sparse.  X is returned
 * as an unsorted sparse vector.  Also, this mexFunction temporarily modifies
 * its input, L, by modifying L->p (in the cs_dfs function) and then restoring
 * it.  This could be corrected by creating a copy of L->p
 * (see cs_dmperm_mex.c), but this would take O(n) time, destroying the
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
    cs Lmatrix, Bmatrix, *L, *B, *X ;
    double *x, *b ;
    csi top, nz, p, *xi ;
    if (nargout > 1 || nargin != 2)
    {
        mexErrMsgTxt ("Usage: x = cs_lsolve(L,b)") ;
    }
    L = cs_mex_get_sparse (&Lmatrix, 1, 1, pargin [0]) ;    /* get L */
    if (mxIsSparse (pargin [1]))
    {
        B = cs_mex_get_sparse (&Bmatrix, 0, 1, pargin [1]) ;/* get sparse b */
        cs_mex_check (0, L->n, 1, 0, 1, 1, pargin [1]) ;
        xi = cs_malloc (2*L->n, sizeof (csi)) ;             /* get workspace */
        x  = cs_malloc (L->n, sizeof (double)) ;
        top = cs_spsolve (L, B, 0, xi, x, NULL, 1) ;        /* x = L\b */
        X = cs_spalloc (L->n, 1, L->n-top, 1, 0) ;          /* create sparse x*/
        X->p [0] = 0 ;
        nz = 0 ;
        for (p = top ; p < L->n ; p++)
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
        b = cs_mex_get_double (L->n, pargin [1]) ;          /* get full b */
        x = cs_mex_put_double (L->n, b, &(pargout [0])) ;   /* x = b */
        cs_lsolve (L, x) ;                                  /* x = L\x */
    }
}
