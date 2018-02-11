#include "cs_mex.h"
/* find nonzero pattern of x=L\sparse(b).  L must be sparse, real, and lower
 * triangular.  b must be a real sparse vector. */

static
void dfsr (csi j, const cs *L, csi *top, csi *xi, csi *w)
{
    csi p ;
    w [j] = 1 ;                                 /* mark node j */
    for (p = L->p [j] ; p < L->p [j+1] ; p++)   /* for each i in L(:,j) */
    {
        if (w [L->i [p]] != 1)                  /* if i is unmarked */
        {
            dfsr (L->i [p], L, top, xi, w) ;    /* start a dfs at i */
        }
    }
    xi [--(*top)] = j ;                         /* push j onto the stack */
}

/* w [0..n-1] == 0 on input, <= 1 on output.  size n */
static
csi reachr (const cs *L, const cs *B, csi *xi, csi *w)
{
    csi p, n = L->n ;
    csi top = n ;                               /* stack is empty */
    for (p = B->p [0] ; p < B->p [1] ; p++)     /* for each i in pattern of b */
    {
        if (w [B->i [p]] != 1)                  /* if i is unmarked */
        {
            dfsr (B->i [p], L, &top, xi, w) ;   /* start a dfs at i */
        }
    }
    return (top) ;                              /* return top of stack */
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs Lmatrix, Bmatrix, *L, *B ;
    double *x ;
    csi i, j, top, *xi ;

    if (nargout > 1 || nargin != 2)
    {
        mexErrMsgTxt ("Usage: x = cs_reachr(L,b)") ;
    }

    /* get inputs */
    L = cs_mex_get_sparse (&Lmatrix, 1, 1, pargin [0]) ;
    B = cs_mex_get_sparse (&Bmatrix, 0, 1, pargin [1]) ;
    cs_mex_check (0, L->n, 1, 0, 1, 1, pargin [1]) ;

    xi = cs_calloc (2*L->n, sizeof (csi)) ;

    top = reachr (L, B, xi, xi + L->n) ;

    pargout [0] = mxCreateDoubleMatrix (L->n - top, 1, mxREAL) ;
    x = mxGetPr (pargout [0]) ;
    for (j = 0, i = top ; i < L->n ; i++, j++) x [j] = xi [i] ;

    cs_free (xi) ;
}
