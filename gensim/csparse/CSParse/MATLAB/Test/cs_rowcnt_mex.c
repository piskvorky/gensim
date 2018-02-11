/* Compute the row counts of the Cholesky factor L of the matrix A.  Uses
 * the lower triangular part of A. */

#include "cs_mex.h"

static
void firstdesc (csi n, csi *parent, csi *post, csi *first, csi *level)
{
    csi len, i, k, r, s ;
    for (i = 0 ; i < n ; i++) first [i] = -1 ;
    for (k = 0 ; k < n ; k++)
    {
        i = post [k] ;      /* node i of etree is kth postordered node */
        len = 0 ;           /* traverse from i towards the root */
        for (r = i ; r != -1 && first [r] == -1 ; r = parent [r], len++)
            first [r] = k ;
        len += (r == -1) ? (-1) : level [r] ;   /* root node or end of path */
        for (s = i ; s != r ; s = parent [s]) level [s] = len-- ;
    }
}

static
csi *rowcnt (cs *A, csi *parent, csi *post) /* return rowcount [0..n-1] */
{
    csi i, j, k, p, q, n, jleaf, *Ap, *Ai, *maxfirst,
        *ancestor, *prevleaf, *w, *first, *level, *rowcount ;
    n = A->n ; Ap = A->p ; Ai = A->i ;                  /* get A */
    w = cs_malloc (5*n, sizeof (csi)) ;                 /* get workspace */
    ancestor = w ; maxfirst = w+n ; prevleaf = w+2*n ; first = w+3*n ;
    level = w+4*n ;
    rowcount = cs_malloc (n, sizeof (csi)) ;    /* allocate result */
    firstdesc (n, parent, post, first, level) ; /* find first and level */
    for (i = 0 ; i < n ; i++)
    {
        rowcount [i] = 1 ;      /* count the diagonal of L */
        prevleaf [i] = -1 ;     /* no previous leaf of the ith row subtree */
        maxfirst [i] = -1 ;     /* max first[j] for node j in ith subtree */
        ancestor [i] = i ;      /* every node is in its own set, by itself */
    }
    for (k = 0 ; k < n ; k++)
    {
        j = post [k] ;          /* j is the kth node in the postordered etree */
        for (p = Ap [j] ; p < Ap [j+1] ; p++)
        {
            i = Ai [p] ;
            q = cs_leaf (i, j, first, maxfirst, prevleaf, ancestor, &jleaf) ;
            if (jleaf) rowcount [i] += (level [j] - level [q]) ;
        }
        if (parent [j] != -1) ancestor [j] = parent [j] ;
    }
    cs_free (w) ;
    return (rowcount) ;
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs *A, Amatrix ;
    double *x ;
    csi i, n, *parent, *post, *rowcount ;

    if (nargout > 1 || nargin != 3)
    {
        mexErrMsgTxt ("Usage: r = rowcnt(A,parent,post)") ;
    }

    /* get inputs */
    A = cs_mex_get_sparse (&Amatrix, 1, 0, pargin [0]) ;
    n = A->n ;

    parent = cs_mex_get_int (n, pargin [1], &i, 0) ;
    post = cs_mex_get_int (n, pargin [2], &i, 1) ;

    rowcount = rowcnt (A, parent, post) ;

    pargout [0] = mxCreateDoubleMatrix (1, n, mxREAL) ;
    x = mxGetPr (pargout [0]) ;
    for (i = 0 ; i < n ; i++) x [i] = rowcount [i] ;

    cs_free (rowcount) ;
}
