#include "cs_mex.h"
/* find an augmenting path starting at column j and extend the match if found */
static
csi augment (csi k, cs *A, csi *jmatch, csi *cheap, csi *w, csi j)
{
    csi found = 0, p, i = -1, *Ap = A->p, *Ai = A->i ;
    /* --- Start depth-first-search at node j ------------------------------- */
    w [j] = k ;                             /* mark j as visited for kth path */
    for (p = cheap [j] ; p < Ap [j+1] && !found ; p++)
    {
        i = Ai [p] ;                        /* try a cheap assignment (i,j) */
        found = (jmatch [i] == -1) ;
    }
    cheap [j] = p ;                         /* start here next time for j */
    /* --- Depth-first-search of neighbors of j ----------------------------- */
    for (p = Ap [j] ; p < Ap [j+1] && !found ; p++)
    {
        i = Ai [p] ;                        /* consider row i */
        if (w [jmatch [i]] == k) continue ; /* skip col jmatch [i] if marked */
        found = augment (k, A, jmatch, cheap, w, jmatch [i]) ;
    }
    if (found) jmatch [i] = j ;             /* augment jmatch if path found */
    return (found) ;
}

/* find a maximum transveral */
static
csi *maxtrans (cs *A)   /* returns jmatch [0..m-1] */
{
    csi i, j, k, n, m, *Ap, *jmatch, *w, *cheap ;
    if (!A) return (NULL) ;                         /* check inputs */
    n = A->n ; m = A->m ; Ap = A->p ;
    jmatch = cs_malloc (m, sizeof (csi)) ;          /* allocate result */
    w = cs_malloc (2*n, sizeof (csi)) ;             /* allocate workspace */
    if (!w || !jmatch) return (cs_idone (jmatch, NULL, w, 0)) ;
    cheap = w + n ;
    for (j = 0 ; j < n ; j++) cheap [j] = Ap [j] ;  /* for cheap assignment */
    for (j = 0 ; j < n ; j++) w [j] = -1 ;          /* all columns unflagged */
    for (i = 0 ; i < m ; i++) jmatch [i] = -1 ;     /* no rows matched yet */
    for (k = 0 ; k < n ; k++) augment (k, A, jmatch, cheap, w, k) ;
    return (cs_idone (jmatch, NULL, w, 1)) ;
}

/* invert a maximum matching */
static csi *invmatch (csi *jmatch, csi m, csi n)
{
    csi i, j, *imatch ;
    if (!jmatch) return (NULL) ;
    imatch = cs_malloc (n, sizeof (csi)) ;
    if (!imatch) return (NULL) ;
    for (j = 0 ; j < n ; j++) imatch [j] = -1 ;
    for (i = 0 ; i < m ; i++) if (jmatch [i] >= 0) imatch [jmatch [i]] = i ;
    return (imatch) ;
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
    csi i, m, n, *imatch, *jmatch ;

    if (nargout > 1 || nargin != 1)
    {
        mexErrMsgTxt ("Usage: p = cr_maxtransr(A)") ;
    }

    /* get inputs */
    A = cs_mex_get_sparse (&Amatrix, 0, 0, pargin [0]) ;
    m = A->m ;
    n = A->n ;

    jmatch = maxtrans (A) ;
    imatch = invmatch (jmatch, m, n) ;  /* imatch = inverse of jmatch */

    pargout [0] = mxCreateDoubleMatrix (1, n, mxREAL) ;
    x = mxGetPr (pargout [0]) ;
    for (i = 0 ; i < n ; i++) x [i] = imatch [i] + 1 ;

    cs_free (jmatch) ;
    cs_free (imatch) ;
}
