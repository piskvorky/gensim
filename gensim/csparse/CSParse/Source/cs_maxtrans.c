#include "cs.h"
/* find an augmenting path starting at column k and extend the match if found */
static void cs_augment (csi k, const cs *A, csi *jmatch, csi *cheap, csi *w,
        csi *js, csi *is, csi *ps)
{
    csi found = 0, p, i = -1, *Ap = A->p, *Ai = A->i, head = 0, j ;
    js [0] = k ;                        /* start with just node k in jstack */
    while (head >= 0)
    {
        /* --- Start (or continue) depth-first-search at node j ------------- */
        j = js [head] ;                 /* get j from top of jstack */
        if (w [j] != k)                 /* 1st time j visited for kth path */
        {
            w [j] = k ;                 /* mark j as visited for kth path */
            for (p = cheap [j] ; p < Ap [j+1] && !found ; p++)
            {
                i = Ai [p] ;            /* try a cheap assignment (i,j) */
                found = (jmatch [i] == -1) ;
            }
            cheap [j] = p ;             /* start here next time j is traversed*/
            if (found)
            {
                is [head] = i ;         /* column j matched with row i */
                break ;                 /* end of augmenting path */
            }
            ps [head] = Ap [j] ;        /* no cheap match: start dfs for j */
        }
        /* --- Depth-first-search of neighbors of j ------------------------- */
        for (p = ps [head] ; p < Ap [j+1] ; p++)
        {
            i = Ai [p] ;                /* consider row i */
            if (w [jmatch [i]] == k) continue ; /* skip jmatch [i] if marked */
            ps [head] = p + 1 ;         /* pause dfs of node j */
            is [head] = i ;             /* i will be matched with j if found */
            js [++head] = jmatch [i] ;  /* start dfs at column jmatch [i] */
            break ;
        }
        if (p == Ap [j+1]) head-- ;     /* node j is done; pop from stack */
    }                                   /* augment the match if path found: */
    if (found) for (p = head ; p >= 0 ; p--) jmatch [is [p]] = js [p] ;
}

/* find a maximum transveral */
csi *cs_maxtrans (const cs *A, csi seed)  /*[jmatch [0..m-1]; imatch [0..n-1]]*/
{
    csi i, j, k, n, m, p, n2 = 0, m2 = 0, *Ap, *jimatch, *w, *cheap, *js, *is,
        *ps, *Ai, *Cp, *jmatch, *imatch, *q ;
    cs *C ;
    if (!CS_CSC (A)) return (NULL) ;                /* check inputs */
    n = A->n ; m = A->m ; Ap = A->p ; Ai = A->i ;
    w = jimatch = cs_calloc (m+n, sizeof (csi)) ;   /* allocate result */
    if (!jimatch) return (NULL) ;
    for (k = 0, j = 0 ; j < n ; j++)    /* count nonempty rows and columns */
    {
        n2 += (Ap [j] < Ap [j+1]) ;
        for (p = Ap [j] ; p < Ap [j+1] ; p++)
        {
            w [Ai [p]] = 1 ;
            k += (j == Ai [p]) ;        /* count entries already on diagonal */
        }
    }
    if (k == CS_MIN (m,n))              /* quick return if diagonal zero-free */
    {
        jmatch = jimatch ; imatch = jimatch + m ;
        for (i = 0 ; i < k ; i++) jmatch [i] = i ;
        for (      ; i < m ; i++) jmatch [i] = -1 ;
        for (j = 0 ; j < k ; j++) imatch [j] = j ;
        for (      ; j < n ; j++) imatch [j] = -1 ;
        return (cs_idone (jimatch, NULL, NULL, 1)) ;
    }
    for (i = 0 ; i < m ; i++) m2 += w [i] ;
    C = (m2 < n2) ? cs_transpose (A,0) : ((cs *) A) ; /* transpose if needed */
    if (!C) return (cs_idone (jimatch, (m2 < n2) ? C : NULL, NULL, 0)) ;
    n = C->n ; m = C->m ; Cp = C->p ;
    jmatch = (m2 < n2) ? jimatch + n : jimatch ;
    imatch = (m2 < n2) ? jimatch : jimatch + m ;
    w = cs_malloc (5*n, sizeof (csi)) ;             /* get workspace */
    if (!w) return (cs_idone (jimatch, (m2 < n2) ? C : NULL, w, 0)) ;
    cheap = w + n ; js = w + 2*n ; is = w + 3*n ; ps = w + 4*n ;
    for (j = 0 ; j < n ; j++) cheap [j] = Cp [j] ;  /* for cheap assignment */
    for (j = 0 ; j < n ; j++) w [j] = -1 ;          /* all columns unflagged */
    for (i = 0 ; i < m ; i++) jmatch [i] = -1 ;     /* nothing matched yet */
    q = cs_randperm (n, seed) ;                     /* q = random permutation */
    for (k = 0 ; k < n ; k++)   /* augment, starting at column q[k] */
    {
        cs_augment (q ? q [k]: k, C, jmatch, cheap, w, js, is, ps) ;
    }
    cs_free (q) ;
    for (j = 0 ; j < n ; j++) imatch [j] = -1 ;     /* find row match */
    for (i = 0 ; i < m ; i++) if (jmatch [i] >= 0) imatch [jmatch [i]] = i ;
    return (cs_idone (jimatch, (m2 < n2) ? C : NULL, w, 1)) ;
}
