#include "cs.h"
/* x=A\b where A is unsymmetric; b overwritten with solution */
csi cs_lusol (csi order, const cs *A, double *b, double tol)
{
    double *x ;
    css *S ;
    csn *N ;
    csi n, ok ;
    if (!CS_CSC (A) || !b) return (0) ;     /* check inputs */
    n = A->n ;
    S = cs_sqr (order, A, 0) ;              /* ordering and symbolic analysis */
    N = cs_lu (A, S, tol) ;                 /* numeric LU factorization */
    x = cs_malloc (n, sizeof (double)) ;    /* get workspace */
    ok = (S && N && x) ;
    if (ok)
    {
        cs_ipvec (N->pinv, b, x, n) ;       /* x = b(p) */
        cs_lsolve (N->L, x) ;               /* x = L\x */
        cs_usolve (N->U, x) ;               /* x = U\x */
        cs_ipvec (S->q, x, b, n) ;          /* b(q) = x */
    }
    cs_free (x) ;
    cs_sfree (S) ;
    cs_nfree (N) ;
    return (ok) ;
}
