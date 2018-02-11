#include "cs.h"
/* x=A\b where A is symmetric positive definite; b overwritten with solution */
csi cs_cholsol (csi order, const cs *A, double *b)
{
    double *x ;
    css *S ;
    csn *N ;
    csi n, ok ;
    if (!CS_CSC (A) || !b) return (0) ;     /* check inputs */
    n = A->n ;
    S = cs_schol (order, A) ;               /* ordering and symbolic analysis */
    N = cs_chol (A, S) ;                    /* numeric Cholesky factorization */
    x = cs_malloc (n, sizeof (double)) ;    /* get workspace */
    ok = (S && N && x) ;
    if (ok)
    {
        cs_ipvec (S->pinv, b, x, n) ;   /* x = P*b */
        cs_lsolve (N->L, x) ;           /* x = L\x */
        cs_ltsolve (N->L, x) ;          /* x = L'\x */
        cs_pvec (S->pinv, x, b, n) ;    /* b = P'*x */
    }
    cs_free (x) ;
    cs_sfree (S) ;
    cs_nfree (N) ;
    return (ok) ;
}
