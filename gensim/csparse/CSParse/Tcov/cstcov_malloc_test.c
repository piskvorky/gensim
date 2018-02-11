#include "cstcov_malloc_test.h"
csi malloc_count = INT_MAX ;

/* wrapper for malloc */
void *cs_malloc (csi n, size_t size)
{
    if (--malloc_count < 0) return (NULL) ; /* pretend to fail */
    return (malloc (CS_MAX (n,1) * size)) ;
}

/* wrapper for calloc */
void *cs_calloc (csi n, size_t size)
{
    if (--malloc_count < 0) return (NULL) ; /* pretend to fail */
    return (calloc (CS_MAX (n,1), size)) ;
}

/* wrapper for free */
void *cs_free (void *p)
{
    if (p) free (p) ;       /* free p if it is not already NULL */
    return (NULL) ;         /* return NULL to simplify the use of cs_free */
}

/* wrapper for realloc */
void *cs_realloc (void *p, csi n, size_t size, csi *ok)
{
    void *pnew ;
    *ok = 0 ;
    if (--malloc_count < 0) return (p) ;    /* pretend to fail */
    pnew = realloc (p, CS_MAX (n,1) * size) ; /* realloc the block */
    *ok = (pnew != NULL) ;
    return ((*ok) ? pnew : p) ;             /* return original p if failure */
}
