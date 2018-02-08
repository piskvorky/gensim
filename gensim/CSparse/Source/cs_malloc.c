#include "cs.h"
#ifdef MATLAB_MEX_FILE
#define malloc mxMalloc
#define free mxFree
#define realloc mxRealloc
#define calloc mxCalloc
#endif

/* wrapper for malloc */
void *cs_malloc (csi n, size_t size)
{
    return (malloc (CS_MAX (n,1) * size)) ;
}

/* wrapper for calloc */
void *cs_calloc (csi n, size_t size)
{
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
    pnew = realloc (p, CS_MAX (n,1) * size) ; /* realloc the block */
    *ok = (pnew != NULL) ;                  /* realloc fails if pnew is NULL */
    return ((*ok) ? pnew : p) ;             /* return original p if failure */
}
