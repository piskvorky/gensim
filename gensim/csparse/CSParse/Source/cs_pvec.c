#include "cs.h"
/* x = b(p), for dense vectors x and b; p=NULL denotes identity */
csi cs_pvec (const csi *p, const double *b, double *x, csi n)
{
    csi k ;
    if (!x || !b) return (0) ;                              /* check inputs */
    for (k = 0 ; k < n ; k++) x [k] = b [p ? p [k] : k] ;
    return (1) ;
}
