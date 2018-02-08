#include "cs.h"
/* x(p) = b, for dense vectors x and b; p=NULL denotes identity */
csi cs_ipvec (const csi *p, const double *b, double *x, csi n)
{
    csi k ;
    if (!x || !b) return (0) ;                              /* check inputs */
    for (k = 0 ; k < n ; k++) x [p ? p [k] : k] = b [k] ;
    return (1) ;
}
