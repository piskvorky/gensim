#include "cs.h"
static csi cs_nonzero (csi i, csi j, double aij, void *other)
{
    return (aij != 0) ;
}
csi cs_dropzeros (cs *A)
{
    return (cs_fkeep (A, &cs_nonzero, NULL)) ;  /* keep all nonzero entries */
} 
