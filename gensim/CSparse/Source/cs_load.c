#include "cs.h"
/* load a triplet matrix from a file */
cs *cs_load (FILE *f)
{
    double i, j ;   /* use double for integers to avoid csi conflicts */
    double x ;
    cs *T ;
    if (!f) return (NULL) ;                             /* check inputs */
    T = cs_spalloc (0, 0, 1, 1, 1) ;                    /* allocate result */
    while (fscanf (f, "%lg %lg %lg\n", &i, &j, &x) == 3)
    {
        if (!cs_entry (T, (csi) i, (csi) j, x)) return (cs_spfree (T)) ;
    }
    return (T) ;
}
