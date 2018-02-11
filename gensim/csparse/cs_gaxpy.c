# include "cs.h"

# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <stdint.h>
# include <time.h>
# include <omp.h>

/* y = A*x+y */
csi cs_gaxpy (const cs *A, const double *x, double *y)
{
  csi p, j, n, *Ap, *Ai ;
  double *Ax ;
  if (!CS_CSC (A) || !x || !y) return (0) ;       /* check inputs */
  n = A->n ; Ap = A->p ; Ai = A->i ; Ax = A->x ;
  for (j = 0 ; j < n ; j++)
    {
      for (p = Ap [j] ; p < Ap [j+1] ; p++)
        {
	  y [Ai [p]] += Ax [p] * x [j] ;
        }
    }
  return (1) ;
}



