#include "cs.h"
#include "mex.h"
cs *cs_mex_get_sparse (cs *A, csi square, csi values, const mxArray *Amatlab) ;
mxArray *cs_mex_put_sparse (cs **A) ;
void cs_mex_check (csi nel, csi m, csi n, csi square, csi sparse, csi values,
    const mxArray *A) ;
csi *cs_mex_get_int (csi n, const mxArray *Imatlab, csi *imax, csi lo) ;
mxArray *cs_mex_put_int (csi *p, csi n, csi offset, csi do_free) ;
double *cs_mex_get_double (csi n, const mxArray *X) ;
double *cs_mex_put_double (csi n, const double *b, mxArray **X) ;
