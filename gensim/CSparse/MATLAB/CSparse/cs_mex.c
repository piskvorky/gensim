#include "cs_mex.h"
/* check MATLAB input argument */
void cs_mex_check (csi nel, csi m, csi n, csi square, csi sparse, csi values,
    const mxArray *A)
{
    csi nnel, mm = mxGetM (A), nn = mxGetN (A) ;
    if (values)
    {
        if (mxIsComplex (A))
        {
            mexErrMsgTxt ("matrix must be real; try CXSparse instead") ;
        }
    }
    if (sparse && !mxIsSparse (A)) mexErrMsgTxt ("matrix must be sparse") ;
    if (!sparse)
    {
        if (mxIsSparse (A)) mexErrMsgTxt ("matrix must be full") ;
        if (values && !mxIsDouble (A)) mexErrMsgTxt ("matrix must be double") ;
    }
    if (nel)
    {
        /* check number of elements */
        nnel = mxGetNumberOfElements (A) ;
        if (m >= 0 && n >= 0 && m*n != nnel) mexErrMsgTxt ("wrong length") ;
    }
    else
    {
        /* check row and/or column dimensions */
        if (m >= 0 && m != mm) mexErrMsgTxt ("wrong dimension") ;
        if (n >= 0 && n != nn) mexErrMsgTxt ("wrong dimension") ;
    }
    if (square && mm != nn) mexErrMsgTxt ("matrix must be square") ;
}

/* get a MATLAB sparse matrix and convert to cs */
cs *cs_mex_get_sparse (cs *A, csi square, csi values, const mxArray *Amatlab)
{
    cs_mex_check (0, -1, -1, square, 1, values, Amatlab) ;
    A->m = mxGetM (Amatlab) ;
    A->n = mxGetN (Amatlab) ;
    A->p = (csi *) mxGetJc (Amatlab) ;
    A->i = (csi *) mxGetIr (Amatlab) ;
    A->x = values ? mxGetPr (Amatlab) : NULL ;
    A->nzmax = mxGetNzmax (Amatlab) ;
    A->nz = -1 ;    /* denotes a compressed-col matrix, instead of triplet */
    return (A) ;
}

/* return a sparse matrix to MATLAB */
mxArray *cs_mex_put_sparse (cs **Ahandle)
{
    cs *A ;
    mxArray *Amatlab ;
    if (!Ahandle || !CS_CSC ((*Ahandle))) mexErrMsgTxt ("invalid sparse matrix") ;
    A = *Ahandle ;
    Amatlab = mxCreateSparse (0, 0, 0, mxREAL) ;
    mxSetM (Amatlab, A->m) ;
    mxSetN (Amatlab, A->n) ;
    mxSetNzmax (Amatlab, A->nzmax) ;
    cs_free (mxGetJc (Amatlab)) ;
    cs_free (mxGetIr (Amatlab)) ;
    mxSetJc (Amatlab, (mwIndex *) A->p) ;  /* assign A->p pointer to MATLAB A */
    mxSetIr (Amatlab, (mwIndex *) A->i) ;
    cs_free (mxGetPr (Amatlab)) ;
    if (A->x == NULL)
    {
        /* A is a pattern only matrix; return all 1's to MATLAB */
        csi i, nz ;
        nz = A->p [A->n] ;
        A->x = cs_malloc (CS_MAX (nz,1), sizeof (double)) ;
        for (i = 0 ; i < nz ; i++)
        {
            A->x [i] = 1 ;
        }
    }
    mxSetPr (Amatlab, A->x) ;
    mexMakeMemoryPersistent (A->p) ;    /* ensure MATLAB does not free A->p */
    mexMakeMemoryPersistent (A->i) ;
    mexMakeMemoryPersistent (A->x) ;
    cs_free (A) ;                       /* frees A struct only, not A->p, etc */
    *Ahandle = NULL ;
    return (Amatlab) ;
}

/* get a MATLAB dense column vector */
double *cs_mex_get_double (csi n, const mxArray *X)
{
    cs_mex_check (0, n, 1, 0, 0, 1, X) ;
    return (mxGetPr (X)) ;
}

/* return a double vector to MATLAB */
double *cs_mex_put_double (csi n, const double *b, mxArray **X)
{
    double *x ;
    csi k ;
    *X = mxCreateDoubleMatrix (n, 1, mxREAL) ;      /* create x */
    x = mxGetPr (*X) ;
    for (k = 0 ; k < n ; k++) x [k] = b [k] ;       /* copy x = b */
    return (x) ;
}

/* get a MATLAB flint array and convert to csi */
csi *cs_mex_get_int (csi n, const mxArray *Imatlab, csi *imax, csi lo)
{
    double *p ;
    csi i, k, *C = cs_malloc (n, sizeof (csi)) ;
    cs_mex_check (1, n, 1, 0, 0, 1, Imatlab) ;
    p = mxGetPr (Imatlab) ;
    *imax = 0 ;
    for (k = 0 ; k < n ; k++)
    {
        i = p [k] ;
        C [k] = i - 1 ;
        if (i < lo) mexErrMsgTxt ("index out of bounds") ;
        *imax = CS_MAX (*imax, i) ;
    }
    return (C) ;
}

/* return an csi array to MATLAB as a flint row vector */
mxArray *cs_mex_put_int (csi *p, csi n, csi offset, csi do_free)
{
    mxArray *X = mxCreateDoubleMatrix (1, n, mxREAL) ;
    double *x = mxGetPr (X) ;
    csi k ;
    for (k = 0 ; k < n ; k++) x [k] = (p ? p [k] : k) + offset ;
    if (do_free) cs_free (p) ;
    return (X) ;
}
