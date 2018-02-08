#ifndef _CS_H
#define _CS_H
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stddef.h>
#ifdef MATLAB_MEX_FILE
#include "mex.h"
#endif
#define CS_VER 3                    /* CSparse Version */
#define CS_SUBVER 2
#define CS_SUBSUB 0
#define CS_DATE "Sept 12, 2017"       /* CSparse release date */
#define CS_COPYRIGHT "Copyright (c) Timothy A. Davis, 2006-2016"

#ifdef MATLAB_MEX_FILE
#undef csi
#define csi mwSignedIndex
#endif
#ifndef csi
#define csi ptrdiff_t
#endif

/* --- primary CSparse routines and data structures ------------------------- */
typedef struct cs_sparse    /* matrix in compressed-column or triplet form */
{
    csi nzmax ;     /* maximum number of entries */
    csi m ;         /* number of rows */
    csi n ;         /* number of columns */
    csi *p ;        /* column pointers (size n+1) or col indices (size nzmax) */
    csi *i ;        /* row indices, size nzmax */
    double *x ;     /* numerical values, size nzmax */
    csi nz ;        /* # of entries in triplet matrix, -1 for compressed-col */
} cs ;

cs *cs_add (const cs *A, const cs *B, double alpha, double beta) ;
csi cs_cholsol (csi order, const cs *A, double *b) ;
cs *cs_compress (const cs *T) ;
csi cs_dupl (cs *A) ;
csi cs_entry (cs *T, csi i, csi j, double x) ;
csi cs_gaxpy (const cs *A, const double *x, double *y) ;
cs *cs_load (FILE *f) ;
csi cs_lusol (csi order, const cs *A, double *b, double tol) ;
cs *cs_multiply (const cs *A, const cs *B) ;
double cs_norm (const cs *A) ;
csi cs_print (const cs *A, csi brief) ;
csi cs_qrsol (csi order, const cs *A, double *b) ;
cs *cs_transpose (const cs *A, csi values) ;
/* utilities */
void *cs_calloc (csi n, size_t size) ;
void *cs_free (void *p) ;
void *cs_realloc (void *p, csi n, size_t size, csi *ok) ;
cs *cs_spalloc (csi m, csi n, csi nzmax, csi values, csi triplet) ;
cs *cs_spfree (cs *A) ;
csi cs_sprealloc (cs *A, csi nzmax) ;
void *cs_malloc (csi n, size_t size) ;

/* --- secondary CSparse routines and data structures ----------------------- */
typedef struct cs_symbolic  /* symbolic Cholesky, LU, or QR analysis */
{
    csi *pinv ;     /* inverse row perm. for QR, fill red. perm for Chol */
    csi *q ;        /* fill-reducing column permutation for LU and QR */
    csi *parent ;   /* elimination tree for Cholesky and QR */
    csi *cp ;       /* column pointers for Cholesky, row counts for QR */
    csi *leftmost ; /* leftmost[i] = min(find(A(i,:))), for QR */
    csi m2 ;        /* # of rows for QR, after adding fictitious rows */
    double lnz ;    /* # entries in L for LU or Cholesky; in V for QR */
    double unz ;    /* # entries in U for LU; in R for QR */
} css ;

typedef struct cs_numeric   /* numeric Cholesky, LU, or QR factorization */
{
    cs *L ;         /* L for LU and Cholesky, V for QR */
    cs *U ;         /* U for LU, R for QR, not used for Cholesky */
    csi *pinv ;     /* partial pivoting for LU */
    double *B ;     /* beta [0..n-1] for QR */
} csn ;

typedef struct cs_dmperm_results    /* cs_dmperm or cs_scc output */
{
    csi *p ;        /* size m, row permutation */
    csi *q ;        /* size n, column permutation */
    csi *r ;        /* size nb+1, block k is rows r[k] to r[k+1]-1 in A(p,q) */
    csi *s ;        /* size nb+1, block k is cols s[k] to s[k+1]-1 in A(p,q) */
    csi nb ;        /* # of blocks in fine dmperm decomposition */
    csi rr [5] ;    /* coarse row decomposition */
    csi cc [5] ;    /* coarse column decomposition */
} csd ;

csi *cs_amd (csi order, const cs *A) ;
csn *cs_chol (const cs *A, const css *S) ;
csd *cs_dmperm (const cs *A, csi seed) ;
csi cs_droptol (cs *A, double tol) ;
csi cs_dropzeros (cs *A) ;
csi cs_happly (const cs *V, csi i, double beta, double *x) ;
csi cs_ipvec (const csi *p, const double *b, double *x, csi n) ;
csi cs_lsolve (const cs *L, double *x) ;
csi cs_ltsolve (const cs *L, double *x) ;
csn *cs_lu (const cs *A, const css *S, double tol) ;
cs *cs_permute (const cs *A, const csi *pinv, const csi *q, csi values) ;
csi *cs_pinv (const csi *p, csi n) ;
csi cs_pvec (const csi *p, const double *b, double *x, csi n) ;
csn *cs_qr (const cs *A, const css *S) ;
css *cs_schol (csi order, const cs *A) ;
css *cs_sqr (csi order, const cs *A, csi qr) ;
cs *cs_symperm (const cs *A, const csi *pinv, csi values) ;
csi cs_updown (cs *L, csi sigma, const cs *C, const csi *parent) ;
csi cs_usolve (const cs *U, double *x) ;
csi cs_utsolve (const cs *U, double *x) ;
/* utilities */
css *cs_sfree (css *S) ;
csn *cs_nfree (csn *N) ;
csd *cs_dfree (csd *D) ;

/* --- tertiary CSparse routines -------------------------------------------- */
csi *cs_counts (const cs *A, const csi *parent, const csi *post, csi ata) ;
double cs_cumsum (csi *p, csi *c, csi n) ;
csi cs_dfs (csi j, cs *G, csi top, csi *xi, csi *pstack, const csi *pinv) ;
csi cs_ereach (const cs *A, csi k, const csi *parent, csi *s, csi *w) ;
csi *cs_etree (const cs *A, csi ata) ;
csi cs_fkeep (cs *A, csi (*fkeep) (csi, csi, double, void *), void *other) ;
double cs_house (double *x, double *beta, csi n) ;
csi cs_leaf (csi i, csi j, const csi *first, csi *maxfirst, csi *prevleaf,
    csi *ancestor, csi *jleaf) ;
csi *cs_maxtrans (const cs *A, csi seed) ;
csi *cs_post (const csi *parent, csi n) ;
csi *cs_randperm (csi n, csi seed) ;
csi cs_reach (cs *G, const cs *B, csi k, csi *xi, const csi *pinv) ;
csi cs_scatter (const cs *A, csi j, double beta, csi *w, double *x, csi mark,
    cs *C, csi nz) ;
csd *cs_scc (cs *A) ;
csi cs_spsolve (cs *G, const cs *B, csi k, csi *xi, double *x,
    const csi *pinv, csi lo) ;
csi cs_tdfs (csi j, csi k, csi *head, const csi *next, csi *post,
    csi *stack) ;
/* utilities */
csd *cs_dalloc (csi m, csi n) ;
csd *cs_ddone (csd *D, cs *C, void *w, csi ok) ;
cs *cs_done (cs *C, void *w, void *x, csi ok) ;
csi *cs_idone (csi *p, cs *C, void *w, csi ok) ;
csn *cs_ndone (csn *N, cs *C, void *w, void *x, csi ok) ;

#define CS_MAX(a,b) (((a) > (b)) ? (a) : (b))
#define CS_MIN(a,b) (((a) < (b)) ? (a) : (b))
#define CS_FLIP(i) (-(i)-2)
#define CS_UNFLIP(i) (((i) < 0) ? CS_FLIP(i) : (i))
#define CS_MARKED(w,j) (w [j] < 0)
#define CS_MARK(w,j) { w [j] = CS_FLIP (w [j]) ; }
#define CS_CSC(A) (A && (A->nz == -1))
#define CS_TRIPLET(A) (A && (A->nz >= 0))
#endif
