#include "cs.h"
int main (void)
{
    cs *T, *A, *Eye, *AT, *C, *D ;
    csi i, m ;
    T = cs_load (stdin) ;               /* load triplet matrix T from stdin */
    printf ("T:\n") ; cs_print (T, 0) ; /* print T */
    A = cs_compress (T) ;               /* A = compressed-column form of T */
    printf ("A:\n") ; cs_print (A, 0) ; /* print A */
    cs_spfree (T) ;                     /* clear T */
    AT = cs_transpose (A, 1) ;          /* AT = A' */
    printf ("AT:\n") ; cs_print (AT, 0) ; /* print AT */
    m = A ? A->m : 0 ;                  /* m = # of rows of A */
    T = cs_spalloc (m, m, m, 1, 1) ;    /* create triplet identity matrix */
    for (i = 0 ; i < m ; i++) cs_entry (T, i, i, 1) ;
    Eye = cs_compress (T) ;             /* Eye = speye (m) */
    cs_spfree (T) ;
    C = cs_multiply (A, AT) ;           /* C = A*A' */
    D = cs_add (C, Eye, 1, cs_norm (C)) ;   /* D = C + Eye*norm (C,1) */
    printf ("D:\n") ; cs_print (D, 0) ; /* print D */
    cs_spfree (A) ;                     /* clear A AT C D Eye */
    cs_spfree (AT) ;
    cs_spfree (C) ;
    cs_spfree (D) ;
    cs_spfree (Eye) ;
    return (0) ;
}
