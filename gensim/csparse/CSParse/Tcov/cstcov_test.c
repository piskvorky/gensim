#include "cs_demo.h"
/* cs_test: read a matrix and run cs_demo2 and cs_demo3, using malloc tests. */

#include "cstcov_malloc_test.h"

int main (int argc, char **argv)
{
    FILE *f ;
    problem *Prob ;
    int trials, ok, demo ;
    if (argc < 2) return (-1) ;
    printf ("cs_test, file: %s\n", argv [1]) ;
    for (demo = 2 ; demo <= 3 ; demo++)
    {
        printf ("demo: %g\n", (double) demo) ;
        for (trials = 0 ; trials < 4000 ; trials++)
        {
            malloc_count = trials ;
            f = fopen (argv [1], "r") ;
            if (!f) return (-1) ;
            Prob = get_problem (f, (demo == 2) ? 1e-14 : 0) ;
            fclose (f) ;
            if (Prob) ok = (demo == 2) ? demo2 (Prob) : demo3 (Prob) ;
            free_problem (Prob) ;
            if (malloc_count > 0) break ;
        }
        printf ("demo %g # trials: %g\n", (double) demo, (double) trials) ;
    }
    return (0) ;
}
