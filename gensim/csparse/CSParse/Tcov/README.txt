CSparse/Tcov:  comprehensive test coverage for CSparse.  Requires Linux.
Type "make" to compile, and then "make run" to run the tests.
The test coverage is in cover.out.  The test output is
printed on stdout, except for cs_test (which prints its output in various
*.out files).

If the test is successful, the last line printed should be
"statements not yet tested: 0", and all printed residuals should be small.

Note that you will get warnings about unused parameters for some functions.
These warnings can be safely ignored.  They are parameters for functions that
are passed to cs_fkeep, and all functions used in this manner must have the
same calling sequence, even if some of the parameters are not used.

Timothy A. Davis, http://www.suitesparse.com
