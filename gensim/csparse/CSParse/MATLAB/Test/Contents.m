% CSparse "textbook" MATLAB M-files and mexFunctions, related to CSparse but
% not a part of CSparse itself.
%
%   M-files:
%
%   chol_downdate    - downdate a Cholesky factorization.
%   chol_left        - left-looking Cholesky factorization.
%   chol_left2       - left-looking Cholesky factorization, more details.
%   chol_right       - right-looking Cholesky factorization.
%   chol_super       - left-looking "supernodal" Cholesky factorization.
%   chol_up          - up-looking Cholesky factorization.
%   chol_update      - update a Cholesky factorization.
%   chol_updown      - update or downdate a Cholesky factorization.
%   cond1est         - 1-norm condition estimate.
%   cs_fiedler       - the Fiedler vector of a connected graph.
%   givens2          - find a Givens rotation.
%   house            - find a Householder reflection.
%   lu_left          - left-looking LU factorization.
%   lu_right         - right-looking LU factorization.
%   lu_rightp        - right-looking LU factorization, with partial pivoting.
%   lu_rightpr       - recursive right-looking LU, with partial pivoting.
%   lu_rightr        - recursive right-looking LU.
%   norm1est         - 1-norm estimate.
%   qr_givens        - Givens-rotation QR factorization.
%   qr_givens_full   - Givens-rotation QR factorization, for full matrices.
%   qr_left          - left-looking Householder QR factorization.
%   qr_right         - right-looking Householder QR factorization.
%
% mexFunctions:
%
%   cs_frand         - generate a random finite-element matrix
%   cs_ipvec         - x(p)=b
%   cs_maxtransr     - recursive maximum matching algorithm
%   cs_pvec          - x=b(p)
%   cs_reach         - non-recursive reach (interface to CSparse cs_reach)
%   cs_reachr        - recursive reach (interface to CSparse cs_reachr)
%   cs_rowcnt        - row counts for sparse Cholesky
%   cs_sparse2       - same as cs_sparse, to test cs_entry function
%
% Extensive test functions, not for normal usage:
%
%   check_if_same    - check if two inputs are identical or not
%   choldn           - Cholesky downdate
%   cholup           - Cholesky update, using Given's rotations
%   cholupdown       - Cholesky update/downdate (Bischof, Pan, and Tang method)
%   cs_q1            - construct Q from Householder vectors
%   cs_test_make     - compiles the CSparse, Demo, and Test mexFunctions.
%   dmperm_test      - test cs_dmperm
%   chol_example     - simple Cholesky factorization example
%   etree_sample     - construct a sample etree and symbolic factorization
%   gqr3             - QR factorization, based on Givens rotations
%   happly           - apply Householder reflection to a vector
%   hmake1           - construct a Householder reflection
%   mynormest1       - estimate norm(A,1), using LU factorization (L*U = P*A*Q).
%   myqr             - QR factorization using Householder reflections
%   another_colormap - try another color map
%   cspy_test        - test cspy and cs_dmspy
%   qr2              - QR factorization based on Householder reflections
%   sample_colormap  - try a colormap for use in cspy
%   signum           - compute and display the sign of a column vector x
%   sqr_example      - test cs_sqr
%   dmspy_test       - test cspy, cs_dmspy, and cs_dmperm
%   test_qr          - test various QR factorization methods
%   test_randperms   - test random permutations
%   testh            - test Householder reflections
%   test_qr1         - test QR factorizations
%   test_qrsol       - test cs_qrsol
%   test_sep         - test cs_sep, and compare with Gilbert's meshpart vtxsep
%   testall          - test all CSparse functions (run tests 1 to 28 below)
%   test1            - test cs_transpose
%   test2            - test cs_sparse
%   test3            - test cs_lsolve, cs_ltsolve, cs_usolve, cs_chol
%   test4            - test cs_multiply
%   test5            - test cs_add
%   test6            - test cs_reach, cs_reachr, cs_lsolve, cs_usolve
%   test7            - test cs_lu
%   test8            - test cs_cholsol, cs_lusol
%   test9            - test cs_qr
%   test10           - test cs_qr
%   test11           - test cs_rowcnt
%   test12           - test cs_qr and compare with svd
%   test13           - test cs_counts, cs_etree
%   test14           - test cs_droptol
%   test15           - test cs_amd
%   test16           - test cs_amd
%   test17           - test cs_qr, cs_qright, cs_q1, cs_qrleft, cs_qrsol
%   test18           - test iterative refinement after backslash
%   test19           - test cs_dmperm, cs_maxtransr, cs_dmspy, cs_scc
%   test20           - test cholupdown
%   test21           - test cs_updown
%   test22           - test cond1est
%   test23           - test cs_dmspy
%   test24           - test cs_fielder
%   test25           - test cs_nd
%   test26           - test cs_dmsol and cs_dmspy
%   test27           - test cs_qr, cs_utsolve, cs_qrsol
%   test28           - test cs_randperm, cs_dmperm

% Example:
%   help chol_update

%   Copyright 2006-2007, Timothy A. Davis
