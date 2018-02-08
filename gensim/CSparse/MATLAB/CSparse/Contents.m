% CSparse: a Concise Sparse matrix Package.
%
%   Matrices used in CSparse must in general be either sparse and real,
%   or dense vectors.  Ordering methods can accept any sparse matrix.
%
%   cs_add       - sparse matrix addition.
%   cs_amd       - approximate minimum degree ordering.
%   cs_chol      - sparse Cholesky factorization.
%   cs_cholsol   - solve A*x=b using a sparse Cholesky factorization.
%   cs_counts    - column counts for sparse Cholesky factor L.
%   cs_dmperm    - maximum matching or Dulmage-Mendelsohn permutation.
%   cs_dmsol     - x=A\b using the coarse Dulmage-Mendelsohn decomposition.
%   cs_dmspy     - plot the Dulmage-Mendelsohn decomposition of a matrix.
%   cs_droptol   - remove small entries from a sparse matrix.
%   cs_esep      - find an edge separator of a symmetric matrix A
%   cs_etree     - elimination tree of A or A'*A.
%   cs_gaxpy     - sparse matrix times vector.
%   cs_lsolve    - solve a sparse lower triangular system L*x=b.
%   cs_ltsolve   - solve a sparse upper triangular system L'*x=b.
%   cs_lu        - sparse LU factorization, with fill-reducing ordering.
%   cs_lusol     - solve Ax=b using LU factorization.
%   cs_make      - compiles CSparse for use in MATLAB.
%   cs_multiply  - sparse matrix multiply.
%   cs_nd        - generalized nested dissection ordering.
%   cs_nsep      - find a node separator of a symmetric matrix A.
%   cs_permute   - permute a sparse matrix.
%   cs_print     - print the contents of a sparse matrix.
%   cs_qr        - sparse QR factorization.
%   cs_qleft     - apply Householder vectors on the left.
%   cs_qright    - apply Householder vectors on the right.
%   cs_qrsol     - solve a sparse least-squares problem.
%   cs_randperm  - random permutation.
%   cs_sep       - convert an edge separator into a node separator.
%   cs_scc       - strongly-connected components of a square sparse matrix.
%   cs_scc2      - cs_scc, or connected components of a bipartite graph.
%   cs_sparse    - convert a triplet form into a sparse matrix.
%   cs_sqr       - symbolic sparse QR factorization.
%   cs_symperm   - symmetric permutation of a symmetric matrix.
%   cs_transpose - transpose a real sparse matrix.
%   cs_updown    - rank-1 update/downdate of a sparse Cholesky factorization.
%   cs_usolve    - solve a sparse upper triangular system U*x=b.
%   cs_utsolve   - solve a sparse lower triangular system U'*x=b.
%   cspy         - plot a matrix in color.
%   ccspy        - plot the connected components of a matrix.

% Example:
%   help cs_add

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

% helper function:
%   cs_must_compile - return 1 if source code f must be compiled, 0 otherwise
