function [C, sym] = get_problem (prefix, name, tol)
% [C, sym] = get_problem(prefix, name,tol)
% read a problem from a file, drop entries with abs value < tol
% tol defaults to zero if not present
%
% Example:
%    [C, sym] = get_problem ('', 'west0067') ;
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

fprintf ('\n------------------- Matrix: %s\n', name) ;

if (nargin < 2)
    tol = 0 ;
end

s = find (name == '/') ;
if (isempty (s))
    s = 0 ;
end

% load the triplet version of the matrix
T = load ([ prefix '/' name(s+1:end) ]) ;

% convert into a sparse matrix and compare with cs_sparse
A  = sparse    (T (:,1)+1, T (:,2)+1, T (:,3)) ;
A2 = cs_sparse (T (:,1)+1, T (:,2)+1, T (:,3)) ;
err = norm (A-A2,1) ;
if (err > 0)
    fprintf ('A difference: %g\n', err) ;
end

[m n] = size (A) ;
nz2 = nnz (A) ;

if (tol > 0)
    A = cs_droptol (A, tol) ;
end

% assume A is symmetric if it is upper or lower triangular
sym = is_sym (A) ;
if (sym)
    C = A + (A' - diag (diag (A))) ;
else
    C = A ;
end

fprintf ('--- Matrix: %d-by-%d, nnz: %d (sym: %d nnz %d), norm: %8.2e\n', ...
    m, n, nnz(A), sym, abs(sym)*nnz(C), norm (C,1)) ;

if (nz2 ~= nnz(A))
    fprintf ('tiny entries dropped: %d\n', nz2 - nnz(A)) 
end
