function x = cs_usolve (U,b)                                                %#ok
%CS_USOLVE solve a sparse upper triangular system U*x=b.
%   x = cs_usolve(U,b) computes x = U\b, U must be lower triangular with a
%   zero-free diagonal.  b must be a column vector.  x is full if b is full.
%   If b is sparse, x is sparse but nonzero pattern of x is NOT sorted (it is
%   returned in topological order).
%
%   Example:
%       Prob = ssget ('HB/arc130') ; A = Prob.A ; n = size (A,1) ;
%       b = rand (n,1);
%       [L U p q] = cs_lu (A) ;
%       x = cs_usolve (U, cs_lsolve (L, b(p))) ;   % x = U \ (L \ b(p)) ;
%       x (q) = x ;
%       norm (A*x-b)
%
%   See also CS_LSOLVE, CS_LTSOLVE, CS_UTSOLVE, MLDIVIDE.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

error ('cs_usolve mexFunction not found') ;
