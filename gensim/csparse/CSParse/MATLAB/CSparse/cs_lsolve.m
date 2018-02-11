function x = cs_lsolve (L,b)                                                %#ok
%CS_LSOLVE solve a sparse lower triangular system L*x=b.
%   x = cs_lsolve(L,b) computes x = L\b, L must be lower triangular with a
%   zero-free diagonal.  b must be a column vector.  x is full if b is full.
%   If b is sparse, x is sparse but the nonzero pattern of x is NOT sorted (it
%   is returned in topological order).
%
%   Example:
%       Prob = ssget ('HB/bcsstk01') ; L = cs_chol (Prob.A) ; n = size (L,1) ;
%       b = rand (n,1) ; x = cs_lsolve (L,b) ; norm (L*x-b)
%
%   See also CS_LTSOLVE, CS_USOLVE, CS_UTSOLVE, MLDIVIDE.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

error ('cs_lsolve mexFunction not found') ;
