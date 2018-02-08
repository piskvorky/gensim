function x = cs_utsolve (U,b)                                               %#ok
%CS_UTSOLVE solve a sparse lower triangular system U'*x=b.
%   x = cs_utsolve(U,b) computes x = U'\b, U must be upper triangular with a
%   zero-free diagonal.  b must be a full vector.
%
%   Example:
%       Prob = ssget ('HB/arc130') ; A = Prob.A ; n = size (A,1) ;
%       b = rand (n,1);
%       [L U p q] = cs_lu (A) ;
%       x = cs_ltsolve (L, cs_utsolve (U, b(q))) ;   % x = L' \ (U' \ b(q)) ;
%       x (p) = x ;
%       norm (A'*x-b)
%
%   See also CS_LSOLVE, CS_LTSOLVE, CS_USOLVE, MLDIVIDE.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

error ('cs_utsolve mexFunction not found') ;
