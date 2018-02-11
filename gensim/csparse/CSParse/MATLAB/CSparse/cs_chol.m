function [L,p] = cs_chol (A,drop)                                           %#ok
%CS_CHOL sparse Cholesky factorization.
%   L = cs_chol(A) is the same as L = chol(A)', using triu(A).
%   [L,p] = cs_chol(A) first orders A with p=cs_amd(A), so that L*L' = A(p,p).
%   A second optional input argument controls whether or not numerically zero
%   entries are removed from L.  cs_chol(A) and cs_chol(A,1) drop them;
%   cs_chol(A,0) keeps them.  They must be kept for cs_updown to work properly.
%
%   Example:
%       Prob = ssget ('HB/bcsstk01') ; A = Prob.A ; [L,p] = cs_chol (A) ;
%       cspy (A (p,p)) ;
%       cspy (L) ;
%
%   See also CS_AMD, CS_UPDOWN, CHOL, AMD, SYMAMD.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

error ('cs_chol mexFunction not found') ;
