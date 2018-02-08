function [s,a,b] = cs_nsep (A)
%CS_NSEP find a node separator of a symmetric matrix A.
%   [s,a,b] = cs_nsep(A) finds a node separator s that splits the graph of A
%   into two parts a and b of roughly equal size.  If A is unsymmetric, use
%   cs_nsep(A|A').  The permutation p = [a b s] is a one-level dissection of A.
%
%   Example:
%       A = delsq (numgrid ('L', 10)) ;    % smaller version as used in 'bench'
%       [s a b] = cs_nsep (A) ; p = [a b s] ;
%       cspy (A (p,p)) ;
%
%   See also CS_SEP, CS_ESEP, CS_ND.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

[a b] = cs_esep (A) ;
[s a b] = cs_sep (A, a, b) ;
