function p = cs_amd (A,order)                                               %#ok
%CS_AMD approximate minimum degree ordering.
%   p = cs_amd(A) finds a minimum degree ordering of A+A'
%   p = cs_amd(A,order):
%       order = 1:  same as cs_amd(A)
%       order = 2:  minimum degree ordering of S'*S where S = A except that
%                   "dense" rows of A are removed from S (a dense row has
%                   10*sqrt(n) or more entries where n = size(A,2)).  Similar
%                   to p = colamd(A), except that colamd does not form A'*A
%                   explicitly.
%       order = 3:  minimum degree ordering of A'*A.  Similar to colamd(A,[n m])
%                   where [m n] = size(A), except that colamd does not form A'*A
%                   explicitly.
%   Example:
%       Prob = ssget ('HB/bcsstk01') ; A = Prob.A ;
%       p = cs_amd (A) ;
%       nnz (chol (A))
%       nnz (chol (A (p,p)))
%
%   See also AMD, COLAMD, SYMAMD.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

error ('cs_amd mexFunction not found') ;
