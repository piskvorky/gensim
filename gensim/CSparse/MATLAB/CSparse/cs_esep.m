function [a,b] = cs_esep (A)
%CS_ESEP find an edge separator of a symmetric matrix A
%   [a,b] = cs_esep(A) finds a edge separator s that splits the graph of A
%   into two parts a and b of roughly equal size.  The edge separator is the
%   set of entries in A(a,b).
%
%   Example:
%       Prob = ssget ('HB/bcsstk01') ; A = Prob.A ;
%       [a,b] = cs_esep (A) ;
%       cspy (A (a,b)) ;
%
%   See also CS_NSEP, CS_SEP, CS_ND, SYMRCM.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

p = symrcm (A) ;
n2 = fix (size(A,1)/2) ;
a = p (1:n2) ;
b = p (n2+1:end) ;
