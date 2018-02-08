function [vnz,rnz,parent,c,leftmost,p,q] = cs_sqr (A)                       %#ok
%CS_SQR symbolic sparse QR factorization.
%   [vnz,rnz,parent,c,leftmost,p] = cs_sqr(A): symbolic QR of A(p,:).
%   [vnz,rnz,parent,c,leftmost,p,q] = cs_sqr(A) computes the symbolic QR
%   factorization of A(p,q).  The fill-reducing ordering q is found via
%   q = cs_amd(A,3).
%
%   vnz is the number of entries in the matrix of Householder vectors, V.
%   rnz is the number of entries in R.  parent is elimination tree.
%   c(i) is the number of entries in R(i,:).  leftmost(i) = min(find(A(i,q))).
%   p is the row permutation used to ensure R has a symbolically zero-free
%   diagonal (it can be larger than m if A is structurally rank deficient).
%   q is the fill-reducing ordering, if requested.
%
%   Example:
%       Prob = ssget ('HB/ibm32') ; A = Prob.A ;
%       [vnz, rnz, parent, c, leftmost, p, q] = cs_sqr (A) ;
%       cspy (A (p,q)) ;
%
%   See also CS_AMD, CS_QR.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

error ('cs_sqr mexFunction not found') ;
