function [L,U,p,q] = cs_lu (A,tol)                                          %#ok
%CS_LU sparse LU factorization, with fill-reducing ordering.
%   [L,U,p] = cs_lu(A) factorizes A(p,:) into L*U using no fill-reducing
%   ordering.
%
%   [L,U,p] = cs_lu(A,tol) factorizes A(p,:) into L*U using no fill-reducing
%   ordering.  Entries on the diagonal are given preference in partial pivoting.
%
%   [L,U,p,q] = cs_lu(A) factorizes A(p,q) into L*U using a fill-reducing
%   ordering q = cs_amd(A,2).  Normal partial pivoting is used.
%
%   [L,U,p,q] = cs_lu(A,tol) factorizes A(p,q) into L*U, using a fill-reducing
%   ordering q = cs_amd(A,1).  Entries on the diagonal are given preference in
%   partial pivoting.  With a pivot tolerance tol, the entries in L have
%   magnitude 1/tol or less.  tol = 1 is normal partial pivoting (with
%   q = cs_amd(A)).  tol = 0 ensures p = q.  0<tol<1 is relaxed partial
%   pivoting; the diagonal is selected if it is >= tol * max(abs(A(:,k))).
%
%   Example:
%       Prob = ssget ('HB/arc130') ; A = Prob.A ; [L,U,p,q] = cs_lu (A) ;
%       cspy (A (p,q)) ; cspy (L+U) ;
%       norm (L*U - A(p,q), 1)
%
%   See also CS_AMD, LU, UMFPACK, AMD, COLAMD.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

error ('cs_lu mexFunction not found') ;
