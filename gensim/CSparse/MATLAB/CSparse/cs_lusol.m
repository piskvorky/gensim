function x = cs_lusol (A,b,order,tol)                                       %#ok
%CS_LUSOL solve Ax=b using LU factorization.
%   x = cs_lusol(A,b) computes x = A\b, where A is sparse and square, and b is a
%   full vector.  The ordering cs_amd(A,2) is used.
%
%   x = cs_lusol(A,b,1) also computes x = A\b, but uses the cs_amd(A) ordering
%   with diagonal preference (tol=0.001).
%
%   x = cs_lusol(A,b,order,tol) allows both the ordering and tolerance to be
%   defined.  The ordering defaults to 1, and tol defaults to 1.
%   ordering: 0: natural, 1: amd(A+A'), 2: amd(S'*S) where S=A except with no
%   dense rows, 3: amd(A'*A).
%
%   Example:
%       Prob = ssget ('HB/arc130') ; A = Prob.A ; n = size(A,1) ;
%       b = rand (n,1) ; x = cs_lusol (A,b) ; norm (A*x-b)
%
%   See also CS_LU, CS_AMD, CS_CHOLSOL, CS_QRSOL, MLDIVIDE.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

error ('cs_lusol mexFunction not found') ;
