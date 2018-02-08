function [V,beta,p,R,q] = cs_qr (A)                                         %#ok
%CS_QR sparse QR factorization (Householder-based).
%   [V,beta,p,R] = cs_qr(A) computes the QR factorization of A(p,:).
%   [V,beta,p,R,q] = cs_qr(A) computes the QR factorization of A(p,q).
%   The V, beta, and p terms represent the Householder vectors and coefficients.
%   The fill-reducing ordering q is found via q = cs_amd(A,3).
%   The orthogonal factor Q can be obtained via
%   Q = cs_qright(V,beta,p,speye(size(V,1))), in which case Q*R=A(:,q) is the
%   resulting factorization (the permutation p is folded into Q).  A must be
%   m-by-n with m >= n.  If A is structurally rank deficient, additional empty
%   rows may have been added to V and R.  Note that V is typically much sparser
%   than Q.
%
%   Example:
%
%       Prob = ssget ('HB/well1033') ; A = Prob.A ; [m n] = size (A) ;
%       b = rand (m,1) ;
%       [V,beta,p,R,q] = cs_qr (A) ; % QR factorization of A(p,q)
%       b1 = cs_qleft (V, beta, p, b) ;
%       x1 = R (1:n,1:n) \ b1 (1:n) ;
%       x1 (q) = x1 ;
%       x2 = A\b ;
%       norm (x1-x2)
%       Q = cs_qright(V,beta,p,speye(size(V,1))) ;  % Note: p accounted for in Q
%       norm (Q*R-A(:,q),1)
%       fprintf ('nnz(R) %d, nnz(V) %d, nnz(Q) %d\n', nnz(R), nnz(V), nnz(Q)) ;
%
%   See also CS_AMD, CS_QRIGHT, CS_QR, CS_DMPERM, QR, COLAMD.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

error ('cs_qr mexFunction not found') ;
