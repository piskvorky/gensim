function X = cs_qleft (V, Beta, p, Y)
%CS_QLEFT apply Householder vectors on the left.
%   X = cs_qleft(V,Beta,p,Y) computes X = Hn*...*H2*H1*P*Y = Q'*Y where Q is
%   represented by the Householder vectors V, coefficients Beta, and
%   permutation p.  p can be [], which denotes the identity permutation.
%
%   Example:
%       Prob = ssget ('HB/well1033') ; A = Prob.A ; [m n] = size (A) ;
%       b = rand (m,1) ;
%       [V,beta,p,R] = cs_qr (A) ; % QR factorization of A(p,:)
%       b1 = cs_qleft (V, beta, p, b) ;
%       x1 = R (1:n,1:n) \ b1 (1:n) ;
%       x2 = A\b ;
%       norm (x1-x2)
%      
%   See also CS_QR, CS_QRIGHT.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

[m2 n] = size (V) ;
[m ny] = size (Y) ;
X = Y ;
if (m2 > m)
    if (issparse (Y))
        X = [X ; sparse(m2-m,ny)] ;
    else
        X = [X ; zeros(m2-m,ny)] ;
    end
end
if (~isempty (p))
    X = X (p,:) ;
end
for k = 1:n
    X = X - V (:,k) * (Beta (k) * (V (:,k)' * X)) ;
end
