function [V,Beta,R] = qr2 (A)
%QR2 QR factorization based on Householder reflections
%
% Example:
%   [V,beta,R] = qr2 (A)
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

[m n] = size (A) ;
V = zeros (m,n) ;
Beta = zeros (1,n) ;
for k = 1:n
    % [v,beta,s] = gallery ('house', A (k:m,k), 2) ;
    [v,beta] = house (A (k:m,k)) ;
    V (k:m,k) = v ;
    Beta (k) = beta ;
    A (k:m,k:n) = A (k:m,k:n) - v * (beta * (v' * A (k:m,k:n))) ;
end
R = triu (A) ;
