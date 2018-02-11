function [L,U,P] = lu_rightpr (A)
%LU_RIGHTPR recursive right-looking LU, with partial pivoting.
%
% Example:
%   [L,U,P] = lu_rightpr (A)
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

n = size (A,1) ;
if (n == 1)
    P = 1 ;
    L = 1 ;
    U = A ;
else
    [x,i] = max (abs (A (1:n,1))) ;                           % partial pivoting
    P1 = eye (n) ;
    P1 ([1 i],:) = P1 ([i 1], :) ;
    A = P1*A ;
    u11 = A (1,1) ;                                           % (6.10)
    u12 = A (1,2:n) ;                                         % (6.11)
    l21 = A (2:n,1) / u11 ;                                   % (6.12)
    [L22,U22,P2] = lu_rightpr (A (2:n,2:n) - l21*u12) ;       % (6.9) or (6.13)
    o = zeros(1,n-1) ;
    L = [ 1 o ; P2*l21 L22 ] ;                                % (6.14)
    U = [ u11 u12 ; o' U22 ] ;
    P = [ 1 o ; o' P2] * P1 ;
end
