function [L,U] = lu_rightr (A)
%LU_RIGHTR recursive right-looking LU.
% Example:
%   [L,U] = lu_rightr (A)
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

n = size (A,1) ;
if (n == 1)
    L = 1 ;
    U = A ;
else
    u11 = A (1,1) ;                                                      % (6.4)
    u12 = A (1,2:n) ;                                                    % (6.5)
    l21 = A (2:n,1) / u11 ;                                              % (6.6)
    [L22,U22] = lu_rightr (A (2:n,2:n) - l21*u12) ;                      % (6.7)
    L = [ 1 zeros(1,n-1) ; l21 L22 ] ;
    U = [ u11 u12 ; zeros(n-1,1) U22 ] ;
end
