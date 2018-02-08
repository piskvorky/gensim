function [v,beta,s] = house (x)
%HOUSE find a Householder reflection.
% Example:
%   [v,beta,s] = house (x)
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

n = length (x) ;
if (n == 1)
    sigma = 0 ;
else
    sigma = x (2:n)' * x (2:n) ;
end
v = x ;
if (sigma == 0)
    s = x (1) ;
    v (1) = 0 ;
    beta = 0 ;
else
    s = sqrt (x(1)^2 + sigma) ;
    if (x (1) <= 0)
        v (1) = x (1) - s ;
    else
        v (1) = -sigma / (x (1) + s) ;
    end
    beta = -1 / (s * v(1)) ;
end
