function [v,beta,xnorm] = hmake1 (x)
%HMAKE1 construct a Householder reflection
% Example:
%   [v,beta,xnorm] = hmake1 (x)
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

n = length (x) ;
if (n == 1)
    v = 1 ;
    xnorm = norm (x) ;
    if (x (1) < 0)
        beta = 2 ;
    else
        beta = 0 ;
    end
    return
end
sigma = x (2:n)'*x(2:n) ;
xnorm = sqrt (x (1)^2 + sigma) ;
v = x ;
if (sigma == 0)
    v (1) = 1 ;
    if (x (1) < 0)
        beta = 2 ;
    else
        beta = 0 ;
    end
else
    if (x (1) <= 0)
        v (1) = x(1) - xnorm ;
    else
        v (1) = -sigma / (x(1) + xnorm) ;
    end
    beta = (2*v(1)^2) / (sigma + v(1)^2) ;
    v = v / v(1) ;
end
