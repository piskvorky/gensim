function [L, w] = chol_updown (L, sigma, w)
%CHOL_UPDOWN update or downdate a Cholesky factorization.
% Example:
%   [L, w] = chol_updown (L, sigma, w)
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

beta = 1 ;
n = size (L,1) ;
if (n == 1)
    L = sqrt (L*L'+sigma*w*w') ;
    return ;
end
for k = 1:n
    alpha = w(k) / L(k,k) ;
    beta2 = sqrt (beta^2 + sigma*alpha^2) ;
    gamma = sigma * alpha / (beta2 * beta) ;
    if (sigma > 0)
        % update
        delta = beta / beta2 ;
        L (k,k) = delta * L (k,k) + gamma * w (k) ;
        w1 = w (k+1:n) ;
        w (k+1:n) = w (k+1:n) - alpha * L (k+1:n,k) ;
        L (k+1:n,k) = delta * L (k+1:n,k) + gamma * w1 ;
    else
        % downdate
        delta = beta2 / beta ;
        L (k,k) = delta * L (k,k) ;
        w (k+1:n) = w (k+1:n) - alpha * L (k+1:n,k) ;
        L (k+1:n,k) = delta * L (k+1:n,k) + gamma * w (k+1:n) ;
    end
    w (k) = alpha ;
    beta = beta2 ;
end
