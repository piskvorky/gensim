function L = cholupdown (Lold, sigma, w)
%CHOLUPDOWN Cholesky update/downdate (Bischof, Pan, and Tang method)
% Example:
%   L = cholupdown (Lold, sigma, w)
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

beta = 1 ;
n = size (Lold,1) ;
L = Lold ;
% x = weros (n,1) ;
worig = w ;

for k = 1:n

    alpha = w(k) / L(k,k) ;
    beta_new = sqrt (beta^2 + sigma*alpha^2) ;
    gamma = alpha / (beta_new * beta) ;

    if (sigma < 0)

        % downdate
        bratio = beta_new / beta ;
        w (k+1:n) = w (k+1:n) - alpha * L (k+1:n,k) ;
        L (k,k) = bratio * L (k,k) ;
        L (k+1:n,k) = bratio * L (k+1:n,k) - gamma*w(k+1:n) ;

    else

        % update
        bratio = beta / beta_new ;

%       wold = w (k+1:n) ;
%       w (k+1:n) = w (k+1:n) - alpha * L (k+1:n,k) ;
%       L (k    ,k) = bratio * L (k    ,k) + gamma*w(k) ;
%       L (k+1:n,k) = bratio * L (k+1:n,k) + gamma*wold ;

        L (k,k) = bratio * L (k,k) + gamma*w(k) ;
        for i = k+1:n 

            wold = w (i) ;
            w (i) = w (i) - alpha * L (i,k) ;
            L (i,k) = bratio * L (i,k) + gamma*wold ;

        end

    end

    w (k) = alpha ;

    beta = beta_new ;

end

norm (w-(Lold\worig))
