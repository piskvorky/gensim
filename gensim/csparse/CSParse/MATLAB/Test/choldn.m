function L = choldn (Lold,w)
%CHOLDN Cholesky downdate
% given Lold and w, compute L so that L*L' = Lold*Lold' - w*w'
%
% Example:
%   L = cholnd (Lold,w)
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

n = size (Lold,1) ;
L = Lold ;

alpha = 1 ;
beta = 1 ;

wold = w ;
wnew = zeros (n,1) ;

for i = 1:n

    a = w (i) / L(i,i) ;
    alpha = alpha - a^2 ;
    if (alpha <= 0)
        error ('not pos def') ;
    end
    beta_new = sqrt (alpha) ;
    b = beta_new / beta ;
    c = (a / (beta*beta_new)) ;
    beta = beta_new ;

    % L (i,i) = b * L (i,i) ;

    wnew (i) = a ;

    for k = i:n
        w (k)   = w (k) - a * L (k,i) ;
        L (k,i) = b * L (k,i) - c * w(k) ;
    end

end

% w
% wnew
disp (wnew - Lold\wold)
