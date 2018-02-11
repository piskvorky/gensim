function est = norm1est (L,U,P,Q)
%NORM1EST 1-norm estimate.
% Example:
%   est = norm1est (L,U,P,Q)
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

n = size (L,1) ;
for k = 1:5
    if (k == 1)
        est = 0 ;
        x = ones (n,1) / n ;
        jold = -1 ;
    else
        j = min (find (abs (x) == norm (x,inf))) ;                          %#ok
        if (j == jold)
            break
        end ;
        x = zeros (n,1) ;
        x (j) = 1 ;
        jold = j ;
    end
    x = Q * (U \ (L \ (P*x))) ;
    est_old = est ;
    est = norm (x,1) ;
    if (k > 1 & est <= est_old)                                             %#ok
        break
    end ;
    s = ones (n,1) ;
    s (find (x < 0)) = -1 ;                                                 %#ok
    x = P' * (L' \ (U' \ (Q'*s))) ;
end
