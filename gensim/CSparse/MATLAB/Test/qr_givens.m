function R = qr_givens (A)
%QR_GIVENS Givens-rotation QR factorization.
% Example:
%   R = qr_givens (A)
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

[m n] = size (A) ;
parent = cs_etree (sparse (A), 'col') ;
A = full (A) ;
for i = 2:m
    k = min (find (A (i,:))) ;                                              %#ok
    if (isempty (k))
        continue ;
    end
    while (k > 0 & k <= min (i-1,n))                                        %#ok
        A ([k i],k:n) = givens2 (A(k,k), A(i,k)) * A ([k i],k:n) ;
        A (i,k) = 0 ;
        k = parent (k) ;
    end
end
R = sparse (A) ;
