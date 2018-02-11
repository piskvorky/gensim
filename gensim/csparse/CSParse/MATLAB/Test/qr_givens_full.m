function R = qr_givens_full (A)
%QR_GIVENS_FULL Givens-rotation QR factorization, for full matrices.
% Example:
%   R = qr_givens_full (A)
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

[m n] = size (A) ;
for i = 2:m
    for k = 1:min(i-1,n)
        A ([k i],k:n) = givens2 (A(k,k), A(i,k)) * A ([k i],k:n) ;
        A (i,k) = 0 ;
    end
end
R = A ;
