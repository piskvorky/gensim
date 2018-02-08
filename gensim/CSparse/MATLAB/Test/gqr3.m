function R = gqr3 (A)
%GQR3 QR factorization, based on Givens rotations
%
% Example:
%   R = gqr3 (A)
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

[m n] = size (A) ;

% parent = cs_etree (sparse (A), 'col') ;

for i = 2:m
    % i
    for k = 1:min(i-1,n)
    % k
        % Givens rotation to zero out A(i,k) using A(k,k)
        G = givens2 (A(k,k), A(i,k)) ;
        A ([k i],k:n) = G * A ([k i],k:n) ;
        A (i,k) = 0 ;
        % fprintf ('A(21,25)=%g\n', A(21,25)) ;
        % if (A(21,25) ~= 0)
            % pause
        % end
    end
end
R = A ;
