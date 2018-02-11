function test15
%TEST15 test cs_amd
%
% Example:
%   test15
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

rand ('state', 0) ;
randn ('state', 0) ;
clf

for trials = 1:100
    n = fix (200 * rand (1)) ;
    d = 0.05 * rand (1) ;
    A = sprandn (n, n, d) ;

    % add a randomly placed dense column
    k = fix (n * rand (1)) ;
    k = max (1, k) ;
    k = min (n, k) ;
    A (:,k) = 1 ;

    try
        p0 = amd (A) ;
    catch
        p0 = symamd (A) ;
    end
    p1 = cs_amd (A) ;

    if (any (sort (p1) ~= 1:n))
        error ('not perm!') ;
    end

    C = A+A' + speye (n) ;
    lnz0 = sum (symbfact (C (p0,p0))) ;
    lnz1 = sum (symbfact (C (p1,p1))) ;
    subplot (1,3,1) ; spy (C)
    subplot (1,3,2) ; spy (C (p0,p0))
    subplot (1,3,3) ; spy (C (p1,p1))
    fprintf ('n %4d nz %6d lnz %6d %6d\n', n, nnz(A), lnz0, lnz1) ;
    drawnow

end

