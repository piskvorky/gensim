function test3
%TEST3 test cs_lsolve, cs_ltsolve, cs_usolve, cs_chol
%
% Example:
%   test3
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

index = ssget ;
[ignore f] = sort (max (index.nrows, index.ncols)) ;
f = f (1:100) ;

clf
% f = f(1)

for i = f
    Prob = ssget (i) ;
    disp (Prob) ;
    A = Prob.A ;
    [m n] = size (A) ;
    if (~isreal (A) | m ~= n)                                               %#ok
        continue
    end

    A = A*A' + 2*n*speye (n) ;
    try
        p = amd (A) ;
    catch
        p = symamd (A) ;
    end
    try
        L0 = chol (A)' ;
    catch
        continue
    end
    b = rand (n,1) ;

    C = A(p,p) ;
    c = condest (C) ;
    fprintf ('condest: %g\n', c) ;

    x1 = L0\b ;
    x2 = cs_lsolve (L0,b) ;
    err = norm (x1-x2,1) ;
    if (err > 1e-12 * c)
        error ('!') ;
    end

    x1 = L0'\b ;
    x2 = cs_ltsolve (L0,b) ;
    err = norm (x1-x2,1) ;
    if (err > 1e-10 * c)
        error ('!') ;
    end

    U = L0' ;

    x1 = U\b ;
    x2 = cs_usolve (U,b) ;
    err = norm (x1-x2,1) ;
    if (err > 1e-10 * c)
        error ('!') ;
    end

    L2 = cs_chol (A) ;
    subplot (2,3,1) ; spy (L0) ;
    subplot (2,3,4) ; spy (L2) ;
    err = norm (L0-L2,1) ;
    if (err > 1e-8 * c)
        error ('!') ;
    end

    L1 = chol (C)' ;
    L2 = cs_chol (C) ;
    subplot (2,3,2) ; spy (L1) ;
    subplot (2,3,5) ; spy (L2) ;
    err = norm (L1-L2,1) ;
    if (err > 1e-8 * c)
        error ('!') ;
    end

    [L3,p] = cs_chol (A) ;
    C = A(p,p) ;
    L4 = chol (C)' ;
    subplot (2,3,3) ; spy (L4) ;
    subplot (2,3,6) ; spy (L3) ;
    err = norm (L4-L3,1) ;
    if (err > 1e-8 * c)
        error ('!') ;
    end

    drawnow

end

