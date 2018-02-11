function test8
%TEST8 test cs_cholsol, cs_lusol
%
% Example:
%   test8
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

index = ssget ;
[ignore f] = sort (max (index.nrows, index.ncols)) ;
f = f (1:100) ;

% f = f(1)

for i = f
    Prob = ssget (i) ;
    disp (Prob) ;
    A = Prob.A ;
    [m n] = size (A) ;
    if (~isreal (A) | m ~= n)                                               %#ok
        continue
    end

    spd = 0 ;
    if (m == n)
        if (nnz (A-A') == 0)
            try
                p = amd (A) ;
            catch
                p = symamd (A) ;
            end
            [R,p] = chol (A (p,p)) ;
            spd = (p == 0) ;
        end
    end

    if (spd)
        C = A ;
    else
        C = A*A' + n*speye (n) ;
        try
            p = amd (C) ;
        catch
            p = symamd (C) ;
        end
        try
            R = chol (C (p,p)) ;
        catch
            continue
        end
    end

    b = rand (n,1) ;

    x1 = C\b ;
    x2 = cs_cholsol (C,b) ;
    r1 = norm (C*x1-b,1) / norm (C,1) ;
    r2 = norm (C*x2-b,1) / norm (C,1) ;
    err = abs (r1-r2) ;
    fprintf ('err %g\n', err) ;
    if (err > 1e-10)
        error ('!') ;
    end

    x2 = cs_lusol (C,b, 1, 0.001) ;
    r2 = norm (C*x2-b,1) / norm (C,1) ;
    err = abs (r1-r2) ;
    fprintf ('err %g (lu with amd(A+A'')\n', err) ;
    if (err > 1e-10)
        error ('!') ;
    end

    if (m ~= n)
        continue ;
    end

    x1 = A\b ;
    r1 = norm (A*x1-b,1) / norm (A,1) ;
    if (r1 < 1e-6)
        x2 = cs_lusol (A,b) ;
        r2 = norm (A*x2-b,1) / norm (A,1) ;
        fprintf ('lu resid %g %g\n', r1, r2) ;
    end
end
