function test11
%TEST11 test cs_rowcnt
%
% Example:
%   test11
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

clear functions
index = ssget ;
[ignore f] = sort (max (index.nrows, index.ncols)) ;
f = f (1:200) ;

for i = f
    Prob = ssget (i, index) ;
    disp (Prob) ;
    A = Prob.A ;
    [m n] = size (A) ;
    if (~isreal (A) | m ~= n)                                               %#ok
        continue
    end

    A = spones (A) ;
    A = A+A' + speye(n) ;

    [cc h pa po R] = symbfact (A) ;
    rc1 = full (sum (R)) ;
    rc2 = cs_rowcnt (A, pa, po) ;
    if (any (rc1 ~= rc2))
        error ('!') ;
    end

    try
        p = amd (A) ;
    catch
        p = symamd (A) ;
    end
    A = A (p,p) ;

    [cc h pa po R] = symbfact (A) ;
    rc1 = full (sum (R)) ;
    rc2 = cs_rowcnt (A, pa, po) ;
    if (any (rc1 ~= rc2))
        error ('!') ;
    end

end

