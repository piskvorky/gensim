function test16
%TEST16 test cs_amd
%
% Example:
%   test16
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

rand ('state', 0) ;
randn ('state', 0) ;
clf

index = ssget ;
[ignore f] = sort (max (index.nrows, index.ncols)) ;
f = f (1:200) ;
skip = 811 ;

% f = 719

for i = f
    if (any (i == skip))
        continue
    end
    Prob = ssget (i) ;
    A = spones (Prob.A) ;
    Aorig = A ;
    [m n] = size (A) ;
    if (m < n)
        A = A' ;
    end
    [m n] = size (A) ;
    if (m ~= n)
        A = A'*A ;
    end

    fprintf ('n %4d nz %d\n', n, nnz (A)) ;

    try
        p0 = amd (A) ;
    catch
        p0 = symamd (A) ;
    end
    fprintf ('symmetric case:\n') ;
    p1 = cs_amd (A) ;

    if (any (sort (p1) ~= 1:n))
        error ('not perm!') ;
    end

    C = A+A' + speye (n) ;
    lnz0 = sum (symbfact (C (p0,p0))) ;
    lnz1 = sum (symbfact (C (p1,p1))) ;
    subplot (2,3,1) ; spy (C)
    subplot (2,3,2) ; spy (C (p0,p0)) ; title ('amd') ;
    subplot (2,3,3) ; spy (C (p1,p1)) ; title ('csamd') ;
    drawnow

    if (lnz0 ~= lnz1)
        fprintf ('----------------- lnz %d %d   %9.4f\n', ...
            lnz0, lnz1, 100*(lnz0-lnz1)/max([1 lnz0])) ;
    end

    if (1)

        p0 = colamd (Aorig) ;
        [m n] = size (Aorig) ;
        fprintf ('m %d n %d\n', m, n) ;

        fprintf ('A''A case, no dense rows (for QR):\n') ;
        p1 = cs_amd (Aorig, 3) ;
        if (any (sort (p1) ~= 1:n))
            error ('not perm!') ;
        end

        subplot (2,3,4) ; spy (Aorig)
        subplot (2,3,5) ; spy (Aorig (:,p0)) ; title ('colamd') ;
        subplot (2,3,6) ; spy (Aorig (:,p1)) ; title ('cs amd(A''A)') ;
        lnz0 = sum (symbfact (Aorig (:,p0), 'col')) ;
        lnz1 = sum (symbfact (Aorig (:,p1), 'col')) ;
        fprintf ('                    A''A: %7d %7d %9.4f\n', ...
            lnz0, lnz1, 100*(lnz0-lnz1)/max([1 lnz0])) ;
        drawnow

        % pause

    end
end

