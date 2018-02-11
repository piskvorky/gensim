function test7
%TEST7 test cs_lu
%
% Example:
%   test7
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

index = ssget ;
[ignore f] = sort (max (index.nrows, index.ncols)) ;
f = f (1:100) ;

clf

for i = f
    Prob = ssget (i) ;
    disp (Prob) ;
    A = Prob.A ;
    if (~isreal (A))
        continue
    end
    [m n] = size (A) ;
    if (m ~= n)
        continue
    end

    [L,U,P] = lu (A) ;

    udiag = full (diag (U)) ;
    umin = min (abs (udiag)) ;
    fprintf ('umin %g\n', umin) ;

    if (umin > 1e-14)

        [L2,U2,p] = cs_lu (A) ;

        subplot (3,4,1) ; spy (A) ;
        subplot (3,4,2) ; spy (A(p,:)) ;
        subplot (3,4,3) ; spy (L2) ;
        subplot (3,4,4) ; spy (U2) ;

        err1 = norm (L*U-P*A,1) ;
        err2 = norm (L2*U2-A(p,:),1) ;
        fprintf ('err %g %g\n', err1, err2) ;
    end

    q = colamd (A) ;

    [L,U,P] = lu (A (:,q)) ;

    udiag = full (diag (U)) ;
    umin = min (abs (udiag)) ;
    fprintf ('umin %g with q\n', umin) ;

    if (umin > 1e-14)

        [L2,U2,p,q2] = cs_lu (A) ;

        subplot (3,4,5) ; spy (A) ;
        subplot (3,4,6) ; spy (A(p,q2)) ;
        subplot (3,4,7) ; spy (L2) ;
        subplot (3,4,8) ; spy (U2) ;

        err1 = norm (L*U-P*A(:,q),1) ;
        err2 = norm (L2*U2-A(p,q2),1) ;
        fprintf ('err %g %g\n', err1, err2) ;
    end


    try
        q = amd (A) ;
    catch
        q = symamd (A) ;
    end

    tol = 0.01 ;

    [L,U,P] = lu (A (q,q), tol) ;

    udiag = full (diag (U)) ;
    umin = min (abs (udiag)) ;
    fprintf ('umin %g with amd q\n', umin) ;

    if (umin > 1e-14)

        [L2,U2,p,q2] = cs_lu (A,tol) ;

        subplot (3,4,9) ; spy (A) ;
        subplot (3,4,10) ; spy (A(p,q2)) ;
        subplot (3,4,11) ; spy (L2) ;
        subplot (3,4,12) ; spy (U2) ;

        err1 = norm (L*U-P*A(q,q),1) ;
        err2 = norm (L2*U2-A(p,q2),1) ;
        lbig = full (max (max (abs (L2)))) ;
        fprintf ('err %g %g lbig %g\n', err1, err2, lbig) ;
        if (lbig > 1/tol)
            error ('L!') ;
        end
    end

    drawnow
    % pause

end
