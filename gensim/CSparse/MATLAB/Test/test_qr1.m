function test_qr1
%TEST_QR1 test QR factorizations
%
% Example:
%   test_qr1
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

index = ssget ;
[ignore f] = sort (max (index.nrows, index.ncols)) ;

for i = f
    Prob = ssget (i,index) ;
    A = Prob.A ;
    if (~isreal (A))
        continue ;
    end

    [m n] = size (A) ;
    if (m < n)
        A = A' ;
    end

    t0 = 0 ;
    k0 = 0 ;
    while (t0 < 0.1)
        tic
        q = colamd (A, [-1 10]) ;
        % [Q,R] = qr (A (:,q)) ;
        R = qr (A (:,q)) ;                                          %#ok
        t = toc ;
        t0 = t0 + t ;
        k0 = k0 + 1 ;
    end
    t0 = t0 / k0 ;

    t1 = 0 ;
    k1 = 0 ;
    while (t1 < 0.1)
        tic
        [V,beta, p, R,q]  = cs_qr (A) ;                                 %#ok
        t = toc ;
        t1 = t1 + t ;
        k1 = k1 + 1 ;
    end
    t1 = t1 / k1 ;

    fprintf (...
        '%25s  MATLAB: %10.4f (%8d)   CS: %10.4f (%8d)  speedup: %8.2f\n', ...
        Prob.name, t0, k0, t1, k1, t0/t1) ;



end

