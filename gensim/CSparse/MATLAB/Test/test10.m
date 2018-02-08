function test10
%TEST10 test cs_qr
%
% Example:
%   test10
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

rand ('state', 0) ;

% f = 185 ;
% f = 449 ;
clf

for trials = 1:100
    
    m = fix (100 * rand (1)) ;
    n = fix (100 * rand (1)) ;
    d = 0.1 * rand (1) ;
    A = sprandn (m, n, d) ;
    [m n] = size (A) ;
    if (m < n)
        A = A' ;
    end
    [m n] = size (A) ;
    sp = sprank (A) ;
    % if (sp < n)
    %   continue ;
    % end

    Aorig = A ;

    % A = A (:, colamd (A)) ;

    tic ;
    R = qr (A) ;
    t1 = toc ;

    % tic ;
    % [Q,R] = qr (A) ;
    % t1 = toc ;

    [c,h,parent] = symbfact (A, 'col') ;                                    %#ok
    rnz = sum (c) ;                                                         %#ok
    tic ;
    [V2,Beta2,p,R2] = cs_qr (sparse(A)) ;
    t2 = toc ;

    C = A ;
    m2 = size (V2,1) ;
    if (m2 > m)
        C = [A ; sparse(m2-m, n)] ;
    end
    C = C (p,:) ;

    [H1,R1] = myqr (C) ;
    err1 = norm (R1-R2,1) / norm (R1) ;
    disp ('err1 = ') ;
    disp (err1) ;
    % [svd(A) svd(R1) svd(full(R2))]
    s1 = svd (full (A)) ;
    s2 = svd (full (R2)) ;
    if (n > 0)
        err2 = norm (s1 - s2) / s1 (1)  ;
        disp ('err2 = ') ;
        disp (err2) ;
    else
        err2 = 0 ;
    end
    fprintf ('%10.6f %10.6f  cs speedup %8.3f sprank %d vs %d\n', t1, t2, t1/t2, sp, n) ;

    % H2 = full (H2)
    % R2 = full (R2)

    subplot (2,4,1) ; spy (A) ;         title ('A colamd') ;
    subplot (2,4,4) ; spy (Aorig) ;     title ('Aorig') ;
    subplot (2,4,2) ; spy (C) ;         title ('A rperm') ;
    subplot (2,4,5) ; spy (abs(R2)>0) ; title ('spqr R, no zeros') ;
    subplot (2,4,6) ; spy (R) ;         title ('matlab R') ;
    subplot (2,4,7) ; spy (R2) ;        title ('spqr R') ;
    subplot (2,4,8) ; spy (V2) ;        title ('spqr H') ;
    drawnow

    if (err2 > 1e-9)
        error ('!') ;
    end

    if (m2 > m)
        fprintf ('added %d rows, sprank %d n %d\n', m2-m, sp, n) ;
    end
end

