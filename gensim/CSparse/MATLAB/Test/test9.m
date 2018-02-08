function test9
%TEST9 test cs_qr
%
% Example:
%   test9
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

rand ('state', 0) ;

index = ssget ;
[ignore f] = sort (max (index.nrows, index.ncols)) ;
f = f (1:100) ;
clf

% f = 185 ;
% f = 449 ;
% f = 186

for i = f
    Prob = ssget (i) ;
    disp (Prob) ;
    A = Prob.A ;
    [m n] = size (A) ;
    if (m < n)
        A = A' ;
    end
    [m n] = size (A) ;
    sp = sprank (A) ;
%    if (sprank (A) < min (m,n))
%       continue
%    end
    Aorig = A ;

    A = A (:, colamd (A)) ;
    s1 = svd (full (A)) ;

    tic ;
    R = qr (A) ;
    t1 = toc ;                                                              %#ok

    % tic ;
    % [Q,R] = qr (A) ;
    % t1 = toc ;

    [c,h,parent] = symbfact (A, 'col') ;
    rnz = sum (c) ;                                                         %#ok
    tic ;
    [V2,Beta2,p,R2] = cs_qr (sparse(A)) ;
    t2 = toc ;                                                              %#ok

    v2 = full (V2) ;
    if (any (spones (v2) ~= spones (V2)))
        error ('got zeros!') ;
    end

    C = A ;
    m2 = size (V2,1) ;
    if (m2 > m)
        C = [A ; sparse(m2-m, n)] ;
    end
    C = C (p,:) ;

%    [H1,R1] = myqr (C) ;
%    err1 = norm (R1-R2,1) / norm (R1)
%    % [svd(A) svd(R1) svd(full(R2))]
%    s2 = svd (full (R2)) ;
%    err2 = norm (s1 - s2) / s1 (1) 
%    fprintf ('%10.6f %10.6f  cs speedup %8.3f sprank %d n %d\n', ...
%       t1, t2, t1/t2, sp, n) ;
%    err2

    % left-looking:
    [V,Beta3,R3] = qr_left (C) ;                                            %#ok
    s3 = svd (full (R2)) ;
    err3 = norm (s1 - s3) / s1 (1) ;
    disp ('err3 = ') ; disp (err3) ;
    if (err3 > 1e-12)
        error ('!') ;
    end

    % right-looking:
    [V,Beta4,R4] = qr_right (C) ;                                           %#ok
    s4 = svd (full (R2)) ;
    err4 = norm (s1 - s4) / s1 (1) ;
    disp ('err4 = ') ; disp (err4) ;
    if (err4 > 1e-12)
        error ('!') ;
    end

    % H2 = full (H2)
    % R2 = full (R2)

    subplot (2,4,1) ; spy (A) ;         title ('A colamd') ;
    subplot (2,4,2) ; spy (C) ;         title ('A rperm') ;
    subplot (2,4,3) ; treeplot (parent) ;
    subplot (2,4,4) ; spy (Aorig) ;     title ('Aorig') ;
    subplot (2,4,5) ; spy (abs(R2)>0) ; title ('spqr R, no zeros') ;
    subplot (2,4,6) ; spy (R) ;         title ('matlab R') ;
    subplot (2,4,7) ; spy (R2) ;        title ('spqr R') ;
    subplot (2,4,8) ; spy (V2) ;        title ('spqr V') ;
    drawnow

%    if (err2 > 1e-9)
%       error ('!') ;
%    end
    if (m2 > m)
        fprintf ('added %d rows, sprank %d n %d\n', m2-m, sp, n) ;
    end
    % pause
end

