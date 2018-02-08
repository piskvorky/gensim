function test28
%TEST28 test cs_randperm, cs_dmperm
%
% Example:
%   test28
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

clear functions

rand ('state', 0) ;
for n = 1:100
    for trials = 1:1000
        p = cs_randperm (n, rand) ;
        if (any (sort (p) ~= 1:n))
            n           %#ok
            p           %#ok
            error ('!')
        end
    end
end

index = ssget ;
[ignore f] = sort (index.nnz) ;

fprintf ('p=dmperm (std, rand, rev)  [p,q,r,s]=dmperm (std, rand, rev)\n') ;

nmat = min (100, length (f)) ;
T1 = zeros (nmat,1) ;
T2 = zeros (nmat,1) ;
T3 = zeros (nmat,1) ;
D1 = zeros (nmat,1) ;
D2 = zeros (nmat,1) ;
D3 = zeros (nmat,1) ;


for k = 1:nmat

    i = f (k) ;
    Prob = ssget (i,index) ;
    A = Prob.A ;
    [m n] = size (A) ;
    fprintf ('%35s: ', Prob.name) ;

    tic
    p = cs_dmperm (A) ;
    t1 = toc ;
    sprank1 = sum (p > 0) ;
    fprintf (' %8.2f', t1) ;
    T1 (k) = t1 ;

    tic
    p = cs_dmperm (A,1) ;
    t2 = toc ;
    sprank2 = sum (p > 0) ;
    fprintf (' %8.2f', t2) ;
    T2 (k) = t2 ;

    tic
    p = cs_dmperm (A,-1) ;
    t3 = toc ;
    sprank3 = sum (p > 0) ;
    fprintf (' %8.2f', t3) ;
    T3 (k) = t3 ;

    if (sprank1 ~= sprank2 | sprank1 ~= sprank3)                            %#ok
        error ('!') ;
    end

    tic
    [p1,q1,r1,s1,cc1,rr1] = cs_dmperm (A) ;                                 %#ok
    d1 = toc ;
    fprintf (' %8.2f', d1) ;
    D1 (k) = d1 ;

    tic
    [p2,q2,r2,s2,cc2,rr2] = cs_dmperm (A,1) ;                               %#ok
    d2 = toc ;
    fprintf (' %8.2f', d2) ;
    D2 (k) = d2 ;

    tic
    [p3,q3,r3,s3,cc3,rr3] = cs_dmperm (A,-1) ;                              %#ok
    d3 = toc ;
    fprintf (' %8.2f\n', d3) ;
    D3 (k) = d3 ;

    if (sprank1 == max (m,n))
        nz1 = nnz (diag (A (p1,q1))) ;
        nz2 = nnz (diag (A (p2,q2))) ;
        nz3 = nnz (diag (A (p3,q3))) ;

        if (nz1 ~= sprank1 | nz2 ~= sprank2 | nz3 ~= sprank3)               %#ok
            error ('!')
        end
    end

    subplot (1,2,1)
    loglog (T1 (1:k), T2 (1:k), 'x', ...
        T1 (1:k), T3 (1:k), 'go', ...
        [1e-5 1e3], [1e-5 1e3], 'r-') ; 
    axis equal

    subplot (1,2,2)
    loglog (D1 (1:k), D2 (1:k), 'x', ...
        D1 (1:k), D3 (1:k), 'go', ...
        [1e-5 1e3], [1e-5 1e3], 'r-') ; 
    axis equal

    drawnow
end
