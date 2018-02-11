function test2
%TEST2 test cs_sparse
%
% Example:
%   test2
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

rand ('state', 0)
% clf

for trial = 1:100
    m = fix (10 * rand (1)) ;
    n = fix (10 * rand (1)) ;
    nz = fix (100 * rand (1)) ;

    i = 1 + fix (m * rand (nz,1)) ;
    j = 1 + fix (n * rand (nz,1)) ;
    x = rand (nz,1) ;

    A = sparse (i,j,x) ;
    B = cs_sparse (i,j,x) ;
    D = cs_sparse2 (i,j,x) ;
    fprintf ('%3d %3d %6d : %6d %6d : %d\n', ...
        m, n, nz, nnz (A), nnz(B), nz-nnz(A)) ;

    err = norm (A-B,1) / norm (A,1) ;
    if (err > 0)
        disp ('err = ') ;
        disp (err) ;
    end
    if (err > 1e-14)
        error ('!') ;
    end

    if (nnz (B-D) > 0)
        error ('!') ;
    end

    if (nnz (A) ~= nnz (B))
        error ('nz!') ;
    end

    if (max (1,nnz (B)) ~= max (1,nzmax (B)))
        nnz (B)
        nzmax (B)
        error ('nzmax!') ;
    end
    % pack


    [m n] = size (A) ;
    p = randperm (m) ;
    q = randperm (n) ;
    C1 = A (p,q) ;
    C2 = cs_permute (A,p,q) ;
    err = norm (C1-C2,1) ;
    if (err > 0)
        error ('!') ;
    end

%    subplot (1,2,1) ; spy (A)
%    subplot (1,2,2) ; spy (C2)
%    drawnow

    x = rand (m,1) ;
    x1 = x (p) ;
    x2 = cs_pvec (x, p) ;

    err = norm (x1-x2,1) ;
    if (err > 0)
        error ('!') ;
    end

    x1 = zeros (m,1) ;
    x1 (p) = x ;                                                            %#ok
    x2 = cs_ipvec (x, p) ;                                                  %#ok

    n = min (m,n) ;
    B = A (1:n, 1:n) ;
    p = randperm (n) ;
    B = B+B' ;

    C1 = triu (B (p,p)) ;
    C2 = cs_symperm (B,p) ;

    try
        pp = amd (C2) ;                                                     %#ok
    catch
        pp = symamd (C2) ;                                                  %#ok
    end

    err = norm (C1-C2,1) ;
    if (err > 0)
        error ('!') ;
    end



end
