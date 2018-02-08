function test_randperms
%TEST_RANDPERMS test random permutations
% Example:
%   test_randperms
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

rand ('state', 0)

for trial = 1:100
    m = fix (30 * rand (1)) ;
    n = fix (30 * rand (1)) ;
    d = rand (1) ;
    A = sprandn (m,n,d) ;

    if (m == 0)
        p = [] ;
    else
        p = randperm (m) ;
    end
    if (n == 0)
        q = [] ;
    else
        q = randperm (n) ;
    end

    C = A(p,q) ;

    Im = speye (m) ;
    In = speye (n) ;
    P = Im (p,:) ;
    Q = In (:,q) ;

    q2 = find (Q) ;
    if (any (q ~= q2'))
        error ('!') 
    end

    p2 = find (P') ;
    if (any (p ~= p2'))
        error ('!') 
    end

    E = P*A*Q ;
    if (norm (C-E,1) ~= 0)
        error ('!') 
    end

    P = sparse (1:m, p, 1) ;
    Q = sparse (q, 1:n, 1) ;

    E = P*A*Q ;
    if (norm (C-E,1) ~= 0)
        error ('2!') 
    end

end

