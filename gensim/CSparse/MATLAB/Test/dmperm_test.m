function dmperm_test
%DMPERM_TEST test cs_dmperm
%
% Example:
%   dmperm_test
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

index = ssget ;

f = find (index.nrows ~= index.ncols) ;
[ignore i] = sort (index.nrows(f) ./ index.ncols(f)) ;
f = [209:211 f(i)] ;

nmat = length(f) ;

tt1 = zeros (1,nmat) ;
tt2 = zeros (1,nmat) ;
tt3 = zeros (1,nmat) ;
tt4 = zeros (1,nmat) ;
mm = zeros (1,nmat) ;
nn = zeros (1,nmat) ;
ss = zeros (1,nmat) ;
me = zeros (1,nmat) ;
ne = zeros (1,nmat) ;

p = cs_dmperm (sparse (1)) ;

for k = 1:length(f)
    
    i = f(k) ;

    Prob = ssget (i)                                                        %#ok

    A = Prob.A ;
    [m n] = size (A) ;
    if (m > n)
        % make sure A is short and fat
        A = A' ;
    end
    % C is tall and thin
    C = A' ;
    [m n] = size (A) ;

    k1 = 0 ;
    t1 = 0 ;
    while (t1 < 1)
        tic
        p = cs_dmperm (A) ;
        t = toc ;
        t1 = t1 + t ;
        k1 = k1 + 1 ;
    end
    t1 = t1 / k1 ;
    s1 = sum (p > 0) ;

    k2 = 0 ;
    t2 = 0 ;
    while (t2 < 1)
        tic
        p = cs_dmperm (C) ;
        t = toc ;
        t2 = t2 + t ;
        k2 = k2 + 1 ;
    end
    t2 = t2 / k2 ;
    s2 = sum (p > 0) ;

    k3 = 0 ;
    t3 = 0 ;
    while (t3 < 1)
        tic
        p = cs_dmperm_orig (A) ;
        t = toc ;
        t3 = t3 + t ;
        k3 = k3 + 1 ;
    end
    t3 = t3 / k3 ;

    k4 = 0 ;
    t4 = 0 ;
    while (t4 < 1)
        tic
        p = cs_dmperm_orig (A') ;
        t = toc ;
        t4 = t4 + t ;
        k4 = k4 + 1 ;
    end
    t4 = t4 / k4 ;

    sprnk = sum (p > 0) ;
    nempty = full (sum (sum (spones (A)) == 0)) ;
    mempty = full (sum (sum (spones (C)) == 0)) ;

    fprintf ('[m %d:%d n %d:%d (%d)]:\n', m, mempty, n, nempty, sprnk) ;
    fprintf ('    A: t1 %10.6f (%6d)   C: t2 %10.6f (%6d) new\n', ...
        t1, k1, t2, k2) ;
    fprintf ('    A: t3 %10.6f (%6d)   C: t4 %10.6f (%6d) orig\n', ...
        t3, k3, t4, k4) ;

    if (s1 ~= sprnk | s2 ~= sprnk)                                          %#ok
        s1                                                                  %#ok
        s2                                                                  %#ok
        sprnk                                                               %#ok
        error ('!') ;
    end

    tt1 (k) = t1 ;
    tt2 (k) = t2 ;
    tt3 (k) = t3 ;
    tt4 (k) = t4 ;
    mm (k) = m ;
    nn (k) = n ;
    ss (k) = sprnk ;
    me (k) = mempty ;
    ne (k) = nempty ;

    clear A C

    semilogy (ss(1:k) ./ nn(1:k), tt1(1:k) ./ tt3(1:k), 'o', ...
        [0 1], [1 1], 'r-') ;
    drawnow

end
