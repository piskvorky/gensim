function test_sep
%TEST_SEP test cs_sep, and compare with Gilbert's meshpart vtxsep
% (requires MESHPART).
%
% Example:
%   test_sep
%
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

clear functions

index = ssget ;
[ignore f] = sort (max (index.nrows, index.ncols)) ;

clf

for k = 1:length(f)
    
    i = f (k) ;
    Prob = ssget (i) ;
    disp (Prob) ;
    A = spones (Prob.A) ;
    [m n] = size (A) ;
    if (m ~= n)
        A = A'*A ;
    end

    A = A|A' ;

    p = symrcm (A) ;

    n = size (A,1) ;
    n2 = fix (n/2) ;
    a = p (1:n2) ;
    b = p ((n2+1):n) ;

    clf

    subplot (2,3,1) ; spy (A) ;
    subplot (2,3,2) ; spy (A (p,p)) ;

    hold on
    plot ([.5 n2+.5 n2+.5 .5 .5], [.5 .5 n2+.5 n2+.5 .5], 'r', 'LineWidth', 2) ;
    hold off

    subplot (2,3,3) ; spy (A (a,b)) ; title ('edge sep') ;
    subplot (2,3,6) ; cs_dmspy (A (a,b)) ; title ('node sep') ;

    [s as bs] = vtxsep (A,a,b) ;                                        %#ok
    [s2 a2 b2] = cs_sep (A,a,b) ;

    p2 = [a2 b2 s2] ;
    B = A (p2,p2) ;
    subplot (2,3,5) ; spy (B) ;
    hold on

    px = [s2 a2 b2] ;
    if (any (sort (px) ~= 1:n))
        px      %#ok
        n       %#ok
        error ('!') ;
    end

    na = length (a2) ;
    nb = length (b2) ;
    ns = length (s2) ;                                                  %#ok

    nab = na + nb ;

    plot ([.5 na+.5 na+.5 .5 .5], [.5 .5 na+.5 na+.5 .5], 'r', 'LineWidth', 2) ;

    plot ([na nab nab na na]+0.5, [na na nab nab na]+0.5, 'r', 'LineWidth', 2) ;

    plot ([.5 nab+.5 nab+.5 .5 .5], [.5 .5 nab+.5 nab+.5 .5], 'g', 'LineWidth', 1) ;

    hold off

    nz1 = nnz (A (a2,b2)) ;
    if (nz1 ~= 0)
        nz1     %#ok
        error ('!') ;
    end

    nz2 = nnz (A (a2,b2)) ;
    if (nz2 ~= 0)
        nz2     %#ok
        error ('!') ;
    end

    if (length (s) ~= length (s2))
        fprintf ('lengths differ: %d %d\n', length (s), length (s2)) ;
    end

    drawnow
    % pause


end
