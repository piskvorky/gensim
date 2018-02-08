function test19
%TEST19 test cs_dmperm, cs_maxtransr, cs_dmspy, cs_scc
%
% Example:
%   test19
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

clear functions

randn ('state', 0) ;
rand ('state', 0) ;

clf

for trials = 1:1000

    m = fix (100 * rand (1)) ;
    n = fix (100 * rand (1)) ;
    % d = 0.1 * rand (1) ;
    d = rand (1) * 4 * max (m,n) / max (m*n,1) ;
    A = sprandn (m,n,d) ;
    S = sprandn (m,m,d) + speye (m) ;

    subplot (2,3,1) ;
    cspy (A) ;

    pp = dmperm (A) ;

    sprnk = sum (pp > 0) ;

    pp2 = cs_dmperm (A) ;
    spr2 = sum (pp2 > 0) ;
    if (spr2 ~= sprnk)
        error ('!')
    end

    pp2 = cs_maxtransr (A) ;
    spr2 = sum (pp2 > 0) ;
    if (spr2 ~= sprnk)
        error ('!')
    end

    [p,q,r,s] = dmperm (A) ;
    C = A (p,q) ;
    % r
    % s

    nk = length (r) - 1 ;

    fprintf ('sprnk: %d  m %d n %d   nb: %d\n', sprnk, m, n, nk) ;

    subplot (2,3,2) ;
    hold off
    spy (C)
    hold on

    for k = 1:nk
        r1 = r(k) ;
        r2 = r(k+1) ;
        c1 = s(k)  ;
        c2 = s(k+1) ;
        plot ([c1 c2 c2 c1 c1]-.5, [r1 r1 r2 r2 r1]-.5, 'g') ;
    end

    [p2,q2,rr2,ss2,cp,rp] = cs_dmperm (A) ;

    if (min (m,n) > 0)
        if (length (rr2) ~= length (r))
            error ('# fine blocks!') ;
        end
    end

    if (rp (4) - 1 ~= sprnk)
        rp              %#ok
        sprnk           %#ok
        error ('!') ;
    end

    if (any (sort (p2) ~= 1:m))
        error ('p2!') ;
    end

    if (any (sort (q2) ~= 1:n))
        error ('q2!') ;
    end

    if (cp (5) ~= n+1)
        error ('cp!') ;
    end

    if (rp (5) ~= m+1)
        error ('rp!') ;
    end

    C = A (p2,q2) ;

    subplot (2,3,3) ; cs_dmspy (A,0) ;

    % hold off
    % spy (C) ;
    % hold on

    % r1 = rp(1) ;
    % r2 = rp(2) ;
    % c1 = cp(1)  ;
    % c2 = cp(2) ;
    % plot ([c1 c2 c2 c1 c1]-.5, [r1 r1 r2 r2 r1]-.5, 'g') ;

    r1 = rp(1) ;
    r2 = rp(2) ;
    c1 = cp(2) ;
    c2 = cp(3) ;
    % plot ([c1 c2 c2 c1 c1]-.5, [r1 r1 r2 r2 r1]-.5, 'g') ;

    B = C (r1:r2-1, c1:c2-1) ;
    if (nnz (diag (B)) ~= size (B,1))
        error ('C1 diag!') ;
    end

    r1 = rp(2) ;
    r2 = rp(3) ;
    c1 = cp(3) ;
    c2 = cp(4) ;
    % plot ([c1 c2 c2 c1 c1]-.5, [r1 r1 r2 r2 r1]-.5, 'r') ;

    B = C (r1:r2-1, c1:c2-1) ;
    if (nnz (diag (B)) ~= size (B,1))
        error ('C2 diag!') ;
    end

    r1 = rp(3) ;
    r2 = rp(4) ;
    c1 = cp(4) ;
    c2 = cp(5) ;
    % plot ([c1 c2 c2 c1 c1]-.5, [r1 r1 r2 r2 r1]-.5, 'g') ;

    B = C (r1:r2-1, c1:c2-1) ;
    if (nnz (diag (B)) ~= size (B,1))
        error ('C3 diag!') ;
    end

    r1 = rp(4) ;                                                            %#ok
    r2 = rp(5) ;                                                            %#ok
    c1 = cp(4) ;                                                            %#ok
    c2 = cp(5) ;                                                            %#ok
    % plot ([c1 c2 c2 c1 c1]-.5, [r1 r1 r2 r2 r1]-.5, 'g') ;

    if (~isempty (S))

        [p1,q1,r0,s0] = dmperm (S) ;
        [p3,r3] = cs_scc (S) ;
        if (length (r3) ~= length (r0))
            error ('scc size!') ;
        end

        if (any (sort (p3) ~= 1:m))
            error ('scc perm!') ;
        end

        nk = length (r0)-1 ;

        subplot (2,3,4) ;
        hold off
        spy (S (p1,q1)) ;
        hold on
        for k = 1:nk
            r1 = r0(k) ;
            r2 = r0(k+1) ;
            c1 = s0(k)  ;
            c2 = s0(k+1) ;
            plot ([c1 c2 c2 c1 c1]-.5, [r1 r1 r2 r2 r1]-.5, 'g') ;
        end

        subplot (2,3,5) ;
        hold off
        spy (S (p3,p3)) ;
        hold on
        for k = 1:nk
            r1 = r3(k) ;
            r2 = r3(k+1) ;
            c1 = r3(k)  ;
            c2 = r3(k+1) ;
            plot ([c1 c2 c2 c1 c1]-.5, [r1 r1 r2 r2 r1]-.5, 'g') ;
        end

    end

    subplot (2,3,6) ;
    cs_dmspy (A) ;
    drawnow
   % pause




end
