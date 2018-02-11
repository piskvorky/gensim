function test6
%TEST6 test cs_reach, cs_reachr, cs_lsolve, cs_usolve
%
% Example:
%   test6
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

rand ('state', 0)
maxerr = 0 ;
clf
for trial = 1:201
    n = fix (100 * rand (1)) ;
    d = 0.1 * rand (1) ;
    L = tril (sprandn (n,n,d),-1) + sprand (speye (n)) ;
    b = sprandn (n,1,d) ;

    for uplo = 0:1

        if (uplo == 1)
            % solve Ux=b instead ;
            L = L' ;
        end

        x = L\b ;
        sr = 1 + cs_reachr (L,b) ;
        sz = 1 + cs_reachr (L,b) ;

        check_if_same (sr,sz) ;

        s2 = 1 + cs_reach (L,b) ;

        if (uplo == 0)
            x3 = cs_lsolve (L,b) ;
        else
            x3 = cs_usolve (L,b) ;
        end
        % cs_lsolve and cs_usolve return sparse vectors with
        % unsorted indices and possibly with explicit zeros.
        x3 = 1 * x3'' ;

        spy ([L b x x3])
        drawnow

        s = sort (sr) ;
        [i j xx] = find (x) ;                                               %#ok
        [i3 j3 xx3] = find (x3) ;                                           %#ok

        if (isempty (i))
            if (~isempty (s))
                i       %#ok
                s       %#ok
                error ('!') ;
            end
        elseif (any (s ~= i))
            i       %#ok
            s       %#ok
            error ('!') ;
        end

        if (isempty (i3))
            if (~isempty (s))
                i3      %#ok
                s       %#ok
                error ('!') ;
            end
        elseif (any (s ~= sort (i3)))
            s       %#ok
            i3      %#ok
            error ('!') ;
        end

        if (any (s2 ~= sr))
            s2      %#ok
            sr      %#ok
            error ('!') ;
        end

        err = norm (x-x3,1) ;
        if (err > 1e-12)
            x       %#ok
            x3      %#ok
            uplo    %#ok
            err     %#ok
            error ('!') 
        end

        maxerr = max (maxerr, err) ;

    end

    drawnow
end
fprintf ('maxerr = %g\n', maxerr) ;
