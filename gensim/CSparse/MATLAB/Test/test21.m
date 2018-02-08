function test21
%TEST21 test cs_updown
%
% Example:
%   test21
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

clear functions
rand ('state', 0) ;
randn ('state', 0) ;

clf

for trials = 1:10
    if (trials <= 1)
        n = trials ;
    else
        n = 1+fix (100 * rand (1)) ;
    end
    fprintf ('n: %d\n', n) ;
    d = 0.1 * rand (1) ;
    A = sprandn (n,n,d) ;
    A = A+A' + 100 * speye (n) ;
    try
        p = amd (A) ;
    catch
        p = symamd (A) ;
    end
    A = sparse (A (p,p)) ;

    try
        L = chol (A)' ;
    catch
        continue ;
    end

    parent = etree (A) ;

    subplot (1,3,1) ;
    spy (A) ;

    if (n > 0)
        subplot (1,3,2) ;
        treeplot (parent) ;
    end

    subplot (1,3,3) ;
    spy (L) ;

    drawnow

    for trials2 = 1:10

        k = 1+fix (n * rand (1)) ;
        if (k <= 0 | k > n)                                                 %#ok
            k = 1 ;
        end

        w = sprandn (L (:,k)) ;
        Anew = A + w*w' ;

        % Lnew = cs_update (L, w, parent) ;
        % err1 = norm (Lnew*Lnew' - Anew, 1) ;

        Lnew = cs_updown (L, w, parent) ;
        err6 = norm (Lnew*Lnew' - Anew, 1) ;

        Lnew = cs_updown (L, w, parent, '+') ;
        err7 = norm (Lnew*Lnew' - Anew, 1) ;

        [Lnew, wnew] = chol_update (L, w) ;
        err2 = norm (Lnew*Lnew' - Anew, 1) ;
        err10 = norm (wnew - (L\w)) ;

        L3 = chol_updown (L, +1, w) ;
        err9 = norm (L3*L3' - Anew, 1) ;



        [L2, wnew] = chol_downdate (Lnew, w) ;
        err3 = norm (L2*L2' - A, 1) ;
        err11 = norm (wnew - (Lnew\w)) ;

        % L2 = cs_downdate (Lnew, w, parent) ;
        % err4 = norm (L2*L2' - A, 1) ;

        L2 = cs_updown (Lnew, w, parent, '-') ;
        err5 = norm (L2*L2' - A, 1) ;

        L2 = chol_updown (Lnew, -1, w) ;
        err8 = norm (L2*L2' - A, 1) ;

        err = max ([err2 err3 err5 err6 err7 err9 err8 err10 err11]) ;

        fprintf ('   k %3d  %6.2e\n', k, err) ;

        if (err > 1e-11)
            err2        %#ok
            err3        %#ok
            err5        %#ok
            err6        %#ok
            err7        %#ok
            err8        %#ok
            err9        %#ok
            err10       %#ok
            err11       %#ok
            pause
        end


    end
    % pause

end
