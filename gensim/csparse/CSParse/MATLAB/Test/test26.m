function test26
%TEST26 test cs_dmsol and cs_dmspy
%
% Example:
%   test26
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

clear functions

randn ('state', 0) ;
rand ('state', 0) ;

clf

ntrials = 1000 ;
e1 = zeros (ntrials,1) ;
e2 = zeros (ntrials,1) ;

for trials = 1:ntrials

    m = fix (100 * rand (1)) ;
    n = fix (100 * rand (1)) ;
    % d = 0.1 * rand (1) ;
    d = rand (1) * 4 * max (m,n) / max (m*n,1) ;
    A = sprandn (m,n,d) ;
    % S = sprandn (m,m,d) + speye (m) ;

    subplot (1,3,2) ; spy (A) ;
    subplot (1,3,3) ; cs_dmspy (A) ;

    b = rand (m,1) ;

    x1 = A\b ;
    x2 = cs_dmsol (A,b) ; 

    err1 = norm (A*x1-b) ;
    err2 = norm (A*x2-b) ;

    lerr1 = log10 (max (err1, eps)) ;
    lerr2 = log10 (max (err2, eps)) ;

    fprintf ('rank: %3d %3d err %6.2e  %6.2e  :   %6.1f\n', ...
        sprank(A), rank(full(A)), err1, err2, lerr1 - lerr2) ;

    if (isnan (err1))
        lerr1 = 10 ;
    end
    if (isnan (err2))
        lerr2 = 10 ;
    end

    if (lerr2 > lerr1 + 5)
        % pause
    end

    e1 (trials) = lerr1 ;
    e2 (trials) = lerr2 ;

    subplot (1,3,1) ; plot (e1, e2, 'o', [-16 10], [-16 10], 'r') ;
    xlabel ('MATLAB error') ;
    ylabel ('dmsol error') ;


    drawnow
    % pause
end
