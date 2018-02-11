function test20
%TEST20 test cholupdown
%
% Example:
%   test20
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

clear functions

rand ('state', 0) ;

for trials = 1:10

    n = fix (100 * rand (1)) ;
    A = rand (n) ;
    A1 = A*A' + n*eye (n) ;

    try
        L1 = chol (A1)' ;
    catch
        continue ;
    end
    err1 = norm (L1*L1'-A1) ;

    w = rand (n,1) ;

    A2 = A1 + w*w' ;

    L2 = chol (A2)' ;
    err2 = norm (L2*L2'-A2) ;

    % try an update
    L2b = cholupdown (L1, +1, w) ;
    err2b = norm (L2b*L2b'-A2) ;

    % try a downdate
    L1b = cholupdown (L2, -1, w) ;                                          %#ok
    err1b = norm (L2b*L2b'-A2) ;


    fprintf ('%3d: %6.2e %6.2e : %6.2e %6.2e\n', n, err1, err2, err2b, err1b) ;
    % pause
end
