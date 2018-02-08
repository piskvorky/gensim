function test12
%TEST12 test cs_qr and compare with svd
%
% Example:
%   test12
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

fprintf ('test 12\n') ;
rand ('state',0) ;
% A = rand (3,4)

for trial = 1:100
    m = fix (100 * rand (1)) ;
    n = fix (100 * rand (1)) ;
    d = .1 * rand (1) ;
    A = sprandn (m,n,d) ;
    fprintf ('m %d n %d nnz %d\n', m, n, nnz(A)) ;
    if (m < n)
        continue ;
    end
    if (m == 0 | n == 0)                                                    %#ok
        continue ;
    end
    % save A A
    fprintf ('[ ') ;
    [V,Beta,p,R] = cs_qr (A) ;
    % [Q,R] = svd (full(A)) ;
    fprintf (']\n') ;

    s1 = svd (full (A)) ;
    s2 = svd (full (R)) ;
    s2 = s2 (1:length(s1)) ;
    err = norm (s1-s2) ; 
    if (length (s1) > 1)
        err = err / s1 (1) ;
    end
    fprintf ('err %g\n', err) ;
    if (err > 1e-12)
        error ('!') ;
    end
end
