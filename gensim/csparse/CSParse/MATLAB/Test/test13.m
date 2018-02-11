function test13
%TEST13 test cs_counts, cs_etree
%
% Example:
%   test13
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

clear functions
randn ('state',0) ;
rand ('state',0) ;

for trial = 1:100
    m = fix (100 * rand (1)) ;
    n = fix (100 * rand (1)) ;
    d = .1 * rand (1) ;
    A = sprandn (n,n,d) ;
    C = sprandn (m,n,d) ;
    A = A+A' ;
    fprintf ('m %4d n %4d nnz(A) %6d nnz(C) %6d\n', m, n, nnz(A), nnz(C)) ;

    [p1,po1] = etree (A) ;

    [p2,po2] = cs_etree (A) ;
    [p3,po3] = cs_etree (A, 'sym') ;
    % po2 = cs_post (p2) ;

    check_if_same (p1,p2) ;
    check_if_same (po1,po2) ;

    check_if_same (p1,p3) ;
    check_if_same (po1,po3) ;

    c1 = symbfact (A) ;
    c2 = cs_counts (A) ;

    % A-A'

    check_if_same (c1,c2) ;

    c2 = cs_counts (triu (A)) ;
    check_if_same (c1,c2) ;
    % pause

    p0 = etree (A, 'col') ;
    % p1 = etree2 (A, 'col') ;  % CHOLMOD
    p2 = cs_etree (A, 'col') ;

    if (~isempty (A))
        check_if_same (p0,p2) ;
    end

    p0 = etree (C, 'col') ;
    % p1 = etree2 (C, 'col') ;  % CHOLMOD
    p2 = cs_etree (C, 'col') ;

    if (~isempty (C))
        check_if_same (p0,p2) ;
    end

    % find etree of A'A, and postorder it
    [m n] = size (A) ;                                                      %#ok
    % full (A)

    [cp0 cpo0] = etree (A, 'col') ;
    % [cp1 cpo1] = etree2 (A, 'col') ;  % CHOLMOD
    [cp2 cpo2] = cs_etree (A, 'col') ;
    % cpo2 = cs_post (cp2) ;

    check_if_same (cp0, cp2) ;
    check_if_same (cpo0, cpo2) ;

    c0 = symbfact (A, 'col') ;
    % c1 = symbfact2 (A, 'col') ;   % CHOLMOD
    c2 = cs_counts (A, 'col') ;

    check_if_same (c0, c2) ;

end
