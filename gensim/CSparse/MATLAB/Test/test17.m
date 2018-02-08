function test17
%TEST17 test cs_qr, cs_qright, cs_q1, cs_qrleft, cs_qrsol
%
% Example:
%   test17
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

clear functions
clf

rand ('state', 0) ;
randn ('state', 0) ;

for trials = 1:100

    m = 1 + fix (10 * rand (1)) ;
    n = 1 + fix (10 * rand (1)) ;
    d = rand (1) ;
%   n = m ;

    A = sprandn (m, n, d) ;
    if (m < n)
        A = A' ;
    end
    [m n] = size (A) ;

    subplot (3,4,1) ; spy (A) ;

    [V1, Beta1, p1, R1, q1] = cs_qr (A) ;
    Q1 = cs_qright (V1, Beta1, p1, speye (size (V1,1))) ;
    Q1b = cs_q1 (V1, Beta1, p1) ;

    err = norm (Q1-Q1b,1) ;
    disp ('err = ') ;
    disp (err) ;
    if (err > 1e-12)
        error ('!') ;
    end

    m2 = size (Q1,1) ;
    A1 = [A ; sparse(m2-m,n)] ;

    subplot (3,4,5) ; spy (A1 (p1,q1)) ;
    subplot (3,4,6) ; spy (V1) ;
    subplot (3,4,7) ; spy (R1) ;
    subplot (3,4,8) ; spy (Q1) ;

    [V3, Beta3, R3] = qr2 (A) ;
    % Q3 = cs_qmake (V3, Beta3) ;
    Q3 = cs_q1 (V3, Beta3) ;
    subplot (3,4,9) ; spy (A) ;
    subplot (3,4,10) ; spy (V3) ;
    subplot (3,4,11) ; spy (R3) ;
    subplot (3,4,12) ; spy (Q3) ;

    err1 = norm (Q1*R1 - A1(:,q1), 1) ;
%   err2 = norm (Q2*R2 - A (:,q2), 1) ;
    err3 = norm (Q3*R3 - A, 1) ;

    fprintf ('m %3d m2 %3d n %3d    ::: %3d     %6.2e %6.2e\n', ...
        m, m2, n, m2-m, err1, err3) ;

    if (err1 > 1e-12)
        error ('!') ;
    end

%    if (err2 > 1e-12)
%       error ('!') ;
%    end

    if (err3 > 1e-12)
        error ('!') ;
    end

try

    b = rand (m,1) ;
    [Q,R] = qr (A (:,q1)) ;
    x = R\(Q'*b) ;
    x (q1) = x ;
    r1 = norm (A*x-b) ;

    x2 = cs_qrsol (A,b) ;
    r2 = norm (A*x2(1:n)-b) ;

    qt = cs_qleft (V1, Beta1, p1, speye(size(V1,1))) ;
    fprintf ('Q''*A-R:  %6.2e\n', norm (qt*A1(:,q1)-R1,1)) ;

    qtb = cs_qleft (V1, Beta1, p1, b) ;
    % [V1, Beta1, p1, R1, q1] = cs_qr (A) ;
    x3 = R1 \ qtb ;
    r3 = norm (A(:,q1)*x3(1:n)-b) ;

    fprintf ('least sq:  %6.2e %6.2e %6.2e diff %6.2e %6.2e\n', ...
        r1, r2, r3, r1-r2, r1-r3) ;

catch
end

    drawnow

    % pause

end
