function test_qrsol
%TEST_QRSOL test cs_qrsol
%
% Example:
%   test_qrsol
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

index = ssget ;
[ignore f] = sort (max (index.nrows, index.ncols)) ;

k = 0 ;
rs1 = zeros (1,0) ;
rs2 = zeros (1,0) ;

for i = f
    Prob = ssget (i,index) ;
    A = Prob.A ;
    if (~isreal (A))
        continue ;
    end

    [m n] = size (A) ;                                                  %#ok
    b = rand (m,1) ;

    x1 = A\b ;
    x2 = cs_qrsol (A,b) ;

    x1 (~isfinite (x1)) = 0 ;
    x2 (~isfinite (x2)) = 0 ;

    r1 = norm (A*x1-b) ;
    r2 = norm (A*x2-b) ;

    k = k + 1 ;
    rs1 (k) = r1 ;
    rs2 (k) = r2 ;

    fprintf ('%30s  MATLAB: %6.2e CS: %6.2e\n', Prob.name, r1, r2) ;

    loglog (rs1, rs2, 'o') ;
    drawnow

    clear A b x1 x2
    % pack

end

