function test18
%TEST18 test iterative refinement after backslash
%
% Example:
%   test18
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

index = ssget ;
[ignore f] = sort (max (index.nrows, index.ncols)) ;
f = f (1:100) ;

clf
% f = f(1)

for i = f
    Prob = ssget (i) ;
    disp (Prob) ;
    A = Prob.A ;
    [m n] = size (A) ;
    if (~isreal (A) | m ~= n)                                               %#ok
        continue
    end

    b = rand (n,1) ;
    x = A\b ;
    r = b - A*x ;
    x = x + A\r ;

    fprintf ('\n%6.2e to %6.2e\n', norm (r), norm (b-A*x)) ;
end
