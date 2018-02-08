function etree_sample
% ETREE_SAMPLE construct a sample etree and symbolic factorization
%
% Example
%   etree_sample
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

clf

% desired etree:
%       1 2 3 4 5 6 7  8  9 10 11
goal = [6 3 8 6 8 7 9 10 10 11  0] ;

o = 0 ;
X = 1 ;
x = 0 ;

A = [
1 o o o o o o o o o o
o 2 o o o o o o o o o
o X 3 o o o o o o o o
o o o 4 o o o o o o o
o o o o 5 o o o o o o
X o o X o 6 o o o o o
X o o x o x 7 o o o o
o X x o X o o 8 o o o
x o o x o X x o 9 o o
x x X X x X x X x 10 o
x x X x X x X X x X 11 ] ;

A = A + tril(A,-1)' ;
disp ('A = ') ;
disp (A)

[count,h,parent,post,R] = symbfact (A) ;
L = R' ;

subplot (2,3,1) ;
spy (A)
title ('A') ;

subplot (2,3,2) ;
etreeplot (A)
title ('etree') ;

% [parent, post] = etree (A) ;

subplot (2,3,3) ;
spy (L)
title ('L, not postordered') ;

n = size (A,1) ;
for k = 1:n
    fprintf ('parent (%d) = %d  goal: %d   ok: %d\n', ...
        k, parent (k), goal (k), goal (k) == parent(k)) ;
end


[count,h,parent2,post2,R] = symbfact (A (post,post)) ;
L = R' ;

subplot (2,3,5) ;
spy (A (post,post))
title ('A postordered') ;

subplot (2,3,6) ;
spy (L)
title ('L postordered') ;
