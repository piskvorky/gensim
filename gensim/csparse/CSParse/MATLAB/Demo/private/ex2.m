function ex2 (n)
%EX2: create an n-by-n 2D mesh, four different ways

% Example:
%   ex2
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

if (nargin < 1)
    n = 30 ;
end

subplot (1,2,1) ;

% method 1: create an n-by-n 2D mesh for the 2nd difference operator
tic
ii = zeros (5*n^2, 1) ;
jj = zeros (5*n^2, 1) ;
xx = zeros (5*n^2, 1) ;
k = 1 ;
for j = 0:n-1
    for i = 0:n-1
        s = j*n+i + 1 ;
        ii (k:k+4) = [(j-1)*n+i j*n+(i-1) j*n+i j*n+(i+1) (j+1)*n+i ] + 1 ;
        jj (k:k+4) = [s s s s s] ;
        xx (k:k+4) = [-1 -1 4 -1 -1] ;
        k = k + 5 ;
    end
end

% remove entries beyond the boundary
keep = find (ii >= 1 & ii <= n^2 & jj >= 1 & jj <= n^2) ;
ii = ii (keep) ;
jj = jj (keep) ;
xx = xx (keep) ;
A = sparse (ii,jj,xx) ;
t1 = toc ; disp (t1) ;
% subplot (2,2,1) ; 
spy (A)
title (sprintf ('%d-by-%d 2D mesh\n', n, n)) ;

% method 2, using no for loops
tic
nn = 1:n^2 ;
i2 = [nn-n ; nn-1 ; nn ; nn+1 ; nn+n] ;
j2 = repmat (nn, 5, 1) ;
x2 = repmat ([-1 -1 4 -1 -1]', 1, n^2) ;
keep = find (i2 >= 1 & i2 <= n^2 & j2 >= 1 & j2 <= n^2) ;
i2 = i2 (keep) ;
j2 = j2 (keep) ;
x2 = x2 (keep) ;
C = sparse (i2,j2,x2) ;
t2 = toc ; disp (t2) ;

% subplot (2,2,2) ; plot (j2) ;
% title ('2D fast j2') ;
disp (A-C) ;

any (ii-i2)
any (jj-jj)

% method 3: create an n-by-n-by-n 3D mesh for the 2nd difference operator
tic
[A, keep, ii, jj, xx] = mesh3d1 (n) ;
ii = ii (keep) ;
jj = jj (keep) ;
xx = xx (keep) ;
t3 = toc ; disp (t3) ;
tic
E = sparse (ii,jj,xx) ;
t3b = toc ; disp (t3b) ;
subplot (1,2,2) ; spy (E) ;
title (sprintf ('%d-by-%d-by-%d 3D mesh\n', n, n, n)) ;

% method 4, using no for loops
tic
nn = 1:n^3 ;
i2 = [nn-n^2 ; nn-n ; nn-1 ; nn ; nn+1 ; nn+n ; nn+n^2] ;
j2 = repmat (nn, 7, 1) ;
x2 = repmat ([-1 -1 -1 6 -1 -1 -1]', 1, n^3) ;
keep = find (i2 >= 1 & i2 <= n^3 & j2 >= 1 & j2 <= n^3) ;
i2 = i2 (keep) ;
j2 = j2 (keep) ;
x2 = x2 (keep) ;
t4 = toc ; disp (t4) ;
tic
F = sparse (i2,j2,x2) ;
t4b = toc ; disp (t4b) ;
disp (E-F) ;
