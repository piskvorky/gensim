function ex_1
%EX_1: four methods for creating the same matrix.
%   (please wait, this can take a while...)

% Example:
%   ex_1
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

n = 1000 ;
nz = 1e5 ;

tic
% method 1: A(i,j) = ...
rand ('state', 0) ;
A = sparse (n,n) ;
for k = 1:nz
    % compute some arbitrary entry and add it into the matrix
    i = 1 + fix (n * rand (1)) ;
    j = 1 + fix (n * rand (1)) ;
    x = rand (1) ;
    A (i,j) = A (i,j) + x ; % VERY slow, esp. if A(i,j) not already nonzero!
end
fprintf ('Method 1: ') ;
toc
A1 = A ;

tic
% method 2: triplet form, one entry at a time
rand ('state', 0) ;
ii = zeros (nz, 1) ;    % preallocate ii, jj, and xx
jj = zeros (nz, 1) ;
xx = zeros (nz, 1) ;
for k = 1:nz
    % compute some arbitrary entry and add it into the matrix
    ii (k) = 1 + fix (n * rand (1)) ;
    jj (k) = 1 + fix (n * rand (1)) ;
    xx (k) = rand (1) ;
end
A = sparse (ii,jj,xx) ;
fprintf ('Method 2: ') ;
toc
A2 = A ;
disp (A1-A2) ;

tic
% method 3: triplet form, one entry at a time, pretend nz is unknown
rand ('state', 0) ;
len = 16 ;
ii = zeros (len, 1) ;
jj = zeros (len, 1) ;
xx = zeros (len, 1) ;
for k = 1:nz
    % compute some arbitrary entry and add it into the matrix
    if (k > len)
        % double the size of ii,jj,xx
        len = 2*len ;
        ii (len) = 0 ;
        jj (len) = 0 ;
        xx (len) = 0 ;
    end
    ii (k) = 1 + fix (n * rand (1)) ;
    jj (k) = 1 + fix (n * rand (1)) ;
    xx (k) = rand (1) ;
end
A = sparse (ii (1:k), jj (1:k), xx (1:k)) ;
fprintf ('Method 3: ') ;
toc

A3 = A ;
disp (A1-A3) ;

tic
% method 4: avoid the for loop
rand ('state', 0) ;
e = rand (3, nz) ;
e (1,:) = 1 + fix (n * e (1,:)) ;
e (2,:) = 1 + fix (n * e (2,:)) ;
A = sparse (e (1,:), e (2,:), e (3,:)) ;
fprintf ('Method 4: ') ;
toc

A4 = A ;
disp (A1-A4) ;
