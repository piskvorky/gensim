function A = frand (n,nel,s)
% A = frand (n,nel,s) creates an n-by-n sparse matrix consisting of nel finite
% elements, each of which are of size s-by-s with random symmetric nonzero
% pattern, plus the identity matrix.
%
% Example:
%   A = frand (100, 100, 4) ; cspy (A)
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

ss = s^2 ;
nz = nel*ss ;
ii = zeros (nz,1) ;
jj = zeros (nz,1) ;
xx = zeros (nz,1) ;
k = 1 ;
for e = 1:nel
    i = 1 + fix (n * rand (s,1)) ;
    i = repmat (i, 1, s) ;
    j = i' ;
    x = rand (s,s) ;
    ii (k:k+ss-1) = i (:) ; 
    jj (k:k+ss-1) = j (:) ; 
    xx (k:k+ss-1) = x (:) ;
    k = k + ss ;
end
A = sparse (ii,jj,xx,n,n) + speye (n) ;
