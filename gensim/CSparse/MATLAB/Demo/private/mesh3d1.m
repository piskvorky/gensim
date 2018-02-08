function [A, keep, ii, jj, xx] = mesh3d1 (n)
% create an n-by-n-by-n 3D mesh for the 2nd difference operator
% Example:
%   A = mesh3d1 (10) ;  % a 10-by-10-by-10 mesh
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

ii = zeros (7*n^3, 1) ;
jj = zeros (7*n^3, 1) ;
xx = zeros (7*n^3, 1) ;
t = 1 ;
for k = 0:n-1
    for j = 0:n-1
        for i = 0:n-1
            s = k*n^2 + j*n+i + 1 ;
            ii (t:t+6) = [
                    (k-1)*n^2 + j*n+i 
                    k*n^2 + (j-1)*n+i 
                    k*n^2 + j*n+(i-1)
                    k*n^2 + j*n+i
                    k*n^2 + j*n+(i+1)
                    k*n^2 + (j+1)*n+i
                    (k+1)*n^2 + j*n+i ]' + 1 ;
            jj (t:t+6) = [s s s s s s s] ;
            xx (t:t+6) = [-1 -1 -1 6 -1 -1 -1] ;
            t = t + 7 ;
        end
    end
end
keep = find (ii >= 1 & ii <= n^3 & jj >= 1 & jj <= n^3) ;
A = sparse (ii (keep), jj (keep), xx (keep)) ;
