function A = mesh2d2 (n)
% create an n-by-n 2D mesh for the 2nd difference operator
% Example:
%   A = mesh2d2 (30) ;  % a 30-by-30 mesh
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

nn = 1:n^2 ;
ii = [nn-n ; nn-1 ; nn ; nn+1 ; nn+n] ;
jj = repmat (nn, 5, 1) ;
xx = repmat ([-1 -1 4 -1 -1]', 1, n^2) ;
keep = find (ii >= 1 & ii <= n^2 & jj >= 1 & jj <= n^2) ;
A = sparse (ii (keep), jj (keep), xx (keep)) ;
