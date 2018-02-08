function A = cs_frand (n,nel,s)                                             %#ok
%CS_FRAND generate a random finite-element matrix
% A = cs_frand (n,nel,s) creates an n-by-n sparse matrix consisting of nel
% finite elements, each of which are of size s-by-s with random symmetric
% nonzero pattern, plus the identity matrix.
%
% Example
%   A = cs_frand (100, 100, 3) ;
% See also cs_demo.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

error ('cs_frand mexFunction not found') ;

