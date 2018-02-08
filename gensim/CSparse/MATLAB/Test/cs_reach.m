function x = cs_reach(L,b)                                                  %#ok
%CS_REACH non-recursive reach (interface to CSparse cs_reach)
% find nonzero pattern of x=L\sparse(b).  L must be sparse, real, and lower
% triangular.  b must be a real sparse vector.
%
% Example:
%   x = cs_reach(L,b)
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

error ('cs_reach mexFunction not found') ;

