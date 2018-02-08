function Problem = UFget (matrix, index)
%UFGET former interface to the UF (now SuiteSparse) Matrix Collection
% This function works but is deprecated.  Use ssget instead.

% Copyright 2017, Timothy A. Davis, http://www.suitesparse.com

% warning ('UFget:deprecated', 'UFget is deprecated; use ssget instead') ;
if (nargin == 0)
    Problem = ssget ;
elseif (nargin == 1)
    Problem = ssget (matrix) ;
elseif (nargin == 2)
    Problem = ssget (matrix, index) ;
else
    error ('usage: Problem = ssget (matrix, index)') ;
end

