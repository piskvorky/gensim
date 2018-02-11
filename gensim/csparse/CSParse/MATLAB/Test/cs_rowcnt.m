function r = cs_rowcnt(A,parent,post)                                       %#ok
%CS_ROWCNT row counts for sparse Cholesky
% Compute the row counts of the Cholesky factor L of the matrix A.  Uses
% the lower triangular part of A.
%
% Example:
%   r = cs_rowcnt(A,parent,post)
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

error ('cs_rowcnt mexFunction not found') ;

