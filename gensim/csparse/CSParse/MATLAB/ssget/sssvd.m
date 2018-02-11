function S = sssvd (matrix, ss_index)                                       %#ok
%SSSVD singular values of a matrix in the SuiteSparse Matrix Collection.
%
% As of Nov 2012, only matrices for which min(size(A)) <= 30401
% have their singular values computed.
%
% Examples:
%   S = sssvd ('HB/arc130')
%   S = sssvd (6)
%   index = ssget
%   S = sssvd (6, index)
%
% S is a struct containing:
%   s       the singular values (a column vector of size min(size(A)))
%   how     a string
%
% See also ssget.

% Copyright 2017, Timothy A. Davis, http://www.suitesparse.com

if (nargin < 2)
    % load the SuiteSparse index
    ss_index = ssget ;
end

% look up the matrix in the SuiteSparse index
[group, matrix, id] = ssget_lookup (matrix, ss_index) ;
if (id == 0)
    error ('invalid matrix') ;
end

% determine where the files go
params = ssget_defaults ;
svddir  = [ params.topdir 'svd' filesep group ] ;
svdfile = [ svddir filesep matrix '_SVD.mat' ] ;
svdurl  = [ params.topurl '/svd/' group '/' matrix '_SVD.mat' ] ;

% make sure the mat/Group directory exists
if (~exist (svddir, 'dir'))
    mkdir (svddir) ;
end

% download the *_SVD.mat file, if not already downloaded
if (~exist (svdfile, 'file'))
    fprintf ('downloading %s\n', svdurl) ;
    fprintf ('to %s\n', svdfile) ;
    tmp = tempname ;                        % download to a temp file first
    try
        websave (tmp, svdurl) ;
    catch me
        error ('SVD not yet computed for this matrix (or URL not found)') ;
    end
    movefile (tmp, svdfile, 'f') ;          % move the new matrix into place
end

% load the SVD, which sets the return value of S
load (svdfile) ;

