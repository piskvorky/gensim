function stats = ssweb (matrix, opts)
%SSWEB opens the URL for a matrix in the SuiteSparse Matrix Collection.
%
%   ssweb(matrix) opens the URL for a matrix.  This parameter can be a string,
%   or an integer.  If it is a string with no "/" character, the web page for a
%   matrix group is displayed.  With no arguments, a list of all the matrix
%   groups is displayed.
%
%   Example:
%
%   If Problem = ssget ('HB/arc130'), the first four examples display
%   the same thing, the web page for the HB/arc130 matrix:
%
%       ssweb (6)
%       ssweb ('HB/arc130')
%       stats = ssweb (6)
%
%   The latter also returns statistics about the matrix or matrix group.
%   To display the web page for the HB (Harwell-Boeing) group:
%
%       ssweb ('HB')
%
%   To display the home page for the ss sparse matrix collection:
%
%       ssweb
%       ssweb (0)
%       ssweb ('')
%
%   The latter two are useful if a second optional parameter is specified.
%   The second optional argument is a string passed as additional parameters to
%   the MATLAB web command.  To use the system web browser instead of the MATLAB
%   browser, for example, use ssweb ('HB/arc130', '-browser').
%
%   See also web, ssget, ssget_defaults.

% Copyright 2009-2017, Timothy A. Davis, http://www.suitesparse.com

params = ssget_defaults ;
ss_index = ssget ;

if (nargin < 1)
    matrix = '' ;
end
if (nargin < 2)
    opts = '' ;
end
opts = [' ' opts] ;

% get the matrix group, name, and id
[group, name, id] = ssget_lookup (matrix, ss_index) ;

url = params.topurl ;

% open the web page for the matrix, group, or whole collection
if (id == 0)
    if (isempty (group))
        eval (['web ' url opts])
    else
        eval (['web ' url '/' group opts])
    end
else
    eval (['web ' url '/' group '/' name opts])
end

% return stats
if (nargout > 0)

    if (id == 0)

        if (isempty (group))

            % return stats about the whole collection
            stats.nmatrices = length (ss_index.nrows) ;
            stats.LastRevisionDate = ss_index.LastRevisionDate ;

        else

            % return stats about one matrix group
            nmat = length (ss_index.nrows) ;
            ngroup = 0 ;
            for i = 1:nmat
                if (strcmp (group, ss_index.Group {i}))
                    ngroup = ngroup + 1 ;
                end
            end
            stats.nmatrices = ngroup ;
            stats.LastRevisionDate = ss_index.LastRevisionDate ;

        end
    else

        % look up the matrix statistics
        stats.Group = group ;
        stats.Name = name ;
        stats.nrows = ss_index.nrows (id) ;
        stats.ncols = ss_index.ncols (id) ;
        stats.nnz = ss_index.nnz (id) ;
        stats.nzero = ss_index.nzero (id) ;
        stats.pattern_symmetry = ss_index.pattern_symmetry (id) ;
        stats.numerical_symmetry = ss_index.numerical_symmetry (id) ;
        stats.isBinary = ss_index.isBinary (id) ;
        stats.isReal = ss_index.isReal (id) ;
        stats.nnzdiag = ss_index.nnzdiag (id) ;
        stats.posdef = ss_index.posdef (id) ;
        stats.amd_lnz = ss_index.amd_lnz (id) ;
        stats.amd_flops = ss_index.amd_flops (id) ;
        stats.amd_vnz = ss_index.amd_vnz (id) ;
        stats.amd_rnz = ss_index.amd_rnz (id) ;
        stats.nblocks = ss_index.nblocks (id) ;
        stats.sprank = ss_index.sprank (id) ;
        stats.RBtype = ss_index.RBtype (id,:) ;
        stats.cholcand = ss_index.cholcand (id) ;
        stats.ncc = ss_index.ncc (id) ;
        stats.isND = ss_index.isND (id) ;
        stats.isGraph = ss_index.isGraph (id) ;

    end
end
