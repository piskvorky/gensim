function Problem = ssget (matrix, ss_index)
%SSGET loads a matrix from the SuiteSparse Matrix Collection.
%   (formerly called the University of Florida Sparse Matrix Collection).
%
%   Problem = ssget(matrix) loads a matrix from the SuiteSparse Matrix
%   Collection, specified as either a unique ID number (1 to the # of matrices
%   in the collection) or as a string (the full name of the matrix).  With no
%   input parameters, index=ssget returns an index of matrices in the
%   collection.  A local copy of the matrix is saved.  If no input or output
%   arguments are provided, the index is printed.  With a 2nd parameter
%   (Problem = ssget (matrix, index)), the index file is not loaded.  This is
%   faster if you are loading lots of matrices.
%
%   Examples:
%       index = ssget ;                     % loads index
%       index = ssget ('refresh') ;         % forces download of new index
%       index = ssget ('update') ;          % same as 'refresh'
%
%       Problem = ssget (6)                 % 4 ways of loading the same Problem
%       Problem = ssget ('HB/arc130')
%       Problem = ssget (6, index)
%       Problem = ssget ('HB/arc130', index)
%
%   See also ssgrep, ssweb, ssget_example, ssget_defaults, websave.

% Copyright 2009-2017, Timothy A. Davis, http://www.suitesparse.com

%-------------------------------------------------------------------------------
% get the parameter settings
%-------------------------------------------------------------------------------

params = ssget_defaults ;

% The ss_index.mat file is used by ssget only, not by ssgui.java.
indexfile = [ params.topdir 'files' filesep 'ss_index.mat' ] ;
indexurl  = [ params.topurl '/files/ss_index.mat' ] ;

% The ssstats.csv file is used by the ssgui.java program.  It is also used by
% the sskinds.m function, which reads the file to find the problem kind for
% each matrix in the collection.
statfile = [ params.topdir 'files' filesep 'ssstats.csv' ] ;
staturl  = [ params.topurl '/files/ssstats.csv' ] ;

%-------------------------------------------------------------------------------
% get the index file (download a new one if necessary)
%-------------------------------------------------------------------------------

refresh = 0 ;
if nargin == 0
    % if the user passed in a zero or no argument at all, return the index file
    matrix = 0 ;
else
    % ssget ('refresh') downloads the latest index file from the web
    if (ischar (matrix))
        if (strcmp (matrix, 'refresh') || strcmp (matrix, 'update'))
            matrix = 0 ;
            refresh = 1 ;
        end
    end
end

if (~refresh)
    try
        % load the existing index file
        if (nargin < 2)
            load (indexfile) ;
        end
        % see if the index file is old; if so, download a fresh copy
        fileinfo = dir (indexfile) ;
        refresh = (fileinfo.datenum + params.refresh < now) ;
    catch
        % oops, no index file, or a refresh is due.  download it.
        refresh = 1 ;
    end
end

err = '' ;      % to catch a download error, if any

if (refresh)
    % a new ss_index.mat file to get access to new matrices (if any)
    try
        % fprintf ('Downloading latest index to the collection.\n') ;

        % get a new ss_index.mat file
        tmp = tempname ;                        % download to a temp file first
        old = [ params.topdir 'files' filesep 'ss_index_old.mat' ] ;
        websave (tmp, indexurl) ;               % download the latest index file
        try
            movefile (indexfile, old, 'f') ;    % keep a backup of the old index
        catch
            % backup failed, continue anyway
        end
        movefile ([tmp '.mat'], indexfile, 'f');% move the new index into place

        % get a new ssstats.csv file
        tmp = tempname ;                        % download to a temp file first
        old = [ params.topdir 'files' filesep 'ssstats_old.csv' ] ;
        websave (tmp, staturl) ;                % download the latest stats file
        try
            movefile (statfile, old, 'f') ;     % keep a backup of the old stats
        catch
            % backup failed, continue anyway
        end
        movefile ([tmp '.csv'], statfile, 'f') ;% move the new index into place

    catch me
        err = me.message ;
    end
    load (indexfile) ;
end

%-------------------------------------------------------------------------------
% return the index file if requested
%-------------------------------------------------------------------------------

if (matrix == 0)
    if (nargout == 0)
        % no output arguments have been passed, so print the index file
        fprintf ('\nSuiteSparse Matrix Collection\n') ;
        fprintf ('(formerly called the Univ of Florida Sparse Matrix Collection)\n') ;
        fprintf ('index:  last revised %s\n', ss_index.LastRevisionDate) ;
        fprintf ('\nLegend:\n') ;
        fprintf ('(p,n)sym:  symmetry of the pattern and values\n') ;
        fprintf ('           (0 = unsymmetric, 1 = symmetric, - = not computed)\n') ;
        fprintf ('type:      real\n') ;
        fprintf ('           complex\n') ;
        fprintf ('           binary:  all entries are 0 or 1\n') ;
        nmat = length (ss_index.nrows) ;
        for j = 1:nmat
            if (mod (j, 25) == 1)
                fprintf ('\n') ;
                fprintf ('ID   Group/Name                nrows-by-  ncols  nonzeros  (p,n)sym  type\n') ;
            end
            s = sprintf ('%s/%s', ss_index.Group {j}, ss_index.Name {j}) ;
            fprintf ('%4d %-30s %7d-by-%7d %9d ', ...
            j, s, ss_index.nrows (j), ss_index.ncols (j), ss_index.nnz (j)) ;
            psym = ss_index.pattern_symmetry (j) ;
            nsym = ss_index.numerical_symmetry (j) ;
            if (psym < 0)
                fprintf ('  -  ') ;
            else
                fprintf (' %4.2f', psym) ;
            end
            if (nsym < 0)
                fprintf ('  -  ') ;
            else
                fprintf (' %4.2f', nsym) ;
            end
            if (ss_index.isBinary (j))
                fprintf (' binary\n') ;
            elseif (~ss_index.isReal (j))
                fprintf (' complex\n') ;
            else
                fprintf (' real\n') ;
            end
        end
    else
        Problem = ss_index ;
    end

    if (~isempty (err))
        fprintf ('\nssget: unable to download latest index; using old one.\n') ;
        disp (err) ;
    end
    return ;
end

%-------------------------------------------------------------------------------
% determine if the matrix parameter is a matrix index or name
%-------------------------------------------------------------------------------

[group, matname, id] = ssget_lookup (matrix, ss_index) ;

if (id == 0)
    error ('invalid matrix') ;
end

%-------------------------------------------------------------------------------
% download the matrix (if needed) and load it into MATLAB

grpdir =  [ params.topdir 'mat' filesep group ] ;
matfile = [ grpdir filesep matname '.mat' ] ;
maturl =  [ params.topurl '/mat/' group '/' matname '.mat' ] ;

if (~exist (grpdir, 'dir'))
    mkdir (grpdir) ;                        % create the Group directory
end

if (~exist (matfile, 'file'))
    fprintf ('downloading %s\n', maturl) ;
    fprintf ('to %s\n', matfile) ;
    tmp = tempname ;                        % download to a temp file first
    websave (tmp, maturl) ;
    movefile ([tmp '.mat'], matfile, 'f') ; % move the new matrix into place
end

load (matfile) ;

