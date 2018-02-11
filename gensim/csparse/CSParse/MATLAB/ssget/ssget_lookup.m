function [group, name, id] = ssget_lookup (matrix, ss_index)
%SSGET_LOOKUP gets the group, name, and id of a matrix.
%   This is a helper function for ssget, sssvd, and ssweb.
%   The 'matrix' argument is a number (matrix id) or string containing
%   group/name, as in 'HB/arc130' for example.
%
%   Example:
%       [group name id] = ssget_lookup (matrix, ss_index)
%
%   See also ssget.

% Copyright 2009-2017, Timothy A. Davis, http://www.suitesparse.com

if (isnumeric (matrix))

    % make sure that the matrix parameter is only one integer value
    % this means that if we get an array, just use the first value
    id = fix (full (matrix (1))) ;

    % if the index is less than one or bigger than the length of the array,
    % then no particular matrix is accessed
    if (id > length (ss_index.nrows) | id < 1)                              %#ok
        id = 0 ;
        group = '' ;
        name = '' ;
    else
        % assign the group and name for the given id
        group = ss_index.Group {matrix} ;
        name = ss_index.Name {matrix} ;
    end

elseif (ischar (matrix))

    % the group and matrix name are in the string as in GroupName/MatrixName

    % find the group index for the file separator
    % check both types of slashes, and a colon
    gi = find (matrix == '/') ;
    if (length (gi) == 0)                                                   %#ok
        gi = find (matrix == '\') ;
    end
    if (length (gi) == 0)                                                   %#ok
        gi = find (matrix == ':') ;
    end

    % if no name divider is in the string, a whole group is specified
    if (length (gi) == 0)                                                   %#ok

        id = 0 ;
        group = matrix ;
        name = '' ;

    else

        % group equals the first part of the string up to the character before
        % the last file separator
        group = matrix (1:gi(end)-1) ;

        % group equals the last section of the string after the last file
        % separator
        name = matrix (gi(end)+1:end) ;

        % validate the given name and group by checking the index for a match
        % refName = strmatch (name, ss_index.Name) ;
        % refGroup = strmatch (group, ss_index.Group) ;
        % id = intersect (refName, refGroup) ;
        refName  = strcmp (name,  ss_index.Name) ;
        refGroup = strcmp (group, ss_index.Group) ;
        id = find (refName & refGroup) ;
        if (length (id) >= 1)
            id = id (1) ;
        else
            % the given group/matrix does not exist in the index file
            id = 0 ;
        end
    end

else

    % there is an error in the argument types passed into the function
    error ('invalid input') ;

end
