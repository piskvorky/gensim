function check_if_same (p1,p2)
%CHECK_IF_SAME check if two inputs are identical or not
%
% Example:
%   check_if_same (1:5, 2:6)
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

    if (isempty (p1))
        if (~isempty (p2))
            p1          %#ok
            p2          %#ok
            error ('empty!') ;
        end
    elseif (any (p1 ~= p2))
        p1              %#ok
        p2              %#ok
        error ('!') ;
    end

