function print_order (order)
% print_order(order) prints the ordering determined by the order parameter
% Example:
%   print_order (0)
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

switch (fix (order))
    case 0
        fprintf ('natural    ') ;
    case 1
        fprintf ('amd(A+A'')  ') ;
    case 2
        fprintf ('amd(S''*S)  ') ;
    case 3
        fprintf ('amd(A''*A)  ') ;
    otherwise
        fprintf ('undefined  ') ;
end
