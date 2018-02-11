function print_resid (A, x, b)
% print_resid (A, x, b), print the relative residual,
% norm (A*x-b,inf) / (norm(A,1)*norm(x,inf) + norm(b,inf))
% Example:
%   print_resid (A, x, b) ;
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

fprintf ('resid: %8.2e\n', ...
    norm (A*x-b,inf) / (norm(A,1)*norm(x,inf) + norm(b,inf))) ;
