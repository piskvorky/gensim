function s = signum (x)
%SIGNUM compute and display the sign of a column vector x
% Example
%   s = signum(x)
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

s = ones (length (x),1) ;
s (find (x < 0)) = -1 ;     %#ok
disp ('s =') ;
disp (s) ;
