function another_colormap
%ANOTHER_COLORMAP try another color map
%
% Example:
%   another_colormap
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

j = jet (128) ;
j = j (48:112, :) ;

% jj = linspace (0,1,64)' ./ sum (jet,2) ;
% j (:,1) = j (:,1) .* jj ;
% j (:,2) = j (:,2) .* jj ;
% j (:,3) = j (:,3) .* jj ;


% white = [1 1 1] ;
% gray = [.5 .5 .5] ;                                                       %#ok

% j = [white ; purple ; j ] ;
disp ('j = ') ;
disp (j)


image (1:size(j,1)) ;
colormap (j) ;
