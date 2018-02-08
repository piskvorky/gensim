function sample_colormap
%SAMPLE_COLORMAP try a colormap for use in cspy
%
% Example:
%   sample_colormap
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

h = jet (64) ;
h = h (64:-1:1,:) ;
h = h (20:end,:) ;

% h = h (17:128,:) ;

% s = sum (jet,2) ;
% h (:,1) = h (:,1) ./ s ;
% h (:,2) = h (:,2) ./ s ;
% h (:,3) = h (:,3) ./ s ;

h (1,:) = [1 1 1] ;     % white
h (2,:) = [1 1 .8] ;    % light yellow

% h
colormap (h) ;

clf
subplot (5,1,1) ;
image (1:size(h,1)) ;

h = rgb2hsv (h) ;
% h (:,3) = linspace (1,0,64) ;
% h= hsv2rgb (h) ;

subplot (5,1,2)  ; plot (h(:,1)) ; axis ([1 64 0 1]) ; ylabel ('red') ;
subplot (5,1,3)  ; plot (h(:,2)) ; axis ([1 64 0 1]) ; ylabel ('green') ;
subplot (5,1,4)  ; plot (h(:,3)) ; axis ([1 64 0 1]) ; ylabel ('blue') ;
subplot (5,1,5)  ; plot (sum(h,2)) ; axis ([1 64 0 3]) ; ylabel ('sum') ;

