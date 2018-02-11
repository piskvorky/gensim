function drawbox (r1,r2,c1,c2,color,w,e)
%DRAWBOX draw a box around a submatrix in the figure.
%   Used by cspy, cs_dmspy, and ccspy.
%   Example:
%       drawbox (r1,r2,c1,c2,color,w,e)
%   See also drawboxes, plot

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

if (r1 == r2 | c1 == c2)                                                    %#ok
    return
end

if (e == 1)
    r1 = r1 - .5 ;
    r2 = r2 - .5 ;
    c1 = c1 - .5 ;
    c2 = c2 - .5 ;
else
    r1 = ceil (r1 / e) - .5 ;
    r2 = ceil ((r2 - 1) / e) + .5 ;
    c1 = ceil (c1 / e) - .5 ;
    c2 = ceil ((c2 - 1) / e) + .5 ;
end

if (c2 > c1 | r2 > r1)                                                      %#ok
    plot ([c1 c2 c2 c1 c1], [r1 r1 r2 r2 r1], color, 'LineWidth', w) ;
end
