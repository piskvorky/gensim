function drawboxes (nb, e, r, s)
%DRAWBOXES: helper function for cs_dmpsy and ccspy
% Example:
%   drawboxes (nb, e, r, s)
%   See also drawbox, plot

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

if (nb > 1)
    if (e == 1)
        r1 = r (1:nb) - .5 ;
        r2 = r (2:nb+1) - .5 ;
        c1 = s (1:nb) - .5 ;
        c2 = s (2:nb+1) - .5 ;
    else
        r1 = ceil (r (1:nb) / e) - .5 ;
        r2 = ceil ((r (2:nb+1) - 1) / e) + .5 ;
        c1 = ceil (s (1:nb) / e) - .5 ;
        c2 = ceil ((s (2:nb+1) - 1) / e) + .5 ;
    end
    kk = find (diff (c1) > 0 | diff (c2) > 0 | diff (r1) > 0 | diff (r2) > 0) ;
    kk = [1 kk+1] ;
    for k = kk
        plot ([c1(k) c2(k) c2(k) c1(k) c1(k)], ...
              [r1(k) r1(k) r2(k) r2(k) r1(k)], 'k', 'LineWidth', 1) ;
    end
end

