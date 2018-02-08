function L = cholup (Lold,w)
%CHOLUP Cholesky update, using Given's rotations
% given Lold and w, compute L so that L*L' = Lold*Lold' + w*w'
% Example:
%   L = cholup (Lold,w)
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

n = size (Lold,1) ;
L = [Lold w] ;

for k = 1:n

    g = givens (L(k,k), L(k,n+1)) ;

    L (:, [k n+1]) = L (:, [k n+1]) * g' ;

    disp ('L:') ;
    disp (L)
    pause
end

L = L (:,1:n) ;
disp (L) ;
