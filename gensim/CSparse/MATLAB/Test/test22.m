function test22
%TEST22 test cond1est
%
% Example:
%   test22
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

index = ssget ;
[ignore f] = sort (max (index.nrows, index.ncols)) ;
f = f (1:200) ;

clf
% f = f(1)

nprob = length (f) ;
C1 = zeros (nprob,1) ;
C2 = zeros (nprob,1) ;
C3 = zeros (nprob,1) ;


for k = 1:length (f)
    
    i = f (k) ;
    Prob = ssget (i) ;
    disp (Prob) ;
    A = Prob.A ;
    [m n] = size (A) ;
    if (~isreal (A) | m ~= n)                                               %#ok
        continue
    end

    c1 = condest (A) ;
    c2 = cond1est (A) ;
    if (c1 == c2)
        err = 0 ;
    else
        err = (c1-c2)/max(1,c1) ;
    end

    c3 = cond (full (A), 1) ;

    fprintf ('%10.4e %10.4e (%10.4e)  :    %10.4e\n', c1, c2, c3, err) ;

    if (err ~= 0)
        % pause
    end

    C1 (k) = c1 ;
    C2 (k) = c2 ;
    C3 (k) = c3 ;

    subplot (1,2,1) ;
    loglog (C1, C2, 'x', [1 1e20], [1 1e20], 'r') ;
    subplot (1,2,2) ;
    loglog (C3, C2, 'x', [1 1e20], [1 1e20], 'r') ;
    drawnow

    % pause

%    if (c3 < c2)
%       c3
%       c2
%       c2-c3
%       pause
%    end

end
