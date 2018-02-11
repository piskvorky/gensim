function cs_demo1 (matrixpath)
%CS_DEMO1 MATLAB version of the CSparse/Demo/cs_demo1.c program.
% Uses both MATLAB functions and CSparse mexFunctions, and compares the two
% results.  This demo also plots the results, which the C version does not do.
%
% Example:
%   cs_demo1
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

if (nargin < 1)
    matrixpath = [] ;
end

if (isempty (matrixpath))
    try
        % older versions of MATLAB do not have an input argument to mfilename
        p = mfilename ('fullpath') ;
        t = strfind (p, '/') ;
        matrixpath = [ p(1:t(end)) '../../Matrix' ] ;
    catch
        % assume we are in the C*Sparse/MATLAB/CSparse/Demo directory
        matrixpath = '../../Matrix' ;
    end
end

t1 = load ([matrixpath '/t1']) ;

T = t1                                                                      %#ok
A  = sparse    (T(:,1)+1, T(:,2)+1, T(:,3))                                 %#ok
A2 = cs_sparse (T(:,1)+1, T(:,2)+1, T(:,3))                                 %#ok
fprintf ('A difference: %g\n', norm (A-A2,1)) ;
% CSparse/Demo/cs_demo1.c also clears the triplet matrix T at this point:
% clear T 
clf
subplot (2,2,1) ; cspy (A) ; title ('A', 'FontSize', 16) ;
AT = A'                                                                     %#ok
AT2 = cs_transpose (A)                                                      %#ok
fprintf ('AT difference: %g\n', norm (AT-AT2,1)) ;
subplot (2,2,2) ; cspy (AT) ; title ('A''', 'FontSize', 16) ;
n = size (A,2) ;
I = speye (n) ;
C = A*AT ;
C2 = cs_multiply (A, AT)                                                    %#ok
fprintf ('C difference: %g\n', norm (C-C2,1)) ;
subplot (2,2,3) ; cspy (C) ; title ('C=A*A''', 'FontSize', 16) ;
cnorm = norm (C,1) ;
D = C + I*cnorm                                                             %#ok
D2 = cs_add (C, I, 1, cnorm)                                                %#ok
fprintf ('D difference: %g\n', norm (D-D2,1)) ;
subplot (2,2,4) ; cspy (D) ; title ('D=C+I*norm(C,1)', 'FontSize', 16) ;
% CSparse/Demo/cs_demo1.c clears all matrices at this point:
% clear A AT C D I
% clear A2 AT2 C2 D2
