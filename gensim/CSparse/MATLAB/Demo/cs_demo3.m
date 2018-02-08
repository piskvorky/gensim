function cs_demo3 (do_pause, matrixpath)
%CS_DEMO3 MATLAB version of the CSparse/Demo/cs_demo3.c program.
%   Cholesky update/downdate.
%
% Example:
%   cs_demo3
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

if (nargin < 2)
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

matrices = { 'HB/bcsstk01', 'HB/bcsstk16' } ;

if (nargin < 1)
    do_pause = 1 ;
end

for i = 1:length(matrices)
    name = matrices {i} ;
    [C sym] = get_problem (matrixpath, name, 1e-14) ;
    demo3 (C, sym, name) ;
    if (do_pause)
        input ('Hit enter to continue: ') ;
    end
end
