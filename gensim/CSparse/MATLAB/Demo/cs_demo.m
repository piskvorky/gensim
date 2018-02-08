function cs_demo (do_pause, matrixpath)
%CS_DEMO run all CSparse demos.
%   cs_demo(0) will run all demos without pausing.
%
% Example:
%   cs_demo
% See also: cs_demo1, cs_demo2, cs_demo3

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

help cs_demo
if (nargin < 1)
    do_pause = 1 ;
end
if (nargin < 2)
    matrixpath = [] ;
end

clf

fprintf ('\n\n-------------------------------------------------------\n') ;
help cs_demo1 ;
cs_demo1 (matrixpath) ;

fprintf ('\n\n-------------------------------------------------------\n') ;
help cs_demo2
cs_demo2 (do_pause, matrixpath) ;

fprintf ('\n\n-------------------------------------------------------\n') ;
help cs_demo3
cs_demo3 (do_pause, matrixpath) ;

fprintf ('\n\n-------------------------------------------------------\n') ;
help private/ex_1
ex_1

fprintf ('\n\n-------------------------------------------------------\n') ;
help private/ex2
ex2

fprintf ('\n\n-------------------------------------------------------\n') ;
help private/ex3
ex3

fprintf ('\nAll CSparse demos finished.\n') ;
