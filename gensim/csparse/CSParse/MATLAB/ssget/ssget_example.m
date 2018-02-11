%SSGET_EXAMPLE a demo for ssget.
%   This example script gets the index file of the SuiteSparse Matrix
%   Collection, and then loads in all symmetric non-binary matrices, in
%   increasing order of number of rows in the matrix.
%
%   Example:
%       ssget_example
%
%   See also ssget, ssweb, ssgrep.

% Copyright 2009-2017, Timothy A. Davis, http://www.suitesparse.com

type ssget_example ;

index = ssget ;
f = find (index.numerical_symmetry == 1 & ~index.isBinary) ;
[y, j] = sort (index.nrows (f)) ;
f = f (j) ;

for i = f
    fprintf ('Loading %s%s%s, please wait ...\n', ...
        index.Group {i}, filesep, index.Name {i}) ;
    Problem = ssget (i,index) ;
    disp (Problem) ;
    spy (Problem.A) ;
    title (sprintf ('%s:%s', Problem.name, Problem.title')) ;
    ssweb (i) ;
    input ('hit enter to continue:') ;
end

