function cs_test_make (force)
%CS_TEST_MAKE compiles the CSparse, Demo, and Test mexFunctions.
%   The current directory must be CSparse/MATLAB/Test to use this function.
%
% Example:
%   cs_test_make
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

try
    % ispc does not appear in MATLAB 5.3
    pc = ispc ;
catch
    % if ispc fails, assume we are on a Windows PC if it's not unix
    pc = ~isunix ;
end

if (~isempty (strfind (computer, '64')))
    fprintf ('Compiling CSparse (64-bit)\n') ;
    mexcmd = 'mex -largeArrayDims' ;
else
    fprintf ('Compiling CSparse (32-bit)\n') ;
    mexcmd = 'mex' ;
end

if (pc)
    here = pwd ;
    cd ../../Include
    mexcmd = [mexcmd ' -I' pwd] ;
    cd (here)
    cd ../CSparse
    mexcmd = [mexcmd ' -I' pwd] ;
    cd (here)
else
    mexcmd = [mexcmd ' -I../../Include -I../CSparse'] ;
end

if (nargin < 1)
    force = 0 ;
end

cd ('../CSparse') ;
[object_files timestamp] = cs_make ;
cd ('../Test') ;

mexfunc = { 'cs_ipvec', 'cs_pvec', 'cs_sparse2', ...
    'cs_reach', 'cs_maxtransr', 'cs_reachr', 'cs_rowcnt', 'cs_frand' } ;

for i = 1:length(mexfunc)
    [s t tobj] = cs_must_compile ('', mexfunc{i}, '_mex', ...
        ['.' mexext], 'cs_test_make.m', force) ;
    if (s | tobj < timestamp)                                               %#ok
        cmd = [mexcmd ' -O -output ' mexfunc{i} ' ' mexfunc{i} '_mex.c ' ...
            object_files] ;
        fprintf ('%s\n', cmd) ;
        eval (cmd) ;
     end
end
