function cs_install (do_pause)
%CS_INSTALL: compile and install CSparse for use in MATLAB.
%   Your current working directory must be CSparse/MATLAB in order to use this
%   function.
%   
%   The directories
%
%       CSparse/MATLAB/CSparse
%       CSparse/MATLAB/Demo
%       CSparse/MATLAB/ssget
%
%   are added to your MATLAB path (see the "pathtool" command to add these to
%   your path permanently, for future MATLAB sessions).
%
%   Next, the MATLAB CSparse demo program, CSparse/MATLAB/cs_demo is executed.
%   To run the demo with pauses so you can see the results, use cs_install(1).
%   To run the full MATLAB test programs for CSparse, run testall in the
%   Test directory.
%
%   Example:
%       cs_install          % install and run demo with no pauses
%       cs_install(1)       % install and run demo with pauses
%
%   See also: cs_demo
%
%   Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

fprintf ('Compiling and installing CSparse\n') ;
if (nargin < 1)
    do_pause = 0 ;
end

if (do_pause)
    input ('Hit enter to continue: ') ;
end
addpath ([pwd '/CSparse']) ;
addpath ([pwd '/Demo']) ;

if (verLessThan ('matlab', '8.4'))
    fprintf ('ssget not installed (MATLAB 8.4 or later required)\n') ;
else
    % install ssget, unless it's already in the path
    try
        % if this fails, then ssget is not yet installed
        index = ssget ;
        fprintf ('ssget already installed:\n') ;
        which ssget
    catch
        index = [ ] ;
    end
    if (isempty (index))
        % ssget is not installed.  Use ./ssget
        fprintf ('Installing ./ssget\n') ;
        try
            addpath ([pwd '/ssget']) ;
        catch me
            disp (me.message) ;
            fprintf ('ssget not installed\n') ;
        end
    end
end

cd ('CSparse') ;
cs_make (1) ;
cd ('../Demo') ;
cs_demo (do_pause)
