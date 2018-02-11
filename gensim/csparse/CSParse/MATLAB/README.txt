CSparse/MATLAB directory, which contains the MATLAB mexFunction interfaces
for CSparse, demos, and tests.  It includes various "textbook" files
that are printed in the book, but not a proper part of CSparse itself.
It also includes "ssget", a MATLAB interface for the UF Sparse Matrix
Collection.

Type the command "cs_install" while in this directory.  It will compile
CSparse, and add the directories:

    CSparse/MATLAB/CSparse
    CSparse/MATLAB/Demo
    CSparse/MATLAB/ssget

to your MATLAB path (see the "pathtool" command to add these to your path
permanently, for future MATLAB sessions).

To run the MATLAB demo programs, run cs_demo in the Demo directory.
To run the MATLAB test programs, run testall in the Test directory.

Timothy A. Davis, http://www.suitesparse.com
