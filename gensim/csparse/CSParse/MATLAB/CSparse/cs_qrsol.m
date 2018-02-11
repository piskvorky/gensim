function x = cs_qrsol (A,b,order)                                           %#ok
%CS_QRSOL solve a sparse least-squares problem.
%   x = cs_qrsol(A,b) solves the over-determined least squares problem to
%   find x that minimizes norm(A*x-b), where b is a full vector and
%   A is m-by-n with m >= n.  If m < n, it solves the underdetermined system
%   Ax=b.  A 3rd input argument specifies the ordering method to use
%   (0: natural, 3: amd(A'*A)).  The default ordering is 3.
%
%   Example:
%       Prob = ssget ('HB/well1033') ; A = Prob.A ; [m n] = size (A) ;
%       b = rand (m,1) ;
%       x1 = cs_qrsol (A,b) ;
%       x2 = A\b ;
%       norm (x1-x2)
%
%   For this example, cs_qrsol is about 3 times faster than A\b in MATLAB 7.3.
%
%   See also CS_QR, CS_AMD, CS_LUSOL, CS_CHOLSOL, MLDIVIDE.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

error ('cs_qrsol mexFunction not found') ;
