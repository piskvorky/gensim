function z = cs_gaxpy (A,x,y)                                               %#ok
%CS_GAXPY sparse matrix times vector.
%   z = cs_gaxpy(A,x,y) computes z = A*x+y where x and y are full vectors.
%
%   Example:
%       Prob = ssget ('HB/arc130') ; A = Prob.A ; [m n] = size (A) ;
%       x = rand (n,1) ; y = rand (m,1) ;
%       z = cs_gaxpy (A, x, y) ;
%       
%   See also PLUS, MTIMES.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

error ('cs_gaxpy mexFunction not found') ;
