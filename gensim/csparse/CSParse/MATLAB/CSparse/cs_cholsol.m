function x = cs_cholsol (A,b,order)                                         %#ok
%CS_CHOLSOL solve A*x=b using a sparse Cholesky factorization.
%   x = cs_cholsol(A,b) computes x = A\b, where A sparse symmetric positive
%   definite, and b is a full vector.  A 3rd input parameter allows the
%   ordering to be modified: 0: natural, 1:amd(A), 2: amd(S'*S) where S=A except
%   with no dense rows, 3:amd(A'*A).  The default ordering option is 1.
%
%   Example:
%       Prob = ssget ('HB/bcsstk01') ; A = Prob.A ; b = rand (size (A,1),1) ;
%       x = cs_cholsol (A,b) ; norm (A*x-b)
%
%   See also CS_CHOL, CS_AMD, CS_LUSOL, CS_QRSOL, MLDIVIDE.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

error ('cs_cholsol mexFunction not found') ;
