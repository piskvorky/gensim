function C = cs_add (A,B,alpha,beta)                                        %#ok
%CS_ADD sparse matrix addition.
%   C = cs_add(A,B,alpha,beta) computes C = alpha*A+beta*B,
%   where alpha and beta default to 1 if not present.
%
%   Example:
%       Prob1 = ssget ('HB/ibm32') ;        A = Prob1.A ;
%       Prob2 = ssget ('Hamrle/Hamrle1') ;  B = Prob2.A ;
%       C = cs_add (A,B) ;
%       D = A+B ;                           % same as C
%
%   See also CS_MULTIPLY, CS_GAXPY, PLUS, MINUS.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

error ('cs_add mexFunction not found') ;
