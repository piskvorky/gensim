function x = cs_dmsol (A,b)
%CS_DMSOL x=A\b using the coarse Dulmage-Mendelsohn decomposition.
%   x = cs_dmsol(A,b) computes x=A\b where A may be rectangular and/or
%   structurally rank deficient, and b is a full vector.
%
%   Example:
%       Prob = ssget ('HB/arc130') ; A = Prob.A ; b = rand (size (A,1),1) ;
%       x = cs_dmsol (A,b) ; norm (A*x-b)
%
%   See also CS_QRSOL, CS_LUSOL, CS_DMPERM, SPRANK, RANK.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

[m n] = size (A) ;
[p q r s cc rr] = cs_dmperm (A) ;
C = A (p,q) ;
b = b (p) ;
x = zeros (n,1) ;
if (rr(3) <= m & cc(4) <= n)                                                %#ok
    x (cc(4):n) = cs_qrsol (C (rr(3):m, cc(4):n), b (rr(3):m)) ;
    b (1:rr(3)-1) = b (1:rr(3)-1) - C (1:rr(3)-1, cc(4):n) * x (cc(4):n) ;
end
if (rr(2) < rr (3) & cc(3) < cc(4))                                         %#ok
    x (cc(3):cc(4)-1) = ...
        cs_lusol (C (rr(2):rr(3)-1, cc(3):cc(4)-1), b (rr(2):rr(3)-1)) ;
    b (1:rr(2)-1) = ...
        b (1:rr(2)-1) - C (1:rr(2)-1, cc(3):cc(4)-1) * x (cc(3):cc(4)-1) ;
end
if (rr(2) > 1 & cc(3) > 1)                                                  %#ok
    x (1:cc(3)-1) = cs_qrsol (C (1:rr(2)-1, 1:cc(3)-1), b (1:rr(2)-1)) ;
end
x (q) = x ;
