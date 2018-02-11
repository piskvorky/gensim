function [p,v,d] = cs_fiedler (A)
%CS_FIEDLER the Fiedler vector of a connected graph.
%   [p,v,d] = cs_fiedler(A) computes the Fiedler vector v (the eigenvector
%   corresponding to the 2nd smallest eigenvalue d of the Laplacian of the graph
%   of A+A').  p is the permutation obtained when v is sorted.  A should be a
%   connected graph.
%
%   Example:
%       [p,v,d] = cs_fiedler (A) ;
%
%   See also CS_SCC, EIGS, SYMRCM, UNMESH.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

n = size (A,1) ;
if (n < 2)
    p = 1 ; v = 1 ; d = 0 ; return ;
end
opt.disp = 0 ;                      % turn off printing in eigs
opt.tol = sqrt (eps) ;
S = A | A' | speye (n) ;            % compute the Laplacian of A
S = diag (sum (S)) - S ;
[v,d] = eigs (S, 2, 'SA', opt) ;    % find the Fiedler vector v
v = v (:,2) ;
d = d (2,2) ;
[ignore p] = sort (v) ;             % sort it to get p
