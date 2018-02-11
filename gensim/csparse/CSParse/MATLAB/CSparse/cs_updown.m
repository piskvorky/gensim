function L = cs_updown (L, c, parent, sigma)                                %#ok
%CS_UPDOWN rank-1 update/downdate of a sparse Cholesky factorization.
%   L = cs_updown(L,c,parent) computes the rank-1 update L = chol(L*L'+c*c')',
%   where parent is the elimination tree of L.  c must be a sparse column
%   vector, and find(c) must be a subset of find(L(:,k)) where k = min(find(c)).
%   L = cs_updown(L,c,parent,'-') is the downdate L = chol(L*L'-c*c').
%   L = cs_updown(L,c,parent,'+') is the update L = chol(L*L'+c*c').
%   Updating/downdating is much faster than refactorizing the matrix with
%   cs_chol or chol.  L must not have an entries dropped due to numerical
%   cancellation (use cs_chol(A,0)).
%
%   Example:
%       Prob = ssget ('HB/bcsstk01') ; A = Prob.A ; n = size (A,1) ;
%       L = cs_chol (A,0) ;
%       parent = cs_etree (A) ;
%       c = sprand (L (:, floor(n/2))) ;
%       L1 = cs_updown (L, c, parent) ;
%       L2 = cs_chol (A + c*c', 0) ;
%       norm (L1-L2, 1)
%
%   See also CS_ETREE, CS_CHOL, ETREE, CHOLUPDATE, CHOL.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

error ('cs_updown mexFunction not found') ;
