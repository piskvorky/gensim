function [p, q, r, s] = ccspy (A, bipartite, res)
%CCSPY plot the connected components of a matrix.
%
%   Example:
%       [p, q, r, s] = ccspy (A, bipartite, res)
%
%   If A is square, [p,q,r,s] = ccspy(A) finds a permutation p so that A(p,q)
%   is permuted into block upper triangular form.  In this case, r=s, p=q and
%   the kth diagonal block is given by A (t,t) where t = r(k):r(k+1)-1. 
%   The diagonal of A is ignored.
%
%   If A is not square (or for [p,q,r,s] = ccspy(A,1)), then the connected
%   components of the bipartite graph of A are found.  A(p,q) is permuted into
%   block diagonal form, where the diagonal blocks are rectangular.  The kth
%   block is given by A(r(k):r(k+1)-1,s(k):s(k+1)-1).  A can be rectangular.
%
%   It then plots the result via cspy, drawing a greenbox around each component.
%   A 3rd input argument (res) controls the resolution (see cspy for a
%   description of the res parameter).
%
%   See also CSPY, CS_DMPERM, DMPERM, CS_SCC, CS_SCC2, CS_DMSPY.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

if (~issparse (A))
    A = sparse (A) ;
end
[m n] = size (A) ;
if (nargin < 3)
    res = 256 ;
end
if (nargin < 2)
    bipartite = [ ] ;
end
if (isempty (bipartite))
    bipartite = (m ~= n) ;
end

% find the strongly connected components
[p1 q r s] = cs_scc2 (A, bipartite) ;
if (nargout > 0)
    p = p1 ;
end

nb = length (r)-1 ;

% plot the result
S = A (p1,q) ;
if (res == 0)
    spy (S) ;
    e = 1 ;
else
    e = cspy (S,res) ;
end
hold on

title (sprintf ('%d-by-%d, strongly connected commponents: %d\n', m, n, nb)) ;

if (~bipartite)
    plot ([.5 .5 n+.5 n+.5], [.5 .5 n+.5 n+.5], 'r') ;
end

drawboxes (nb, e, r, s) ;

drawbox (1,m+1,1,n+1,'k',1,e) ;
hold off
