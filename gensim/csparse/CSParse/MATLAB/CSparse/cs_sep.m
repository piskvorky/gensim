function [s,as,bs] = cs_sep (A,a,b)
%CS_SEP convert an edge separator into a node separator.
%   [s,as,bs] = cs_sep (A,a,b) converts an edge separator into a node separator.
%   [a b] is a partition of 1:n, thus the edges in A(a,b) are an edge separator
%   of A.  s is the node separator, consisting of a node cover of the edges of
%   A(a,b).  as and bs are the sets a and b with s removed.
%
%   Example:
%       type cs_nsep ; % to see a simple example of use in cs_nsep.m
%
%   See also CS_DMPERM, CS_NSEP, CS_ESEP, CS_ND.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

[p q r s cc rr] = cs_dmperm (A (a,b)) ;
s = [(a (p (1:rr(2)-1))) (b (q (cc(3):(cc(5)-1))))] ;
w = ones (1, size (A,1)) ;
w (s) = 0 ;
as = a (find (w (a))) ;                                                     %#ok
bs = b (find (w (b))) ;                                                     %#ok
