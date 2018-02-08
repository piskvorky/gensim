function demo3 (C, sym, name)
%DEMO3: Cholesky update/downdate
%
% Example:
%   demo3 (C, 1, 'name of system')
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

clf
subplot (2,2,1) ; cspy (C) ;
title (name, 'FontSize', 16, 'Interpreter', 'none') ;
drawnow

[m n] = size (C) ;
if (m ~= n | ~sym)                                                          %#ok
    return ;
end

b = rhs (n) ;
fprintf ('chol then update/downdate ') ;
print_order (0) ;

tic ;
[L,p] = cs_chol (C) ;
t = toc ;
fprintf ('\nchol  time: %8.2f\n', t) ;

subplot (2,2,2) ; cspy (L) ; title ('L') ;
drawnow

tic ;
x = b (p) ;
x = cs_lsolve (L,x) ;
x = cs_ltsolve (L,x) ;
x (p) = x ;
t = toc ;
fprintf ('solve time: %8.2f\n', t) ;

fprintf ('original: ') ;
print_resid (C, x, b) ;

k = fix (n/2) ;
w = L(k,k) * sprand (L (:,k)) ;

parent = cs_etree (C (p,p)) ;

tic ;
L2 = cs_updown (L, w, parent, '+') ;
t1 = toc ;
fprintf ('update:   time: %8.2f\n', t1) ;

subplot (2,2,3) ; cspy (L2) ; title ('updated L') ;
subplot (2,2,4) ; cspy (L-L2) ; title ('L - updated L') ;
drawnow

tic ;
x = b (p) ;
x = cs_lsolve (L2,x) ;
x = cs_ltsolve (L2,x) ;
x (p) = x ;
t = toc ;

w2 = sparse (n,1) ;
w2 (p) = w ;                    % w2 = P'*w
wt = cs_transpose (w2) ;
ww = cs_multiply (w2,wt) ;
E = cs_add (C, ww, 1, 1) ;      % E = C + w2*w2' ;

fprintf ('update:   time: %8.2f (incl solve) ', t1+t) ;
print_resid (E, x, b) ;

tic
[L,p2] = cs_chol (E) ;
x = b (p2) ;
x = cs_lsolve (L,x) ;
x = cs_ltsolve (L,x) ;
x (p2) = x ;
t = toc ;
fprintf ('rechol:   time: %8.2f (incl solve) ', t) ;
print_resid (E, x, b) ;

tic ;
L3 = cs_updown (L2, w, parent, '-') ;
t1 = toc ;
fprintf ('downdate: time: %8.2f\n', t1) ;

tic ;
x = b (p) ;
x = cs_lsolve (L3,x) ;
x = cs_ltsolve (L3,x) ;
x (p) = x ;
t = toc ;
fprintf ('downdate: time: %8.2f (incl solve) ', t1+t) ;
print_resid (C, x, b) ;
