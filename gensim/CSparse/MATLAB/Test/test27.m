function test27
%TEST27 test cs_qr, cs_utsolve, cs_qrsol
%
% Example:
%   test27
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

clear functions
rand ('state', 0) ;

Prob = ssget ('HB/ibm32') ;
A = Prob.A ;
A = A (1:10,:) ;
[m n] = size (A) ;

[V,Beta,p,R,q] = cs_qr (A') ;

b = rand (m,1) ;

Rm = R (1:m,1:m) ;

bq = b (q) ;
rtbq = Rm' \ bq ;

rt2 = cs_utsolve (Rm, bq) ;

norm (rtbq - rt2)

x = [rt2 ; zeros(n-m,1)] ;

for k = m:-1:1
    x = x - V(:,k) * (Beta (k) * (V (:,k)' * x)) ;
end

x (p) = x ;

norm (A*x-b)

x2 = cs_qrsol (A,b) ;

norm (A*x2-b)
norm (x-x2)
