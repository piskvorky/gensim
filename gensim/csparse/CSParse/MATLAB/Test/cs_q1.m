function Q = cs_q1 (V, Beta, p)
%CS_Q1 construct Q from Householder vectors
% Example:
%   Q = cs_q1 (V, beta, p)
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

[m n] = size (V) ;
Q = speye (m) ;
if (nargin > 2)
    Q = Q (:,p) ;
end
for i = 1:m
    for k = 1:n
        Q (i,:) = Q (i,:) - ((Q(i,:) * V(:,k)) * Beta(k)) * V(:,k)' ;
    end
end
