function est = mynormest1 (L, U, P, Q)
%MYNORMEST1 estimate norm(A,1), using LU factorization (L*U = P*A*Q).
%
% Example:
%   est = mynormest1 (L, U, P, Q)
% See also: testall

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

n = size (L,1) ;
est = 0 ;
S = zeros (n,1) ;

for k = 1:5

    if k == 1
        x = ones (n,1) / n ;
    else

        j = find (abs (x) == max (abs (x))) ;
        j = j (1) ;
        x = zeros (n,1) ;
        x (j) = 1 ;

        % fprintf ('eka: k %d j %d est %g\n', k, j, est) ;
    end


    % x=A\x, but use the existing P*A*Q=L*U factorization

    x = Q * (U \ (L \ (P*x))) ;

    est_old = est ;
    est = sum (abs (x)) ;

    unchanged = 1 ;
    for i = 1:n
        if (x (i) >= 0)
            s = 1 ;
        else
            s = -1 ;
        end
        if (s ~= S (i))
            S (i) = s ;
            unchanged = 0 ;
        end
    end

    if (any (S ~= signum (x)))
        S'                                                                  %#ok
        signum(x)'                                                          %#ok
        error ('Hey!') ;
    end

    if k > 1 & (est <= est_old | unchanged)                                 %#ok
        break ;
    end
    x = S ;

    % x=A'\x, but use the existing P*A*Q=L*U factorization
    x = P' * (L' \ (U' \ (Q'*x))) ;

    if k > 1
        jnew = find (abs (x) == max (abs (x))) ;
        if (jnew == j)
            break ;
        end
    end 

end

for k = 1:n
    x (k) = power (-1, k+1) * (1 + ((k-1)/(n-1))) ;
end

% x=A\x, but use the existing P*A*Q=L*U factorization
x = Q * (U \ (L \ (P*x))) ;

est_new = 2 * sum (abs (x)) / (3 * n) ;
if (est_new > est)
    est = est_new ;
end




function s = signum (x)
%SIGNUM compute sign of x
s = ones (length (x),1) ;
s (find (x < 0)) = -1 ;     %#ok
