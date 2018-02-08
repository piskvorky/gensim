function [s,M,H] = cspy (A,res)
%CSPY plot a matrix in color.
%   cspy(A) plots a matrix, in color, with a default resolution of
%   256-by-256.  cspy(A,res) changes the resolution to res.  Zero entries are
%   white.  Entries with tiny absolute value are light orange.  Entries with
%   large magnitude are black.  Entries in the midrange (the median of the
%   log10 of the nonzero values, +/- one standard deviation) range from light
%   green to deep blue.  With no inputs, the color legend of cspy is plotted.
%   [s,M,H] = cspy(A) returns the scale factor s, the image M, and colormap H.
%
%   The matrix A can be full or sparse, and either numeric (double, single,
%   integer) or character type, and either complex or real.
%
%   Example
%       A = delsq (numgrid ('L', 10)) ;
%       cspy (A) ;
%
%   See also CS_DMSPY, SPY.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

if nargin < 2
    res = 256 ;
end

h = jet (64) ;
h = h (64:-1:1,:) ;
h = h (30:end,:) ;
hmax = size (h,1) ;

h (1,:)  = [1 1 1] ;    % white for zero
h (2,:)  = [1 .9 .5] ;  % light orange for tiny entries
h (hmax,:) = [0 0 0] ;  % black for very large entries
colormap (h) ;

if (nargin == 0)
    image (1:hmax) ;
    title ('cspy color map') ;
    return
end

% convert complex, integers, and strings to real double
if (~isreal (A) | ~isa (A, 'double') | ~issparse (A))                       %#ok
    A = sparse (abs (double (A))) ;
end

[m1 n1] = size (A) ;
if (m1 == 0 | n1 == 0)                                                      %#ok
    A (1,1) = 0 ;
end
[m1 n1] = size (A) ;

S = cs_thumb (A,res) ;      % get the thumbnail of the matrix
[m n] = size (S) ;
[i j x] = find (S) ;
x = log10 (x) ;

if (isempty (x))
    S = zeros (size (S)) ;
else
    med = median (x) ;
    sdev = std (x) ;
    big = med + sdev ;
    tiny = med - sdev ;
    imid = find (x > tiny & x < big) ;
    itiny = find (x <= tiny) ;
    ibig = find (x >= big) ;
    x (imid) = 1 + ceil ((hmax-2) * (x (imid) - tiny) / (big - tiny)) ;
    x (itiny) = 1 ;                                                         %#ok
    x (ibig) = hmax-1 ;                                                     %#ok
    S = full (1 + sparse (i,j,x,m,n)) ;

%   title (sprintf ('tiny: %-8.2g   median: %-8.2g   big: %-8.2g\n', ...
%       10^tiny, 10^med, 10^big)) ;
end

% draw the matrix
image (S) ;
axis equal ;
axis ([-1 n+1 -1 m+1]) ;
axis off

% draw a box around the whole matrix
e = ceil (max (m1,n1) / max (m,n)) ;    % scale factor
hold on
drawbox (1,m1+1,1,n1+1,'k',1,e) ;
hold off

% return results
if (nargout > 0)
    s = e ;
end
if (nargout > 1)
    M = S ;             % image
end
if (nargout > 2)
    H = h ;             % colormap
end

