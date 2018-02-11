function [s, t, tobj] = cs_must_compile (srcdir, f, suffix, obj, hfile, force)
%CS_MUST_COMPILE return 1 if source code f must be compiled, 0 otherwise
%   Used by cs_make, and MATLAB/Test/cs_test_make.m.
%
%   Example:
%        none, not meant for end users.
%   See also: CS_MAKE.

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

dobj = dir ([f obj]) ;
if (force | isempty (dobj))                                                 %#ok
    s = 1 ;
    t = Inf ;
    tobj = -1 ;
    return
end
dsrc = dir ([srcdir f suffix '.c']) ;
dh = dir (hfile) ;
t = max (datenum (dsrc.date), datenum (dh.date)) ;
tobj = datenum (dobj.date) ;
s = (tobj < t) ;
