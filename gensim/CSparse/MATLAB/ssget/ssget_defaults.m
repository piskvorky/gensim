function params = ssget_defaults
%SSGET_DEFAULTS returns default parameter settings for ssget.
%   Usage:  params = ssget_defaults ;
%   Returns the default parameter settings for ssget.
%   Edit this file to change these settings.
%
%   params.topurl: URL for the SuiteSparse Matrix Collection,
%       default is https://sparse.tamu.edu.  You shouldn't need to change this,
%       but you can edit this URL to access the collection from a mirror.
%
%   params.topdir: your directory for your copy of the collection.  The default
%       is the directory containing this M-file.  If you modify this file and
%       use (for example):
%           params.topdir = '/users/me/mystuff/' ;
%       then all of your copies of the matrices will reside there.  The MATLAB
%       *.mat files will be in /users/me/mystuff/mat/, Matrix Market files go
%       in /users/me/mystuff/MM, and Rutherford-Boeing files are in
%       /users/me/mystuff/RB.
%
%   params.refresh:  how many days should elapse before re-downloading the
%       index file (for obtaining access to new matrices in the collection).
%       default is 30 days.  Use 'inf' to never refresh.
%
%   Example:
%       params = ssget_defaults ;
%
%   See also ssget.

% Copyright 2009-2017, Timothy A. Davis, http://www.suitesparse.com

% decode the current directory for this M-file
s = which (mfilename) ;
i = find (s == filesep) ;
this = s (1:i(end)) ;

params.topurl = 'https://sparse.tamu.edu' ;
params.topdir = this ;
params.refresh = 30 ;

if (params.topdir (end) ~= filesep)
    % ensure a file separator appears at the end of the local directory path
    params.topdir = [params.topdir filesep] ;
end

