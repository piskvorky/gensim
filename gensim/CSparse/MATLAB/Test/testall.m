function testall
%TESTALL test all CSparse functions (run tests 1 to 28 below)
%
% Example:
%   testall
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

h = waitbar (0, 'CSparse') ;

cs_test_make        % compile all CSparse, Demo, Text, and Test mexFunctions

ntests = 28 ;

testwait (1,  ntests, h) ; test1  ;
testwait (2,  ntests, h) ; test2  ;
testwait (3,  ntests, h) ; test3  ;
testwait (4,  ntests, h) ; test4  ;
testwait (5,  ntests, h) ; test5  ;
testwait (6,  ntests, h) ; test6  ;
testwait (7,  ntests, h) ; test7  ;
testwait (8,  ntests, h) ; test8  ;
testwait (9,  ntests, h) ; test9  ;
testwait (10, ntests, h) ; test10 ;
testwait (11, ntests, h) ; test11 ;
testwait (12, ntests, h) ; test12 ;
testwait (13, ntests, h) ; test13 ;
testwait (14, ntests, h) ; test14 ;
testwait (15, ntests, h) ; test15 ;
testwait (16, ntests, h) ; test16 ;
testwait (17, ntests, h) ; test17 ;
testwait (18, ntests, h) ; test18 ;
testwait (19, ntests, h) ; test19 ;
testwait (20, ntests, h) ; test20 ;
testwait (21, ntests, h) ; test21 ;
testwait (22, ntests, h) ; test22 ;
testwait (23, ntests, h) ; test23 ;
testwait (24, ntests, h) ; test24 ;
testwait (25, ntests, h) ; test25 ;
testwait (26, ntests, h) ; test26 ;
testwait (27, ntests, h) ; test27 ;
testwait (28, ntests, h) ; test28 ;

close (h)
fprintf ('All CSparse tests passed\n') ;

function testwait (n,ntests,h)
fprintf ('\n------------------------ test%d\n', n) ;
waitbar (n/(ntests+1), h, sprintf ('CSparse test %d of %d\n', n, ntests)) ;

