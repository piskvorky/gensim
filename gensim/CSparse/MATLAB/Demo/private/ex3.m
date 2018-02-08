function ex3
%EX3: create 2D and 3D meshes using mesh2d1, mesh2d2, mesh3d1, mesh3d2.

% Example:
%   ex3
% See also: cs_demo

% Copyright 2006-2012, Timothy A. Davis, http://www.suitesparse.com

t1 = zeros (50,1) ;
t2 = zeros (50,1) ;
t3 = zeros (50,1) ;
t4 = zeros (50,1) ;

fprintf ('run times for each method, given n:\n') ;
for n = 2:50

    tic ;
    A = mesh2d1 (n) ;
    t1 (n) = toc ;

    tic
    B = mesh2d2 (n) ;
    t2 (n) = toc ;

    tic
    C = mesh3d1 (n) ;
    t3 (n) = toc ;

    tic
    D = mesh3d2 (n) ;
    t4 (n) = toc ;

    fprintf ('%3d: %8.3f %8.3f %8.3f %8.3f\n', n, t1(n), t2(n), t3(n), t4(n)) ;

    subplot (2,2,1) ; spy (A) ; title ('2D mesh, method 1') ;
    subplot (2,2,2) ; spy (B) ; title ('2D mesh, method 2') ;
    subplot (2,2,3) ; spy (C) ; title ('3D mesh, method 1') ;
    subplot (2,2,4) ; spy (D) ; title ('3D mesh, method 2') ;
    drawnow
end
