ssget:  MATLAB and Java interfaces to the SuiteSparse Matrix Collection
(Formerly the University of Florida Sparse Matrix Collection).
Copyright 2005-2017, Timothy A. Davis, http://www.suitesparse.com,
Texas A&M University.

REQUIREMENTS:

    Java JRE 1.8.0 or later is required for the ssgui Java program.
    ssget requires a recent version of MATLAB (R2014b or later).

See http://www.suitesparse.com
for a single archive file with all the files listed below:

    ssget/README.txt            this file

    for Java:
    ssget/ssgui.java            a stand-alone Java interface to the collection
    ssget/ssgui.jar             the compiled ssgui program
    ssget/sshelp.html           help for ssgui
    ssget/Makefile              for compiling ssgui.java into ssgui.jar
    ssget/files/ssstats.csv     matrix statistics file for ssgui.java and
                                sskinds.m

    for MATLAB:
    ssget/Contents.m            help for ssget in MATLAB
    ssget/ssget_defaults.m      default parameter settings for ssget.m
    ssget/ssget_example.m       demo for ssget
    ssget/ssget_lookup.m        get the group, name, and id of a matrix
    ssget/ssget.m               primary user interface
    ssget/ssgrep.m              searches for matrices by name
    ssget/sskinds.m             returns the 'kind' for all matrices
    ssget/ssweb.m               opens the URL for a matrix or collection
    ssget/files/ss_index.mat    index to the SuiteSparse Matrix Collection

    download directories:
    ssget/MM                    for Matrix Market files
    ssget/RB                    for Rutherford/Boeing files
    ssget/mat                   for *.mat files
    ssget/files                 for *.png icon images of the matrices, and the
                                index files ss_index.mat and sstats.csv
    ssget/svd                   singular values (for smaller matrices)

    ssget/Doc                   ChangeLog and license

--------------------------------------------------------------------------------
For the Java ssgui program:
--------------------------------------------------------------------------------

    To run the ssgui on Windows or Mac OS X, just double-click the ssgui.jar
    file.  Or, on any platform, type the following command in your command
    window:

        java -jar ssgui.jar

    or just type 'make run' on Unix/Linux/MacOS.

    If that doesn't work, then you need to install the Java JDK or JRE and add
    it to your path.  See http://java.sun.com/javase/downloads/index.jsp and
    http://java.sun.com/javase/6/webnotes/install/index.html for more
    information.  For Linux, you should be able to install Java using your
    package manager.

    The ssgui.jar file is the compiled version of ssgui.java.  If you modify
    ssgui.java, you can recompile it (for Unix/Linux/Mac/Solaris) by typing
    the command:

        make

    ssgui.java contains default parameter settings; edit the file
    and recompile to change them:

        sssite: URL for the SuiteSparse Matrix Collection
        default is sssite = "https://sparse.tamu.edu" ;

        ssarchive: directory containing your copy of the collection.
        If blank, then it defaults to the directory containing ssgui.

        refresh: refresh time, in days, for updating the index.  use INF to
        never refresh.  Default is 30.

        proxy_server: HTTP proxy server. If none (default), then leave blank.

        proxy port: default is 80 if left blank

--------------------------------------------------------------------------------
For the ssget.m MATLAB interface:
--------------------------------------------------------------------------------

    To install the MATLAB package, just add the path containing the ssget
    directory to your MATLAB path.  Type "pathtool" in MATLAB for more details.

    For a simple example of use, type this command in MATLAB:

        ssget_example

    The MATLAB statement

        Problem = ssget ('HB/arc130')

    (for example), will download a sparse matrix called HB/arc130 (a laser
    simulation problem) and load it into MATLAB.  You don't need to use your
    web browser to load the matrix.  The statement

        Problem = ssget (6)

    will also load same the HB/arc130 matrix.  Each matrix in the collection
    has a unique identifier, in the range 1 to the number of matrices.  As new
    matrices are added, the id's of existing matrices will not change.

    To view an index of the entire collection, just type

        ssget

    in MATLAB.  To modify your download directory, edit the ssget_defaults.m
    file (note that this will not modify the download directory for the
    ssgui java program, however).  The ssget_defaults.m file contains
    the following default settings:

        params.topurl: URL for the SuiteSparse Matrix Collection,
        default is https://sparse.tamu.edu.

        params.topdir: your directory for your copy of the collection.  The
        default is the directory containing ssget.m.  If you modify this file
        and use (for example):
            params.topdir = '/users/me/mystuff/' ;
        then all of your copies of the matrices will reside there.  The MATLAB
        *.mat files will be in /users/me/mystuff/mat/, Matrix Market files go
        in /users/me/mystuff/MM, and Rutherford-Boeing files are in
        /users/me/mystuff/RB.

        params.refresh:  how many days should elapse before re-downloading the
        index file (for obtaining access to new matrices in the collection).
        default is 30 days.  Use 'inf' to never refresh.

    If you use both the Java ssgui program, and the MATLAB ssget.m, you will
    need to modify the default settings in both places.

    To open a URL of the entire collection, just type

        ssweb

    To open the URL of a group of matrices in the collection:

        ssweb ('HB')

    To open the web page for one matrix, use either of these formats:

        ssweb ('HB/arc130')
        ssweb (6)

    To download a new index, to get access to new matrices:

        ssget ('refresh')

    (by default, using ssget downloads the index every 90 days anyway).

    To search for matrices

For more information on how the matrix statistics were created, see
http://www.suitesparse.com.
