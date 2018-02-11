#------------------------------------------------------------------------------
# CSparse Makefile
#------------------------------------------------------------------------------

C:
	( cd Lib ; $(MAKE) )
	( cd Demo ; $(MAKE) )

all: C cov

library:
	( cd Lib ; $(MAKE) )

# compile the static libraries only
static:
	( cd Lib    ; $(MAKE) static )

cov:
	( cd Tcov ; $(MAKE) )

clean:
	( cd Lib ; $(MAKE) clean )
	( cd Demo ; $(MAKE) clean )
	( cd Tcov ; $(MAKE) clean )
	( cd MATLAB/CSparse ; $(RM) *.o )
	( cd MATLAB/Test    ; $(RM) *.o )

purge:
	( cd Lib ; $(MAKE) purge )
	( cd Demo ; $(MAKE) purge )
	( cd Tcov ; $(MAKE) purge )
	( cd MATLAB/CSparse ; $(RM) *.o *.mex* )
	( cd MATLAB/Test    ; $(RM) *.o *.mex* )

distclean: purge

install: library
	( cd Lib ; $(MAKE) install )

uninstall:
	( cd Lib ; $(MAKE) uninstall )

