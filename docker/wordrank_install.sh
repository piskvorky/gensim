#!/bin/bash

printf "1. clean up workspace\n"
./clean.sh

printf "\n2. install glove to construct cooccurrence matrix\n"
wget http://nlp.stanford.edu/software/GloVe-1.0.tar.gz # if failed, check http://nlp.stanford.edu/projects/glove/ for the original version
tar -xvzf GloVe-1.0.tar.gz; rm GloVe-1.0.tar.gz
patch -p0 -i glove.patch
cd glove; make clean all; cd ..

printf "\n3. install hyperwords for evaluation\n"
hg clone -r 56 https://bitbucket.org/omerlevy/hyperwords
patch -p0 -i hyperwords.patch

printf "\n4. build wordrank\n"
#export CC=icc CXX=icpc
export CC=gcc CXX=g++      # uncomment this line if you don't have an Intel compiler, but with gcc all #pragma simd are ignored as of now
cmake .
make clean all
