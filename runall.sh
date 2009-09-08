#!/bin/bash

./build_database.py 2>&1 | tee ~/xrehurek/results/build_database.log

for lang in cze ita eng fre rus ger;
do
	./build_tfidf.py $lang 2>&1 | tee ~/xrehurek/results/build_${lang}.log
	./gensim.py lsi $lang 2>&1 | tee ~/xrehurek/results/gensim_${lang}_lsi.log
	./gensim.py tfidf $lang 2>&1 | tee ~/xrehurek/results/gensim_${lang}_tfidf.log
	./gensim.py rp $lang 2>&1 | tee ~/xrehurek/results/gensim_${lang}_rp.log
done
