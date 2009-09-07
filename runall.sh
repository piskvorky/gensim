#!/bin/bash

./build_database.py

for lang in cze ita eng fre rus ger;
do
	./build_tfidf.py $lang 2>&1 | tee ~/xrehurek/results/build_$lang.log
	./gensim.py lsi $lang 2>&1 | tee ~/xrehurek/results/gensim_$lang_lsi.log
	./gensim.py tfidf $lang 2>&1 | tee ~/xrehurek/results/gensim_$lang_tfidf.log
	./gensim.py rp $lang 2>&1 | tee ~/xrehurek/results/gensim_$lang_rp.log
done
