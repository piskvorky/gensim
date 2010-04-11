#!/bin/bash

# set python path, so that python can find and import gensim modules

export PYTHONPATH=$PYTHONPATH:~/xrehurek


# language is set to 'any', meaning all articles are processed for similarity in one go, regardless of their language
# set language to 'eng', 'fre', 'rus' etc. to only process a specific subset of articles. language is determined from metadata.


./gensim_build.py any 2>&1 | tee ~/xrehurek/results/gensim_build.log

for method in tfidf lsi;
do
	./gensim_genmodel.py any $method 2>&1 | tee ~/xrehurek/results/gensim_genmodel_${method}.log
	./gensim_xml.py any $method 2>&1 | tee ~/xrehurek/results/gensim_xml_${method}.log
done
