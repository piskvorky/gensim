#!/bin/bash

#BIN_PATH=/data/dmlcz/xrehurek/gensim/dmlcz
BIN_PATH=.

# set python path, so that python can find and import gensim modules
export PYTHONPATH=$PYTHONPATH:~/xrehurek

# language is set to 'any', meaning all articles are processed for similarity in one go, regardless of their language
# set language to 'eng', 'fre', 'rus' etc. to only process a specific subset of articles. language is determined from metadata.
language=any

# ========== parse all article sources, build article co-occurence matrix ======
${BIN_PATH}/gensim_build.py $language 2>&1 | tee ~/xrehurek/results/gensim_build.log

# ========== build transformation models =======================================
# tfidf and rp are very memory-undemanding, so process both in parallel 
for method in tfidf rp;
do
	( ${BIN_PATH}/gensim_genmodel.py $language $method 2>&1 | tee ~/xrehurek/results/gensim_genmodel_${method}.log ) &
done
wait

method=lsi
${BIN_PATH}/gensim_genmodel.py $language $method 2>&1 | tee ~/xrehurek/results/gensim_genmodel_${method}.log

# =========== generate output xml files ========================================
# generate xml files for all methods at once, in parallel, to save time. 
# FIXME if out of memory, move tfidf out of the loop (tfidf uses a lot of memory)
for method in tfidf lsi rp;
do
    ( ${BIN_PATH}/gensim_xml.py $language $method 2>&1 | tee ~/xrehurek/results/gensim_xml_${method}.log ) &
done
wait
