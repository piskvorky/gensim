#!/bin/bash

# full path to gensim executables
BIN_PATH=~/xrehurek/gensim/dmlcz

# intermediate data will be stored to this dir
RESULT_PATH=~/xrehurek/results

# set python path, so that python can find and import gensim modules
export PYTHONPATH=~/xrehurek:${PYTHONPATH}

# Language is set to 'any', meaning all articles are processed for similarity in
# one go, regardless of their language.
# Set language to 'eng', 'fre', 'rus' etc. to only process a specific subset of
# articles (an article's language is determined from its metadata).
language=any


# ========== parse all article sources, build article co-occurence matrix ======
${BIN_PATH}/gensim_build.py ${language} 2>&1 | tee ${RESULT_PATH}/gensim_build.log


# ========== build transformation models =======================================
for method in tfidf rp;
do
	( ${BIN_PATH}/gensim_genmodel.py ${language} ${method} 2>&1 | tee ${RESULT_PATH}/gensim_genmodel_${method}.log ) &
done
wait

method=lsi
${BIN_PATH}/gensim_genmodel.py ${language} ${method} 2>&1 | tee ${RESULT_PATH}/gensim_genmodel_${method}.log


# =========== generate output xml files ========================================
# generate xml files for all methods at once, in parallel, to save time.
# NOTE if out of memory, move tfidf out of the loop (tfidf uses a lot of memory here)
for method in tfidf lsi rp;
do
    ( ${BIN_PATH}/gensim_xml.py ${language} ${method} 2>&1 | tee ${RESULT_PATH}/gensim_xml_${method}.log ) &
done
wait
