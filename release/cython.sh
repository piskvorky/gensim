#
# Regenerate all C/CPP files using Cython and commit them to the release branch
#
set -euxo pipefail

git checkout release-${RELEASE}

cython -2 gensim/corpora/_mmreader.pyx
cython -2 gensim/_matutils.pyx
cython -2 gensim/models/nmf_pgd.pyx
cython -2 gensim/models/fasttext_inner.pyx
cython -2 gensim/models/doc2vec_inner.pyx
cython -2 gensim/models/word2vec_inner.pyx
cython -2 gensim/models/_utils_any2vec.pyx
cython -2 --cplus gensim/models/word2vec_corpusfile.pyx
cython -2 --cplus gensim/models/doc2vec_corpusfile.pyx
cython -2 --cplus gensim/models/fasttext_corpusfile.pyx

git add gensim/corpora/_mmreader.c gensim/_matutils.c gensim/models/nmf_pgd.c gensim/models/fasttext_inner.c gensim/models/doc2vec_inner.c gensim/models/word2vec_inner.c gensim/models/_utils_any2vec.c gensim/models/word2vec_corpusfile.cpp gensim/models/doc2vec_corpusfile.cpp gensim/models/fasttext_corpusfile.cpp
git commit -m "regenerated C/CPP files with Cython"
