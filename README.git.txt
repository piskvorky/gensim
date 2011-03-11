This is my working version of gensim. I keep it synchronized with the upstream
svn one at assembla.
I have added some functional tests and utility functions to it. But the main
reason I'm using the library is to replicate (Gabrilovich & Markovitch, 2006,
2007b, 2009)  Explicit semantic analisis (ESA).

For other implementations try:
C#: http://www.srcco.de/v/wikipedia-esa
java: airhead research library. However the lack of sparse matrix support on
java linear algebra libraries make java a poor choice.

Currently (as of 27 Aug 2010) , gensim can parse wikipedia from xml wiki dumps quite efficiently.
However, our ESA code uses a different parsing that we did before (following the
method section of the paper).

We use here a parsing from March 2008.

Our parsings have three advantages:
1- THey consider centrality measures, and this is not currently easy to do with
 the xml dumps directly
2-
3- We did an unsupervised name entity recognition parsing (NER) using openNLP.
THis is parallelized on 8 cores using java code, see ri.larkc.eu:8087/tools.
We could have used 

NOTE:
Because example corpora are big, the repository ignores the data folder. Our
parsing is available online at: (TODO)
download it and place it under (TODO)

folder structure:

/acme
    contains my working scripts
    
/data/corpora
    contains corpora.

/parsing
    tfidf/preprocessing/porter in /parsing adapted from Mathieu Blondel:
    git clone http://www.mblondel.org/code/tfidf.git

how to replicate the paper
--------------------------
code is in /acme/lee-wiki

First you need to create the tfidf space.
There's a flag. Set createCorpus =  True.
The corpus creation takes about 1hr, with profuse logging.
This is faster than parsing the corpus from xml (about 16 hrs) because we do not
do any xml filtering, stopword removal etc (it's already done on the .cor file).

Once the sparse matrix is on disk, it's faster to read the serialized objects than to
parse the corpus again.

References:
------------
E. Gabrilovich and S. Markovitch (2009) "Wikipedia-based Semantic Interpretation
for Natural Language Processing", Journal of artificial intelligence research, Volume 34, pages 443-498
doi:10.1613/jair.2669
