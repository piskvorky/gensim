#!/usr/bin/env python
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
USAGE: %(program)s LANGUAGE METHOD
    Generate similar.xml files, using a previously built model for METHOD.

Example: ./gensim_xml.py eng lsi
"""


import logging
import sys
import os.path

from gensim.corpora import dmlcorpus, MmCorpus
from gensim.similarities import MatrixSimilarity, SparseMatrixSimilarity

import gensim_build


# set to True to do everything EXCEPT actually writing out similar.xml files to disk.
# similar.xml files are NOT written if DRY_RUN is true.
DRY_RUN = False

# how many 'most similar' documents to store in each similar.xml?
MIN_SCORE = 0.0  # prune based on similarity score (all below MIN_SCORE are ignored)
MAX_SIMILAR = 10  # prune based on rank (at most MAX_SIMILAR are stored). set to 0 to store all of them (no limit).

# if there are no similar articles (after the pruning), do we still want to generate similar.xml?
SAVE_EMPTY = True

# xml template for similar articles
ARTICLE = """
    <article weight="%(score)f">
        <authors>
            <author>%(author)s</author>
        </authors>
        <title>%(title)s</title>
        <suffix>%(suffix)s</suffix>
        <links>
            <link source="%(source)s" id="%(intId)s" path="%(pathId)s"/>
        </links>
    </article>"""

# template for the whole similar.xml file (will be filled with multiple ARTICLE instances)
SIMILAR = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<related>%s
</related>
"""


def generateSimilar(corpus, index, method):
    for docNo, topSims in enumerate(index):  # for each document
        # store similarities to the following file
        outfile = os.path.join(corpus.articleDir(docNo), 'similar_%s.xml' % method)

        articles = []  # collect similars in this list
        for docNo2, score in topSims:  # for each most similar article
            if score > MIN_SCORE and docNo != docNo2:  # if similarity is above MIN_SCORE and not identity (=always maximum similarity, boring)
                source, (intId, pathId) = corpus.documents[docNo2]
                meta = corpus.getMeta(docNo2)
                suffix, author, title = '', meta.get('author', ''), meta.get('title', '')
                articles.append(ARTICLE % locals())  # add the similar article to output
                if len(articles) >= MAX_SIMILAR:
                    break

        # now `articles` holds multiple strings in similar_*.xml format
        if SAVE_EMPTY or articles:
            output = ''.join(articles)  # concat all similars to one string
            if not DRY_RUN:  # only open output files for writing if DRY_RUN is false
                logging.info("generating %s (%i similars)" % (outfile, len(articles)))
                outfile = open(outfile, 'w')
                outfile.write(SIMILAR % output)  # add xml headers and print to file
                outfile.close()
            else:
                logging.info("would be generating %s (%i similars):%s\n" % (outfile, len(articles), output))
        else:
            logging.debug("skipping %s (no similar found)" % outfile)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logging.info("running %s" % ' '.join(sys.argv))

    program = os.path.basename(sys.argv[0])

    # check and process input arguments
    if len(sys.argv) < 3:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    language = sys.argv[1]
    method = sys.argv[2].strip().lower()

    logging.info("loading corpus mappings")
    config = dmlcorpus.DmlConfig('%s_%s' % (gensim_build.PREFIX, language),
                                 resultDir=gensim_build.RESULT_DIR, acceptLangs=[language])

    logging.info("loading word id mapping from %s" % config.resultFile('wordids.txt'))
    id2word = dmlcorpus.DmlCorpus.loadDictionary(config.resultFile('wordids.txt'))
    logging.info("loaded %i word ids" % len(id2word))

    corpus = dmlcorpus.DmlCorpus.load(config.resultFile('.pkl'))
    input = MmCorpus(config.resultFile('_%s.mm' % method))
    assert len(input) == len(corpus), "corpus size mismatch (%i vs %i): run ./gensim_genmodel.py again" % (len(input), len(corpus))

    # initialize structure for similarity queries
    if method == 'lsi' or method == 'rp':  # for these methods, use dense vectors
        index = MatrixSimilarity(input, numBest=MAX_SIMILAR + 1, numFeatures=input.numTerms)
    else:
        index = SparseMatrixSimilarity(input, numBest=MAX_SIMILAR + 1)

    index.normalize = False  # do not normalize query vectors during similarity queries (the index is already built normalized, so it would be a no-op)
    generateSimilar(corpus, index, method)  # for each document, print MAX_SIMILAR nearest documents to a xml file, in dml-cz specific format

    logging.info("finished running %s" % program)
