#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# author Radim Rehurek, radimrehurek@seznam.cz

"""
Estimate parameters for Latent Dirichlet Allocation (LDA) and/or perform inference.

How to estimate (model will be written to datafile.model): 
./lda_estimate.py k datafile
example: ./lda_estimate.py 200 ~/ldadata/trndocs.dat

How to do inference (document likelihoods and gammas will be written to datafile.lda_inferred):
./lda_infer.py modelfile datafile
example: ./lda_infer.py ~/ldadata/trndocs.dat.model ~/ldadata/newdocs.dat
"""

import logging
import sys
import os.path

from ldamodel import LdaModel

PRINT_TOPICS = 10 # when printing model topics, how many top words to print out?


class CorpusLow(object):
    """
    List_Of_Words corpus handles input in GibbsLda++ format.
    
    Quoting http://gibbslda.sourceforge.net/#3.2_Input_Data_Format :
    Both data for training/estimating the model and new data (i.e., previously 
    unseen data) have the same format as follows:

    [M]
    [document1]
    [document2]
    ...
    [documentM]

    in which the first line is the total number for documents [M]. Each line 
    after that is one document. [documenti] is the ith document of the dataset 
    that consists of a list of Ni words/terms.

    [documenti] = [wordi1] [wordi2] ... [wordiNi]

    in which all [wordij] (i=1..M, j=1..Ni) are text strings and they are separated by the blank character.
    """
    def __init__(self, fname, id2word = None, line2words = lambda line: line.strip().split(' ')):
        logging.info("loading corpus from %s" % fname)
        
        self.fname = fname # input file, see class doc for format
        self.line2words = line2words # how to translate lines into words (simply split on space by default)
        self.numDocs = int(open(fname).readline()) # the first line in input data is the number of documents (integer). throws exception on bad input.
        
        if not id2word:
            # build a list of all word types in the corpus (distinct words)
            logging.info("extracting vocabulary from the corpus")
            allTerms = set()
            self.useWordIds = False # return documents as (word, wordCount) 2-tuples
            for doc in self:
                allTerms.update(word for word, wordCnt in doc)
            allTerms = sorted(allTerms) # sort the list of all words; rank in that list = word's integer id
            self.id2word = dict(zip(xrange(len(allTerms)), allTerms)) # build a mapping of word id(int) -> word (string)
        else:
            logging.info("using provided word mapping (%i ids)" % len(id2word))
            self.id2word = id2word
        self.word2id = dict((v, k) for k, v in self.id2word.iteritems())
        self.numTerms = len(self.word2id)
        self.useWordIds = True # return documents as (wordIndex, wordCount) 2-tuples
        
        logging.info("loaded corpus with %i documents and %i terms from %s" % 
                     (self.numDocs, self.numTerms, fname))

    def __len__(self):
        return self.numDocs
    
    def __iter__(self):
        for lineNo, line in enumerate(open(self.fname)):
            if lineNo > 0: # ignore the first line = number of documents
                # convert document line to words, using the function supplied in constructor
                words = self.line2words(line)
                
                if self.useWordIds:
                    uniqWords = set(words).intersection(self.word2id.iterkeys()) # all distinct terms in this document, ignore unknown words
                    # the following creates a unique list of words *in the same order*
                    # as they were in the input. when iterating over the documents,
                    # the (word, count) pairs will appear in the same order as they
                    # were in the input (bar duplicates), which looks better. 
                    # if this was not needed, we might as well have used useWords = set(words)
                    useWords, marker = [], set()
                    for word in words:
                        if (word in uniqWords) and (word not in marker):
                            useWords.append(word)
                            marker.add(word)
                    # construct a list of (wordIndex, wordFrequency) 2-tuples
                    doc = zip(map(self.word2id.get, useWords), map(words.count, useWords)) # suboptimal but speed of this whole function is irrelevant
                else:
                    uniqWords = set(words)
                    # construct a list of (word, wordFrequency) 2-tuples
                    doc = zip(uniqWords, map(words.count, uniqWords)) # suboptimal but that's irrelevant at this point
                
                # return the document, then forget it and move on to the next one
                # note that this way, only one doc is stored in memory at a time, not the whole corpus
                yield doc
    
    def saveAsBlei(self, fname = None):
        """
        Save the corpus in a format compatible with Blei's LDA-C.
        """
        if fname is None:
            fname = self.fname + '.blei'
        
        logging.info("converting corpus to Blei's format: %s" % fname)
        fout = open(fname, 'w')
        for doc in self:
            fout.write("%i %s\n" % (len(doc), ' '.join("%i:%i" % p for p in doc)))
        fout.close()
        
        # write out vocabulary, in a format compatible with Blei's topics.py script
        fnameVocab = fname + '.vocab'
        logging.info("saving vocabulary to %s" % fnameVocab)
        fout = open(fnameVocab + '.vocab', 'w')
        for word, wordId in sorted(self.word2id.iteritems(), key = lambda item: item[1]):
            fout.write("%s\n" % (word))
        fout.close()
#endclass CorpusLow


# ============= main entry point ================
if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG)
    logging.info("running %s" % " ".join(sys.argv))
    
    program = os.path.basename(sys.argv[0])
    
    if 'estimate' in program:
        # make sure we have enough cmd line parameters
        if len(sys.argv) < 3:
            print globals()["__doc__"]
            sys.exit(1)
        
        # parse cmd line
        k = int(sys.argv[1])
        datafile = sys.argv[2]
        
        # load corpus and run estimation
        corpus = CorpusLow(datafile)
        #corpus.saveAsBlei()
        model = LdaModel.fromCorpus(corpus, id2word = corpus.id2word, numTopics = k)
        model.save(datafile + '.model')
        if PRINT_TOPICS:
            logging.info("printing topics (top %i words)" % PRINT_TOPICS)
            model.printTopics(numWords = PRINT_TOPICS)
            print '=' * 40
    elif 'infer' in program:
        # make sure we have enough cmd line parameters
        if len(sys.argv) < 3:
            print globals()["__doc__"]
            sys.exit(1)
        
        # parse cmd line
        modelfile = sys.argv[1]
        datafile = sys.argv[2]
        
        # load model and perform inference
        corpus = CorpusLow(datafile)
        model = LdaModel.load(modelfile)
        model.infer(corpus) # save inference to datafile.lda_inferred
    else:
        print globals()["__doc__"]
        sys.exit(1)
    
    logging.info("finished running %s" % program)
