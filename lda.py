#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# author Radim Rehurek, radimrehurek@seznam.cz

"""
Estimate parameters for Latent Dirichlet Allocation (LDA) and/or perform inference.
The code is based on Blei's C implementation, see http://www.cs.princeton.edu/~blei/lda-c/

estimation (model will be written to datafile.model): 
./lda_estimate.py k datafile [alpha=50.0/k]
example: ./lda_estimate.py 200 ~/ldadata/trndocs.dat

inference (document likelihoods and gammas will be written to datafile.infer):
./lda_infer.py modelfile datafile
example: ./lda_infer.py ~/ldadata/trndocs.dat.model ~/ldadata/newdocs.dat
"""

import logging
import sys
import cPickle # for persistent storage of LDA models
import string
import os.path

import numpy # for array, vector operations, broadcasting etc.
from scipy.special import gammaln, digamma # gamma function utils
from scipy.maxentropy import logsumexp # sum of logarithms

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
    def __init__(self, fname, line2words = lambda line: line.strip().split(' ')):
        logging.info("loading corpus from %s" % fname)
        
        self.fname = fname # input file, see class doc for format
        self.line2words = line2words # how to translate lines into words (simply split on space by default)
        self.numDocs = int(open(fname).readline()) # the first line in input data is the number of documents (integer). throws exception on bad input.
        self.useWordIds = False # return documents as (word, wordCount) 2-tuples
        
        # build a list of all word types in the corpus (distinct words)
        logging.info("extracting vocabulary from the corpus")
        allTerms = set()
        for doc in self:
            allTerms.update(word for word, wordCnt in doc)
        allTerms = sorted(allTerms) # sort the list of all words; rank in that list = word's integer id
        self.word2id = dict(zip(allTerms, xrange(len(allTerms)))) # build a mapping of word (string) -> its id (int)
        self.id2word = dict((v, k) for k, v in self.word2id.iteritems())
        self.useWordIds = True # return documents as (wordIndex, wordCount) 2-tuples
        self.numTerms = len(self.word2id)
        
        logging.info("loaded corpus with %i documents and %i terms from %s" % 
                     (self.numDocs, self.numTerms, fname))

    def __len__(self):
        return self.numDocs
    
    def __iter__(self):
        for lineNo, line in enumerate(open(self.fname)):
            if lineNo > 0: # ignore the first line
                # convert document line to words, using the function supplied in constructor
                words = self.line2words(line)
                uniqWords = set(words) # all distinct terms in this document
                
                if self.useWordIds:
                    # construct a list of (wordIndex, wordFrequency) 2-tuples
                    doc = zip(map(self.word2id.get, uniqWords), map(words.count, uniqWords)) # suboptimal but that's irrelevant at this point
                else:
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
        
        # write out vocabulary
        fnameVocab = fname + '.vocab'
        logging.info("saving vocabulary to %s" % fnameVocab)
        fout = open(fnameVocab + '.vocab', 'w')
        for word, wordId in sorted(self.word2id.iteritems(), key = lambda item: item[1]):
            fout.write("%s\n" % (word))
        fout.close()
#endclass CorpusLow


class LdaModel(object):
    def __init__(self, numTerms, numTopics = 200, alpha = None):
        # store user-supplied parameters
        self.numTerms = numTerms # size of vocabulary
        self.numTopics = numTopics # number of latent topics
        self.ESTIMATE_ALPHA = alpha is None
        if alpha is None: # no alpha supplied by user => get some initial estimate
            alpha = 50.0 / numTopics # initial estimate is 50 / numTopics, as suggested in Steyvers&Griffiths: Probabilistic Topic Models
        self.alpha = alpha # dirichlet prior

        # set EM constants
        self.EM_MAX_ITER = 1000
        self.EM_CONVERGED = 1e-4
        self.VAR_MAX_ITER = 20
        self.VAR_CONVERGED = 1e-6
    
    @staticmethod
    def load(fname):
        logging.info("loading LdaModel from %s" % fname)
        return cPickle.load(open(fname))

    def save(self, fname):
        logging.info("saving %s to %s" % (self, fname))
        f = open(fname, 'w')
        cPickle.dump(self, f)
        f.close()

    def __str__(self):
        return "LdaModel(numTerms=%s, numTopics=%s, alpha=%s (estimated=%s))" % \
                (self.numTerms, self.numTopics, self.alpha, self.ESTIMATE_ALPHA)

    @staticmethod
    def fromCorpus(corpus, numTopics):
        """
        Run LDA parameter estimation from a training corpus, using the EM algorithm.
        """
        
        # initialize the model
        logging.info("initializing LDA model")
        model = LdaModel(corpus.numTerms, numTopics, alpha)
#        model.classWord = 1.0 / model.numTerms + numpy.zeros((model.numTopics, model.numTerms)) # set all counts to 1/numTerms + add noise from (0, 1)
        model.classWord = 1.0 / model.numTerms + numpy.random.rand(model.numTopics, model.numTerms) # set all counts to 1/numTerms + add noise from (0, 1)
        model.mle(estimateAlpha = False) # update word probabilities with these initial counts; don't do anything with alpha
        
        # set up temporary vars needed for EM
        likelihood = likelihoodOld = converged = numpy.NAN
        
        # main EM loop
        for i in xrange(model.EM_MAX_ITER):
            logging.info("starting EM iteration #%i, converged=%s, likelihood=%s" % 
                         (i, converged, likelihood))

            if numpy.isfinite(converged) and (converged <= model.EM_CONVERGED): # solution good enough?
                logging.info("EM converged in %i iterations" % i)
                break
            
#            model.printTopics(corpus.id2word, numWords = 10)
            
            # initialize help structures for this iteration
            model.classWord[:] = 0.0
            model.alphaSuffStats = 0.0
            model.numDocs = 0
            
            # E step: iterate over individual documents, using old logProbW and updating word counts
            logging.info("performing E step")
            likelihood = numpy.sum(model.docEStep(docNo, doc) for docNo, doc in enumerate(corpus))
            assert numpy.isfinite(likelihood), "bad likelihood %s" % likelihood

            # M step -- compute logProbW based on topic-word counts
            logging.info("performing M step")
            model.mle(estimateAlpha = model.ESTIMATE_ALPHA)
            
            # check for convergence
            converged = numpy.divide(likelihoodOld - likelihood, likelihoodOld)
            if converged < 0.0: # FIXME what is this for?
                self.VAR_MAX_ITER *= 2
            logging.info("finished iteration #%i: likelihood %f, likelihoodOld %f, converged %f" % 
                         (i, likelihood, likelihoodOld, converged))
            likelihoodOld = likelihood
            
        return model

    def docEStep(self, docNo, doc):
        # posterior inference
        likelihood, phi, gamma = self.inference(doc)
        
        # update sufficient statistics
        for n, (wordIndex, wordCount) in enumerate(doc):
            self.classWord[:, wordIndex] += wordCount * phi[n]
        
        # update stats for estimating alpha
        self.alphaSuffStats += numpy.sum(digamma(gamma)) - self.numTopics * digamma(numpy.sum(gamma))
        self.numDocs += 1
        logging.debug("doc #%i likelihood=%.3f" % (docNo, likelihood))
        
        return likelihood

    def mle(self, estimateAlpha):
        """
        Bring conditional word probabilities up to date with classWord counts.
        """
        marginal = numpy.log(numpy.sum(self.classWord, axis = 1)) 
        marginal.shape += (1,)
        logProbW = numpy.log(self.classWord) - marginal
        self.logProbW = numpy.where(numpy.isfinite(logProbW), logProbW, -100.0) # replace log(0) with -100.0
        
        if estimateAlpha:
            self.alpha = self.optAlpha()
        
        logging.debug("updated model to %s" % self)
    
    def optAlpha(self, MAX_ALPHA_ITER = 1000, NEWTON_THRESH = 1e-5):
        """
        Estimate alpha.
        """
        initA = 100.0
        logA = numpy.log(initA)
        
        for i in xrange(MAX_ALPHA_ITER):
            a = numpy.exp(logA)
            if not numpy.isfinite(a):
                initA = initA * 10.0
                logging.warning("alpha is NaN; new init alpha=%f" % initA)
                a = initA
                logA = numpy.log(a)
            f = self.numDocs * (gammaln(self.numTopics * a) - self.numTopics * gammaln(a)) + (a - 1) * self.alphaSuffStats
            df = self.alphaSuffStats + self.numDocs * (self.numTopics * digamma(self.numTopics * a) - self.numTopics * digamma(a))
            d2f = self.numDocs * (self.numTopics * self.numTopics * trigamma(self.numTopics * a) - self.numTopics * trigamma(a))
            logA -= df / (d2f * a + df)
            logging.debug("alpha maximization: f=%f, df=%f" % (f, df))
            if numpy.abs(df) <= NEWTON_THRESH:
                break
        result = numpy.exp(logA)
        logging.info("using new alpha %s" % result)
        return result

    def inference(self, doc):
        # note: the original C version had extra pointer arguments that were used 
        # to return the help structures; i return them explicitly as a result tuple

        # init help structures
        totalWords = sum(count for word, count in doc)
        gamma = numpy.zeros(self.numTopics) + self.alpha + 1.0 * totalWords / self.numTopics
        phi = numpy.zeros(shape = (len(doc), self.numTopics)) + 1.0 / self.numTopics
        likelihood = likelihoodOld = converged = numpy.NAN
        
        # compute posterior dirichlet
        for i in xrange(self.VAR_MAX_ITER):
#            logging.debug("inference step #%s, converged=%s, likelihood=%s, likelikelihoodOld=%s" % 
#                          (i, converged, likelihood, likelihoodOld))
            
            if numpy.isfinite(converged) and converged <= self.VAR_CONVERGED:
                logging.debug("document converged in %i iterations" % i)
                break
            
            for n, (wordIndex, wordCount) in enumerate(doc):
                # compute phi vars, in log space, to prevent numerical nastiness
                newPhi = digamma(gamma) + self.logProbW[:, wordIndex] # vector operation

                # convert phi and update gamma
                tmp = numpy.exp(newPhi - logsumexp(newPhi))
                gamma += wordCount * (tmp - phi[n])
                phi[n] = tmp
            
            likelihood = self.computeLikelihood(doc, phi, gamma)
            assert numpy.isfinite(likelihood)
            converged = numpy.divide(likelihoodOld - likelihood, likelihoodOld)
            likelihoodOld = likelihood
        return likelihood, phi, gamma

    def computeLikelihood(self, doc, phi, gamma):
        gammaSum = numpy.sum(gamma)
        digSum = digamma(gammaSum)
        dig = digamma(gamma) - digSum # precompute the difference; in the original C version this was done on the fly (multiple times)
        
        likelihood = gammaln(self.alpha * self.numTopics) - \
                     self.numTopics * gammaln(self.alpha) - \
                     gammaln(gammaSum)
        
        likelihood += numpy.sum((self.alpha - 1) * dig + gammaln(gamma) - (gamma - 1) * dig) # note: array broadcast
        
        for n, (wordIndex, wordCount) in enumerate(doc):
            partial = phi[n] * (dig - numpy.log(phi[n]) + self.logProbW[:, wordIndex]) # array broadcast
            partial = numpy.where(numpy.isfinite(partial), partial, 0.0) # silently replace NaNs (from 0 * log(0) in phi) with 0.0
            likelihood += wordCount * numpy.sum(partial)
        
        return likelihood
    
    def printTopics(self, id2word, numWords = 10):
        """
        Print the top 'numTerms' words for each topics.
        Which words are at the top is determined by a sort of tf-idf score, where
        the word gets higher score with higher mass in this topic and lower score
        if it's frequent in the whole corpus  (geometric mean across all topic 
        masses).
        See Blei&Laffery: "Topic Models", 2009
        """
        # compute inverse of the geometric mean across all topics
        idf = numpy.sum(self.logProbW, axis = 0) / self.numTopics # note: vector operation
        assert idf.shape == (self.numTerms,), "idf.shape=%s" % idf.shape
        
        # print significant words, one topic after another
        for i in xrange(self.numTopics):
            # determine the score of all words, in this topic
            scores = numpy.exp(self.logProbW[i]) * (self.logProbW[i] - idf)
            assert scores.shape == (self.numTerms,), scores.shape
            
            # link scores with the actual strings; this is done through the id2word mapping supplied as parameter
            termScores = zip(scores, self.logProbW[i], map(id2word.get, xrange(len(scores))))
            
            # sort words -- words with best scores come first; keep only the best numWords
            best = sorted(termScores, reverse = True)[:numWords]
           
            # print best numWords, with a space between words
            print "topic #%i: %s" % (i, ' '.join('%s:%.3f' % (word, prob) for (score, prob, word) in best))
#endclass LdaModel



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
        if len(sys.argv) > 3:
            alpha = float(sys.argv[3])
        else:
            alpha = None
        
        # load corpus and run estimation
        corpus = CorpusLow(datafile)
        #corpus.saveAsBlei()
        model = LdaModel.fromCorpus(corpus, numTopics = k)
        model.save(datafile + '.model')
    elif 'infer' in program:
        # make sure we have enough cmd line parameters
        if len(sys.argv) < 3:
            print globals()["__doc__"]
            sys.exit(1)
        
        # parse cmd line
        modelfile = sys.argv[1]
        datafile = sys.argv[2]
        
        # load model and perform inference
        model = LdaModel.load(modelfile)
        # TODO inference
    else:
        print globals()["__doc__"]
        sys.exit(1)
    
    logging.info("finished running %s" % program)
