#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# author Radim Rehurek, radimrehurek@seznam.cz


import logging
import cPickle # for persistent storage of LDA models

import numpy # for arrays, array broadcasting etc.
from scipy.special import gammaln, digamma, polygamma # gamma function utils
from scipy.maxentropy import logsumexp # sum of logarithms

trigamma = lambda x: polygamma(1, x) # second derivative of the gamma fnc


class LdaModel(object):
    """
    Objects of this class allow building and maintaining a model of Latent Dirichlet
    Allocation.
    
    The code is based on Blei's C implementation, see http://www.cs.princeton.edu/~blei/lda-c/
    
    This Python code uses numpy heavily, and is about 4-5x slower than the original 
    C version. The up side is that it is much more straightforward and concise, 
    using vector operations ala MATLAB, easily pluggable/extensible etc.
    
    The main functions are LdaModel.fromCorpus() and LdaModel.infer().
    Both accept a corpus (an iterable returning documents and len()) and realize
    the tasks of parameter estimation/LDA inference respectively.
    
    The model can be persistently saved via the load/save methods.
    """
    def __init__(self, id2word, numTopics = 200, alpha = None):
        """
        Initialize the model's hyperparameters and constants.
        
        id2word is a mapping between word ids (integers) to words (strings). It is
        used to determine the vocabulary size, as well as for debugging and topic 
        printing.
        
        numTopics is the number of requested topics.
        
        alpha is either None (to be estimated during training) or a fixed number 
        from (0.0, 1.0).
        """
        # store user-supplied parameters
        self.numTerms = len(id2word) # size of vocabulary
        self.id2word = id2word
        self.numTopics = numTopics # number of latent topics
        self.ESTIMATE_ALPHA = alpha is None
        if alpha is None: # no alpha supplied by user => get some initial estimate
            alpha = 10.0 / numTopics # initial estimate is 50 / numTopics, as suggested in Steyvers&Griffiths: Probabilistic Topic Models
        self.alpha = min(0.99999, max(0.00001, alpha)) # dirichlet prior; make sure it's within bounds

        # set EM training constants
        self.EM_MAX_ITER = 1000 # maximum number of EM iterations; usually converges much earlier
        self.EM_CONVERGED = 1e-4 # relative difference between two iterations; if lower than this, stop the EM training 
        self.VAR_MAX_ITER = 20 # maximum number of document inference iterations
        self.VAR_CONVERGED = 1e-6 # relative difference between document inference iterations needed to stop sooner than VAR_MAX_ITER
    
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
    
    def initializeFromCorpus(self, corpus, numInitDocs = 1):
        """
        Initialize the model word counts from the corpus. Each topic will be initialized
        from numInitDocs random documents.
        """
        logging.info("initializing model with %i random document(s) per topic" % numInitDocs)
        result = numpy.ones(shape = (self.numTopics, self.numTerms)) # add-one smooth
        # next we precompute the all the random document indices, so that we can 
        # update the counts in a single sweep over the corpus. 
        # all this drama is because the corpus doesn't necessarily support 
        # random access (indexing) -- it only supports sequential iteration over 
        # the documents (streaming).
        initDocs = numpy.random.randint(0, len(corpus), (self.numTopics, numInitDocs)) # get document indices
        
        # go over the corpus, updating counts
        for i, doc in enumerate(corpus):
            for k in xrange(self.numTopics):
                if i in initDocs[k]: # do we want to initialize this topic with this document?
                    for wordIndex, wordCount in doc: # for each word in the document...
                        result[k, wordIndex] += wordCount # ...add its count to the word-topic count
        return result
    
    @staticmethod
    def fromCorpus(corpus, numTopics, initMode = 'random'):
        """
        Run LDA parameter estimation from a training corpus, using the EM algorithm.
        """
        # initialize the model
        logging.info("initializing LDA model with '%s'" % initMode)
        model = LdaModel(corpus.id2word, numTopics)

        # set initial word counts
        if initMode == 'seeded':
            counts = model.initializeFromCorpus(corpus, numInitDocs = 2)
        else:
            counts = 1.0 / model.numTerms + numpy.random.rand(model.numTopics, model.numTerms) # add noise from <0, 1)
        model.classWord = counts
            
        # update model parameters with these initial counts; don't do anything with alpha
        model.mle(estimateAlpha = False)
        
        # set up temporary vars needed for EM
        likelihood = likelihoodOld = converged = numpy.NAN
        
        # main EM loop
        for i in xrange(model.EM_MAX_ITER):
            logging.info("starting EM iteration #%i, converged=%s, likelihood=%s" % 
                         (i, converged, likelihood))

            if numpy.isfinite(converged) and (converged <= model.EM_CONVERGED): # solution good enough?
                logging.info("EM converged in %i iterations" % i)
                break
            
            # initialize help structures for this iteration
            model.classWord[:] = 0.0 # reset counts
            model.alphaSuffStats = 0.0 # reset alpha stats
            model.numDocs = 0
            
            # E step: iterate over individual documents, using old beta and updating new word counts
            logging.info("performing E step #%i" % i)
            likelihood = sum(model.docEStep(doc) for doc in corpus)
            assert numpy.isfinite(likelihood), "bad likelihood %s" % likelihood

            # M step -- update alpha and beta
            logging.info("performing M step #%i" % i)
            model.mle(estimateAlpha = model.ESTIMATE_ALPHA)
            
            # check for convergence
            converged = numpy.divide(likelihoodOld - likelihood, likelihoodOld)
            logging.info("finished iteration #%i: likelihood %f, likelihoodOld %f, converged %f" % 
                         (i, likelihood, likelihoodOld, converged))
            likelihoodOld = likelihood
        
        return model

    def docEStep(self, doc):
        # posterior inference
        likelihood, phi, gamma = self.inference(doc)
        
        # update sufficient statistics
        for n, (wordIndex, wordCount) in enumerate(doc):
            self.classWord[:, wordIndex] += wordCount * phi[n]
        self.alphaSuffStats += numpy.sum(digamma(gamma)) - self.numTopics * digamma(numpy.sum(gamma))
        self.numDocs += 1
        
        return likelihood

    def mle(self, estimateAlpha):
        """
        Maximum likelihood estimate.
        
        Bring conditional word probabilities up to date with word-topic counts, 
        estimate new alpha (if requested).
        """
        marginal = numpy.log(numpy.sum(self.classWord, axis = 1)) 
        marginal.shape += (1,) # alter array shape from (n,) to (n, 1); needed for correct broadcasting on the following line
        logProbW = numpy.log(self.classWord) - marginal # (logarithm of) the beta parameter = word-topic distribution
        self.logProbW = numpy.where(numpy.isfinite(logProbW), logProbW, -100.0) # replace log(0) with -100.0
        
        if estimateAlpha:
            self.alpha = self.optAlpha()
        
        logging.debug("updated model to %s" % self)
    
    def optAlpha(self, MAX_ALPHA_ITER = 1000, NEWTON_THRESH = 1e-5):
        """
        Estimate alpha.
        """
        initA = 100.0
        logA = numpy.log(initA) # keep computations in log space
        logging.debug("optimizing old alpha %s" % self.alpha)
        
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
#            logging.debug("alpha maximization: f=%f, df=%f" % (f, df))
            if numpy.abs(df) <= NEWTON_THRESH:
                break
        result = numpy.exp(logA) # convert back from log space
        logging.info("estimated old alpha %s to new alpha %s" % (self.alpha, result))
        return result

    def inference(self, doc):
        """
        Perform inference on a single document.
        
        Return 3-tuple of (likelihood of this document, word-topic distribution
        phi, topic distribution gamma).
        
        A document is simply a bag-of-words collection which supports len() and 
        iteration over (wordIndex, wordCount) 2-tuples.
        """
        # init help structures
        totalWords = sum(wordCount for _, wordCount in doc)
        gamma = numpy.zeros(self.numTopics) + self.alpha + 1.0 * totalWords / self.numTopics
        phi = numpy.zeros(shape = (len(doc), self.numTopics)) + 1.0 / self.numTopics
        likelihood = likelihoodOld = converged = numpy.NAN
        
        # compute posterior dirichlet
        for i in xrange(self.VAR_MAX_ITER):
#            logging.debug("inference step #%s, converged=%s, likelihood=%s, likelikelihoodOld=%s" % 
#                          (i, converged, likelihood, likelihoodOld))
            
            if numpy.isfinite(converged) and converged <= self.VAR_CONVERGED:
#                logging.debug("document converged in %i iterations" % i)
                break
            
            for n, (wordIndex, wordCount) in enumerate(doc):
                # compute phi vars, in log space, to prevent numerical nastiness
                tmp = digamma(gamma) + self.logProbW[:, wordIndex] # vector operation

                # convert phi and update gamma
                newPhi = numpy.exp(tmp - logsumexp(tmp))
                gamma += wordCount * (newPhi - phi[n])
                phi[n] = newPhi
            
            likelihood = self.computeLikelihood(doc, phi, gamma)
            assert numpy.isfinite(likelihood)
            converged = numpy.divide(likelihoodOld - likelihood, likelihoodOld)
            likelihoodOld = likelihood
        return likelihood, phi, gamma

    def computeLikelihood(self, doc, phi, gamma):
        """
        Compute the document likelihood, given all model parameters.
        """
        gammaSum = numpy.sum(gamma)
        digSum = digamma(gammaSum)
        dig = digamma(gamma) - digSum # precompute the difference
        
        likelihood = gammaln(self.alpha * self.numTopics) - \
                     self.numTopics * gammaln(self.alpha) - \
                     gammaln(gammaSum)
        
        likelihood += numpy.sum((self.alpha - 1) * dig + gammaln(gamma) - (gamma - 1) * dig) # array broadcast
        
        for n, (wordIndex, wordCount) in enumerate(doc):
            partial = phi[n] * (dig - numpy.log(phi[n]) + self.logProbW[:, wordIndex]) # array broadcast
            partial = numpy.where(numpy.isfinite(partial), partial, 0.0) # silently replace NaNs (from 0 * log(0) in phi) with 0.0
            likelihood += wordCount * numpy.sum(partial)
        
        return likelihood
    
    def infer(self, corpus):
        """
        Perform inference on a corpus of documents.
        
        This means that a standard inference step is taken for each document from 
        the corpus and the results are saved into file corpus.fname.lda_inferred.
        
        The output format of this file is one doc per line:
        doc_likelihood[TAB]topic1:prob ... topicK:prob[TAB]word1:topic ... wordN:topic
        
        Topics are sorted by probability, words are in the same order as in the input.
        
        The model itself is not affected in any way (it is read-only=const for this method).
        """
        fname = corpus.fname + '.lda_inferred'
        logging.info("writing inferences to %s" % fname)
        fout = open(fname, 'w')
        for doc in corpus:
            # do the inference
            likelihood, phi, gamma = self.inference(doc)
            
            # for the printing, sort topics in decreasing order of probability
            sumGamma = sum(gamma)
            topicVals = [(1.0 * val / sumGamma, topic) for  topic, val in enumerate(gamma)]
            gammaStr = ["%i:%.4f" % (topic, val) for (val, topic) in sorted(topicVals, reverse = True)]
            
            # for word topics, don't store the full topic distribution phi for 
            # each word -- only pick out the most probable topic, and store its 
            # index, ie. an integer <0, K-1> for each word.
            # information about second-best topic, numerical differences in the 
            # distribution etc. are lost.
            words = [self.id2word[wordIndex] for wordIndex, _ in doc]
            bestTopics = numpy.argmax(phi, axis = 1)
            assert len(words) == len(bestTopics)
            phiStr = ["%s:%s" % pair for pair in zip(words, bestTopics)]
            
            fout.write("%s\t%s\t%s\n" % (likelihood, ' '.join(gammaStr), ' '.join(phiStr)))
        fout.close()
    
    def printTopics(self, numWords = 10):
        """
        Print the top 'numTerms' words for each topic, along with the log of their 
        probability.
        
        Which words are 'at the top' is determined by a sort of TF-IDF score, 
        where the word gets higher score if it's probable in this topic (the TF 
        part) and lower score if it's probable across the whole corpus (the IDF 
        part).
        
        The exact formula is taken from Blei&Laffery: "Topic Models", 2009
        """
        # compute the geometric mean of words' probability across all topics
        idf = numpy.sum(self.logProbW, axis = 0) / self.numTopics # note: vector operation
        assert idf.shape == (self.numTerms,), "idf.shape=%s" % idf.shape
        
        # print top words, one topic after another
        for i in xrange(self.numTopics):
            # determine the score of all words, in this topic
            scores = numpy.exp(self.logProbW[i]) * (self.logProbW[i] - idf)
            assert scores.shape == (self.numTerms,), scores.shape
            
            # link scores with the actual strings
            termScores = zip(scores, self.logProbW[i], map(self.id2word.get, xrange(len(scores))))
            
            # sort words -- words with best scores come first; keep only the best numWords
            best = sorted(termScores, reverse = True)[:numWords]
           
            # print best numWords, with a space between the words
            print "topic #%i: %s" % (i, ' '.join('%s:%.3f' % (word, prob) for (score, prob, word) in best))
#endclass LdaModel
