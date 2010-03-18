#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# author Radim Rehurek, radimrehurek@seznam.cz

"""
This module encapsulates functionality for the Latent Dirichlet Allocation algorithm.

It allows both model estimation from a training corpus and inference on new, 
unseen documents.

The implementation is based on Blei et al., Latent Dirichlet Allocation, 2003,
and on Blei's LDA-C software in particular. This means it uses variational EM
inference rather than Gibbs sampling to estimate model parameters.
"""


import logging

import numpy # for arrays, array broadcasting etc.
from scipy.special import gammaln, digamma, polygamma # gamma function utils
from scipy.maxentropy import logsumexp # log of sum

from gensim import interfaces


trigamma = lambda x: polygamma(1, x) # second derivative of the gamma fnc


class LdaModel(interfaces.TransformationABC):
    """
    Objects of this class allow building and maintaining a model of Latent Dirichlet
    Allocation.
    
    The code is based on Blei's C implementation, see http://www.cs.princeton.edu/~blei/lda-c/ .
    
    This Python code uses numpy heavily, and is about 4-5x slower than the original 
    C version. The up side is that it is much more straightforward and concise, 
    using vector operations ala MATLAB, easily pluggable/extensible etc.
    
    The constructor estimates model parameters based on a training corpus:
    
    >>> lda = LdaModel(corpus, numTopics = 10)
    
    You can then infer topic distributions on new, unseen documents:
    
    >>> doc_lda = lda[doc_bow]
    
    Model persistency is achieved via its load/save methods.
    """
    def __init__(self, corpus, id2word = None, numTopics = 200, alpha = None, initMode = 'random'):
        """
        Initialize the model based on corpus.
        
        `id2word` is a mapping from word ids (integers) to words (strings). It is
        used to determine the vocabulary size, as well as for debugging and topic 
        printing.
        
        `numTopics` is the number of requested topics.
        
        `alpha` is either None (to be estimated during training) or a number 
        between (0.0, 1.0).
        """
        # store user-supplied parameters
        self.id2word = id2word
        if self.id2word is None:
            logging.info("no word id mapping provided; initializing from corpus, assuming identity")
            maxId = -1
            for document in corpus:
                maxId = max(maxId, max([-1] + [fieldId for fieldId, _ in document]))
            self.numTerms = 1 + maxId
            self.id2word = dict(zip(xrange(self.numTerms), xrange(self.numTerms)))
        else:
            self.numTerms = 1 + max([-1] + self.id2word.keys())
        self.numTopics = numTopics # number of latent topics
        
        # internal constants; can be manually changed after having called the init
        self.ESTIMATE_ALPHA = alpha is None
        if alpha is None: # no alpha supplied by user => get some initial estimate
            alpha = 10.0 / numTopics # initial estimate is 50 / numTopics, as suggested in Steyvers&Griffiths: Probabilistic Topic Models
        self.alpha = min(0.99999, max(0.00001, alpha)) # dirichlet prior; make sure it's within bounds

        # set EM training constants
        self.EM_MAX_ITER = 50 # maximum number of EM iterations; usually converges much earlier
        self.EM_CONVERGED = 0.0001 # relative difference between two iterations; if lower than this, stop the EM training 
        self.VAR_MAX_ITER = 20 # maximum number of document inference iterations
        self.VAR_CONVERGED = 0.000001 # relative difference between document inference iterations needed to stop sooner than VAR_MAX_ITER
        
        if corpus is not None:
            self.initialize(corpus, initMode)
    

    def __str__(self):
        return "LdaModel(numTerms=%s, numTopics=%s, alpha=%s (estimated=%s))" % \
                (self.numTerms, self.numTopics, self.alpha, self.ESTIMATE_ALPHA)


    def initialize(self, corpus, initMode = 'random'):
        """
        Run LDA parameter estimation from a training corpus, using the EM algorithm.
        
        After the model has been initialized, you can infer topic distribution over
        other, different corpora, using this estimated model.
        
        `initMode` can be either 'random', for a fast random initialization of 
        the model parameters, or 'seeded', for an initialization based on a handful
        of real documents. The 'seeded' mode requires a sweep over the entire 
        corpus, and is thus much slower.
        """
        # initialize the model
        logging.info("initializing LDA model with '%s'" % initMode)

        # set initial word counts
        if self.numTerms == 0:
            raise ValueError("cannot compute LDA over an empty collection (no terms)")
        if initMode == 'seeded':
            counts = self.countsFromCorpus(corpus, numInitDocs = 2)
        else:
            counts = 1.0 / self.numTerms + numpy.random.rand(self.numTopics, self.numTerms) # add noise from <0, 1)
        self.classWord = counts
            
        # update model parameters with these initial counts; don't do anything with alpha
        self.mle(estimateAlpha = False)
        
        # set up temporary vars needed for EM
        likelihood = likelihoodOld = converged = numpy.NAN
        
        # main EM loop
        for i in xrange(self.EM_MAX_ITER):
            logging.info("starting EM iteration #%i, converged=%s, likelihood=%s" % 
                         (i, converged, likelihood))

            if numpy.isfinite(converged) and (converged <= self.EM_CONVERGED): # solution good enough?
                logging.info("EM converged in %i iterations" % i)
                break
            
            # initialize help structures for this iteration
            self.classWord[:] = 0.0 # reset counts
            self.alphaSuffStats = 0.0 # reset alpha stats
            self.numDocs = 0
            
            # E step: iterate over individual documents, using old beta and updating new word counts
            logging.info("performing E step #%i" % i)
            likelihood = sum(self.docEStep(doc) for doc in corpus)
            assert numpy.isfinite(likelihood), "bad likelihood %s" % likelihood

            # M step -- update alpha and beta
            logging.info("performing M step #%i" % i)
            self.mle(estimateAlpha = self.ESTIMATE_ALPHA)
            
            # check for convergence
            converged = numpy.divide(likelihoodOld - likelihood, likelihoodOld)
            logging.info("finished iteration #%i: likelihood %f, likelihoodOld %f, converged %f" % 
                         (i, likelihood, likelihoodOld, converged))
            likelihoodOld = likelihood


    def docEStep(self, doc):
        """
        Find optimizing parameters for phi and gamma, and update sufficient statistics.
        """
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
        
        This maximizes the lower bound on log likelihood wrt. to the alpha and beta
        parameters.
        """
        marginal = numpy.log(self.classWord.sum(axis = 1)) 
        marginal.shape += (1,) # alter array shape from (n,) to (n, 1); needed for correct broadcasting on the following line
        logProbW = numpy.log(self.classWord) - marginal # (logarithm of) the beta parameter = word-topic distribution
        self.logProbW = numpy.where(numpy.isfinite(logProbW), logProbW, -100.0) # replace log(0) with -100.0 (almost zero probability)
        
        if estimateAlpha:
            self.alpha = self.optAlpha()
        
        logging.debug("updated model to %s" % self)
    
    
    def optAlpha(self, MAX_ALPHA_ITER = 1000, NEWTON_THRESH = 1e-5):
        """
        Estimate new Dirichlet priors (actually just one scalar shared across all
        topics).
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
        phi, expected word counts gamma (~topic distribution)).
        
        A document is simply a bag-of-words collection which supports len() and 
        iteration over (wordIndex, wordCount) 2-tuples.
        
        The model itself is not affected in any way (this function is read-only aka 
        const).
        """
        # init help structures
        totalWords = sum(wordCount for _, wordCount in doc)
        gamma = numpy.zeros(self.numTopics) + self.alpha + 1.0 * totalWords / self.numTopics
        phi = numpy.zeros(shape = (len(doc), self.numTopics)) + 1.0 / self.numTopics
        likelihood = likelihoodOld = converged = numpy.NAN
        
        # variational estimate
        for i in xrange(self.VAR_MAX_ITER):
#            logging.debug("inference step #%s, converged=%s, likelihood=%s, likelikelihoodOld=%s" % 
#                          (i, converged, likelihood, likelihoodOld))
            
            if numpy.isfinite(converged) and converged <= self.VAR_CONVERGED:
                logging.debug("document converged in %i iterations" % i)
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
        gammaSum = gamma.sum()
        digSum = digamma(gammaSum)
        dig = digamma(gamma) - digSum # precompute the difference
        
        likelihood = gammaln(self.alpha * self.numTopics) - \
                     self.numTopics * gammaln(self.alpha) - \
                     gammaln(gammaSum)
        
        likelihood += numpy.sum((self.alpha - 1) * dig + gammaln(gamma) - (gamma - 1) * dig) # array broadcast
        
        for n, (wordIndex, wordCount) in enumerate(doc):
            partial = phi[n] * (dig - numpy.log(phi[n]) + self.logProbW[:, wordIndex]) # array broadcast
            partial = numpy.where(numpy.isfinite(partial), partial, 0.0) # silently replace NaNs (from 0 * log(0) in phi) with 0.0
            likelihood += wordCount * partial.sum()
        
        return likelihood
    
    
    def countsFromCorpus(self, corpus, numInitDocs = 1):
        """
        Initialize the model word counts from the corpus. Each topic will be initialized
        from `numInitDocs` random documents.
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
    
    
    def infer(self, corpus):
        """
        Perform inference on a corpus of documents.
        
        This means that a standard inference step is taken for each document from 
        the corpus and the results are saved into file corpus.fname.lda_inferred.
        
        The output format of this file is one doc per line::
        doc_likelihood[TAB]topic1:prob ... topicK:prob[TAB]word1:topic ... wordN:topic
        
        Topics are sorted by probability, words are in the same order as in the input.
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
    
    
    def getTopicsMatrix(self):
        """
        Transform topic-word distribution via a tf-idf score and return it instead
        of the simple self.logProbW word-topic probabilities.
        
        The transformation is a sort of TF-IDF score, where the word gets higher 
        score if it's probable in this topic (the TF part) and lower score if 
        it's probable across the whole corpus (the IDF part).
        
        The exact formula is taken from Blei&Laffery: "Topic Models", 2009
        
        The returned matrix is of the same shape as logProbW.
        """
        # compute the geometric mean of words' probability across all topics
        idf = self.logProbW.sum(axis = 0) / self.numTopics # note: vector operation
        assert idf.shape == (self.numTerms,), "idf.shape=%s" % idf.shape
        
        # transform the weights, one topic after another
        result = numpy.empty_like(self.logProbW)
        for i in xrange(self.numTopics):
            # determine the score of all words, in this topic
            scores = numpy.exp(self.logProbW[i]) * (self.logProbW[i] - idf)
            assert scores.shape == (self.numTerms,), scores.shape
            result[i] = scores
        
        return result
        
    
    def printTopics(self, numWords = 10):
        """
        Print the top `numTerms` words for each topic, along with the log of their 
        probability. 
        
        Uses getTopicsMatrix() method to determine the 'top words'.
        """
        # determine the score of all words, in all topics
        transformed = self.getTopicsMatrix()
        
        # print top words, one topic after another
        for i, scores in enumerate(transformed):
            # link scores with the actual words (strings)
            termScores = zip(scores, self.logProbW[i], map(self.id2word.get, xrange(len(scores))))
            
            # sort words -- words with the best scores come first; keep only the best numWords
            best = sorted(termScores, reverse = True)[:numWords]
           
            # print best numWords, with a space separating each word:prob entry
            print "topic #%i: %s" % (i, ' '.join('%s:%.3f' % (word, prob) for (score, prob, word) in best))
    

    def __getitem__(self, bow):
        """
        Return topic distribution for the given document, as a list of 
        (topic_id, topic_value) 2-tuples.
        
        Ignore topics with very low probability (below 0.001).
        """
        likelihood, phi, gamma = self.inference(bow)
        gamma -= self.alpha # subtract topic prior, to get the expected number of words for each topic
        sumGamma = gamma.sum()
        if numpy.allclose(sumGamma, 0): # if there were no topics found, return nothing (eg for empty documents)
            return []
        topicDist = gamma / sumGamma # convert to proper distribution
        return [(topicId, topicValue) for topicId, topicValue in enumerate(topicDist)
                if topicValue >= 0.001] # ignore topics with prob < 0.001
#endclass LdaModel

