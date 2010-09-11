#!/usr/bin/env python
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
import itertools

logger = logging.getLogger('ldamodel')
logger.setLevel(logging.INFO)


import numpy # for arrays, array broadcasting etc.
numpy.seterr(divide='ignore') # ignore 0*log(0) errors

from scipy.special import gammaln, digamma, polygamma # gamma function utils
from scipy.maxentropy import logsumexp # log of sum
try:
    from scipy import weave
except ImportError:
    logger.warning("scipy.weave not available; falling back to pure Python, LDA will be *much* slower")

from gensim import interfaces, utils



trigamma = lambda x: polygamma(1, x) # second derivative of the gamma fnc


class LdaState(utils.SaveLoad):
    def __init__(self):
        self.alphaSuffStats = 0.0 # reset alpha stats
        self.numDocs = 0
        self.likelihood = 0.0

    def reset(self, mat):
        self.classWord = numpy.zeros_like(mat) # reset counts
        self.alphaSuffStats = 0.0 # reset alpha stats
        self.numDocs = 0
        self.likelihood = 0.0
    
    def merge(self, other):
        self.classWord += other.classWord
        self.alphaSuffStats += other.alphaSuffStats
        self.numDocs += other.numDocs
        self.likelihood += other.likelihood
#endclass LdaState


class LdaModel(interfaces.TransformationABC):
    """
    Objects of this class allow building and maintaining a model of Latent Dirichlet
    Allocation.
    
    The code is based on Blei's C implementation, see http://www.cs.princeton.edu/~blei/lda-c/ .
    
    This Python code uses numpy heavily, and is about 4-5x slower than the original 
    C version. The up side is that it is streamed (documents come in sequentially,
    no random indexing) and runs in constant memory w.r.t. the number of documents 
    (input corpus size).
    
    The constructor estimates model parameters based on a training corpus:
    
    >>> lda = LdaModel(corpus, numTopics=10)
    
    You can then infer topic distributions on new, unseen documents, with:
    
    >>> doc_lda = lda[doc_bow]
    
    Model persistency is achieved via its load/save methods.
    """
    def __init__(self, corpus=None, numTopics=200, id2word=None, distributed=False, 
                 chunks=1000, alpha=None, initMode='random', dtype=numpy.float32):
        """
        `numTopics` is the number of requested latent topics to be extracted from
        the training corpus. 
        
        `id2word` is a mapping from word ids (integers) to words (strings). It is
        used to determine the vocabulary size, as well as for debugging and topic 
        printing.
        
        `initMode` can be either 'random', for a fast random initialization of 
        the model parameters, or 'seeded', for an initialization based on a handful
        of real documents. The 'seeded' mode requires an extra sweep over the entire 
        input corpus, and is thus much slower.

        `alpha` is either None (to be estimated during training) or a number 
        between (0.0, 1.0).
        
        Turn on `distributed` to force distributed computing (see the web tutorial
        on how to set up a cluster).
        
        Example:
        
        >>> lda = LdaModel(corpus, numTopics=100)
        >>> print lda[doc_tfidf] # get topic probability distribution for a documents
        >>> lda.addDocuments(corpus2) # update LDA with additional documents
        >>> print lda[doc_tfidf]
        
        """
        # store user-supplied parameters
        self.id2word = id2word
        if self.id2word is None:
            logger.info("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dictFromCorpus(corpus)
            self.numTerms = len(self.id2word)
        else:
            self.numTerms = 1 + max([-1] + self.id2word.keys())
        if self.numTerms == 0:
            raise ValueError("cannot compute LDA over an empty collection (no terms)")
        
        self.distributed = bool(distributed)
        self.numTopics = int(numTopics)
        self.chunks = int(chunks)
        self.state = LdaState()
        
        # initialize wordtype/topic counts
        if initMode == 'seeded': # init from corpus (slow)
            self.state.classWord = self.countsFromCorpus(corpus, numInitDocs=2)
        elif initMode == 'random': # init with 1/k+noise
            self.state.classWord = 1.0 / self.numTerms + numpy.random.rand(self.numTopics, self.numTerms) # add noise from <0, 1)
        else:
            raise NotImplementedError("LDA initialization mode '%s' not supported" % str(initMode))
        self.state.classWord = self.state.classWord.astype(dtype)
        
        # internal algorithm constants
        self.estimate_alpha = alpha is None
        if self.estimate_alpha: # no alpha supplied by user => get some initial estimate
            alpha = 10.0 / numTopics # n / numTopics, as suggested in Steyvers&Griffiths: Probabilistic Topic Models
        self.alpha = min(0.99999, max(0.00001, alpha)) # dirichlet prior; make sure it's within bounds

        # EM training constants
        self.EM_MAX_ITER = 50 # maximum number of EM iterations; usually converges earlier
        self.EM_CONVERGED = 1e-4 # relative difference between two iterations; if lower than this, stop the EM training 
        self.VAR_MAX_ITER = 20 # maximum number of document inference iterations
        self.VAR_CONVERGED = 1e-6 # relative difference between document inference iterations needed to stop sooner than VAR_MAX_ITER
        
        if not distributed:
            logger.info("using serial LDA version on this node")
            self.dispatcher = None
        else:
            try:
                import Pyro
                ns = Pyro.naming.locateNS()
                dispatcher = Pyro.core.Proxy('PYRONAME:gensim.lda_dispatcher@%s' % ns._pyroUri.location)
                dispatcher._pyroOneway.add("exit")
                logger.debug("looking for dispatcher at %s" % str(dispatcher._pyroUri))
                dispatcher.initialize(id2word=self.id2word, numTopics=numTopics, 
                                      chunks=chunks, alpha=alpha, distributed=False)
                self.dispatcher = dispatcher
                logger.info("using distributed version with %i workers" % len(dispatcher.getworkers()))
            except Exception, err:
                # distributed version was specifically requested, so this is an error state
                logger.error("failed to initialize distributed LDA (%s)" % err)
                raise RuntimeError("failed to initialize distributed LDA (%s)" % err)

        if corpus is not None:
            self.addDocuments(corpus)
    

    def __str__(self):
        return "LdaModel(numTerms=%s, numTopics=%s, alpha=%s (estimated=%s))" % \
                (self.numTerms, self.numTopics, self.alpha, self.estimate_alpha)


    def reset(self, logProbW):
        self.state.reset(logProbW)
        self.logProbW = numpy.asfortranarray(logProbW)

    
    def addDocuments(self, corpus, chunks=None):
        """
        Run LDA parameter estimation on a training corpus, using the EM algorithm.
        
        This effectively updates the underlying LDA model on new documents from 
        `corpus` (or initializes the model if this is the first call).
        """
        if chunks is None:
            chunks = self.chunks
        
        likelihoodOld = converged = numpy.NAN
        self.mle(estimateAlpha = False)
        
        # main EM loop: iterate over the supplied corpus multiple times, until convergence
        for i in xrange(self.EM_MAX_ITER):
            logger.info("starting EM iteration #%i, converged=%s, likelihood=%s" % 
                         (i, converged, self.state.likelihood))

            if likelihoodOld < 1e-6 or numpy.isfinite(converged) and (converged <= self.EM_CONVERGED): # solution good enough?
                logger.info("EM converged in %i iterations" % i)
                break
        
            # initialize a new iteration
            if self.dispatcher:
                logger.info('initializing workers for a new EM iteration')
                self.dispatcher.reset(self.logProbW)
            else:
                self.state.reset(self.logProbW)
    
            # E step: iterate over the corpus, using old beta and updating new word counts
            # proceed in chunks of `chunks` documents
            chunk_no, chunker = -1, itertools.groupby(enumerate(corpus), key = lambda (docno, doc): docno / chunks)
            for chunk_no, (key, group) in enumerate(chunker):
                if self.dispatcher:
                    # distributed version: add this job to the job queue, so workers can munch on it
                    logger.info('PROGRESS: iteration %i, dispatched documents up to #%i' % (i, chunk_no * chunks))
                    logger.debug("creating job #%i" % chunk_no)
                    job = [doc for docno, doc in group]
                    self.dispatcher.putjob(job) # this will eventually block, because the queue has a small finite size
                    del job
                else:
                    # serial version, there is only one "worker" (myself) => process the job directly
                    logger.info('PROGRESS: iteration %i, document #%i' % (i, chunk_no * chunks))
                    self.docEStep(doc for docno, doc in group)
            
            # wait for all workers to finish (distributed version only)
            if self.dispatcher:
                logger.info("reached the end of input; now waiting for all remaining jobs to finish")
                import time
                while self.dispatcher.jobsdone() <= chunk_no:
                    time.sleep(0.5) # check every half a second
                logger.info("all jobs finished, downloading iteration statistics")
                self.state = self.dispatcher.getstate()
                
            likelihood = self.state.likelihood # likelihood of the training corpus
            assert numpy.isfinite(likelihood), "bad likelihood %s" % likelihood

            # check for convergence
            converged = numpy.divide(likelihoodOld - likelihood, likelihoodOld)
            logger.info("finished E step #%i: likelihood %f, likelihoodOld %f, converged %f" % 
                         (i, likelihood, likelihoodOld, converged))
            likelihoodOld = likelihood
            
            # M step -- update alpha and beta
            logger.info("performing M step #%i" % i)
            self.mle(estimateAlpha = self.estimate_alpha)
        
            # log some debug info about topics found so far
            self.printTopics()
        #endfor iteration


    def docEStep(self, corpus):
        """
        Find optimizing parameters for phi and gamma, and update sufficient statistics.
        """
        self.logProbW = numpy.asfortranarray(self.logProbW)
        for doc in corpus:
            # posterior inference
            likelihood, phi, gamma = self.inference(doc)
            
            # update sufficient statistics
            for n, (wordIndex, wordCount) in enumerate(doc):
                self.state.classWord[:, wordIndex] += wordCount * phi[n]
            self.state.alphaSuffStats += numpy.sum(digamma(gamma)) - self.numTopics * digamma(numpy.sum(gamma))
            self.state.likelihood += likelihood
            self.state.numDocs += 1


    def inference(self, doc):
        """
        Perform inference on a single document.
        
        Return 3-tuple of `(likelihood of this document, word-topic distribution
        phi, expected word counts gamma (~topic distribution))`.
        
        A document is simply a bag-of-words collection which supports len() and 
        iteration over (wordIndex, wordCount) 2-tuples.
        
        The model itself is not affected in any way (this function is read-only 
        aka const).
        """
        # init help structures
        totalWords = sum(wordCount for _, wordCount in doc)
        gamma = numpy.zeros(self.numTopics) + self.alpha + 1.0 * totalWords / self.numTopics
        phi = numpy.zeros(shape = (len(doc), self.numTopics)) + 1.0 / self.numTopics
        likelihood = likelihoodOld = converged = numpy.NAN
        assert self.logProbW.flags.f_contiguous # assert column-major format; cannot afford conversion at this low level
        
        # variational estimate
        for i in xrange(self.VAR_MAX_ITER):
#            logger.debug("inference step #%s, converged=%s, likelihood=%s, likelikelihoodOld=%s" % 
#                          (i, converged, likelihood, likelihoodOld))
            
            if numpy.isfinite(converged) and converged <= self.VAR_CONVERGED:
                logger.debug("document converged in %i iterations" % i)
                break
            
            for n, (wordIndex, wordCount) in enumerate(doc):
                # compute phi vars, in log space, to prevent numerical nastiness
                phin, tmp = phi[n], digamma(gamma) + self.logProbW[:, wordIndex]
                
                # convert phi and update gamma
                try:
                    code = """
                    const int n = Ntmp[0];
                    double newphi, tmpSum = 0.0;
                    for (int k = 0; k < n; k++)
                        tmpSum += exp(tmp[k]);
                    tmpSum = log(tmpSum);
                    for (int i = 0; i < n; i++) {
                        newphi = exp(tmp[i] - tmpSum);
                        gamma[i] += wordCount * (newphi - phin[i]);
                        phin[i] = newphi;
                    } 
                    """
                    weave.inline(code, ['phin', 'tmp', 'gamma', 'wordCount'])
                except:
                    newPhi = numpy.exp(tmp - numpy.log(numpy.sum(numpy.exp(tmp))))
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
        
        likelihood += numpy.sum((self.alpha - 1) * dig + gammaln(gamma) - (gamma - 1) * dig)
        
        for n, (wordIndex, wordCount) in enumerate(doc):
            try:
                phin, lprob = phi[n], self.logProbW[:, wordIndex] # only references; stride must be 1!
                code = """
                const int num_terms = Nphin[0];
                double result = 0.0;
                for (int i=0; i < num_terms; i++) {
                    if (phin[i] > 1e-8 || phin[i] < -1e-8)
                        result += phin[i] * (dig[i] - log(phin[i]) + lprob[i]);
                }
                return_val = wordCount * result;
                """
                likelihood += weave.inline(code, ['dig', 'phin', 'lprob', 'wordCount'])
            except:
                partial = phi[n] * (dig - numpy.log(phi[n]) + self.logProbW[:, wordIndex])
                partial[numpy.isnan(partial)] = 0.0 # replace NaNs (from 0 * log(0) in phi) with 0.0
                likelihood += wordCount * numpy.sum(partial)
        return likelihood
    

    def mle(self, estimateAlpha):
        """
        Maximum likelihood estimate.
        
        This maximizes the lower bound on log likelihood wrt. to the alpha and beta
        parameters.
        """
        marginal = numpy.log(self.state.classWord.sum(axis = 1)).reshape(self.numTopics, 1)
        self.logProbW = numpy.log(self.state.classWord)
        self.logProbW -= marginal
        self.logProbW = numpy.where(numpy.isfinite(self.logProbW), self.logProbW, -100.0) # replace log(0) with -100.0 (almost zero probability)
        
        if estimateAlpha:
            self.alpha = self.optAlpha()
        
        logger.debug("updated model to %s" % self)
    
    
    def optAlpha(self, MAX_ALPHA_ITER=1000, NEWTON_THRESH=1e-5):
        """
        Estimate new Dirichlet priors (actually just one scalar shared across all
        topics).
        """
        initA = 100.0
        logA = numpy.log(initA) # keep computations in log space
        logger.debug("optimizing old alpha %s" % self.alpha)
        
        for i in xrange(MAX_ALPHA_ITER):
            a = numpy.exp(logA)
            if not numpy.isfinite(a):
                initA = initA * 10.0
                logger.warning("alpha is NaN; new init alpha=%f" % initA)
                a = initA
                logA = numpy.log(a)
            s = self.state
            f = s.numDocs * (gammaln(self.numTopics * a) - self.numTopics * gammaln(a)) + (a - 1) * s.alphaSuffStats
            df = s.alphaSuffStats + s.numDocs * (self.numTopics * digamma(self.numTopics * a) - self.numTopics * digamma(a))
            d2f = s.numDocs * (self.numTopics * self.numTopics * trigamma(self.numTopics * a) - self.numTopics * trigamma(a))
            logA -= df / (d2f * a + df)
#            logger.debug("alpha maximization: f=%f, df=%f" % (f, df))
            if numpy.abs(df) <= NEWTON_THRESH:
                break
        result = numpy.exp(logA) # convert back from log space
        logger.info("estimated old alpha %s to new alpha %s" % (self.alpha, result))
        return result


    def probs2scores(self, numTopics=10):
        """
        Transform topic-word probability distribution into more human-friendly 
        scores, in hopes these scores make topics more easily interpretable.
        
        The transformation is a sort of TF-IDF score, where the word gets higher 
        score if it's probable in this topic (the TF part) and lower score if 
        it's probable across all topics (the IDF part).
        
        The exact formula is taken from Blei&Lafferty: "Topic Models", 2009
        
        The `numTopics` transformed scores are yielded iteratively, one topic after
        another.
        """
        logger.info("computing the word-topic salience matrix for %i topics" % numTopics)
        # compute the geometric mean of words' probability across all topics
        idf = self.logProbW.sum(axis = 0) / self.numTopics # compute the mean in log space
        
        # transform the probabilities to weights, one topic after another
        for probs in self.logProbW[:numTopics]:
            yield numpy.exp(probs) * (probs - idf)
        
    
    def printTopics(self, numTopics=5, numWords=10):
        """
        Print the top `numTerms` words for `numTopics` topics, along with the 
        log of their probability. 
        
        Uses the `probs2scores()` method to determine what the 'top words' are.
        """
        # determine the score of all words in the selected topics
        numTopics = min(numTopics, self.numTopics) # cannot print more topics than computed...
        
        # print top words, one topic after another
        for i, scores in enumerate(self.probs2scores(numTopics)):
            # link scores with the actual words (strings)
            termScores = zip(scores, self.logProbW[i], map(self.id2word.get, xrange(len(scores))))
            
            # sort words -- words with the best scores come first; keep only the best numWords
            best = sorted(termScores, reverse=True)[:numWords]
           
            # print best numWords, with a space separating each word:prob entry
            logger.info("topic #%i: %s" % 
                        (i, ' '.join('%s:%.3f' % (word, prob) for (score, prob, word) in best)))
    

    def countsFromCorpus(self, corpus, numInitDocs=1):
        """
        Initialize the model word counts from the corpus. Each topic will be 
        initialized from `numInitDocs` randomly selected documents. The corpus 
        is only iterated over once.
        """
        logger.info("initializing model with %i random document(s) per topic" % numInitDocs)
        
        # next we precompute the all the random document indices, so that we can 
        # update the counts in a single sweep over the corpus. 
        # all this drama is because the corpus doesn't necessarily support 
        # random access -- it only supports sequential iteration over 
        # the documents (streaming).
        initDocs = numpy.random.randint(0, len(corpus), (self.numTopics, numInitDocs)) # get document indices
        result = numpy.ones(shape = (self.numTopics, self.numTerms)) # add-one smoothing
        
        # go over the corpus once, updating the counts
        for i, doc in enumerate(corpus):
            for k in xrange(self.numTopics):
                if i in initDocs[k]: # do we want to initialize this topic with this document?
                    for wordIndex, wordCount in doc: # for each word in the document...
                        result[k, wordIndex] += wordCount # ...add its count to the word-topic count
        return result
        
    
    def __getitem__(self, bow, eps=0.001):
        """
        Return topic distribution for the given document `bow`, as a list of 
        (topic_id, topic_probability) 2-tuples.
        
        Ignore topics with very low probability (below 0.001).
        """
        # if the input vector is in fact a corpus, return a transformed corpus as result
        is_corpus, corpus = utils.isCorpus(bow)
        if is_corpus:
            return self._apply(corpus)
        
        likelihood, phi, gamma = self.inference(bow)
        gamma -= self.alpha # subtract topic prior, to get the expected number of words for each topic
        sumGamma = gamma.sum()
        if numpy.allclose(sumGamma, 0): # if there were no topics found, return nothing (eg for empty documents)
            return []
        topicDist = gamma / sumGamma # convert to proper distribution
        return [(topicId, topicValue) for topicId, topicValue in enumerate(topicDist)
                if topicValue >= eps] # ignore topics with prob < 0.001
#endclass LdaModel

