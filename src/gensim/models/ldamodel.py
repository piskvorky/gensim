#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
This module encapsulates functionality for the Latent Dirichlet Allocation algorithm.

It allows both model estimation from a training corpus and inference on new, 
unseen documents.

The implementation is based on **Blei et al., Latent Dirichlet Allocation, 2003**,
and on Blei's LDA-C software [1]_ in particular. This means it uses variational EM
inference rather than Gibbs sampling to estimate model parameters. NumPy is used 
heavily here, but is still much slower than the original C version. The up side 
is that it is **streamed** (documents come in sequentially, no random indexing), runs 
in **constant memory** w.r.t. the number of documents (input corpus size) and is 
**distributed** (makes use of a cluster of machines, if available).

.. [1] http://www.cs.princeton.edu/~blei/lda-c/

"""


import sys, logging
import itertools
import time

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
    """
    Encapsulate information returned by distributed computation of the E training step. 
    
    Objects of this class are sent over the network at the end of each corpus
    iteration, so try to keep this class lean to reduce traffic.
    """
    def __init__(self):
        self.reset()

    def reset(self, mat=None):
        """
        Prepare the state for a new iteration.
        """
        self.classWord = numpy.zeros_like(mat) # reset counts
        self.alphaSuffStats = 0.0 # reset alpha stats
        self.numDocs = 0
        self.likelihood = 0.0
    
    def merge(self, other):
        """
        Merge the result of an E step from one node with that of another node.
        
        The merging is trivial and after merging all cluster nodes, we have the 
        exact same result as if the computation was run on a single node (no 
        approximation).
        """
        self.classWord += other.classWord
        self.alphaSuffStats += other.alphaSuffStats
        self.numDocs += other.numDocs
        self.likelihood += other.likelihood
#endclass LdaState



class LdaModel(interfaces.TransformationABC):
    """
    Objects of this class allow building and maintaining a model of Latent Dirichlet
    Allocation.
    
    The constructor estimates model parameters based on a training corpus:
    
    >>> lda = LdaModel(corpus, numTopics=10)
    
    You can then infer topic distributions on new, unseen documents, with:
    
    >>> doc_lda = lda[doc_bow]
    
    Model persistency is achieved via its load/save methods.
    """
    def __init__(self, corpus=None, numTopics=200, id2word=None, distributed=False, 
                 chunks=None, alpha=None, initMode='random', dtype=numpy.float64):
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
        self.state = LdaState()
        self.chunks = chunks
        
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
            # set up distributed version
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
                logger.error("failed to initialize distributed LDA (%s)" % err)
                raise RuntimeError("failed to initialize distributed LDA (%s)" % err)

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None:
            self.addDocuments(corpus)
    

    def __str__(self):
        return "LdaModel(numTerms=%s, numTopics=%s, alpha=%s (estimated=%s))" % \
                (self.numTerms, self.numTopics, self.alpha, self.estimate_alpha)


    def reset(self, logProbW):
        self.state.reset(logProbW)
        self.logProbW = logProbW

    
    def addDocuments(self, corpus, chunks=None):
        """
        Run LDA parameter estimation on a training corpus, using the EM algorithm.
        
        This effectively updates the underlying LDA model on new documents from 
        `corpus` (or initializes the model if this is the first call).
        """
        if chunks is None:
            # in serial version `chunks` only affect the frequency of debug printing, so use whatever
            # in distributed version, `chunks` is a trade-off between network overhead/memory footprint; what is optimal?
            chunks = self.chunks or 100
        logger.info("using chunks of %i documents" % chunks)
        likelihoodOld = converged = numpy.NAN
        self.mle(estimateAlpha = False)
        
        # main EM loop: iterate over the supplied corpus multiple times, until convergence
        for i in xrange(self.EM_MAX_ITER):
            logger.info("starting EM iteration #%i, converged=%s, likelihood=%s" % 
                         (i, converged, self.state.likelihood))
            start = time.time()
            
            # initialize a new iteration
            if self.dispatcher:
                logger.info('initializing workers for a new EM iteration')
                self.dispatcher.reset(self.logProbW, self.alpha)
            else:
                self.reset(self.logProbW)
    
            # E step: iterate over the corpus, using old beta and updating new word counts
            # proceed in chunks of `chunks` documents
            chunk_no, chunker = -1, itertools.groupby(enumerate(corpus), key = lambda (docno, doc): docno / chunks)
            for chunk_no, (key, group) in enumerate(chunker):
                if self.dispatcher:
                    # distributed version: add this job to the job queue, so workers can munch on it
                    logger.info('PROGRESS: iteration %i, dispatched documents up to #%i' % (i, chunk_no * chunks))
                    logger.debug("creating job #%i" % chunk_no)
                    self.dispatcher.putjob([doc for docno, doc in group]) # this will eventually block until some jobs finish, because the queue has a small finite length
                else:
                    # serial version, there is only one "worker" (myself) => process the job directly
                    logger.info('PROGRESS: iteration %i, document #%i' % (i, chunk_no * chunks))
                    self.docEStep(doc for docno, doc in group)
            
            # wait for all workers to finish (distributed version only)
            if self.dispatcher:
                logger.info("reached the end of input; now waiting for all remaining jobs to finish")
                self.state = self.dispatcher.getstate()
            
            # check for convergence
            likelihood = self.state.likelihood # likelihood of the training corpus
            assert numpy.isfinite(likelihood), "bad likelihood %s" % likelihood
            converged = numpy.divide(likelihoodOld - likelihood, likelihoodOld)
            logger.info("finished E step #%i: likelihood %f, likelihoodOld %f, converged %f" % 
                         (i, likelihood, likelihoodOld, converged))
            if self.state.likelihood < likelihoodOld:
                # quit before M step, to keep the old logProbW and alpha
                logger.warning("iteration diverged! returning early")
                break
            likelihoodOld = likelihood
            
            # M step -- update alpha and beta (logProbW)
            logger.info("performing M step #%i" % i)
            self.mle(estimateAlpha = self.estimate_alpha)
        
            if likelihoodOld > -1e-6 or numpy.isfinite(converged) and (converged <= self.EM_CONVERGED): # solution good enough?
                logger.info("EM converged in %i iterations" % (i + 1))
                break
            
            logger.info("iteration #%i took %.2fs" % (i, time.time() - start))
            # log some debug info about topics found so far
            self.printTopics()
        #endfor EM loop


    def docEStep(self, corpus):
        """
        Find optimizing parameters for phi and gamma, and update sufficient statistics.
        """
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
        
        # variational estimate
        for i in xrange(self.VAR_MAX_ITER):
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
                        tmpSum += exp(TMP1(k));
                    tmpSum = log(tmpSum);
                    for (int i = 0; i < n; i++) {
                        newphi = exp(TMP1(i) - tmpSum);
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
                phin, lprob = phi[n], self.logProbW[:, wordIndex]
                code = """
                const int num_terms = Nphin[0];
                double result = 0.0;
                for (int i=0; i < num_terms; i++) {
                    if (phin[i] > 1e-8 || phin[i] < -1e-8)
                        result += phin[i] * (dig[i] - log(phin[i]) + LPROB1(i));
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
    
    
    def optAlpha(self, max_iter=1000, newton_thresh=1e-5):
        """
        Estimate new topic priors (actually just one scalar shared across all
        topics).
        """
        initA = 100.0
        logA = numpy.log(initA) # keep computations in log space
        logger.debug("optimizing old alpha %s" % self.alpha)
        
        for i in xrange(max_iter):
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
            logger.debug("alpha maximization: f=%f, df=%f" % (f, df))
            if numpy.abs(df) <= newton_thresh:
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
        
        The exact formula is taken from **Blei&Lafferty: "Topic Models", 2009**.
        
        The `numTopics` transformed scores are yielded iteratively, one topic after
        another.
        """
        logger.info("computing the word-topic salience matrix for %i topics" % numTopics)
        # compute the geometric mean of words' probability across all topics
        idf = self.logProbW.sum(axis = 0) / self.numTopics # compute the mean in log space
        
        # transform the probabilities to weights, one topic after another
        for probs in self.logProbW[:numTopics]:
            yield numpy.exp(probs) * (probs - idf)
    
    
    def printTopics(self, numTopics=5, numWords=10, pretty=True):
        """
        Print the top `numTerms` words for `numTopics` topics, along with the 
        log of their probability. 
        
        If `pretty` is set, use the `probs2scores()` to determine what the 'top 
        words' are. Otherwise, order the words directly by their word-topic probability.
        """
        # determine the score of all words in the selected topics
        numTopics = min(numTopics, self.numTopics) # cannot print more topics than computed...
        if pretty:
            scores = self.probs2scores(numTopics)
        else:
            scores = self.logProbW[:numTopics]
        
        # print top words, one topic after another
        for i, scores in enumerate(scores):
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
        
        Ignore topics with very low probability (below `eps`).
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

#####################

from gensim.corpora import WikiCorpus, MmCorpus
import numpy as n
from scipy.special import gammaln, psi

n.random.seed(100000001)
meanchangethresh = 0.001



def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return(psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])


class OnlineLDA(interfaces.TransformationABC):
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """

    def __init__(self, vocab, K, D, alpha, eta, tau0, kappa=0.0):
        """
        Arguments:
        K: Number of topics
        vocab: A set of words to recognize. When analyzing documents, any word
           not in this set will be ignored.
        D: Total number of documents in the population. For a fixed corpus,
           this is the size of the corpus. In the truly online setting, this
           can be an estimate of the maximum number of documents that
           could ever be seen.
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.

        Note that if you pass the same set of D documents in every time and
        set kappa=0 this class can also be used to do batch VB.
        """
        self._vocab = dict((word, wordid) for wordid, word in enumerate(vocab))
        self._K = K
        self._W = len(self._vocab)
        self._D = D
        self._alpha = alpha
        self._eta = eta
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._updatect = 0

        # Initialize the variational distribution q(beta|lambda)
        self._lambda = 1*n.random.gamma(100., 1./100., (self._K, self._W))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)

    def do_e_step(self, docs):
        """
        Given a mini-batch of documents, estimates the parameters
        gamma controlling the variational distribution over the topic
        weights for each document in the mini-batch.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns a tuple containing the estimated values of gamma,
        as well as sufficient statistics needed to update lambda.
        """
        logger.info("performing E step on %i documents" % len(docs))

        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        gamma = 1*n.random.gamma(100., 1./100., (len(docs), self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        sstats = n.zeros(self._lambda.shape)
        # Now, for each document d update that document's gamma and phi
        it = 0
        meanchange = 0
        for d, doc in enumerate(docs):
            if d % 10000 == 0:
                logger.info("PROGRESS: at document #%i/#%i" % (d, len(docs)))
            # These are mostly just shorthand
            ids = [id for id, _ in doc]
            cts = numpy.array([cnt for _, cnt in doc])
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self._expElogbeta[:, ids]
            # The optimal phi_{dwk} is proportional to 
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
            # Iterate between gamma and phi until convergence
            for it in xrange(100):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self._alpha + expElogthetad * \
                    n.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = n.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    break
            gamma[d, :] = gammad
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats[:, ids] += n.outer(expElogthetad.T, cts/phinorm)

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        sstats = sstats * self._expElogbeta

        return((gamma, sstats))

    def update_lambda(self, docs):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        variational parameter matrix lambda.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns gamma, the parameters to the variational distribution
        over the topic weights theta for the documents analyzed in this
        update.

        Also returns an estimate of the variational bound for the
        entire corpus for the OLD setting of lambda based on the
        documents passed in. This can be used as a (possibly very
        noisy) estimate of held-out likelihood.
        """

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        (gamma, sstats) = self.do_e_step(docs)
        # Estimate held-out likelihood for current values of lambda.
        bound = self.approx_bound(docs, gamma)
        # Update lambda based on documents.
        self._lambda = self._lambda * (1-rhot) + \
            rhot * (self._eta + self._D * sstats / len(docs))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._updatect += 1

        return(gamma, bound)

    def approx_bound(self, docs, gamma):
        """
        Estimates the variational bound over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.

        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """
        logger.info("computing likelihood of %i documents" % len(docs))

        score = 0
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        # E[log p(docs | theta, beta)]
        for d, doc in enumerate(docs):
            if d % 10000 == 0:
                logger.info("PROGRESS: at document #%i/#%i" % (d, len(docs)))
            gammad = gamma[d, :]
            ids = [id for id, _ in doc]
            cts = numpy.array([cnt for _, cnt in doc])
            phinorm = n.zeros(len(ids))
            for i in xrange(len(ids)):
                temp = Elogtheta[d, :] + self._Elogbeta[:, ids[i]]
                tmax = max(temp)
                phinorm[i] = n.log(sum(n.exp(temp - tmax))) + tmax
            score += n.sum(cts * phinorm)
#             oldphinorm = phinorm
#             phinorm = n.dot(expElogtheta[d, :], self._expElogbeta[:, ids])
#             print oldphinorm
#             print n.log(phinorm)
#             score += n.sum(cts * n.log(phinorm))

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += n.sum((self._alpha - gamma)*Elogtheta)
        score += n.sum(gammaln(gamma) - gammaln(self._alpha))
        score += sum(gammaln(self._alpha*self._K) - gammaln(n.sum(gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score * self._D / len(docs)

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + n.sum((self._eta-self._lambda)*self._Elogbeta)
        score = score + n.sum(gammaln(self._lambda) - gammaln(self._eta))
        score = score + n.sum(gammaln(self._eta*self._W) - 
                              gammaln(n.sum(self._lambda, 1)))

        return(score)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level = logging.INFO)
    logging.info("running %s" % ' '.join(sys.argv))

    # The number of documents to analyze each iteration
    vocab = WikiCorpus.loadDictionary('/Users/kofola/gensim/results/wiki10_en_wordids.txt')
    corpus = MmCorpus('/Users/kofola/gensim/results/wiki10_en_bow.mm')
    sumcnts = sum(sum(cnt for _, cnt in doc) for doc in corpus)
    logger.info("running LDA on %i documents, %i total tokens" % 
                (len(corpus), sumcnts))
    
    batchsize = 100000
    D = 100000 # total number of docs
    K = 100 # number of topics
    iterations = int(sys.argv[1])

    # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
    olda = OnlineLDA(vocab.values(), K, D, 1./K, 1./K, 1., kappa=0.0)
    # Run until we've seen D documents. (Feel free to interrupt *much*
    # sooner than this.)
    for iteration in range(0, iterations):
        # maybe select only a subset of corpus here (to simulate their "stochastic" approach)
        docset = list(itertools.islice(corpus, 10000))
        # Give them to online LDA
        (gamma, bound) = olda.update_lambda(docset)
        # Compute an estimate of held-out perplexity
        perwordbound = bound * len(docset) / (D * sumcnts)
        print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
            (iteration, olda._rhot, numpy.exp(-perwordbound))

        # Save lambda, the parameters to the variational distributions
        # over topics, and gamma, the parameters to the variational
        # distributions over topic weights for the articles analyzed in
        # the last iteration.
        if (iteration % 10 == 0):
            olda.save('lda-%i.pkl' % iteration)
            numpy.savetxt('lambda-%d.dat' % iteration, olda._lambda)
            numpy.savetxt('gamma-%d.dat' % iteration, gamma)
    
    logging.info("finished running %s" % program)
