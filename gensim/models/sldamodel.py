import logging
import numbers
import os

import numpy as np
import six
from scipy.special import gammaln, psi  
from scipy.special import polygamma
from six.moves import xrange
from collections import defaultdict

from gensim import interfaces, utils, matutils
from gensim.matutils import dirichlet_expectation
from gensim.models import basemodel, CoherenceModel

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp
    
logger = logging.getLogger('gensim.models.sldamodel')

def update_dir_prior(prior, N, logphat, rho):
    """
    Updates a given prior using Newton's method, described in
    **Huang: Maximum Likelihood Estimation of Dirichlet Distribution Parameters.**
    http://jonathan-huang.org/research/dirichlet/dirichlet.pdf
    """
    dprior = np.copy(prior)  # TODO: unused var???
    gradf = N * (psi(np.sum(prior)) - psi(prior) + logphat)

    c = N * polygamma(1, np.sum(prior))
    q = -N * polygamma(1, prior)

    b = np.sum(gradf / q) / (1 / c + np.sum(1 / q))

    dprior = -(gradf - b) / q

    if all(rho * dprior + prior > 0):
        prior += rho * dprior
    else:
        logger.warning("updated prior not positive")

    return prior

class sLdaState(utils.SaveLoad):
    """
    Encapsulate information for distributed computation of LdaModel objects.
    Objects of this class are sent over the network, so try to keep them lean to
    reduce traffic.
    """

    def __init__(self, eta, shape, dtype=np.float32):
        self.eta = eta.astype(dtype, copy=False)
        self.sstats = np.zeros(shape, dtype=dtype)
        self.numdocs = 0
        self.dtype = dtype

    def reset(self):
        self.sstats[:] = 0.0
        self.numdocs = 0

    def merge(self, other):
        """
        Merge the result of an E step from one node with that of another node
        (summing up sufficient statistics).
        The merging is trivial and after merging all cluster nodes, we have the
        exact same result as if the computation was run on a single node (no
        approximation).
        """
        self.sstats += other.sstats
        self.numdocs += other.numdocs
    
    def blend(self, rhot, other, targetsize=None):
        """
        Given sLdaState `other`, merge it with the current state. Stretch both to
        `targetsize` documents before merging, so that they are of comparable
        magnitude.
        Merging is done by average weighting: in the extremes, `rhot=0.0` means
        `other` is completely ignored; `rhot=1.0` means `self` is completely ignored.
        This procedure corresponds to the stochastic gradient update from Hoffman
        et al., algorithm 2 (eq. 14).
        """
        assert other is not None
        if targetsize is None:
            targetsize = self.numdocs

        # stretch the current model's expected n*phi counts to target size
        if self.numdocs == 0 or targetsize == self.numdocs:
            scale = 1.0
        else:
            scale = 1.0 * targetsize / self.numdocs
        self.sstats *= (1.0 - rhot) * scale

        # stretch the incoming n*phi counts to target size
        if other.numdocs == 0 or targetsize == other.numdocs:
            scale = 1.0
        else:
            logger.info("merging changes from %i documents into a model of %i documents", other.numdocs, targetsize)
            scale = 1.0 * targetsize / other.numdocs
        self.sstats += rhot * scale * other.sstats

        self.numdocs = targetsize
    
    def blend2(self, rhot, other, targetsize=None):
        """
        Alternative, more simple blend.
        """
        assert other is not None
        if targetsize is None:
            targetsize = self.numdocs

        self.sstats += other.sstats
        self.numdocs = targetsize

    def get_lambda(self):
        return self.eta + self.sstats

    def get_Elogbeta(self):
        return dirichlet_expectation(self.get_lambda())

class sLdaModel(interfaces.TransformationABC, basemodel.BaseTopicModel):
    
    
    def __init__(self, corpus=None, id2word=None, num_topics=100, 
                 chunksize=500, passes=1, iterations=50, decay=0.5, offset=1.0,
                 alpha=0.05, eta=None, update_every=1, 
                 eval_every=10, gamma_threshold=0.001, random_state=None):
        distributed = False
        self.dispatcher = None
        self.numworkers = 1
        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError(
                "at least one of corpus/id2word must be specified, to establish input space dimensionality"
            )
        
        if self.id2word is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        elif len(self.id2word) > 0:
            self.num_terms = 1 + max(self.id2word.keys())
        else:
            self.num_terms = 0

        if self.num_terms == 0:
            raise ValueError("cannot compute the author-topic model over an empty collection (no terms)")

        logger.info('Vocabulary consists of %d words.', self.num_terms)
        
        self.dtype = np.float64
        self.corpus = corpus
        self.num_topics = int(num_topics)
        self.chunksize = chunksize
        self.decay = decay
        self.offset = offset
        self.alpha = np.ascontiguousarray(alpha, self.dtype)
        self.beta = np.ascontiguousarray(beta, self.dtype)
        self.iterations = iterations
        self.random_state = utils.get_random_state(random_state)
        self._likelihoods = list()
        
        self.passes = passes
        self.update_every = update_every
        self.eval_every = eval_every
        
        self.iterations = iterations
        self.gamma_threshold = gamma_threshold
        
    def save(self, fname, *args, **kwargs):
        super(sLdaModel, self).save(fname, *args, **kwargs)
        
    def load(self, cls, *args, **kwargs):
        result = super(sLdaModel, cls).load(fname, *args, **kwargs)
        
    def do_estep(self, chunk, state=None):
        if state is None:
            state = self.state
        gamma, sstats = self.inference(chunk, collect_sstats=True)
        state.sstats += sstats
        state.numdocs += gamma.shape[0]  
        assert gamma.dtype == self.dtype
        return gamma
    
    def do_mstep(self, rho, other, extra_pass=False):
        logger.debug("updating topics")
        diff = np.log(self.expElogbeta)
        self.state.blend(rho, other)
        diff -= self.state.get_Elogbeta()
        self.sync_state()

        self.print_topics(5)
        logger.info("topic diff=%f, rho=%f", np.mean(np.abs(diff)), rho)

        if self.optimize_eta:
            self.update_eta(self.state.get_lambda(), rho)

        if not extra_pass:
            self.num_updates += other.numdocs
    
    def inference(self, chunk, collect_sstats=False):
        raise NotImplementedError
        
    def update_eta(self, lambdat, rho):
        """
        Update parameters for the Dirichlet prior on the per-topic
        word weights `eta` given the last `lambdat`.
        """
        N = float(lambdat.shape[0])
        logphat = (sum(dirichlet_expectation(lambda_) for lambda_ in lambdat) / N).reshape((self.num_terms,))
        assert logphat.dtype == self.dtype

        self.eta = update_dir_prior(self.eta, N, logphat, rho)

        assert self.eta.dtype == self.dtype
        return self.eta
    
    def update(self, corpus, chunksize=None, decay=None, offset=None,
               passes=None, update_every=None, eval_every=None, iterations=None,
               gamma_threshold=None, chunks_as_numpy=False):
        
        raise NotImplementedError
