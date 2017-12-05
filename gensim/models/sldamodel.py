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
from gensim.models.ldamodel import LdaState as state

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
    dprior = np.copy(prior)
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

class SLdaModel(interfaces.TransformationABC, basemodel.BaseTopicModel):


    def __init__(self, corpus=None, responses=None, id2word=None, num_topics=100,
                 chunksize=500, passes=1, iterations=50, decay=0.5, offset=1.0,
                 alpha='symmetric', eta=None, update_every=1,
                 eval_every=10, gamma_threshold=0.001, minimum_probability=0.01,
                 minimum_phi_value=0.01, random_state=None):
        distributed = False
        self.dispatcher = None
        self.numworkers = 1
        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError(
                "at least one of corpus/id2word must be specified, to establish input space dimensionality"
            )

        if responses is None:
            raise ValueError(
                "A response variable must be specified, to establish mapping parameters"
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
        self.responses = np.array(responses)
        self.num_topics = int(num_topics)
        self.chunksize = chunksize
        self.decay = decay
        self.offset = offset
        self.minimum_probability = minimum_probability
        self.num_updates = 0

        self.iterations = iterations
        self.random_state = utils.get_random_state(random_state)
        self._likelihoods = list()

        self.passes = passes
        self.update_every = update_every
        self.eval_every = eval_every
        self.minimum_phi_value = minimum_phi_value
        self.gamma_threshold = gamma_threshold

        self.alpha, self.optimize_alpha = self.init_dir_prior(alpha, 'alpha')

        assert self.alpha.shape == (self.num_topics,), "Invalid alpha shape. Got shape %s, but expected (%d, )" % (str(self.alpha.shape), self.num_topics)

        if isinstance(eta, six.string_types):
            if eta == 'asymmetric':
                raise ValueError("The 'asymmetric' option cannot be used for eta")

        self.eta, self.optimize_eta = self.init_dir_prior(eta, 'eta')

def init_dir_prior(self, prior, name):
    if prior is None:
        prior = 'symmetric'

    if name == 'alpha':
        prior_shape = self.num_topics
    elif name == 'eta':
        prior_shape = self.num_terms
    else:
        raise ValueError("'name' must be 'alpha' or 'eta'")

    is_auto = False

    if isinstance(prior, six.string_types):
        if prior == 'symmetric':
            logger.info("using symmetric %s at %s", name, 1.0 / prior_shape)
            init_prior = np.asarray([1.0 / self.num_topics for i in xrange(prior_shape)])
        elif prior == 'asymmetric':
            init_prior = np.asarray([1.0 / (i + np.sqrt(prior_shape)) for i in xrange(prior_shape)])
            init_prior /= init_prior.sum()
            logger.info("using asymmetric %s %s", name, list(init_prior))
        elif prior == 'auto':
            is_auto = True
            init_prior = np.asarray([1.0 / self.num_topics for i in xrange(prior_shape)])
            if name == 'alpha':
                logger.info("using autotuned %s, starting with %s", name, list(init_prior))
        else:
            raise ValueError("Unable to determine proper %s value given '%s'" % (name, prior))
    elif isinstance(prior, list):
        init_prior = np.asarray(prior)
    elif isinstance(prior, np.ndarray):
        init_prior = prior
    elif isinstance(prior, np.number) or isinstance(prior, numbers.Real):
        init_prior = np.asarray([prior] * prior_shape)
    else:
        raise ValueError("%s must be either a np array of scalars, list of scalars, or scalar" % name)

    return init_prior, is_auto


    def sync_state(self):
        self.expElogbeta = np.exp(state.get_Elogbeta())
        assert self.expElogbeta.dtype == self.dtype

    def clear(self):
        """Clear model state (free up some memory). Used in the distributed algo."""
        self.state = None
        self.Elogbeta = None

    def save(self, fname, *args, **kwargs):
        kwargs['mmap'] = kwargs.get('mmap', None)
        result = super(sLdaModel, self).save(fname, *args, **kwargs)

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
        state.blend(rho, other)
        diff -= state.get_Elogbeta()
        self.sync_state()

        self.print_topics(5)
        logger.info("topic diff=%f, rho=%f", np.mean(np.abs(diff)), rho)

        if self.optimize_eta:
            self.update_eta(state.get_lambda(), rho)

        if not extra_pass:
            self.num_updates += other.numdocs

    def inference(self, chunk, responses, collect_sstats=False):
        try:
            len(chunk)
        except TypeError:
            chunk = list(chunk)
        if len(chunk) > 1:
            logger.debug("performing inference on a chunk of %i documents", len(chunk))

        # Initialize the variational distribution q(theta|gamma) for the chunk
        gamma = self.random_state.gamma(100., 1. / 100., (len(chunk), self.num_topics)).astype(self.dtype, copy=False)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)

        assert Elogtheta.dtype == self.dtype
        assert expElogtheta.dtype == self.dtype

        if collect_sstats:
            sstats = np.zeros_like(self.expElogbeta, dtype=self.dtype)
        else:
            sstats = None
        converged = 0

        for d, doc in enumerate(chunk):
            if len(doc) > 0 and not isinstance(doc[0][0], six.integer_types + (np.integer,)):
                ids = [int(idx) for idx, _ in doc]
            else:
                ids = [idx for idx, _ in doc]
            cts = np.array([cnt for _, cnt in doc], dtype=self.dtype)
            y = self.responses[d]
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self.expElogbeta[:, ids]
            # The optimal phi_{dwk} is proportional to expElogthetad_k * expElogbetad_w.
            # phinorm is the normalizer.
            phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100

            # Iterate between gamma and phi until convergence
            for _ in xrange(self.iterations):
                lastgamma = gammad
                gammad = self.alpha + expElogthetad * np.dot(cts / phinorm, expElogbetad.T)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = np.dot(expElogthetad, expElogbetad) + 1e-100
                meanchange = np.mean(abs(gammad - lastgamma))
                if meanchange < self.gamma_threshold:
                    converged += 1
                    break
            gamma[d, :] = gammad
            assert gammad.dtype == self.dtype
            if collect_sstats:
                sstats[:, ids] += np.outer(expElogthetad.T, cts / phinorm)

        if len(chunk) > 1:
            logger.debug("%i/%i documents converged within %i iterations", converged, len(chunk), self.iterations)

        if collect_sstats:
            sstats *= self.expElogbeta
            assert sstats.dtype == self.dtype

        assert gamma.dtype == self.dtype
        return gamma, sstats

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
