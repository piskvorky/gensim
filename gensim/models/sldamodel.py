import numpy as np
from scipy.special import gammaln, psi
from numpy.linalg import solve

eps = 1e-100


class sLDA:
    def __init__(self, docs, responses, vocab, num_topic, alpha=0.1, sigma=1):
        self.docs = docs
        self.responses = np.array(responses)
        self.vocab = vocab

        self.n_doc = len(docs)
        self.n_voca = len(vocab)
        self.n_topic = num_topic

        # topic proportion for document
        self.gamma = np.random.gamma(100., 1. / 100, (self.n_doc, self.n_topic))
        # word-topic distribution
        self.beta = np.random.gamma(100., 1. / 100, (self.n_topic, self.n_voca))
        self.beta /= np.sum(self.beta, 1)[:, np.newaxis]
        # coefficient & covariance
        self.eta = np.zeros(self.n_topic)
        self.sigma = sigma
        self.alpha = alpha
        self.dir_prior = 0.01
        self.ssword = np.random.gamma(100., 1. / 100, (self.n_topic, self.n_voca))
        self.EA = np.zeros((self.n_doc, self.n_topic))
        self.EAA = np.zeros((self.n_topic, self.n_topic))
        self.small_iter = 10

    def do_e_step(self):
        self.EA = np.zeros((self.n_doc, self.n_topic))
        self.EAA = np.zeros((self.n_topic, self.n_topic))

        e_log_beta = np.log(self.beta)

        for di in xrange(self.n_doc):
            doc = self.docs[di]
            doc_len = len(doc)
            y = self.responses[di]

            phi = self.gamma[di, :][:, np.newaxis] * self.beta[:, doc]  # K * N
            phi /= np.sum(phi, 1)[:, np.newaxis]

            e_log_theta = psi(self.gamma[di, :]) - psi(np.sum(self.gamma[di, :]))
            # exact update phi
            # for i in xrange(self.small_iter):
            # for wi in xrange(len(doc)):
            #     word = doc[wi]
            #     phi_sum = np.sum(phi, 1) - phi[:,wi]    # K dim
            #     rss = (2. * np.dot(phi_sum, self.eta) * self.eta + (self.eta ** 2))/(2.* (np.float(doc_len)**2.) * (self.sigma**2.) )
            #     phi[:,wi] = np.exp(e_log_theta + e_log_beta[:,word] + (y/(np.float(doc_len)*(self.sigma**2.)))*self.eta - rss )
            #     phi[:,wi] /= phi[:,wi].sum()

            # approximately update phi
            phi_sum = np.sum(phi, 1)[:, np.newaxis] - phi
            rss = (2. * np.dot(self.eta, phi_sum)[:, np.newaxis] * self.eta + (self.eta ** 2)) / (
            2. * (np.float(doc_len) ** 2.) * (self.sigma ** 2.))
            rss = rss.T  # K x len
            phi = np.exp(e_log_theta[:, np.newaxis] + e_log_beta[:, doc] + ((y / (
            np.float(doc_len) * (self.sigma ** 2.))) * self.eta)[:, np.newaxis] - rss + eps)
            phi /= np.sum(phi, 0)

            # update gamma
            self.gamma[di, :] = np.sum(phi, 1) + self.alpha

            # ssword for updating bata
            for wi in xrange(len(doc)):
                word = doc[wi]
                self.ssword[:, word] += phi[:, wi]

            phi_norm = np.sum(phi, 1)
            phi_norm /= phi_norm.sum()

            self.EA[di, :] = phi_norm
            self.EAA += np.outer(phi_norm, phi_norm)

    def do_m_step(self):
        # update beta
        self.beta = self.ssword / np.sum(self.ssword, 1)[:, np.newaxis]
        self.ssword = np.zeros(self.beta.shape) + self.dir_prior

        self.eta = solve(self.EAA, np.dot(self.EA.T, self.responses))

        # compute mean absolute error
        mae = np.abs(np.dot(self.EA, self.eta) - self.responses).sum()
        return mae

    def heldoutEstep(self, max_iter, heldout):
        gamma = np.random.gamma(100., 1. / 100, (len(heldout), self.n_topic))

        for iter in xrange(max_iter):
            for di in xrange(len(heldout)):
                doc = heldout[di]
                doc_len = len(doc)

                phi = np.zeros([self.n_topic, doc_len])

                e_log_theta = psi(self.gamma[di, :]) - psi(np.sum(self.gamma[di, :]))
                phi = self.beta[:, doc] * np.exp(e_log_theta)[:, np.newaxis]
                phi /= np.sum(phi, 0)

                # update gamma
                gamma[di, :] = np.sum(phi, 1) + self.alpha

        return gamma
