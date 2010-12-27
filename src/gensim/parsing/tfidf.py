# -*- coding: utf-8 -*-

from math import log

import numpy as np
import scipy.sparse as sp

def tokenize(text):
    return text.split()

def tc(dataset, tokenizer=tokenize):
    """ term counts. this function creates the vocab (type to id dict)
    And also docs, this is the basic global freq dict.
    Note that this is not a bag of words (bow) format,
    here the bag contains the actual word, not the id.
    The ids are calculated a posteriori.

    input: a dataset (array of texts)

    If we wanted compatibility with gensim, this function
    could easyly output the bow..
    """
    vocab = {}
    docs = []

    for doc in dataset:
        d = {} # token => count

        for term in tokenizer(doc):
            vocab[term] = 1
            d[term] = d.get(term, 0) + 1

        docs.append(d)

    sorted_terms = sorted(vocab.keys())
    vocab = dict([(t, i) for i, t in enumerate(sorted_terms)])

    return docs, vocab



def bow_from_tc(term_counts, vocab):
    """
    returns: a bow (bag of words). A bow is list of a tuples (corpusid, localfq)
    """
    bow = []
    for word, count in term_counts.iteritems():
        bow.append(( vocab[word], count))
    return bow

def tf_from_tc(term_counts):
    """ Normalized local FQ (dividing by sum, not vector norms)
    """
    docs = []

    for doc in term_counts:
        d = {}
        length = sum(doc.values())
        for term, count in doc.items():
            d[term] = float(count) / length
        docs.append(d)

    return docs


def idc_from_tc(term_counts):
    """   contextual diversity (n of docs the term appears in, in the corpus
    """
    t = {}
    for doc in term_counts:
        for term in doc:
            t[term] = t.get(term, 0) + 1
    return t

def idf_from_tc(term_counts):
    """ inverse document frequency (related to contextual diversity)
    denominator of the tfidf formula
    """
    n_docs = len(term_counts)
    idf = {}
    idc = idc_from_tc(term_counts)
    for term in idc:
        idf[term] = log(n_docs*1.0/(idc[term]),10)# NOTE changed to log(x,10)
    return idf

def tf_mul_idf(tf, idf):
    docs = []

    for doc in tf:
        d = {}
        for term in doc:
            d[term] = doc[term] * idf[term]
        docs.append(d)

    return docs

def to_vector(idf_dict, vocab):
    ret = np.zeros(len(idf_dict))
    for term, idx in vocab.items():
        ret[idx] = idf_dict[term]
    return ret

def to_sparse_matrix(tfidf_dict, vocab):
    tfm = sp.lil_matrix((len(vocab), len(tfidf_dict)), dtype=np.double)

    for j, doc in enumerate(tfidf_dict):
        for term in doc:
            try:
                i = vocab[term]
                tfm[i,j] = doc[term]
            except KeyError:
                pass

    return tfm

def inverse_vocab(vocab):
    """
    Converts a vocab dictionary term => index to index => term
    """
    return dict((i,t) for t,i in vocab.items())

def vocab_array(vocab):
    """
    Converts vocab dictionary to vocab array
    """
    return np.char.array(sorted(vocab.keys(), 
                                lambda a,b: cmp(vocab[a],vocab[b])))

def vocab_dict(vocab):
    """
    Converts vocab array to vocab dictionary
    """
    return dict((term,i) for i,term in enumerate(vocab))

def replace_vocab(td, oldvocab, newvocab):
    """
    td: V x X term-document matrix
    oldvocab: dictionary
    newvocab: dictionary
    """
    newtd = np.zeros((len(newvocab),td.shape[1]))
    for term in newvocab:
        try:
            newtd[newvocab[term]] = td[oldvocab[term]]
        except KeyError:
            newtd[newvocab[term]] = 0
    return newtd

class tfidf(object):
    """
    dataset is an list of strings
    """
    def __init__(self, dataset, tokenizer=tokenize):
        self._dataset = dataset
        self._tokenizer = tokenizer

    def as_dict(self):
        term_counts, vocab = tc(self._dataset, self._tokenizer)
        tf = term_counts # NOTE before it was: tf_from_tc(term_counts) but
        # tf from tc normalizes, and this doesn't replicate miislita counts
        idf = idf_from_tc(term_counts)
        return tf_mul_idf(tf, idf), vocab

    def as_sparse_matrix(self):
        tfidf_dict, vocab = self.as_dict()
        return to_sparse_matrix(tfidf_dict, vocab), vocab

    def as_array(self):
        tfm, vocab = self.as_sparse_matrix()
        return tfm.toarray(), vocab

    def get_idf(self):
        term_counts, vocab = tc(self._dataset, self._tokenizer)
        idf = idf_from_tc(term_counts)
        return idf, vocab
