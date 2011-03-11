#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import with_statement

import sys
import sys, os, os.path
import cPickle, random,itertools

import logging
logger = logging.getLogger('cossim')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logger.setLevel(logging.DEBUG)

import numpy
import matplotlib.pyplot as plt
plt.ioff()

from gensim import similarities
from gensim import matutils



def rmse(a1, a2):
    assert a1.shape == a2.shape
    diff = a1 - a2
    return numpy.sqrt(1.0 * numpy.multiply(diff, diff).sum() / a1.size)


def cossim(model, corpus):
    a1 = numpy.asmatrix(numpy.zeros((len(corpus), len(corpus)), dtype = numpy.float32))
    logger.info("creating index")
    index = similarities.MatrixSimilarity(model[corpus], numBest = None, numFeatures = model.numTopics)
    logger.info("computing cossims")
    for row, sims in enumerate(index):
        a1[row] = sims
        if row % 1000 == 0:
            logger.debug("created cossim of %i/%i documents" % (row + 1, len(corpus)))
    return a1


def cossim2(model, corpus):
    u = model.projection.u
    s = model.projection.s
    p = numpy.diag(1.0 / numpy.diag(s)) * u.T
    logger.info("constructing vt")
    ak = numpy.asmatrix(numpy.column_stack(p * matutils.sparse2full(doc, model.numTerms).reshape(model.numTerms, 1) for doc in corpus))
#    logger.info("reconstructing rank-k ak")
#    ak = u * (s * vt)
    logger.info("normalizing ak for cossim")
    lens = numpy.sqrt(numpy.sum(numpy.multiply(ak, ak), axis = 0))
    ak = ak / lens
    logger.debug("first few lens: %s" % (lens[:10]))
    logger.info("computing cossims")
    result = ak.T * ak
    return result



def sdiff(s1, s2):
   return numpy.abs(s1-s2) / (numpy.abs(s2))

def diff(u1, s1, u2, s2, scale = False):
    udiff = 1.0 - numpy.abs(numpy.diag(u1.T * u2))
    if scale:
        udiff = (udiff * s2) / numpy.sum(s2) # weight errors by singular values from s2
    degs = numpy.arcsin(udiff) / numpy.pi * 180
#    print r"%.3f$^{\circ}$" % degs
    return sdiff(s1, s2), degs


def fig_sdiff(ns, cols, labels):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for n, col, label in zip(ns, cols, labels):
        ax1.plot(n, color=col, label = label)
    ax1.set_xlabel('singular values $i$')
    ax1.set_ylabel('relative error $r_i$')
    ax1.legend(loc=0)
    plt.ylim(ymin=-.01)
    return fig


def fig_udiff(ns, labels):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for n, label in zip(ns, labels):
        ax1.plot(n, label = label)
    ax1.set_xlabel('singular vectors $i$')
    ax1.set_ylabel('angular error $r_i$')
    ax1.legend(loc=0)
    return fig


def copys():
    """Cut out U from results, leave only S (to save bandwidth in scp)."""
    for fname in os.listdir(Result.froot):
        if fname.startswith('wiki_p'):
            result = Result('', '', fname)
            logging.info("copying %s" % fname)
            with open(fname + '.s', 'w') as fout:
                cPickle.dump([None, result.s], fout, protocol=-1)


def stos(s):
    """wall-clock times to (x hours, y minutes)"""
    h = int(s / 3600)
    m = s - h * 3600
    return h, int(m/60)



class Result(object):
    froot = '/Users/kofola/svn/gensim/trunk/src'

    def __init__(self, name, marker, fname):
        if not fname.endswith('.s'): # prepend "*" for local macbook results (not asteria)
            name = '* ' + name
        self.name = name
        self.fname = fname
        self.marker = marker
        if fname:
            self.s = self.getS()
        else:
            self.s = 14 + 2 * numpy.random.rand(400) # experiment not finished yet; plot some noise

    def getS(self, maxvals = 400):
        fin = os.path.join(Result.froot, self.fname)
        logging.info("loading run from %s" % fin)
        obj = cPickle.load(open(fin))
        try:
            s = obj.projection.s
        except:
            s = obj[1]
        return s[:maxvals]
#endclass Result

def plotS(results, truth=None, labelx='factor $i$', labely=None):
    if labely is None:
        if truth:
            labely='relative error $r_i$'
        else:
            labely='singular value $s_i$ (log scale)'

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(labelx)
    ax1.set_ylabel(labely)

    m = 100.0
    for pos, result in enumerate(results):
        if truth:
            s = abs(result.s-truth.s)/truth.s
        else:
            s = result.s
        m = min(m, min(s))
        ax1.plot(s, result.marker, label=result.name, markevery = 20)

    if not truth:
        ax1.semilogy(basey=3, subsy=[32, 64, 128, 256])
    ax1.legend(loc=0)
    plt.ylim(ymin=m-1.0)
    return fig



def exp1():
    """Oversampling experiment. Factors=400, chunks=20k.
    """
    results = [
        Result('P1, $l=0$ [10h36m]', '-kx', 'wiki_p1_f400_c20k_e0.pkl.s'),
        Result('P1, $l=200$ [21h17m]', '-k*', 'wiki_p1_f600_c20k_e0.pkl.s'),
        Result('P1, $l=400$ [32h40m]', '-k+', 'wiki_p1_f800_c20k_e0.pkl.s'),

        Result('P12, $l=0$ [6h30m] ', '-rx', 'wiki_p12_f400_c20k_e100_pi2.pkl.s'), # (100,2)
#        Result('P12, $l=0$ [6h23m] (200,1) ', '-ro', 'wiki_p12_f400_c20k_e200_pi1.pkl.s'),
#        Result('P12, $l=0$ [5h31m] (100,1) ', '-rs', 'wiki_p12_f400_c20k_e100_pi1.pkl.s'),
#        Result('P12, $l=200$ (100,1) [8h37m]', '-r*', 'wiki_p12_f600_c20k_e100_pi1.pkl.s'),
        Result('P12, $l=200$ [9h21m]', '-r*', 'wiki_p12_f600_c20k_e100_pi2.pkl.s'), # (100,2)
#        Result('P12, $l=400$ []', '-r+', ''), # pada s pameti

        Result('P2, $l=0$ [2h8m]', '-gx', 'wiki_p2_f400_c20k_e0_pi0.pkl.s'),
        Result('P2, $l=200$ [2h28m]', '-g*', 'wiki_p2_f400_c20k_e200_pi0.pkl.s'),
        Result('P2, $l=400$ [2h54m]', '-g+', 'wiki_p2_f400_c20k_e400_pi0.pkl.s'),

        Result('P2, $l=400$, $q=3$ [7h57m]', '-m+', 'wiki_p2_f400_c20k_e400_pi3.pkl.s'),

    ]
    fout = os.path.join(Result.froot, 'experiment1.eps')
    logging.info("saving figure with %i runs to %s" % (len(results), fout))
    plotS(results).savefig(fout)

    results = [
        Result('P2, $l=0$, $q=0$ [2h8m]', '-gx', 'wiki_p2_f400_c20k_e0_pi0.pkl.s'),
        Result('P2, $l=200$, $q=0$ [2h28m]', '-g*', 'wiki_p2_f400_c20k_e200_pi0.pkl.s'),
        Result('P2, $l=400$, $q=0$ [2h54m]', '-g+', 'wiki_p2_f400_c20k_e400_pi0.pkl.s'),

        Result('P2, $l=0$, $q=1$ [3h6m]', '-bx', 'wiki_p2_f400_c20k_e0_pi1.pkl.s'),
        Result('P2, $l=200$, $q=1$ [3h53m]', '-b*', 'wiki_p2_f400_c20k_e200_pi1.pkl.s'),
        Result('P2, $l=400$, $q=1$ [4h6m]', '-b+', 'wiki_p2_f400_c20k_e400_pi1.pkl.s'),

        Result('P2, $l=0$, $q=3$ [4h49m]', '-mx', 'wiki_p2_f400_c20k_e0_pi3.pkl.s'),
        Result('P2, $l=200$, $q=3$ [5h41m]', '-m*', 'wiki_p2_f400_c20k_e200_pi3.pkl.s'),
        Result('P2, $l=400$, $q=3$ [7h57m]', '-m+', 'wiki_p2_f400_c20k_e400_pi3.pkl.s'),
    ]
    fout = os.path.join(Result.froot, 'experiment1pi.eps')
    logging.info("saving figure with %i runs to %s" % (len(results), fout))
    plotS(results).savefig(fout)



def exp2():
    """Effects of chunk size on P1 and P12. Factors=400.
    """
    results = [
        Result('P1, chunks 10k [13h14m]', '-kx', 'wiki_p1_f400_c10k_e0.pkl.s'),
        Result('P1, chunks 20k [10h36m]', '-r*', 'wiki_p1_f400_c20k_e0.pkl.s'),
        Result('P1, chunks 40k [9h29m]', '-g+', 'wiki_p1_f400_c40k_e0.pkl.s'),

        Result('P12, chunks 10k [9h35m]', '-bx', 'wiki_p12_f400_c10k_e100_pi2.pkl.s'),
        Result('P12, chunks 20k [6h30m]', '-m*', 'wiki_p12_f400_c20k_e100_pi2.pkl.s'),
        Result('P12, chunks 40k [4h42m]', '-c+', 'wiki_p12_f400_c40k_e100_pi2.pkl.s'),

        Result('P2, $l=400$, $q=3$ [7h57m]', '-m+', 'wiki_p2_f400_c20k_e400_pi3.pkl.s'),
    ]

    fout = os.path.join(Result.froot, 'experiment2.eps')
    logging.info("saving figure with %i runs to %s" % (len(results), fout))
    plotS(results).savefig(fout)
#    plotS(results, truth=results[0]).savefig(os.path.join(Result.froot, 'experiment2r.eps'))


def exp3():
    """P1 input order experiment. k=400, chunks of !40k!, not 20k.
    """
    results = [
        Result('P1 [9h29m]', '-k|', 'wiki_p1_f400_c40k_e0.pkl.s'),
        Result('P1, shuffled1 [10h40m]', '-b*', 'wiki_p1_f400_c40k_e0_shuffled1.pkl.s'),
        Result('P1, shuffled2 [10h57m]', '-g+', 'wiki_p1_f400_c40k_e0_shuffled2.pkl.s'),
        Result('P1, shuffled3 [10h9m]', '-cx', 'wiki_p1_f400_c40k_e0_shuffled3.pkl.s'),

        Result('P2, $l=400$, $q=3$ [7h57m]', '-m+', 'wiki_p2_f400_c20k_e400_pi3.pkl.s'),
    ]

    fout = os.path.join(Result.froot, 'experiment3.eps')
    logging.info("saving figure with %i runs to %s" % (len(results), fout))
    plotS(results).savefig(fout)


def exp4():
    """Distributed LSI for P1, P12. k=400, c=20k.
    """
    results = [
        Result('P1, 1 node [10h36m]', '-kx', 'wiki_p1_f400_c20k_e0.pkl.s'),
        Result('P1, 2 nodes [6h0m]', '-k*', 'wiki_p1_w2_f400_c20k_e0.pkl.s'),
        Result('P1, 4 nodes [3h18m]', '-k+', 'wiki_p1_w4_f400_c20k_e0.pkl.s'),

        Result('P12, 1 node [5h30m]', '-gx', 'wiki_p12_f400_c20k_e100_pi2.pkl.s'),
        Result('P12, 2 nodes [2h53m]', '-g*', 'wiki_p12_w2_f400_c20k_e100_pi2.pkl.s'),
#        Result('P12, 4 nodes, $l=100$ [2h29m]', '-g+', 'wiki_p12_w4_f500_c20k_e100_pi2.pkl.s'),
        Result('P12, 3 nodes, $l=200$ [3h1m]', '-go', 'wiki_p12_w3_f600_c20k_e100_pi2.pkl.s'),
        Result('P12, 4 nodes, [1h41m]', '-g+', 'wiki_p12_w5_f400_c20k_e100_pi2.pkl.s'),
#        Result('P12, $l=100$, 4 nodes [2h12m]', '-ro', 'wiki_p12_w5_f500_c20k_e100_pi2.pkl.s'),

        Result('P2, $l=400$, $q=3$ [7h57m]', '-m+', 'wiki_p2_f400_c20k_e400_pi3.pkl.s'),
    ]

    fout = os.path.join(Result.froot, 'experiment4.eps')
    logging.info("saving figure with %i runs to %s" % (len(results), fout))
    plotS(results).savefig(fout)



def exp():
    """
    Generate all experiment graphs for the NIPS paper.
    """
    exp1()
    exp2()
    exp3()
    exp4()


def replotECIR(data, labels, markers=['-bx', '-g*', '-r+', '-co', '-ms', '-k^', '-yv', '-b>'],
               labelx='factor $i$', labely=None):
    if labely is None:
        labely='relative error $r_i$'

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(labelx)
    ax1.set_ylabel(labely)

    m = 100.0
    for s, label, marker in itertools.izip(data, labels, markers):
        m = min(m, min(s))
        ax1.plot(s, marker, label=label, markevery=10)

    ax1.legend(loc=0)
    plt.ylim(ymin=m-.001)
    return fig
# y4 = numpy.cumsum([float(part.split(',')[1]) for part in d4.split(' ')])


def piechart(fracs, labels, explode=None):
    f = plt.figure(figsize=(5,5))
    ax1 = f.add_subplot(111)
    ax1.pie(fracs, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, pctdistance=0.7)
    return f


def typeset(topics, topN=10):
    t2 = []
    for topicid in topics:
        topic = lda.expElogbeta[topicid]
        topic = topic / topic.sum()
        bestn = numpy.argsort(topic)[::-1][:topN]
        beststr = [lda.id2word[id].replace('$', '\\$') for id in bestn]
        t2.append(beststr)
    for i in xrange(topN):
        print ' & '.join('$'+bs[i]+'$' if '$' in bs[i] else bs[i] for bs in t2) + r' \\'

# ndoc = lambda doc: [(old2new[id], val) for id, val in doc if id in old2new]

def plotpie(docid):
    mix = lda[ndoc(mm[docid])]
    top = [(cat, frac) for cat, frac in mix if frac > 0.09]
    fracs, cats = [frac for _, frac in top], [cat for cat, _ in top]
    f = cc.piechart(fracs + [1.0-sum(fracs)], labels=["topic %i" % cat for cat in cats] + ['other'])
    return f
