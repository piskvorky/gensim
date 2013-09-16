#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Module for deep learning.

TODO pick word that doesn't belong: https://github.com/dhammack/Word2VecExample/blob/master/main.py

"""


import logging
import sys
import os
import heapq
import time

from numpy import zeros, zeros_like, exp, dot, outer, random, float32 as REAL, seterr, array, uint8

# from nltk.tokenize import sent_tokenize
from gensim import utils

logger = logging.getLogger(__name__)

SENTENCE_MARK = '</s>'  # use the same sentence delimiter as word2vec


try:
    _ = profile
except:
    # not running under line_profiler => ignore @profile decorators
    profile = lambda x: x


def text2sentences(text):
    """
    Iterate over individual sentences in `text`, yielding each as a list of utf8-encoded tokens.

    """
    if not isinstance(text, unicode):
        text = unicode(text, 'utf8', 'strict')
    for sentence in sent_tokenize(text):
        yield [word.encode('utf8') for word in utils.tokenize(sentence, lower=True) if 2 <= len(word) <= 15] + [SENTENCE_MARK]


def dumb_preprocess(text):
    """Trivial preprocessor: entire text = 1 sentence, split words on whitespace."""
    return [text.split() + [SENTENCE_MARK]]



class Vocab(object):
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        if hasattr(self, 'seq'):
            return "seq%s<count:%i, left:%r, right:%r>" % (self.seq, self.count, self.left, self.right)
        else:
            return "%s<count:%i, index:%s, code:%s, point:%s>" % (getattr(self, 'debug', '???'), self.count, getattr(self, 'index', '--'), getattr(self, 'code', '--'), getattr(self, 'point', '--'))



class Word2Vec(object):
    def __init__(self, texts=None, layer1_size=100, alpha=0.025, window=5, preprocess=text2sentences, min_count=5, seed=None):
        """
        Initialize a model with the document sequence `texts`. Each document is
        a string that will be used for training.

        If you don't supply `texts`, the model is left uninitialized -- use if
        you want to initialize it in some other way.

        """
        self.vocab = {}  # mapping from a word (string) to a Vocab object
        self.layer1_size = int(layer1_size)
        self.alpha = float(alpha)
        self.window = int(window)
        self.preprocess = preprocess
        self.seed = seed
        self.min_count = min_count
        if texts is not None:
            self.build_vocab(texts)
            self.train_model(texts)


    def build_vocab(self, texts):
        """
        Build vocabulary from a sequence of documents (strings).

        """
        logger.info("collecting all words and their counts")
        text_no, vocab = -1, {}
        total_sentences = lambda: vocab.get(SENTENCE_MARK, Vocab()).count
        total_words = lambda: sum(v.count for v in vocab.itervalues())
        for text_no, text in enumerate(texts):
            if text_no % 10000 == 0:
                logger.info("PROGRESS: at document #%i, total %i sentences, %i words and %i word types" %
                    (text_no, total_sentences(), total_words(), len(vocab)))
            for sentence in self.preprocess(text):
                for word in sentence:
                    v = vocab.setdefault(word, Vocab(index=len(vocab)))  # FIXME remove debug to save memory
                    v.count += 1
        logger.info("collected %i word types from a corpus of %i words and %i sentences" %
            (len(vocab), total_words(), total_sentences()))
        self.vocab = {}
        for word, v in vocab.iteritems():
            if v.count >= self.min_count:
                v.index = len(self.vocab)
                self.vocab[word] = v
        logger.info("total %i word types after removing those with count<%s" % (len(self.vocab), self.min_count))

        # add info about each word's Huffman encoding
        self.create_binary_tree()


    def create_binary_tree(self):
        """
        Create a binary Huffman tree using stored vocabulary word counts. Frequent words
        will have shorter binary codes.

        This is equivalent to the original word2vec code except the special "</s>"
        word is not treated specially here -- it's sorted and processed like any other word.

        """
        logger.info("constructing a huffman tree from %i words" % len(self.vocab))

        # build the huffman tree
        heap = self.vocab.values()
        heapq.heapify(heap)
        for i in xrange(len(self.vocab) - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, Vocab(count=min1.count + min2.count, seq=i, left=min1, right=min2))

        # recurse over the tree, assigning a binary code to each vocabulary word
        max_depth, stack = 0, [(heap[0], [], [])]
        while stack:
            node, codes, points = stack.pop()
            if not hasattr(node, 'seq'):
                # leaf node => store its path from the root
                node.code = array(codes, dtype=uint8)
                node.point = array(points, dtype=int)
                # print node
                max_depth = max(len(codes), max_depth)
            else:
                # inner node => continue recursion
                stack.append((node.left, codes + [0], points + [node.seq]))
                stack.append((node.right, codes + [1], points + [node.seq]))

        logger.info("built huffman tree with maximum node depth %i" % max_depth)


    @profile
    def train_sentence(self, words, alpha):
        """
        Update skip-gram hierarchical softmax model by training on a single sentence,
        where `sentence` is a list of Vocab objects (or None, where the corresponding
        word is not in the vocabulary).

        """
        for pos, word in enumerate(words):
            if word is None:
                continue  # OOV word in the input sentence => skip
            reduced_window = random.randint(self.window)  # `b` in the original word2vec code

            # now go over all words from the (reduced) window, predicting each one in turn
            start = max(0, pos - self.window + reduced_window)
            for pos2, word2 in enumerate(words[start : pos + self.window + 1 - reduced_window], start):
                if pos2 == pos or word2 is None:
                    # don't train on OOV words and on the `word` itself
                    continue
                l1 = self.syn0[word2.index]
                # avoid python loops and push as much work into numpy's C routines as possible
                l2a = self.syn1[word.point]  # 2d matrix of shape (codelen x size)
                fa = 1.0 / (1.0 + exp(-dot(l1, l2a.T)))  # vector of size codelen
                ga = (1 - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
                self.syn1[word.point] += outer(ga, l1)  # update hidden -> output weights

                # TODO negative sampling here

                l1 += dot(ga, l2a)  # update input -> hidden weights


    def train_model(self, texts, reset=True):
        logger.info("training model with %i words and %i features" % (len(self.vocab), self.layer1_size))
        random.seed(self.seed)
        if reset:
            # reset all projection weights
            self.syn0 = ((random.rand(len(self.vocab), self.layer1_size) - 0.5) / self.layer1_size).astype(dtype=REAL)
            self.syn1 = zeros_like(self.syn0)

        # iterate over documents, training the model one sentence at a time
        total_sentences = self.vocab.get(SENTENCE_MARK, Vocab()).count
        alpha = self.alpha
        word_count, sentences, start = 0, 0, time.clock()
        for text_no, text in enumerate(texts):
            for sentence in self.preprocess(text):
                if sentences % 100 == 0:
                    # decrease learning rate as the training progresses
                    alpha = max(0.0001, self.alpha * (1 - 1.0 * sentences / total_sentences))

                    # print progress and training stats
                    elapsed = time.clock() - start
                    logger.info("PROGRESS: at document #%i, sentence #%i/%i=%.2f%%, alpha %f, %.1f words per second" %
                        (text_no, sentences, total_sentences, 100.0 * sentences / total_sentences, alpha, word_count / elapsed if elapsed else 0.0))
                words = [self.vocab.get(word, None) for word in sentence[:-1]]
                self.train_sentence(words, alpha=alpha)
                word_count += len(filter(None, words))  # don't consider OOV words for the statistics
                sentences += 1
        logger.info("training took %.1fs" % (time.clock() - start))


class Texts(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in open(self.fname):
            yield line.strip()

    def save_as_word2vec(self):
        with open(self.fname + '.cleaned', 'wb') as fout:
            for text in self:
                for sentence in text2sentences(text):
                    fout.write(' '.join(sentence[:-1]) + '\n')


TEXTS = Texts('/Users/kofola/workspace/word2vec/cleaned')



if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s" % " ".join(sys.argv))

    seterr(all='raise')

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 1:
        print globals()['__doc__'] % locals()
        sys.exit(1)

    w = Word2Vec(TEXTS, layer1_size=20, preprocess=dumb_preprocess, seed=1)

    logging.info("finished running %s" % program)
