#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Module for deep learning via the skip-gram model from https://code.google.com/p/word2vec/ .

"""

import logging
import sys
import os
import heapq
import time

from numpy import zeros_like, empty, exp, dot, outer, random, dtype,\
    float32 as REAL, seterr, array, uint8, vstack, argsort, fromstring

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc

logger = logging.getLogger(__name__)

MIN_ALPHA = 0.0001  # don't allow learning rate to drop below this threshold


def dumb_preprocess(text):
    """
    Trivial preprocessor: 1 line = 1 sentence; words = whitespace delimited tokens.

    This corresponds to preprocessing done in the original word2vec implementation.

    """
    for sentence in text.split('\n'):
        yield sentence.split()


def text2sentences(text):
    """
    Iterate over individual sentences in `text`, yielding each as a list of utf8-encoded tokens.

    Sentence splitting = NLTK algo; word splitting = maximal contiguous alphabetic
    sequences, between 2 and 15 characters long, lowercased.

    """
    from nltk.tokenize import sent_tokenize
    if not isinstance(text, unicode):
        text = unicode(text, 'utf8', 'strict')
    for sentence in sent_tokenize(text):
        yield [word.encode('utf8') for word in utils.tokenize(sentence, lower=True) if 2 <= len(word) <= 15]


class Vocab(object):
    """A single vocabulary item, used in constructing binary trees (incl. both word leaves and inner nodes)."""
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "<" + ', '.join(vals) + ">"


class Word2Vec(utils.SaveLoad):
    """
    Model for training, using, storing and loading neural networks described in https://code.google.com/p/word2vec/

    The model can be stored/loaded via its `save()` and `load()` methods, or stored in a format
    compatible with the original word2vec implementation via `save_word2vec_format()`.

    """
    def __init__(self, texts=None, layer1_size=100, alpha=0.025, window=5, preprocess=text2sentences, min_count=5, seed=None):
        """
        Initialize a model with the document sequence `texts`. Each document is
        a string that will be used for training.

        If you don't supply `texts`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.

        """
        self.vocab = {}  # mapping from a word (string) to a Vocab object
        self.index2word = []  # map from a word's matrix index (int) to word (string)
        self.layer1_size = int(layer1_size)
        self.alpha = float(alpha)
        self.window = int(window)
        self.preprocess = preprocess
        self.seed = seed
        self.min_count = min_count
        if texts is not None:
            self.build_vocab(texts)
            self.train_model(texts)


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
            heapq.heappush(heap, Vocab(count=min1.count + min2.count, index=i + len(self.vocab), left=min1, right=min2))

        # recurse over the tree, assigning a binary code to each vocabulary word
        if heap:
            max_depth, stack = 0, [(heap[0], [], [])]
            while stack:
                node, codes, points = stack.pop()
                if node.index < len(self.vocab):
                    # leaf node => store its path from the root
                    node.code, node.point = codes, points
                    max_depth = max(len(codes), max_depth)
                else:
                    # inner node => continue recursion
                    points = array(list(points) + [node.index - len(self.vocab)], dtype=int)
                    stack.append((node.left, array(list(codes) + [0], dtype=uint8), points))
                    stack.append((node.right, array(list(codes) + [1], dtype=uint8), points))

            logger.info("built huffman tree with maximum node depth %i" % max_depth)


    def build_vocab(self, texts):
        """
        Build vocabulary from a sequence of documents (strings). Each document will
        be split into sentences (and words) using the `preprocess` method supplied
        in the constructor.

        """
        logger.info("collecting all words and their counts")
        text_no, vocab = -1, {}
        total_words = lambda: sum(v.count for v in vocab.itervalues())
        for text_no, text in enumerate(texts):
            if text_no % 10000 == 0:
                logger.info("PROGRESS: at document #%i, processed %i words and %i word types" %
                    (text_no, total_words(), len(vocab)))
            for sentence in self.preprocess(text):
                for word in sentence:
                    v = vocab.setdefault(word, Vocab())
                    v.count += 1
        logger.info("collected %i word types from a corpus of %i words" %
            (len(vocab), total_words()))

        # assign a unique index to each word
        self.vocab, self.index2word = {}, []
        for word, v in vocab.iteritems():
            if v.count >= self.min_count:
                v.index = len(self.vocab)
                self.index2word.append(word)
                self.vocab[word] = v
        logger.info("total %i word types after removing those with count<%s" % (len(self.vocab), self.min_count))

        # add info about each word's Huffman encoding
        self.create_binary_tree()


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
                # work on the entire tree at once, to push as much work into numpy's C routines as possible (performance)
                l2a = self.syn1[word.point]  # 2d matrix, codelen x layer1_size
                fa = 1.0 / (1.0 + exp(-dot(l1, l2a.T)))  #  propagate hidden => output
                ga = (1 - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
                self.syn1[word.point] += outer(ga, l1)  # propagate hidden -> output

                # TODO add negative sampling?

                l1 += dot(ga, l2a)  # propagate input -> hidden


    def train_model(self, texts, reset=True):
        logger.info("training model with %i words and %i features" % (len(self.vocab), self.layer1_size))
        random.seed(self.seed)
        if reset:
            # reset all projection weights
            self.syn0 = ((random.rand(len(self.vocab), self.layer1_size) - 0.5) / self.layer1_size).astype(dtype=REAL)
            self.syn1 = zeros_like(self.syn0)

        # iterate over documents, training the model one sentence at a time
        total_words = sum(v.count for v in self.vocab.itervalues())
        alpha = self.alpha
        word_count, sentences, start = 0, 0, time.clock()
        for text_no, text in enumerate(texts):
            for sentence in self.preprocess(text):
                if sentences % 100 == 0:
                    # decrease learning rate as the training progresses
                    alpha = max(MIN_ALPHA, self.alpha * (1 - 1.0 * word_count / total_words))

                    # print progress and training stats
                    elapsed = time.clock() - start
                    logger.info("PROGRESS: at document #%i, sentence #%i, %.2f%% words, alpha %f, %.0f words per second" %
                        (text_no, sentences, 100.0 * word_count / total_words, alpha, word_count / elapsed if elapsed else 0.0))
                words = [self.vocab.get(word, None) for word in sentence]
                self.train_sentence(words, alpha=alpha)
                word_count += len(filter(None, words))  # don't consider OOV words for the statistics
                sentences += 1
        logger.info("training took %.1fs" % (time.clock() - start))
        self.init_sims()


    def save_word2vec_format(self, fname, binary=False):
        """
        Store the input-hidden weight matrix in the same format used by the original
        word2vec-tool, for compatibility.

        """
        logger.info("storing %sx%s projection weights into %s" % (len(self.vocab), self.layer1_size, fname))
        assert (len(self.vocab), self.layer1_size) == self.syn0.shape
        with open(fname, 'wb') as fout:
            fout.write("%s %s\n" % self.syn0.shape)
            # store in sorted order: most frequent words at the top
            for word, vocab in sorted(self.vocab.iteritems(), key=lambda item: -item[1].count):
                row = self.syn0[vocab.index]
                if binary:
                    fout.write("%s %s\n" % (word, row.tostring()))
                else:
                    fout.write("%s %s\n" % (word, ' '.join("%f" % val for val in row)))


    @classmethod
    def load_word2vec_format(cls, fname, binary=False):
        """
        Load the input-hidden weight matrix from the original word2vec-tool format.

        Note that the information loaded is incomplete (the binary tree is missing),
        so while you can query for word similarity etc., you cannot continue training
        with a model loaded this way.

        """
        logger.info("loading projection weights from %s" % (fname))
        with open(fname) as fin:
            header = fin.readline()
            vocab_size, layer1_size = map(int, header.split())  # throws for invalid file format
            result = Word2Vec(layer1_size=layer1_size)
            result.syn0 = empty((vocab_size, layer1_size), dtype=REAL)
            if binary:
                binary_len = dtype(REAL).itemsize * layer1_size
                for line_no in xrange(vocab_size):
                    # mixed text and binary: read text first, then binary
                    word = []
                    while True:
                        ch = fin.read(1)
                        if ch == ' ':
                            word = ''.join(word)
                            break
                        word.append(ch)
                    result.vocab[word] = Vocab(index=line_no)
                    result.index2word.append(word)
                    result.syn0[line_no] = fromstring(fin.read(binary_len), dtype=REAL)
                    fin.read(1)  # newline
            else:
                for line_no, line in enumerate(fin):
                    parts = line.split()
                    assert len(parts) == layer1_size + 1
                    word, weights = parts[0], map(REAL, parts[1:])
                    result.vocab[word] = Vocab(index=line_no)
                    result.index2word.append(word)
                    result.syn0[line_no] = weights
        logger.info("loaded %s matrix from %s" % (result.syn0.shape, fname))
        result.init_sims()
        return result


    def most_similar(self, positive=[], negative=[], topn=10):
        """
        Find the top-N most similar words. Positive words contribute positively towards the
        similarity, negative words negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given words, and corresponds to the `word-analogy`
        script in the original word2vec implementation.

        Example:
        >>> trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
        [('queen', 0.50882536), ...]

        """
        self.init_sims()

        # add weights for each word, if not already present
        positive = [(word, 1.0) if isinstance(word, basestring) else word for word in positive]
        negative = [(word, -1.0) if isinstance(word, basestring) else word for word in negative]
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if word in self.vocab:
                mean.append(weight * matutils.unitvec(self.syn0[self.vocab[word].index]))
                all_words.add(self.vocab[word].index)
            else:
                logger.warning("word '%s' not in vocabulary; ignoring it" % word)
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)
        dists = dot(self.syn0norm, mean)
        best = argsort(dists)[::-1][:topn + len(all_words)]
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], dists[sim]) for sim in best if sim not in all_words]
        return result[:topn]


    def init_sims(self):
        if getattr(self, 'syn0norm', None) is None:
            logger.info("precomputing L2-norms of word weight vectors")
            self.syn0norm = vstack(matutils.unitvec(vec) for vec in self.syn0).astype(REAL)



if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s" % " ".join(sys.argv))

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    infile = sys.argv[1]

    seterr(all='raise')  # don't ignore numpy errors

    w = Word2Vec(Texts(infile), layer1_size=20, preprocess=dumb_preprocess, seed=1, min_count=0)
    w.save(infile + '.model')
    w.save_word2vec_format(infile + '.model.bin', binary=True)
    w.save_word2vec_format(infile + '.model.txt', binary=False)

    logging.info("finished running %s" % program)
