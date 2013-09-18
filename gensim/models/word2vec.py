#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Module for deep learning via *hierarchical softmax skip-gram* from [1]_.
The algorithm is ported from the C package https://code.google.com/p/word2vec/ .

Initialize a model with e.g.::

>>> model = Word2Vec(sentences, size=100, window=5, min_count=5)

Store/load a model with::

>>> model.save(fname)
>>> model = Word2Vec.load(fname)

The model can also be instantiated from an existing, trained file on disk in word2vec format::

>>> model = Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)  # text format
>>> model = Word2Vec.load_word2vec_format('/tmp/vectors.bin', binary=True)  # binary format

.. [1] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.

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


class Vocab(object):
    """A single vocabulary item, used internally for constructing binary trees (incl. both word leaves and inner nodes)."""
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
    Class for training, using and evaluating neural networks described in https://code.google.com/p/word2vec/

    The model can be stored/loaded via its `save()` and `load()` methods, or stored in a format
    compatible with the original word2vec implementation via `save_word2vec_format()`.

    """
    def __init__(self, sentences=None, size=100, alpha=0.025, window=5, min_count=5, seed=1):
        """
        Initialize a model from `sentences`. Each sentence is a list of words
        (utf8 strings) that will be used for training.

        If you don't supply `sentences`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.

        `size` is the dimensionality of the feature vectors.
        `window` is the maximum distance between the current and predicted word within a sentence.
        `alpha` is the initial learning rate (will linearly drop to zero as training progresses).
        `seed` = for the random number generator.
        `min_count` = ignore all words with total frequency lower than this.

        """
        self.vocab = {}  # mapping from a word (string) to a Vocab object
        self.index2word = []  # map from a word's matrix index (int) to word (string)
        self.layer1_size = int(size)
        self.alpha = float(alpha)
        self.window = int(window)
        self.seed = seed
        self.min_count = min_count
        if sentences is not None:
            self.build_vocab(sentences)
            self.reset_weights()
            self.train_model(sentences)


    def create_binary_tree(self):
        """
        Create a binary Huffman tree using stored vocabulary word counts. Frequent words
        will have shorter binary codes. Called internally from `build_vocab()`.

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


    def build_vocab(self, sentences):
        """Build vocabulary from a sequence of sentences."""
        logger.info("collecting all words and their counts")
        sentence_no, vocab = -1, {}
        total_words = lambda: sum(v.count for v in vocab.itervalues())
        for sentence_no, sentence in enumerate(sentences):
            if sentence_no % 10000 == 0:
                logger.info("PROGRESS: at sentence #%i, processed %i words and %i word types" %
                    (sentence_no, total_words(), len(vocab)))
            for word in sentence:
                v = vocab.setdefault(word, Vocab())
                v.count += 1
        logger.info("collected %i word types from a corpus of %i words and %i sentences" %
            (len(vocab), total_words(), sentence_no + 1))

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
        word is not in the vocabulary). Called internally from `train_model())`.

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
                fa = 1.0 / (1.0 + exp(-dot(l1, l2a.T)))  #  propagate hidden -> output
                ga = (1 - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
                self.syn1[word.point] += outer(ga, l1)  # learn hidden -> output

                # TODO add negative sampling?

                l1 += dot(ga, l2a)  # learn input -> hidden


    def train_model(self, sentences, total_words=None):
        """
        Train the model on a sequence of sentences, updating its existing neural weights.
        Each sentence is a list of utf8 strings.

        """
        logger.info("training model with %i words and %i features" % (len(self.vocab), self.layer1_size))

        # iterate over documents, training the model one sentence at a time
        total_words = total_words or sum(v.count for v in self.vocab.itervalues())
        alpha = self.alpha
        word_count, sentence_no, start = 0, -1, time.clock()
        for sentence_no, sentence in enumerate(sentences):
            if sentence_no % 100 == 0:
                # decrease learning rate as the training progresses
                alpha = max(MIN_ALPHA, self.alpha * (1 - 1.0 * word_count / total_words))

                # print progress and training stats
                elapsed = time.clock() - start
                logger.info("PROGRESS: at sentence #%i, %.2f%% words, alpha %f, %.0f words per second" %
                    (sentence_no, 100.0 * word_count / total_words, alpha, word_count / elapsed if elapsed else 0.0))
            words = [self.vocab.get(word, None) for word in sentence]  # replace OOV words with None
            self.train_sentence(words, alpha=alpha)
            word_count += len(filter(None, words))  # don't consider OOV words for the statistics
        logger.info("training took %.1fs" % (time.clock() - start))


    def reset_weights(self):
        """Reset all projection weights, but keep the existing vocabulary."""
        random.seed(self.seed)
        self.syn0 = ((random.rand(len(self.vocab), self.layer1_size) - 0.5) / self.layer1_size).astype(dtype=REAL)
        self.syn1 = zeros_like(self.syn0)


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
            result = Word2Vec(size=layer1_size)
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
                    result.vocab[word] = Vocab(index=line_no, count=vocab_size - line_no)
                    result.index2word.append(word)
                    result.syn0[line_no] = fromstring(fin.read(binary_len), dtype=REAL)
                    fin.read(1)  # newline
            else:
                for line_no, line in enumerate(fin):
                    parts = line.split()
                    assert len(parts) == layer1_size + 1
                    word, weights = parts[0], map(REAL, parts[1:])
                    result.vocab[word] = Vocab(index=line_no, count=vocab_size - line_no)
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

        Example::

          >>> trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
          [('queen', 0.50882536), ...]

        """
        self.init_sims()

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [(word, 1.0) if isinstance(word, basestring) else word for word in positive]
        negative = [(word, -1.0) if isinstance(word, basestring) else word for word in negative]
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if word in self.vocab:
                mean.append(weight * matutils.unitvec(self.syn0[self.vocab[word].index]))
                all_words.add(self.vocab[word].index)
            else:
                logger.warning("word '%s' not in vocabulary; ignoring it" % word)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)
        dists = dot(self.syn0norm, mean)
        if not topn:
            return dists
        best = argsort(dists)[::-1][:topn + len(all_words)]
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], dists[sim]) for sim in best if sim not in all_words]
        return result[:topn]


    def init_sims(self):
        if getattr(self, 'syn0norm', None) is None:
            logger.info("precomputing L2-norms of word weight vectors")
            self.syn0norm = vstack(matutils.unitvec(vec) for vec in self.syn0).astype(REAL)


    def accuracy(self, questions, restrict_vocab=30000):
        """
        Compute accuracy of the model. `questions` is a filename where lines are
        4-tuples of words, split into sections by ": SECTION NAME" lines.
        See https://code.google.com/p/word2vec/source/browse/trunk/questions-words.txt for an example.

        The accuracy is reported (=printed to log and returned as list) for each
        section separately, plus there's one aggregate summary at the end.

        Use `restrict_vocab` to ignore all questions containing a word whose frequency
        is not in the top-N most frequent words (default top 30000).

        This method corresponds to the `compute-accuracy` script of the original C word2vec.

        """
        ok_vocab = dict(sorted(self.vocab.iteritems(), key=lambda item: -item[1].count)[:restrict_vocab])
        ok_index = set(v.index for v in ok_vocab.itervalues())

        def log_accuracy(section):
            correct, incorrect = section['correct'], section['incorrect']
            if correct + incorrect > 0:
                logger.info("%s: %.1f%% (%i/%i)" %
                    (section['section'], 100.0 * correct / (correct + incorrect),
                    correct, correct + incorrect))

        sections, section = [], None
        for line_no, line in enumerate(open(questions)):
            if line.startswith(': '):
                # a new section starts => store the old section
                if section:
                    sections.append(section)
                    log_accuracy(section)
                section = {'section': line.lstrip(': ').strip(), 'correct': 0, 'incorrect': 0}
            else:
                if not section:
                    raise ValueError("missing section header before line #%i in %s" % (line_no, questions))
                try:
                    a, b, c, expected = [word.lower() for word in line.split()]  # TODO assumes vocabulary uses lowercase, too...
                except:
                    logger.info("skipping invalid line #%i in %s" % (line_no, questions))
                if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                    logger.debug("skipping line #%i with OOV words: %s" % (line_no, line))
                    continue

                predicted, ignore = None, set(self.vocab[v].index for v in [a, b, c])
                # go over predicted words, starting from the most likely, but ignoring OOV words and input words
                for index in argsort(self.most_similar(positive=[b, c], negative=[a], topn=False))[::-1]:
                    if index in ok_index and index not in ignore:
                        predicted = self.index2word[index]
                        if predicted != expected:
                            logger.debug("%s: expected %s, predicted %s" % (line.strip(), expected, predicted))
                        break
                section['correct' if predicted == expected else 'incorrect'] += 1
        if section:
            # store the last section, too
            sections.append(section)
            log_accuracy(section)

        total = {'section': 'total', 'correct': sum(s['correct'] for s in sections), 'incorrect': sum(s['incorrect'] for s in sections)}
        log_accuracy(total)
        sections.append(total)
        return sections



class BrownCorpus(object):
    """Yield sentences from the Brown corpus (part of NLTK data)."""
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            for line in open(fname):
                # each file line is a single sentence in the Brown corpus
                # each token is WORD/POS_TAG
                token_tags = [t.split('/') for t in line.split() if len(t.split('/')) == 2]
                # ignore words with non-alphabetic tags like ",", "!" etc (punctuation, weird stuff)
                words = ["%s/%s" % (token.lower(), tag[:2]) for token, tag in token_tags if tag[:2].isalpha()]
                if not words:  # don't bother sending out empty sentences
                    continue
                yield words


class Text8Corpus(object):
    """Yield sentences from the "text8" corpus, unzipped from http://mattmahoney.net/dc/text8.zip ."""
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        # the entire corpus is one gigantic line -- there are no sentence marks at all
        # so just split the sequence of tokens arbitrarily: 1 sentence = 1000 tokens
        sentence, rest, max_sentence_length = [], '', 1000
        with open(self.fname) as fin:
            while True:
                text = rest + fin.read(8192)  # avoid loading the entire file (=1 line) into RAM
                if text == rest:  # EOF
                    sentence.extend(rest.split()) # return the last chunk of words, too (may be shorter/longer)
                    if sentence:
                        yield sentence
                    break
                last_token = text.rfind(' ')  # the last token may have been split in two... keep it for the next iteration
                words, rest = (text[:last_token].split(), text[last_token:].strip()) if last_token >= 0 else ([], text)
                sentence.extend(words)
                while len(sentence) >= max_sentence_length:
                    yield sentence[:max_sentence_length]
                    sentence = sentence[max_sentence_length:]



if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s" % " ".join(sys.argv))

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    infile, outfile = sys.argv[1:3]

    seterr(all='raise')  # don't ignore numpy errors

    w = Word2Vec(BrownCorpus(infile), size=20, min_count=5)
    w.save(outfile + '.model')
    w.save_word2vec_format(outfile + '.model.bin', binary=True)
    w.save_word2vec_format(outfile + '.model.txt', binary=False)

    logging.info("finished running %s" % program)
