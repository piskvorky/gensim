#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import logging
from collections import defaultdict

logger = logging.getLogger("gensim.models.phrases")


class Phrases(object):
    """
    Class learns from sentences interables. It does so by collecting statistics
    on words and joining common  adjacent words with the '_' character.

    It can be used to generate phrases on the fly.

    """
    def __init__(self, sentences, min_count=5, threshold=100,
                 max_vocab_size=500000000):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is
		a list of words (unicode strings) that will be used for training.

        The `sentences` iterable can be simply a list, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.
        See :class:`BrownCorpus`, :class:`Text8Corpus` or :class:`LineSentence` in
        the :module:`word2vec` module for such examples.

        `min_count` ignore all words with total frequency lower than this. By default (`min_count`=5).

        `threshold` represents a threshold for forming the phrases (higher means less phrases). By default (`threshold`=100).
        `max_vocab_size`is the maximum size of the vocabulary. Used to control the pruning of less common words.
        By default 500 million words).

        """
        self.min_count = min_count
        self.threshold = threshold
        self.max_vocab_size = max_vocab_size
        self.vocab = Phrases._learn_vocab(sentences, max_vocab_size)
        self.train_words = len(self.vocab)

    @staticmethod
    def _learn_vocab(sentences, max_vocab_size):

        sentence_no, vocab = -1, {}
        total_words = 0
        logging.info("collecting all words and their counts")
        vocab = defaultdict(int)
        min_reduce = 1
        for sentence_no, sentence in enumerate(sentences):
            if sentence_no % 10000 == 0:
                logger.info("PROGRESS: at sentence #%i, processed %i words and %i word types" %
                            (sentence_no, total_words, len(vocab)))

            for bigram in zip(sentence, sentence[1:]):
                word = bigram[0]
                bigram_word = "%s_%s" % bigram
                total_words += 1
                vocab[word] += 1
                vocab[bigram_word] += 1

            if len(sentence) > 0:    # add last word skipped by previous loop
                word = sentence[-1]
                vocab[word] += 1

            if len(vocab) > max_vocab_size * 0.7:
                to_delete = []
                for w in vocab.iterkeys():
                    if vocab[w] <= min_reduce:
                        to_delete.append(w)
                for w in to_delete:
                    del vocab[w]
                min_reduce += 1

        logger.info("collected %i word types from a corpus of %i words (unigram + bigrams) and %i sentences" %
                    (len(vocab), total_words, sentence_no + 1))

        return vocab

    def __getitem__(self, sentences):
        """
        Return a iterable of the original sentences with common words / phrases
        joined by '_'
        Example::

          >>> sentences = Text8Corpus(path_to_corpus)
          >>> bigram = Phrases(sentences, min_count=5, threshold=100)
          >>> for sentence in phrases[sentences]:
          >>>>    print(' '.join(s))
            he refuted nechaev other anarchists sometimes identified as pacifist anarchists advocated complete
            nonviolence leo_tolstoy

        """
        for sentence_no, s in enumerate(sentences):
            if sentence_no % 10000 == 0:
                logger.info("PROGRESS: at sentence #%i" % (sentence_no))

            new_s = []
            last_bigram = False
            for bigram in zip(s, s[1:]):
                bigram_word = "%s_%s" % bigram

                if all(uni in self.vocab for uni in bigram) and \
                       bigram_word in self.vocab and not last_bigram:

                    pa = float(self.vocab[bigram[0]])
                    pb = float(self.vocab[bigram[1]])
                    pab = float(self.vocab[bigram_word])
                    score = (pab - self.min_count) / pa / pb * self.train_words
                    # logger.debug("score for %s: %.2f" % (bigram_word, score))

                    if score > self.threshold:
                        new_s.append(bigram_word)
                        last_bigram = True
                        continue

                if not last_bigram:
                    new_s.append(bigram[0])
                    last_bigram = False
                else:
                    last_bigram = False

            if len(s) > 0:
                w = s[-1]
                if w in self.vocab and not last_bigram:
                    new_s.append(w)
            yield new_s


if __name__ == '__main__':
    import sys
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, format="%(asctime)s\t%(levelname)-8s\t%(filename)s:%(lineno)-4d\t%(message)s")
    from gensim.models.word2vec import Text8Corpus

    sentences = Text8Corpus("/Users/miguel/Downloads/text8")

    # test_doc = LineSentence('test/test_data/testcorpus.txt')
    bigram = Phrases(sentences, min_count=5, threshold=100)
    for s in bigram[sentences]:
        print ' '.join(s)
