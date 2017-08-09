#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automatically detect common phrases (multiword expressions) from a stream of sentences.

The phrases are collocations (frequently co-occurring tokens). See [1]_ for the
exact formula.

For example, if your input stream (=an iterable, with each value a list of token strings) looks like:

>>> print(list(sentence_stream))
[[u'the', u'mayor', u'of', u'new', u'york', u'was', u'there'],
 [u'machine', u'learning', u'can', u'be', u'useful', u'sometimes'],
 ...,
]

you'd train the detector with:

>>> phrases = Phrases(sentence_stream)

and then create a performant Phraser object to transform any sentence (list of token strings) using the standard gensim syntax:

>>> bigram = Phraser(phrases)
>>> sent = [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there']
>>> print(bigram[sent])
[u'the', u'mayor', u'of', u'new_york', u'was', u'there']

(note `new_york` became a single token). As usual, you can also transform an entire
sentence stream using:

>>> print(list(bigram[any_sentence_stream]))
[[u'the', u'mayor', u'of', u'new_york', u'was', u'there'],
 [u'machine_learning', u'can', u'be', u'useful', u'sometimes'],
 ...,
]

You can also continue updating the collocation counts with new sentences, by:

>>> bigram.add_vocab(new_sentence_stream)

These **phrase streams are meant to be used during text preprocessing, before
converting the resulting tokens into vectors using `Dictionary`**. See the
:mod:`gensim.models.word2vec` module for an example application of using phrase detection.

The detection can also be **run repeatedly**, to get phrases longer than
two tokens (e.g. `new_york_times`):

>>> trigram = Phrases(bigram[sentence_stream])
>>> sent = [u'the', u'new', u'york', u'times', u'is', u'a', u'newspaper']
>>> print(trigram[bigram[sent]])
[u'the', u'new_york_times', u'is', u'a', u'newspaper']

.. [1] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean.
       Distributed Representations of Words and Phrases and their Compositionality.
       In Proceedings of NIPS, 2013.

"""

import sys
import os
import logging
import warnings
from collections import defaultdict
import itertools as it
from functools import partial
from math import log

from six import iteritems, string_types, next

from gensim import utils, interfaces

logger = logging.getLogger(__name__)


def _is_single(obj):
    """
    Check whether `obj` is a single document or an entire corpus.
    Returns (is_single, new) 2-tuple, where `new` yields the same
    sequence as `obj`.

    `obj` is a single document if it is an iterable of strings.  It
    is a corpus if it is an iterable of documents.
    """
    obj_iter = iter(obj)
    try:
        peek = next(obj_iter)
        obj_iter = it.chain([peek], obj_iter)
    except StopIteration:
        # An empty object is a single document
        return True, obj
    if isinstance(peek, string_types):
        # It's a document, return the iterator
        return True, obj_iter
    else:
        # If the first item isn't a string, assume obj is a corpus
        return False, obj_iter


class Phrases(interfaces.TransformationABC):
    """
    Detect phrases, based on collected collocation counts. Adjacent words that appear
    together more frequently than expected are joined together with the `_` character.

    It can be used to generate phrases on the fly, using the `phrases[sentence]`
    and `phrases[corpus]` syntax.

    """
    def __init__(self, sentences=None, min_count=5, threshold=10.0,
                 max_vocab_size=40000000, delimiter=b'_', progress_per=10000,
                 scoring='default'):
        """
        Initialize the model from an iterable of `sentences`. Each sentence must be
        a list of words (unicode strings) that will be used for training.

        The `sentences` iterable can be simply a list, but for larger corpora,
        consider a generator that streams the sentences directly from disk/network,
        without storing everything in RAM. See :class:`BrownCorpus`,
        :class:`Text8Corpus` or :class:`LineSentence` in the :mod:`gensim.models.word2vec`
        module for such examples.

        `min_count` ignore all words and bigrams with total collected count lower
        than this.

        `threshold` represents a score threshold for forming the phrases (higher means
        fewer phrases). A phrase of words `a` followed by `b` is accepted if the score of the
        phrase is greater than threshold. see the `scoring' setting

        `max_vocab_size` is the maximum size of the vocabulary. Used to control
        pruning of less common words, to keep memory under control. The default
        of 40M needs about 3.6GB of RAM; increase/decrease `max_vocab_size` depending
        on how much available memory you have.

        `delimiter` is the glue character used to join collocation tokens, and
        should be a byte string (e.g. b'_').

        `scoring` specifies how potential phrases are scored for comparison to the `threshold`
        setting. two settings are available:
        'default': from "Efficient Estimaton of Word Representations in Vector Space" by
            Mikolov, et. al.:
            (count(worda followed by wordb) - min_count) * N /
            (count(worda) * count(wordb)) > threshold`, where `N` is the total vocabulary size.
        'npmi': normalized pointwise mutual information, from "Normalized (Pointwise) Mutual
            Information in Colocation Extraction" by Gerlof Bouma:
            ln(prop(worda followed by wordb) / (prop(worda)*prop(wordb))) /
            - ln(prop(worda followed by wordb)
            where prop(n) is the count of n / the count of everything in the entire corpus
        'npmi' is more robust when dealing with common words that form part of common bigrams, and
            ranges from -1 to 1, but is slower to calculate than the default

        """
        if min_count <= 0:
            raise ValueError("min_count should be at least 1")

        if threshold <= 0 and scoring == 'default':
            raise ValueError("threshold should be positive for default scoring")
        if scoring == 'npmi' and (threshold < -1 or threshold > 1):
            raise ValueError("threshold should be between -1 and 1 for npmi scoring")

        if not (scoring == 'default' or scoring == 'npmi'):
            raise ValueError('unknown scoring function "' + scoring + '" specified')

        self.min_count = min_count
        self.threshold = threshold
        self.max_vocab_size = max_vocab_size
        self.vocab = defaultdict(int)  # mapping between utf8 token => its count
        self.min_reduce = 1  # ignore any tokens with count smaller than this
        self.delimiter = delimiter
        self.progress_per = progress_per
        self.scoring = scoring
        self.corpus_word_count = 0

        if sentences is not None:
            self.add_vocab(sentences)

    def __str__(self):
        """Get short string representation of this phrase detector."""
        return "%s<%i vocab, min_count=%s, threshold=%s, max_vocab_size=%s>" % (
            self.__class__.__name__, len(self.vocab), self.min_count,
            self.threshold, self.max_vocab_size)

    @staticmethod
    def learn_vocab(sentences, max_vocab_size, delimiter=b'_', progress_per=10000):
        """Collect unigram/bigram counts from the `sentences` iterable."""
        sentence_no = -1
        total_words = 0
        logger.info("collecting all words and their counts")
        vocab = defaultdict(int)
        min_reduce = 1
        for sentence_no, sentence in enumerate(sentences):
            if sentence_no % progress_per == 0:
                logger.info("PROGRESS: at sentence #%i, processed %i words and %i word types" %
                            (sentence_no, total_words, len(vocab)))
            sentence = [utils.any2utf8(w) for w in sentence]
            for bigram in zip(sentence, sentence[1:]):
                vocab[bigram[0]] += 1
                vocab[delimiter.join(bigram)] += 1
                total_words += 1

            if sentence:  # add last word skipped by previous loop
                word = sentence[-1]
                vocab[word] += 1
                total_words += 1

            if len(vocab) > max_vocab_size:
                utils.prune_vocab(vocab, min_reduce)
                min_reduce += 1

        logger.info("collected %i word types from a corpus of %i words (unigram + bigrams) and %i sentences" %
                    (len(vocab), total_words, sentence_no + 1))
        return min_reduce, vocab, total_words

    def add_vocab(self, sentences):
        """
        Merge the collected counts `vocab` into this phrase detector.

        """
        # uses a separate vocab to collect the token counts from `sentences`.
        # this consumes more RAM than merging new sentences into `self.vocab`
        # directly, but gives the new sentences a fighting chance to collect
        # sufficient counts, before being pruned out by the (large) accummulated
        # counts collected in previous learn_vocab runs.
        min_reduce, vocab, total_words = \
        self.learn_vocab(sentences, self.max_vocab_size, self.delimiter, self.progress_per)

        self.corpus_word_count += total_words
        if len(self.vocab) > 0:
            logger.info("merging %i counts into %s", len(vocab), self)
            self.min_reduce = max(self.min_reduce, min_reduce)
            for word, count in iteritems(vocab):
                self.vocab[word] += count
            if len(self.vocab) > self.max_vocab_size:
                utils.prune_vocab(self.vocab, self.min_reduce)
                self.min_reduce += 1
            logger.info("merged %s", self)
        else:
            # in common case, avoid doubling gigantic dict
            logger.info("using %i counts as vocab in %s", len(vocab), self)
            self.vocab = vocab

    def export_phrases(self, sentences, out_delimiter=b' ', as_tuples=False):
        """
        Generate an iterator that contains all phrases in given 'sentences'

        Example::

          >>> sentences = Text8Corpus(path_to_corpus)
          >>> bigram = Phrases(sentences, min_count=5, threshold=100)
          >>> for phrase, score in bigram.export_phrases(sentences):
          ...     print(u'{0}\t{1}'.format(phrase, score))

            then you can debug the threshold with generated tsv
        """

        vocab = self.vocab
        threshold = self.threshold
        delimiter = self.delimiter  # delimiter used for lookup
        min_count = self.min_count
        scoring = self.scoring
        corpus_word_count = self.corpus_word_count

        if scoring == 'default':
            scoring_function = \
            partial(self.original_scorer, len_vocab=float(len(vocab)), min_count=float(min_count))
        elif scoring == 'npmi':
            scoring_function = \
            partial(self.npmi_scorer, corpus_word_count=corpus_word_count)
        # no else here to catch unknown scoring function, check is done in Phrases.__init__

        for sentence in sentences:
            s = [utils.any2utf8(w) for w in sentence]
            last_bigram = False

            for word_a, word_b in zip(s, s[1:]):
                # last bigram check was moved here to save a few CPU cycles
                if word_a in vocab and word_b in vocab and not last_bigram:
                    bigram_word = delimiter.join((word_a, word_b))
                    if bigram_word in vocab:
                        count_a = float(vocab[word_a])
                        count_b = float(vocab[word_b])
                        count_ab = float(vocab[bigram_word])
                        score = scoring_function(count_a, count_b, count_ab)
                        # logger.debug("score for %s: (pab=%s - min_count=%s) / pa=%s / pb=%s * vocab_size=%s = %s",
                        #     bigram_word, pab, self.min_count, pa, pb, len(self.vocab), score)
                        # added mincount check because if the scorer doesn't contain min_count
                        # it would not be enforced otherwise
                        if score > threshold and count_ab >= min_count:
                            if as_tuples:
                                yield ((word_a, word_b), score)
                            else:
                                yield (out_delimiter.join((word_a, word_b)), score)
                            last_bigram = True
                            continue
                last_bigram = False

    def __getitem__(self, sentence):
        """
        Convert the input tokens `sentence` (=list of unicode strings) into phrase
        tokens (=list of unicode strings, where detected phrases are joined by u'_').

        If `sentence` is an entire corpus (iterable of sentences rather than a single
        sentence), return an iterable that converts each of the corpus' sentences
        into phrases on the fly, one after another.

        Example::

          >>> sentences = Text8Corpus(path_to_corpus)
          >>> bigram = Phrases(sentences, min_count=5, threshold=100)
          >>> for sentence in phrases[sentences]:
          ...     print(u' '.join(s))
            he refuted nechaev other anarchists sometimes identified as pacifist anarchists advocated complete
            nonviolence leo_tolstoy

        """
        warnings.warn("For a faster implementation, use the gensim.models.phrases.Phraser class")

        is_single, sentence = _is_single(sentence)
        if not is_single:
            # if the input is an entire corpus (rather than a single sentence),
            # return an iterable stream.
            return self._apply(sentence)

        s, new_s = [utils.any2utf8(w) for w in sentence], []
        last_bigram = False
        vocab = self.vocab
        threshold = self.threshold
        delimiter = self.delimiter
        min_count = self.min_count
        for word_a, word_b in zip(s, s[1:]):
            if word_a in vocab and word_b in vocab:
                bigram_word = delimiter.join((word_a, word_b))
                if bigram_word in vocab and not last_bigram:
                    pa = float(vocab[word_a])
                    pb = float(vocab[word_b])
                    pab = float(vocab[bigram_word])
                    score = (pab - min_count) / pa / pb * len(vocab)
                    # logger.debug("score for %s: (pab=%s - min_count=%s) / pa=%s / pb=%s * vocab_size=%s = %s",
                    #     bigram_word, pab, self.min_count, pa, pb, len(self.vocab), score)
                    if score > threshold:
                        new_s.append(bigram_word)
                        last_bigram = True
                        continue

            if not last_bigram:
                new_s.append(word_a)
            last_bigram = False

        if s:  # add last word skipped by previous loop
            last_token = s[-1]
            if not last_bigram:
                new_s.append(last_token)

        return [utils.to_unicode(w) for w in new_s]

    # calculation of score based on original mikolov word2vec paper
    # len_vocab and min_count set so functools.partial works
    @staticmethod
    def original_scorer(worda_count, wordb_count, bigram_count, len_vocab=0.0, min_count=0.0):
        return (bigram_count - min_count) / worda_count / wordb_count * len_vocab

    # normalized PMI, requires corpus size
    @staticmethod
    def npmi_scorer(worda_count, wordb_count, bigram_count, corpus_word_count=0.0):
        pa = worda_count / corpus_word_count
        pb = wordb_count / corpus_word_count
        pab = bigram_count / corpus_word_count
        return log(pab / (pa * pb)) / -log(pab)


def pseudocorpus(source_vocab, sep):
    """Feeds source_vocab's compound keys back to it, to discover phrases"""
    for k in source_vocab:
        if sep not in k:
            continue
        unigrams = k.split(sep)
        for i in range(1, len(unigrams)):
            yield [sep.join(unigrams[:i]), sep.join(unigrams[i:])]


class Phraser(interfaces.TransformationABC):
    """
    Minimal state & functionality to apply results of a Phrases model to tokens.

    After the one-time initialization, a Phraser will be much smaller and
    somewhat faster than using the full Phrases model.

    Reflects the results of the source model's `min_count`, `threshold`, and
    `scoring` settings. (You can tamper with those & create a new Phraser to try
    other values.)

    """
    def __init__(self, phrases_model):
        self.threshold = phrases_model.threshold
        self.min_count = phrases_model.min_count
        self.delimiter = phrases_model.delimiter
        self.scoring = phrases_model.scoring
        self.phrasegrams = {}
        corpus = pseudocorpus(phrases_model.vocab, phrases_model.delimiter)
        logger.info('source_vocab length %i', len(phrases_model.vocab))
        count = 0
        for bigram, score in phrases_model.export_phrases(corpus, self.delimiter, as_tuples=True):
            if bigram in self.phrasegrams:
                logger.info('Phraser repeat %s', bigram)
            self.phrasegrams[bigram] = (phrases_model.vocab[self.delimiter.join(bigram)], score)
            count += 1
            if not count % 50000:
                logger.info('Phraser added %i phrasegrams', count)
        logger.info('Phraser built with %i %i phrasegrams', count, len(self.phrasegrams))

    def __getitem__(self, sentence):
        """
        Convert the input tokens `sentence` (=list of unicode strings) into phrase
        tokens (=list of unicode strings, where detected phrases are joined by u'_'
        (or other configured delimiter-character).

        If `sentence` is an entire corpus (iterable of sentences rather than a single
        sentence), return an iterable that converts each of the corpus' sentences
        into phrases on the fly, one after another.

        """
        is_single, sentence = _is_single(sentence)
        if not is_single:
            # if the input is an entire corpus (rather than a single sentence),
            # return an iterable stream.
            return self._apply(sentence)

        s, new_s = [utils.any2utf8(w) for w in sentence], []
        last_bigram = False
        phrasegrams = self.phrasegrams
        delimiter = self.delimiter
        for word_a, word_b in zip(s, s[1:]):
            bigram_tuple = (word_a, word_b)
            if phrasegrams.get(bigram_tuple, (-1, -1))[1] > self.threshold and not last_bigram:
                bigram_word = delimiter.join((word_a, word_b))
                new_s.append(bigram_word)
                last_bigram = True
                continue

            if not last_bigram:
                new_s.append(word_a)
            last_bigram = False

        if s:  # add last word skipped by previous loop
            last_token = s[-1]
            if not last_bigram:
                new_s.append(last_token)

        return [utils.to_unicode(w) for w in new_s]


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s" % " ".join(sys.argv))

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    infile = sys.argv[1]

    from gensim.models import Phrases  # for pickle
    from gensim.models.word2vec import Text8Corpus
    sentences = Text8Corpus(infile)

    # test_doc = LineSentence('test/test_data/testcorpus.txt')
    bigram = Phrases(sentences, min_count=5, threshold=100)
    for s in bigram[sentences]:
        print(utils.to_utf8(u' '.join(s)))
