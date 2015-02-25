#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Deep learning via the distributed memory and distributed bag of words models from
[1]_, using either hierarchical softmax or negative sampling [2]_ [3]_.

**Make sure you have a C compiler before installing gensim, to use optimized (compiled)
doc2vec training** (70x speedup [blog]_).

Initialize a model with e.g.::

>>> model = Doc2Vec(sentences, size=100, window=8, min_count=5, workers=4)

Persist a model to disk with::

>>> model.save(fname)
>>> model = Doc2Vec.load(fname)  # you can continue training with the loaded model!

The model can also be instantiated from an existing file on disk in the word2vec C format::

  >>> model = Doc2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format
  >>> model = Doc2Vec.load_word2vec_format('/tmp/vectors.bin', binary=True)  # C binary format

.. [1] Quoc Le and Tomas Mikolov. Distributed Representations of Sentences and Documents. http://arxiv.org/pdf/1405.4053v2.pdf
.. [2] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
.. [3] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality.
       In Proceedings of NIPS, 2013.
.. [blog] Optimizing word2vec in gensim, http://radimrehurek.com/2013/09/word2vec-in-python-part-two-optimizing/

"""

import logging
import os

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from numpy import zeros, random, sum as np_sum

logger = logging.getLogger(__name__)

from gensim import utils  # utility fnc for pickling, common scipy operations etc
from gensim.models.word2vec import Word2Vec, Vocab, train_cbow_pair, train_sg_pair

try:
    from gensim.models.doc2vec_inner import train_sentence_dbow, train_sentence_dm, FAST_VERSION
except:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1

    def train_sentence_dbow(model, sentence, lbls, alpha, work=None, train_words=True, train_lbls=True):
        """
        Update distributed bag of words model by training on a single sentence.

        The sentence is a list of Vocab objects (or None, where the corresponding
        word is not in the vocabulary. Called internally from `Doc2Vec.train()`.

        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from doc2vec_inner instead.

        """
        neg_labels = []
        if model.negative:
            # precompute negative labels
            neg_labels = zeros(model.negative + 1)
            neg_labels[0] = 1.0

        for label in lbls:
            if label is None:
                continue  # OOV word in the input sentence => skip
            for word in sentence:
                if word is None:
                    continue  # OOV word in the input sentence => skip
                train_sg_pair(model, word, label, alpha, neg_labels, train_words, train_lbls)

        return len([word for word in sentence if word is not None])

    def train_sentence_dm(model, sentence, lbls, alpha, work=None, neu1=None, train_words=True, train_lbls=True):
        """
        Update distributed memory model by training on a single sentence.

        The sentence is a list of Vocab objects (or None, where the corresponding
        word is not in the vocabulary. Called internally from `Doc2Vec.train()`.

        This is the non-optimized, Python version. If you have a C compiler, gensim
        will use the optimized version from doc2vec_inner instead.

        """
        lbl_indices = [lbl.index for lbl in lbls if lbl is not None]
        lbl_sum = np_sum(model.syn0[lbl_indices], axis=0)
        lbl_len = len(lbl_indices)
        neg_labels = []
        if model.negative:
            # precompute negative labels
            neg_labels = zeros(model.negative + 1)
            neg_labels[0] = 1.

        for pos, word in enumerate(sentence):
            if word is None:
                continue  # OOV word in the input sentence => skip
            reduced_window = random.randint(model.window)  # `b` in the original doc2vec code
            start = max(0, pos - model.window + reduced_window)
            window_pos = enumerate(sentence[start : pos + model.window + 1 - reduced_window], start)
            word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
            l1 = np_sum(model.syn0[word2_indices], axis=0) + lbl_sum  # 1 x layer1_size
            if word2_indices and model.cbow_mean:
                l1 /= (len(word2_indices) + lbl_len)
            neu1e = train_cbow_pair(model, word, word2_indices, l1, alpha, neg_labels, train_words, train_words)
            if train_lbls:
                model.syn0[lbl_indices] += neu1e

        return len([word for word in sentence if word is not None])


class LabeledSentence(object):
    """
    A single labeled sentence = text item.
    Replaces "sentence as a list of words" from Word2Vec.

    """
    def __init__(self, words, labels):
        """
        `words` is a list of tokens (unicode strings), `labels` a
        list of text labels associated with this text.

        """
        self.words = words
        self.labels = labels

    def __str__(self):
        return '%s(%s, %s)' % (self.__class__.__name__, self.words, self.labels)


class Doc2Vec(Word2Vec):
    """Class for training, using and evaluating neural networks described in http://arxiv.org/pdf/1405.4053v2.pdf"""
    def __init__(self, sentences=None, size=300, alpha=0.025, window=8, min_count=5,
                 sample=0, seed=1, workers=1, min_alpha=0.0001, dm=1, hs=1, negative=0,
                 dm_mean=0, train_words=True, train_lbls=True, **kwargs):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        LabeledSentence object that will be used for training.

        The `sentences` iterable can be simply a list of LabeledSentence elements, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.

        If you don't supply `sentences`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.

        `dm` defines the training algorithm. By default (`dm=1`), distributed memory is used.
        Otherwise, `dbow` is employed.

        `size` is the dimensionality of the feature vectors.

        `window` is the maximum distance between the current and predicted word within a sentence.

        `alpha` is the initial learning rate (will linearly drop to zero as training progresses).

        `seed` = for the random number generator.

        `min_count` = ignore all words with total frequency lower than this.

        `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
                default is 0 (off), useful value is 1e-5.

        `workers` = use this many worker threads to train the model (=faster training with multicore machines).

        `hs` = if 1 (default), hierarchical sampling will be used for model training (else set to 0).

        `negative` = if > 0, negative sampling will be used, the int for negative
        specifies how many "noise words" should be drawn (usually between 5-20).

        `dm_mean` = if 0 (default), use the sum of the context word vectors. If 1, use the mean.
        Only applies when dm is used.

        """
        Word2Vec.__init__(self, size=size, alpha=alpha, window=window, min_count=min_count,
                          sample=sample, seed=seed, workers=workers, min_alpha=min_alpha,
                          sg=(1+dm) % 2, hs=hs, negative=negative, cbow_mean=dm_mean, **kwargs)
        self.train_words = train_words
        self.train_lbls = train_lbls
        if sentences is not None:
            self.build_vocab(sentences)
            self.train(sentences)

    @staticmethod
    def _vocab_from(sentences):
        sentence_no, vocab = -1, {}
        total_words = 0
        for sentence_no, sentence in enumerate(sentences):
            if sentence_no % 10000 == 0:
                logger.info("PROGRESS: at item #%i, processed %i words and %i word types" %
                            (sentence_no, total_words, len(vocab)))
            sentence_length = len(sentence.words)
            for label in sentence.labels:
                total_words += 1
                if label in vocab:
                    vocab[label].count += sentence_length
                else:
                    vocab[label] = Vocab(count=sentence_length)
            for word in sentence.words:
                total_words += 1
                if word in vocab:
                    vocab[word].count += 1
                else:
                    vocab[word] = Vocab(count=1)
        logger.info("collected %i word types from a corpus of %i words and %i items" %
                    (len(vocab), total_words, sentence_no + 1))
        return vocab

    def _prepare_sentences(self, sentences):
        for sentence in sentences:
            # avoid calling random_sample() where prob >= 1, to speed things up a little:
            sampled = [self.vocab[word] for word in sentence.words
                       if word in self.vocab and (self.vocab[word].sample_probability >= 1.0 or
                                                  self.vocab[word].sample_probability >= random.random_sample())]
            yield (sampled, [self.vocab[word] for word in sentence.labels if word in self.vocab])

    def _get_job_words(self, alpha, work, job, neu1):
        if self.sg:
            return sum(train_sentence_dbow(self, sentence, lbls, alpha, work, self.train_words, self.train_lbls) for sentence, lbls in job)
        else:
            return sum(train_sentence_dm(self, sentence, lbls, alpha, work, neu1, self.train_words, self.train_lbls) for sentence, lbls in job)

    def __str__(self):
        return "Doc2Vec(vocab=%s, size=%s, alpha=%s)" % (len(self.index2word), self.layer1_size, self.alpha)

    def save(self, *args, **kwargs):
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm'])  # don't bother storing the cached normalized vectors
        super(Doc2Vec, self).save(*args, **kwargs)


class LabeledBrownCorpus(object):
    """Iterate over sentences from the Brown corpus (part of NLTK data), yielding
    each sentence out as a LabeledSentence object."""
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            for item_no, line in enumerate(utils.smart_open(fname)):
                line = utils.to_unicode(line)
                # each file line is a single sentence in the Brown corpus
                # each token is WORD/POS_TAG
                token_tags = [t.split('/') for t in line.split() if len(t.split('/')) == 2]
                # ignore words with non-alphabetic tags like ",", "!" etc (punctuation, weird stuff)
                words = ["%s/%s" % (token.lower(), tag[:2]) for token, tag in token_tags if tag[:2].isalpha()]
                if not words:  # don't bother sending out empty sentences
                    continue
                yield LabeledSentence(words, ['%s_SENT_%s' % (fname, item_no)])


class LabeledLineSentence(object):
    """Simple format: one sentence = one line = one LabeledSentence object.

    Words are expected to be already preprocessed and separated by whitespace,
    labels are constructed automatically from the sentence line number."""
    def __init__(self, source):
        """
        `source` can be either a string (filename) or a file object.

        Example::

            sentences = LineSentence('myfile.txt')

        Or for compressed files::

            sentences = LineSentence('compressed_text.txt.bz2')
            sentences = LineSentence('compressed_text.txt.gz')

        """
        self.source = source

    def __iter__(self):
        """Iterate through the lines in the source."""
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for item_no, line in enumerate(self.source):
                yield LabeledSentence(utils.to_unicode(line).split(), ['SENT_%s' % item_no])
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.smart_open(self.source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), ['SENT_%s' % item_no])
