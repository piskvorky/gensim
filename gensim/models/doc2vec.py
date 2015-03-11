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

from numpy import zeros, random, sum as np_sum, add as np_add, concatenate
from six import string_types

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
        for label in lbls:
            if label is None:
                continue  # OOV word in the input sentence => skip
            for word in sentence:
                if word is None:
                    continue  # OOV word in the input sentence => skip
                train_sg_pair(model, word, label, alpha, train_words, train_lbls)

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
            neu1e = train_cbow_pair(model, word, word2_indices, l1, alpha, train_words, train_words)
            if train_lbls:
                model.syn0[lbl_indices] += neu1e

        return len([word for word in sentence if word is not None])


def train_sentence_dm_concat(model, sentence, lbls, alpha, work=None, neu1=None, train_words=True, train_lbls=True):
        """
        Update distributed memory model by training on a single sentence.

        The sentence is a list of Vocab objects (or None, where the corresponding
        word is not in the vocabulary. Called internally from `Doc2Vec.train()`.

        This is the non-optimized, Python version. If you have a C compiler, gensim
        will use the optimized version from doc2vec_inner instead.

        """
        lbl_indices = [lbl.index for lbl in lbls if lbl is not None]
        if len(lbl_indices) != model.dm_lbl_count:
            return  # skip doc without expected lbl(s)

        null_word = model.vocab['\0']
        pre_pad_count = int((model.window + 1) / 2)
        post_pad_count = int(model.window / 2)
        padded_sentence_indices = (
            (pre_pad_count * [null_word.index])  # pre-padding
            + [word.index for word in sentence if word is not None]  # elide out-of-Vocabulary words
            + (post_pad_count * [null_word.index])  # post-padding
        )

        for pos in range(pre_pad_count, len(padded_sentence_indices) - post_pad_count):
            l1_indices = (
                lbl_indices  # doc vector(s)
                + padded_sentence_indices[pos - pre_pad_count : pos]  # preceding words
                + padded_sentence_indices[pos + 1 : pos + 1 + post_pad_count]  # following words
            )
            word = model.vocab[model.index2word[padded_sentence_indices[pos]]]
            l1 = model.syn0[l1_indices].ravel()  # numpy advanced-indexing: copy; flatten to 1d
            neu1e = train_cbow_pair(model, word, None, l1, alpha, True, False)

            if not train_lbls:
                # trim lbl indices/errors
                l1_indices = l1_indices[len(lbl_indices):] 
                neu1e = neu1e[len(lbl_indices) * model.vector_size:]
            if not train_words:
                # trim word-vector indices/errors
                l1_indices = l1_indices[:-model.window]
                neu1e = neu1e[:-model.window * model.vector_size]
            if l1_indices:
                # if indices left to train, do so
                np_add.at(model.syn0, l1_indices, neu1e.reshape(len(l1_indices), model.vector_size))

        return len(padded_sentence_indices) - pre_pad_count - post_pad_count


def infer_vector_dbow(model, document, alpha=0.025, min_alpha=0.0001, steps=50):
    """
    Infer a vector for given post-bulk training document, in the 'dbow' model.

    Document should be a list of tokens.

    No cythonized alternative yet.
    """
    if not hasattr(model, 'neg_labels'):
        model.pretrain()

    vector = model.seeded_vector(' '.join(document))
    sentence = next(model._prepare_sentences([LabeledSentence(document, [])]))[0]

    for i in range(steps):
        for word in sentence:
            if word is None:
                continue  # OOV word in the input sentence => skip
            neu1e = train_sg_pair(model, word, vector, alpha, False, False)
            vector += neu1e
        alpha = ((alpha - min_alpha) / (steps - i)) + min_alpha

    return vector


def infer_vector_dm(model, document, alpha=0.025, min_alpha=0.0001, steps=50):
    """
    Infer a vector representation for the given post-training document, in the 'dm' model.

    Document should be a list of tokens.

    No cythonized alternative yet.
    """
    if not hasattr(model, 'neg_labels'):
        model.pretrain()

    vector = model.seeded_vector(' '.join(document))
    sentence = next(model._prepare_sentences([LabeledSentence(document, [])]))[0]

    for i in range(steps):

        for pos, word in enumerate(sentence):
            if word is None:
                continue  # OOV word in the input sentence => skip
            reduced_window = random.randint(model.window)  # `b` in the original doc2vec code
            start = max(0, pos - model.window + reduced_window)
            window_pos = enumerate(sentence[start : pos + model.window + 1 - reduced_window], start)
            word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
            l1 = np_sum(model.syn0[word2_indices], axis=0) + vector  # 1 x layer1_size
            if word2_indices and model.cbow_mean:
                l1 /= (len(word2_indices) + 1)
            neu1e = train_cbow_pair(model, word, None, l1, alpha, False, False)
            vector += neu1e  # learn input -> hidden

        alpha = ((alpha - min_alpha) / (steps - i)) + min_alpha

    return vector


def infer_vector_dm_concat(model, document, alpha=0.025, min_alpha=0.0001, steps=50):
    """
    Infer a vector representation for the given post-training document, in the 'dm_concat' model.

    Document should be a list of tokens.

    No cythonized alternative yet.
    """
    if not hasattr(model, 'neg_labels'):
        model.pretrain()

    vector = model.seeded_vector(' '.join(document))
    sentence = next(model._prepare_sentences([LabeledSentence(document, [])]))[0]

    null_word = model.vocab['\0']
    pre_pad_count = int((model.window + 1) / 2)
    post_pad_count = int(model.window / 2)
    padded_sentence_indices = (
        (pre_pad_count * [null_word.index])  # pre-padding
        + [word.index for word in sentence if word is not None]  # elide out-of-Vocabulary words
        + (post_pad_count * [null_word.index])  # post-padding
    )

    for i in range(steps):

        for pos in range(pre_pad_count, len(padded_sentence_indices)-post_pad_count):
            word = model.vocab[model.index2word[padded_sentence_indices[pos]]]
            l1 = concatenate([
                    [vector],  # doc vector-in-training
                    model.syn0[padded_sentence_indices[pos - pre_pad_count : pos]],  # preceding words
                    model.syn0[padded_sentence_indices[pos + 1 : pos + 1 + post_pad_count]],  # following words
            ]).ravel()

            neu1e = train_cbow_pair(model, word, None, l1, alpha, False, False)

            vector += neu1e[:model.vector_size]  # train doc vector only

        alpha = ((alpha - min_alpha) / (steps - i)) + min_alpha

    return vector


class LabeledSentence(object):
    """
    A single labeled sentence = text item.
    Replaces "sentence as a list of words" from Word2Vec.

    """
    def __init__(self, words, labels):
        """
        `words` is a list of tokens (unicode strings), 
        `labels` a list of text labels associated with this text
        or a single string label.

        """
        if isinstance(labels, string_types):
          labels = (labels,)
        self.words = words
        self.labels = labels

    def __str__(self):
        return '%s(%s, %s)' % (self.__class__.__name__, self.words, self.labels)


class Doc2Vec(Word2Vec):
    """Class for training, using and evaluating neural networks described in http://arxiv.org/pdf/1405.4053v2.pdf"""
    def __init__(self, sentences=None, size=300, alpha=0.025, window=8, min_count=5,
                 sample=0, seed=1, workers=1, min_alpha=0.0001, dm=1, hs=1, negative=0,
                 dm_mean=0, dm_concat=0, dm_lbl_count=1, train_words=True, train_lbls=True, **kwargs):
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

        `dm_concat` = if 1, use concatenation of context vectors rather than sum/average;
        default is 0 (off).

        `dm_lbl_count` = expected constant number of sentence lbls per sentence, when using
        dm_concat mode; default is 1.

        """
        Word2Vec.__init__(self, size=size, alpha=alpha, window=window, min_count=min_count,
                          sample=sample, seed=seed, workers=workers, min_alpha=min_alpha,
                          sg=(1+dm) % 2, hs=hs, negative=negative, cbow_mean=dm_mean, 
                          null_word=dm_concat, **kwargs)
        self.train_words = train_words
        self.train_lbls = train_lbls
        self.dm_concat = dm_concat
        self.dm_lbl_count = dm_lbl_count
        if sentences is not None:
            self.build_vocab(sentences)
            self.train(sentences)

    def reset_weights(self):
        if self.dm_concat:
            # expand l1 size to match concatenated lbls+words length
            self.layer1_size = (self.dm_lbl_count + self.window) * self.vector_size
            logger.info("using concatenative %d-dimensional layer1"% (self.layer1_size))
        Word2Vec.reset_weights(self)

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
                    vocab[label] = Vocab(count=sentence_length)  # FIXME: doc-labels for short docs can be culled by min_count
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
        elif self.dm_concat:
            return sum(train_sentence_dm_concat(self, sentence, lbls, alpha, work, neu1, self.train_words, self.train_lbls) for sentence, lbls in job)
        else:
            return sum(train_sentence_dm(self, sentence, lbls, alpha, work, neu1, self.train_words, self.train_lbls) for sentence, lbls in job)

    def infer_vector(self, document, alpha=0.025, min_alpha=0.0001, steps=50):
        """
        Infer a vector for given post-bulk training document.

        Document should be a list of tokens.
        """
        if self.sg:
            return infer_vector_dbow(self, document, alpha, min_alpha, steps)
        elif self.dm_concat:
            return infer_vector_dm_concat(self, document, alpha, min_alpha, steps)
        else:
            return infer_vector_dm(self, document, alpha, min_alpha, steps)

    def __str__(self):
        return "Doc2Vec(%id, sg=%i, hs=%i, negative=%i, dm_concat=%i)" % (self.vector_size, self.sg, self.hs, self.negative, self.dm_concat)

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
