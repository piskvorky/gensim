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

from collections import namedtuple

from numpy import zeros, random, sum as np_sum, add as np_add, concatenate,\
    repeat as np_repeat, array, float32 as REAL, empty, ones

logger = logging.getLogger(__name__)

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.models.word2vec import Word2Vec, Vocab, train_cbow_pair, train_sg_pair, train_sentence_sg
from six.moves import xrange

try:
    from gensim.models.doc2vec_inner import train_sentence_dbow, train_sentence_dm, train_sentence_dm_concat,\
                                            FAST_VERSION
except:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1

    def train_sentence_dbow(model, word_vocabs, doclbl_vocabs, alpha, work=None,
                            train_words=False, learn_doclbls=True, learn_words=True, learn_hidden=True,
                            word_vectors=None, word_locks=None, doclbl_vectors=None, doclbl_locks=None):
        """
        Update distributed bag of words model by training on a single sentence.

        The sentence is a list of Vocab objects (or None, where the corresponding
        word is not in the vocabulary. Called internally from `Doc2Vec.train()`.

        If train_words is True, simultaneously train word-to-word (not just doc-to-word)
        examples, exactly as per Word2Vec skip-gram training. (Without this option,
        word vectors are neither consulted nor updated during DBOW doc vector training.)

        If learn_words is True, training examples will cause word vectors to be
        updated. If learn_hidden is True, training examples will update the internal
        hidden layer weights.

        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from doc2vec_inner instead.

        """
        if doclbl_vectors is None:
            doclbl_vectors = model.doclbl_syn0
        if doclbl_locks is None:
            doclbl_locks = model.doclbl_syn0_lockf

        if train_words and learn_words:
            train_sentence_sg(model, word_vocabs, alpha, work)  # TODO: adapt for word_vectors/word_locks
        for doclbl in doclbl_vocabs:
            if doclbl is None:
                continue  # OOV token => skip
            for word in word_vocabs:
                if word is None:
                    continue  # OOV word in the input sentence => skip
                train_sg_pair(model, word, doclbl, alpha, learn_vectors=learn_doclbls,
                              learn_hidden=learn_hidden, context_vectors=doclbl_vectors,
                              context_locks=doclbl_locks)

        return len([word for word in word_vocabs if word is not None])

    def train_sentence_dm(model, word_vocabs, doclbl_vocabs, alpha, work=None, neu1=None,
                          learn_doclbls=True, learn_words=True, learn_hidden=True,
                          word_vectors=None, word_locks=None, doclbl_vectors=None, doclbl_locks=None):
        """
        Update distributed memory model by training on a single sentence.

        The sentence is a list of Vocab objects (or None, where the corresponding
        word is not in the vocabulary. Called internally from `Doc2Vec.train()`.

        This is the non-optimized, Python version. If you have a C compiler, gensim
        will use the optimized version from doc2vec_inner instead.

        """
        if word_vectors is None:
            word_vectors = model.syn0
        if word_locks is None:
            word_locks = model.syn0_lockf
        if doclbl_vectors is None:
            doclbl_vectors = model.doclbl_syn0
        if doclbl_locks is None:
            doclbl_locks = model.doclbl_syn0_lockf

        doclbl_indices = [doclbl.index for doclbl in doclbl_vocabs if doclbl is not None]
        doclbl_sum = np_sum(doclbl_vectors[doclbl_indices], axis=0)
        doclbl_len = len(doclbl_indices)

        for pos, word in enumerate(word_vocabs):
            if word is None:
                continue  # OOV word in the input sentence => skip
            reduced_window = random.randint(model.window)  # `b` in the original doc2vec code
            start = max(0, pos - model.window + reduced_window)
            window_pos = enumerate(word_vocabs[start : pos + model.window + 1 - reduced_window], start)
            word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
            l1 = np_sum(word_vectors[word2_indices], axis=0) + doclbl_sum  # 1 x layer1_size
            if word2_indices and model.cbow_mean:
                l1 /= (len(word2_indices) + doclbl_len)
            neu1e = train_cbow_pair(model, word, word2_indices, l1, alpha, learn_vectors=False, learn_hidden=True)
            if word2_indices and not model.cbow_mean:
                neu1e /= (len(word2_indices) + doclbl_len)
            if learn_doclbls:
                doclbl_vectors[doclbl_indices] += \
                    neu1e * np_repeat(doclbl_locks[doclbl_indices],model.vector_size).reshape(-1,model.vector_size)
            if learn_words:
                word_vectors[word2_indices] += \
                    neu1e * np_repeat(word_locks[word2_indices],model.vector_size).reshape(-1,model.vector_size)

        return len([word for word in word_vocabs if word is not None])


    def train_sentence_dm_concat(model, word_vocabs, doclbl_vocabs, alpha, work=None, neu1=None,
                                 learn_doclbls=True, learn_words=True, learn_hidden=True,
                                 word_vectors=None, word_locks=None, doclbl_vectors=None, doclbl_locks=None):
        """
        Update distributed memory model by training on a single sentence, using a
        concatenation of the context window word vectors (rather than a sum or average).

        The sentence is a list of Vocab objects (or None, where the corresponding
        word is not in the vocabulary. Called internally from `Doc2Vec.train()`.

        This is the non-optimized, Python version. If you have a C compiler, gensim
        will use the optimized version from doc2vec_inner instead.

        """
        if word_vectors is None:
            word_vectors = model.syn0
        if word_locks is None:
            word_locks = model.syn0_lockf
        if doclbl_vectors is None:
            doclbl_vectors = model.doclbl_syn0
        if doclbl_locks is None:
            doclbl_locks = model.doclbl_syn0_lockf

        doclbl_indices = [doclbl.index for doclbl in doclbl_vocabs if doclbl is not None]
        doclbl_len = len(doclbl_indices)
        if doclbl_len != model.dm_lbl_count:
            return 0 # skip doc without expected doclbl(s)

        null_word = model.vocab['\0']
        pre_pad_count = model.window
        post_pad_count = model.window
        padded_sentence_indices = (
            (pre_pad_count * [null_word.index])  # pre-padding
            + [word.index for word in word_vocabs if word is not None]  # elide out-of-Vocabulary words
            + (post_pad_count * [null_word.index])  # post-padding
        )

        for pos in range(pre_pad_count, len(padded_sentence_indices) - post_pad_count):
            word_context_indices = (
                padded_sentence_indices[pos - pre_pad_count : pos]  # preceding words
                + padded_sentence_indices[pos + 1 : pos + 1 + post_pad_count]  # following words
            )
            word_context_len = len(word_context_indices)
            predict_word = model.vocab[model.index2word[padded_sentence_indices[pos]]]
            # numpy advanced-indexing copies; concatenate, flatten to 1d
            l1 = concatenate((doclbl_vectors[doclbl_indices], word_vectors[word_context_indices])).ravel()
            neu1e = train_cbow_pair(model, predict_word, None, l1, alpha, learn_hidden=learn_hidden, learn_vectors=False)

            # filter by locks and shape for addition to source vectors
            e_locks = concatenate((doclbl_locks[doclbl_indices], word_locks[word_context_indices]))
            neu1e_r = (neu1e.reshape(-1,model.vector_size)
                       * np_repeat(e_locks,model.vector_size).reshape(-1,model.vector_size))

            if learn_doclbls:
                np_add.at(doclbl_vectors, doclbl_indices, neu1e_r[:doclbl_len])
            if learn_words:
                np_add.at(word_vectors, word_context_indices, neu1e_r[doclbl_len:])

        return len(padded_sentence_indices) - pre_pad_count - post_pad_count


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

class Doclbl(namedtuple('Doclbl', 'index, word_count, doc_count')):
    """A document label discovered during the initial vocabulary
    scan. (The document-vector equivalent of a Vocab object.)"""
    __slots__ = ()
    def repeat(self, word_count):
        return self._replace(word_count=self.word_count + word_count, doc_count=self.doc_count + 1)


class Doc2Vec(Word2Vec):
    """Class for training, using and evaluating neural networks described in http://arxiv.org/pdf/1405.4053v2.pdf"""
    def __init__(self, sentences=None, size=300, alpha=0.025, window=8, min_count=5,
                 sample=0, seed=1, workers=1, min_alpha=0.0001, dm=1, hs=1, negative=0,
                 dbow_words=0, dm_mean=0, dm_concat=0, dm_lbl_count=1, **kwargs):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        LabeledSentence object that will be used for training.

        The `sentences` iterable can be simply a list of LabeledSentence elements, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.

        If you don't supply `sentences`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.

        `dm` defines the training algorithm. By default (`dm=1`), 'distributed memory' (PV-DM) is used.
        Otherwise, `distributed bag of words` (PV-DBOW) is employed.

        `size` is the dimensionality of the feature vectors.

        `window` is the maximum distance between the current and predicted word within a sentence.

        `alpha` is the initial learning rate (will linearly drop to zero as training progresses).

        `seed` = for the random number generator. Only runs with a single worker will be
        deterministically reproducible because of the ordering randomness in multi-threaded runs.

        `min_count` = ignore all words with total frequency lower than this.

        `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
                default is 0 (off), useful value is 1e-5.

        `workers` = use this many worker threads to train the model (=faster training with multicore machines).

        `hs` = if 1 (default), hierarchical sampling will be used for model training (else set to 0).

        `negative` = if > 0, negative sampling will be used, the int for negative
        specifies how many "noise words" should be drawn (usually between 5-20).

        `dm_mean` = if 0 (default), use the sum of the context word vectors. If 1, use the mean.
        Only applies when dm is used in non-concatenative mode.

        `dm_concat` = if 1, use concatenation of context vectors rather than sum/average;
        default is 0 (off). Note concatenation results in a much-larger model, as the input
        is no longer the size of one (sampled or arithmatically combined) word vector, but the
        size of the label(s) and all words in the context strung together.

        `dm_lbl_count` = expected constant number of sentence lbls per sentence, when using
        dm_concat mode; default is 1.

        `dbow_words` if set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW
        doc-vector training; default is 0 (faster training of doc-vectors only.

        """
        Word2Vec.__init__(self, size=size, alpha=alpha, window=window, min_count=min_count,
                          sample=sample, seed=seed, workers=workers, min_alpha=min_alpha,
                          sg=(1+dm) % 2, hs=hs, negative=negative, cbow_mean=dm_mean, 
                          null_word=dm_concat, **kwargs)
        self.doclbls = {}  # mapping from doclbl (string) to Doclbl object
        self.index2doclbl = []  # map from doclbl's int index to string
        self.dbow_words = dbow_words
        self.dm_concat = dm_concat
        self.dm_lbl_count = dm_lbl_count
        if sentences is not None:
            self.build_vocab(sentences)
            self.train(sentences)

    def reset_weights(self):
        if self.dm_concat:
            # expand l1 size to match concatenated lbls+words length
            self.layer1_size = (self.dm_lbl_count + (2 * self.window)) * self.vector_size
            logger.info("using concatenative %d-dimensional layer1"% (self.layer1_size))
        Word2Vec.reset_weights(self)
        self.doclbl_syn0 = empty((len(self.doclbls), self.vector_size), dtype=REAL)
        for i in xrange(len(self.doclbls)):
            # construct deterministic seed from word AND seed argument
            self.doclbl_syn0[i] = self.seeded_vector(self.index2doclbl[i] + str(self.seed))        
        self.doclbl_syn0_lockf = ones(len(self.doclbls), dtype=REAL)  # zeros suppress learning

    def _vocab_from(self, sentences):
        sentence_no, vocab = -1, {}
        total_words = 0
        for sentence_no, sentence in enumerate(sentences):
            if sentence_no % 10000 == 0:
                logger.info("PROGRESS: at item #%i, processed %i words and %i word types" %
                            (sentence_no, total_words, len(vocab)))
            sentence_length = len(sentence.words)
            for label in sentence.labels:
                total_words += 1
                if label in self.doclbls:
                    self.doclbls[label] = Doclbl[label].repeat(sentence_length)
                else:
                    self.doclbls[label] = Doclbl(sentence_no, sentence_length, 1)
                    self.index2doclbl.append(label)
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
            yield (self._tokens_to_vocabs(sentence.words),
                   self._tokens_to_vocabs(sentence.labels, sample=False, source_dict=self.doclbls))

    def _tokens_to_vocabs(self, tokens, sample=True, source_dict=None):
        if source_dict is None:
            source_dict = self.vocab
        if sample:
            return [source_dict[token] for token in tokens if token in source_dict
                    and (source_dict[token].sample_probability >= 1.0 or
                         source_dict[token].sample_probability >= random.random_sample())]
        else:
            return [source_dict[token] for token in tokens if token in source_dict]

    def _get_job_words(self, alpha, work, job, neu1):
        if self.sg:
            return sum(train_sentence_dbow(self, sentence, lbls, alpha, work, train_words=self.dbow_words)
                                           for sentence, lbls in job)
        elif self.dm_concat:
            return sum(train_sentence_dm_concat(self, sentence, lbls, alpha, work, neu1) for sentence, lbls in job)
        else:
            return sum(train_sentence_dm(self, sentence, lbls, alpha, work, neu1) for sentence, lbls in job)

    def infer_vector(self, document, alpha=0.1, min_alpha=0.0001, steps=5):
        """
        Infer a vector for given post-bulk training document.

        Document should be a list of (word) tokens.
        """
        doclbl_vectors = empty((1, self.vector_size), dtype=REAL)
        doclbl_vectors[0] = self.seeded_vector(' '.join(document))
        doclbl_locks = ones(1, dtype=REAL)
        doclbl_vocabs = [Doclbl(0,0,0) for doclbl in doclbl_vectors]
        word_vocabs = self._tokens_to_vocabs(document)

        work = zeros(self.layer1_size, dtype=REAL)
        if not self.sg:
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)

        for i in range(steps):
            if self.sg:
                train_sentence_dbow(self, word_vocabs, doclbl_vocabs, alpha, work,
                                    learn_words=False, learn_hidden=False,
                                    doclbl_vectors=doclbl_vectors, doclbl_locks=doclbl_locks)
            elif self.dm_concat:
                train_sentence_dm_concat(self, word_vocabs, doclbl_vocabs, alpha, work, neu1,
                                         learn_words=False, learn_hidden=False,
                                         doclbl_vectors=doclbl_vectors, doclbl_locks=doclbl_locks)
            else:
                train_sentence_dm(self, word_vocabs, doclbl_vocabs, alpha, work, neu1,
                                  learn_words=False, learn_hidden=False,
                                  doclbl_vectors=doclbl_vectors, doclbl_locks=doclbl_locks)
            alpha = ((alpha - min_alpha) / (steps - i)) + min_alpha

        return doclbl_vectors[0]

    def get_docvec(self, doclbl):
        return self.doclbl_syn0[self.doclbls[doclbl].index]

    @property
    def docs(self):
        class DocsView(object):
            def __init__(self, doc2vec):
                self.model = doc2vec
            def __getitem__(self, doclbl):
                return self.model.get_docvec(doclbl)
        return DocsView(self)

    def __str__(self):
        return "Doc2Vec(%id, sg=%i, hs=%i, negative=%i, dm_concat=%i)" % (self.vector_size, self.sg, self.hs, self.negative, self.dm_concat)

    @property
    def compact_name(self):
        segments = []
        if self.sg:
            segments.append('dbow')  # PV-DBOW (skip-gram-style)
            if self.dbow_words:
                segments.append('w')  # also training words
        else:
            segments.append('dm')  # PV-DM...
            if self.dm_concat:
                segments.append('c')  # ...with concatenative context layer
            else:
                if self.cbow_mean:
                    segments.append('m')
                else:
                    segments.append('s')
        segments.append('_')
        segments.append('d%d' % self.vector_size)  # dimensions
        if self.negative:
            segments.append('n%d' % self.negative)  # negative samples
        if self.hs:
            segments.append('hs')
        if not self.sg or (self.sg and self.dbow_words):
            segments.append('w%d' % self.window)  # window size, when relevant
        if self.min_count > 1:
            segments.append('mc%d' % self.min_count)
        if self.sample > 0:
            segments.append('s%d' % self.sample)
        return ''.join(segments)

    def save(self, *args, **kwargs):
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm'])  # don't bother storing the cached normalized vectors
        super(Doc2Vec, self).save(*args, **kwargs)  ### TODO: save doclbl fields


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
