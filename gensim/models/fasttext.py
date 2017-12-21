#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Chinmaya Pancholi <chinmayapancholi13@gmail.com>, Shiva Manne <s.manne@rare-technologies.com>
# Copyright (C) 2017 RaRe Technologies s.r.o.

"""Learn word representations via fasttext's "skip-gram and CBOW models", using either
hierarchical softmax or negative sampling [1]_.

Notes
-----
There are more ways to get word vectors in Gensim than just FastText.
See wrappers for VarEmbed and WordRank or Word2Vec

This module allows training a word embedding from a training corpus with the additional ability
to obtain word vectors for out-of-vocabulary words.

For a tutorial on gensim's native fasttext, refer to the noteboook -- [2]_

**Make sure you have a C compiler before installing gensim, to use optimized (compiled) fasttext training**

.. [1] P. Bojanowski, E. Grave, A. Joulin, T. Mikolov
       Enriching Word Vectors with Subword Information. In arXiv preprint arXiv:1607.04606.
       https://arxiv.org/abs/1607.04606

.. [2] https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/FastText_Tutorial.ipynb

"""

import logging
import struct

import numpy as np
from numpy import zeros, ones, vstack, random, sum as np_sum, empty, float32 as REAL

# from gensim.models.word2vec import Word2Vec, train_sg_pair, train_cbow_pair
from gensim.models.word2vec import Word2VecVocab, Word2VecTrainables
from gensim.models.keyedvectors import FastTextKeyedVectors, Vocab
# from gensim.models.wrappers.fasttext import FastTextKeyedVectors
from gensim.models.wrappers.fasttext import compute_ngrams, ft_hash
from gensim.models.base_any2vec import BaseAny2VecModel
from gensim.models import word2vec_utils

from types import GeneratorType
from six import iteritems
from gensim.utils import deprecated
from gensim import utils

logger = logging.getLogger(__name__)

try:
    from gensim.models.fasttext_inner import train_batch_sg, train_batch_cbow
    from gensim.models.fasttext_inner import FAST_VERSION, MAX_WORDS_IN_BATCH
    logger.debug('Fast version of Fasttext is being used')

except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    raise RuntimeError("Support for Python/Numpy implementations has been continued.")

FASTTEXT_FILEFORMAT_MAGIC = 793712314


class FastText(BaseAny2VecModel):
    """Class for training, using and evaluating word representations learned using method
    described in [1]_ aka Fasttext.

    The model can be stored/loaded via its :meth:`~gensim.models.fasttext.FastText.save()` and
    :meth:`~gensim.models.fasttext.FastText.load()` methods, or loaded in a format compatible with the original
    fasttext implementation via :meth:`~gensim.models.fasttext.FastText.load_fasttext_format()`.

    """
    def __init__(self, sentences=None, sg=0, hs=0, size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, word_ngrams=1, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                 negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0, min_n=3, max_n=6, sorted_vocab=1,
                 bucket=2000000, trim_rule=None, batch_words=MAX_WORDS_IN_BATCH, callbacks=()):
        """Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (unicode strings) that will be used for training.

        Parameters
        ----------
        sentences : iterable of iterables
            The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
            If you don't supply `sentences`, the model is left uninitialized -- use if you plan to initialize it
            in some other way.
        sg : int {1, 0}
            Defines the training algorithm. If 1, CBOW is used, otherwise, skip-gram is employed.
        size : int
            Dimensionality of the feature vectors.
        window : int
            The maximum distance between the current and predicted word within a sentence.
        alpha : float
            The initial learning rate.
        min_alpha : float
            Learning rate will linearly drop to `min_alpha` as training progresses.
        seed : int
            Seed for the random number generator. Initial vectors for each word are seeded with a hash of
            the concatenation of word + `str(seed)`. Note that for a fully deterministically-reproducible run,
            you must also limit the model to a single worker thread (`workers=1`), to eliminate ordering jitter
            from OS thread scheduling. (In Python 3, reproducibility between interpreter launches also requires
            use of the `PYTHONHASHSEED` environment variable to control hash randomization).
        min_count : int
            Ignores all words with total frequency lower than this.
        max_vocab_size : int
            Limits the RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
            Set to `None` for no limit.
        sample : float
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        workers : int
            Use these many worker threads to train the model (=faster training with multicore machines).
        hs : int {1,0}
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
        negative : int
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        cbow_mean : int {1,0}
            If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
        hashfxn : function
            Hash function to use to randomly initialize weights, for increased training reproducibility.
        iter : int
            Number of iterations (epochs) over the corpus.
        trim_rule : function
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            Note: The rule, if given, is only used to prune vocabulary during build_vocab() and is not stored as part
            of the model.
        sorted_vocab : int {1,0}
            If 1, sort the vocabulary by descending frequency before assigning word indexes.
        batch_words : int
            Target size (in words) for batches of examples passed to worker threads (and
            thus cython routines).(Larger batches will be passed if individual
            texts are longer than 10000 words, but the standard cython code truncates to that maximum.)
        min_n : int
            Min length of char ngrams to be used for training word representations.
        max_n : int
            Max length of char ngrams to be used for training word representations. Set `max_n` to be
            lesser than `min_n` to avoid char ngrams being used.
        word_ngrams : int {1,0}
            If 1, uses enriches word vectors with subword(ngrams) information.
            If 0, this is equivalent to word2vec.
        bucket : int
            Character ngrams are hashed into a fixed number of buckets, in order to limit the
            memory usage of the model. This option specifies the number of buckets used by the model.

        Examples
        --------
        Initialize and train a `FastText` model

        >>> from gensim.models import FastText
        >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
        >>>
        >>> model = FastText(sentences, min_count=1)
        >>> say_vector = model['say']  # get vector for word
        >>> of_vector = model['of']  # get vector for out-of-vocab word


        """
        self.callbacks = callbacks

        if FAST_VERSION == -1:
            logger.warning('Slow version of %s is being used', __name__)
        else:
            logger.debug('Fast version of %s is being used', __name__)

        self.sg = int(sg)
        if size % 4 != 0:
            logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.alpha = float(alpha)
        self.min_alpha_yet_reached = float(alpha)  # To warn user if alpha increases
        self.window = int(window)
        self.random = random.RandomState(seed)
        self.min_alpha = float(min_alpha)
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.running_training_loss = 0
        self.word_ngrams = word_ngrams
        if self.word_ngrams <= 1 and max_n == 0:
            bucket = 0

        self.wv = FastTextKeyedVectors()
        self.vocabulary = FastTextVocab(
            max_vocab_size=max_vocab_size, min_count=min_count, sample=sample,
            sorted_vocab=sorted_vocab, null_word=null_word, min_n=min_n, max_n=max_n)
        self.trainables = FastTextTrainables(
            vector_size=size, seed=seed, alpha=alpha, min_alpha=min_alpha, hs=hs, negative=negative,
            hashfxn=hashfxn, bucket=bucket)

        super(FastText, self).__init__(
            workers=workers, vector_size=size, epochs=iter, callbacks=callbacks, batch_words=batch_words)

        if sentences is not None:
            if isinstance(sentences, GeneratorType):
                raise TypeError("You can't pass a generator as the sentences argument. Try an iterator.")
            self.build_vocab(sentences, trim_rule=trim_rule)
            self.train(
                sentences, total_examples=self.vocabulary.corpus_count, epochs=self.epochs,
                start_alpha=self.trainables.alpha, end_alpha=self.trainables.min_alpha)
        else:
            if trim_rule is not None:
                logger.warning(
                    "The rule, if given, is only used to prune vocabulary during build_vocab() "
                    "and is not stored as part of the model. Model initialized without sentences. "
                    "trim_rule provided, if any, will be ignored.")

    def build_vocab(self, sentences, keep_raw_vocab=False, trim_rule=None, progress_per=10000, update=False):
        """Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        Parameters
        ----------
        sentences : iterable of iterables
            The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
        keep_raw_vocab : bool
            If not true, delete the raw vocabulary after the scaling is done and free up RAM.
        trim_rule : function
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            Note: The rule, if given, is only used to prune vocabulary during build_vocab() and is not stored as part
            of the model.
        progress_per : int
            Indicates how many words to process before showing/updating the progress.
        update: bool
            If true, the new words in `sentences` will be added to model's vocab.

        Example
        -------
        Train a model and update vocab for online training

        >>> from gensim.models import FastText
        >>> sentences_1 = [["cat", "say", "meow"], ["dog", "say", "woof"]]
        >>> sentences_2 = [["dude", "say", "wazzup!"]]
        >>>
        >>> model = FastText(min_count=1)
        >>> model.build_vocab(sentences_1)
        >>> model.train(sentences_1, total_examples=model.corpus_count, epochs=model.iter)
        >>> model.build_vocab(sentences_2, update=True)
        >>> model.train(sentences_2, total_examples=model.corpus_count, epochs=model.iter)

        """
        if update:
            if not len(self.wv.vocab):
                raise RuntimeError(
                    "You cannot do an online vocabulary-update of a model which has no prior vocabulary. "
                    "First build the vocabulary of your model with a corpus "
                    "before doing an online update.")
            self.vocabulary.old_vocab_len = len(self.vocabulary.vocab)
            self.trainables.old_hash2index_len = len(self.trainables.hash2index)

        super(FastText, self).build_vocab(
            sentences, keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, progress_per=progress_per, update=update)

    def _set_keyedvectors(self):
        self.wv.vectors = self.trainables.vectors
        self.wv.vector_size = self.trainables.vector_size
        self.wv.vectors_vocab = self.trainables.vectors_vocab
        self.wv.vectors_ngrams = self.trainables.vectors_ngrams
        self.wv.ngrams = self.trainables.ngrams
        self.wv.hash2index = self.trainables.hash2index
        self.wv.min_n = self.vocabulary.min_n
        self.wv.max_n = self.vocabulary.max_n
        self.wv.ngrams_word = self.vocabulary.ngrams_word
        self.wv.vocab = self.vocabulary.vocab
        self.wv.index2word = self.vocabulary.index2word

    def _set_params_from_kv(self):
        self.trainables.vectors = self.wv.vectors
        self.trainables.vector_size = self.wv.vector_size
        self.trainables.vectors_vocab = self.wv.vectors_vocab
        self.trainables.vectors_ngrams = self.wv.vectors_ngrams
        self.trainables.ngrams = self.wv.ngrams
        self.trainables.hash2index = self.wv.hash2index
        self.vocabulary.min_n = self.wv.min_n
        self.vocabulary.max_n = self.wv.max_n
        self.vocabulary.ngrams_word = self.wv.ngrams_word
        self.vocabulary.vocab = self.wv.vocab
        self.vocabulary.index2word = self.wv.index2word

    def _get_job_params(self):
        """Return the paramter required for each batch."""
        return word2vec_utils._get_job_params(self)

    def _update_job_params(self, job_params, progress, cur_epoch):
        return word2vec_utils._update_job_params(self, job_params, progress, cur_epoch)

    def _get_thread_working_mem(self):
        return word2vec_utils._get_thread_working_mem(self)

    def _raw_word_count(self, job):
        """Return the number of words in a given job."""
        return word2vec_utils._raw_word_count(job)

    def _set_train_params(self, **kwargs):
        self.trainables.alpha = self.alpha
        self.trainables.min_alpha = self.min_alpha
        return

    def _check_training_sanity(self, epochs=None, total_examples=None, total_words=None, **kwargs):
        return word2vec_utils._check_training_sanity(
            self, epochs=epochs, total_examples=total_examples, total_words=total_words)

    def _clear_post_train(self):
        """Resets certain properties of the model, post training. eg. `kv.syn0norm`"""
        self.wv.vectors_norm = None
        self.wv.vectors_vocab_norm = None
        self.wv.vectors_ngrams_norm = None

    def _do_train_job(self, sentences, alpha, inits):
        """Train a single batch of sentences. Return 2-tuple `(effective word count after
        ignoring unknown words and sentence length trimming, total word count)`.

        Parameters
        ----------
        sentences : iterable of iterables
            The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
        alpha : float
            The current learning rate.
        inits : (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Each worker's private work memory.

        Returns
        -------
        (int, int)
            Tuple of (effective word count after ignoring unknown words and sentence length trimming, total word count)

        """
        work, neu1 = inits
        tally = 0
        if self.sg:
            tally += train_batch_sg(self, sentences, alpha, work, neu1)
        else:
            tally += train_batch_cbow(self, sentences, alpha, work, neu1)

        return tally, self._raw_word_count(sentences)

    def train(self, sentences, total_examples=None, total_words=None,
              epochs=None, start_alpha=None, end_alpha=None,
              word_count=0, queue_factor=2, report_delay=1.0):
        """Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        For FastText, each sentence must be a list of unicode strings. (Subclasses may accept other examples.)

        To support linear learning-rate decay from (initial) alpha to min_alpha, and accurate
        progress-percentage logging, either total_examples (count of sentences) or total_words (count of
        raw words in sentences) **MUST** be provided (if the corpus is the same as was provided to
        :meth:`~gensim.models.fasttext.FastText.build_vocab()`, the count of examples in that corpus
        will be available in the model's :attr:`corpus_count` property).

        To avoid common mistakes around the model's ability to do multiple training passes itself, an
        explicit `epochs` argument **MUST** be provided. In the common and recommended case,
        where :meth:`~gensim.models.fasttext.FastText.train()` is only called once,
        the model's cached `iter` value should be supplied as `epochs` value.

        Parameters
        ----------
        sentences : iterable of iterables
            The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
        total_examples : int
            Count of sentences.
        total_words : int
            Count of raw words in sentences.
        epochs : int
            Number of iterations (epochs) over the corpus.
        start_alpha : float
            Initial learning rate.
        end_alpha : float
            Final learning rate. Drops linearly from `start_alpha`.
        word_count : int
            Count of words already trained. Set this to 0 for the usual
            case of training on all words in sentences.
        queue_factor : int
            Multiplier for size of queue (number of workers * queue_factor).
        report_delay : float
            Seconds to wait before reporting progress.

        Examples
        --------
        >>> from gensim.models import FastText
        >>> sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
        >>>
        >>> model = FastText(min_count=1)
        >>> model.build_vocab(sentences)
        >>> model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

        """

        super(FastText, self).train(
            sentences, total_examples=self.vocabulary.corpus_count, epochs=self.epochs,
            start_alpha=self.trainables.alpha, end_alpha=self.trainables.min_alpha)
        self.trainables.get_vocab_word_vecs(vocabulary=self.vocabulary)
        self._set_keyedvectors()

    def init_sims(self, replace=False):
        """
        init_sims() resides in KeyedVectors because it deals with syn0 mainly, but because syn1 is not an attribute
        of KeyedVectors, it has to be deleted in this class, and the normalizing of syn0 happens inside of KeyedVectors
        """
        if replace and hasattr(self.trainables, 'syn1'):
            del self.trainables.syn1
        return self.wv.init_sims(replace)

    def clear_sims(self):
        """
        Removes all L2-normalized vectors for words from the model.
        You will have to recompute them using init_sims method.
        """

        self._clear_post_train()

    @deprecated("Method will be removed in 4.0.0, use self.wv.__getitem__() instead")
    def __getitem__(self, words):
        """
        Deprecated. Use self.wv.__getitem__() instead.
        Refer to the documentation for `gensim.models.KeyedVectors.__getitem__`
        """
        return self.wv.__getitem__(words)

    @deprecated("Method will be removed in 4.0.0, use self.wv.__contains__() instead")
    def __contains__(self, word):
        """
        Deprecated. Use self.wv.__contains__() instead.
        Refer to the documentation for `gensim.models.KeyedVectors.__contains__`
        """
        return self.wv.__contains__(word)

    @classmethod
    def load_fasttext_format(cls, model_file, encoding='utf8'):
        """
        Load the input-hidden weight matrix from the fast text output files.

        Note that due to limitations in the FastText API, you cannot continue training
        with a model loaded this way, though you can query for word similarity etc.

        `model_file` is the path to the FastText output files.
        FastText outputs two model files - `/path/to/model.vec` and `/path/to/model.bin`

        Expected value for this example: `/path/to/model` or `/path/to/model.bin`,
        as gensim requires only `.bin` file to load entire fastText model.

        """
        model = cls()
        if not model_file.endswith('.bin'):
            model_file += '.bin'
        model.file_name = model_file
        model.load_binary_data(encoding=encoding)
        return model

    def load_binary_data(self, encoding='utf8'):
        """Loads data from the output binary file created by FastText training"""
        with utils.smart_open(self.file_name, 'rb') as f:
            self.load_model_params(f)
            self.load_dict(f, encoding=encoding)
            self.load_vectors(f)

    def load_model_params(self, file_handle):
        magic, version = self.struct_unpack(file_handle, '@2i')
        if magic == FASTTEXT_FILEFORMAT_MAGIC:  # newer format
            self.new_format = True
            dim, ws, epoch, min_count, neg, _, loss, model, bucket, minn, maxn, _, t = \
                self.struct_unpack(file_handle, '@12i1d')
        else:  # older format
            self.new_format = False
            dim = magic
            ws = version
            epoch, min_count, neg, _, loss, model, bucket, minn, maxn, _, t = self.struct_unpack(file_handle, '@10i1d')
        # Parameters stored by [Args::save](https://github.com/facebookresearch/fastText/blob/master/src/args.cc)
        self.trainables.vector_size = dim
        self.vector_size = dim
        self.window = ws
        self.epochs = epoch
        self.vocabulary.min_count = min_count
        self.negative = neg
        self.hs = loss == 1
        self.sg = model == 2
        self.trainables.bucket = bucket
        self.vocabulary.min_n = minn
        self.vocabulary.max_n = maxn
        self.vocabulary.sample = t

    def load_dict(self, file_handle, encoding='utf8'):
        vocab_size, nwords, nlabels = self.struct_unpack(file_handle, '@3i')
        # Vocab stored by [Dictionary::save](https://github.com/facebookresearch/fastText/blob/master/src/dictionary.cc)
        if nlabels > 0:
            raise NotImplementedError("Supervised fastText models are not supported")
        logger.info("loading %s words for fastText model from %s", vocab_size, self.file_name)

        self.struct_unpack(file_handle, '@1q')  # number of tokens
        if self.new_format:
            pruneidx_size, = self.struct_unpack(file_handle, '@q')
        for i in range(vocab_size):
            word_bytes = b''
            char_byte = file_handle.read(1)
            # Read vocab word
            while char_byte != b'\x00':
                word_bytes += char_byte
                char_byte = file_handle.read(1)
            word = word_bytes.decode(encoding)
            count, _ = self.struct_unpack(file_handle, '@qb')

            self.vocabulary.vocab[word] = Vocab(index=i, count=count)
            self.vocabulary.index2word.append(word)

        assert len(self.vocabulary.vocab) == nwords, (
            'mismatch between final vocab size ({} words), '
            'and expected number of words ({} words)'.format(len(self.vocabulary.vocab), nwords))
        if len(self.vocabulary.vocab) != vocab_size:
            # expecting to log this warning only for pretrained french vector, wiki.fr
            logger.warning(
                "mismatch between final vocab size (%s words), and expected vocab size (%s words)",
                len(self.vocabulary.vocab), vocab_size
            )

        if self.new_format:
            for j in range(pruneidx_size):
                self.struct_unpack(file_handle, '@2i')

    def load_vectors(self, file_handle):
        if self.new_format:
            self.struct_unpack(file_handle, '@?')  # bool quant_input in fasttext.cc
        num_vectors, dim = self.struct_unpack(file_handle, '@2q')
        # Vectors stored by [Matrix::save](https://github.com/facebookresearch/fastText/blob/master/src/matrix.cc)
        assert self.vector_size == dim, (
            'mismatch between vector size in model params ({}) and model vectors ({})'
            .format(self.vector_size, dim)
        )
        float_size = struct.calcsize('@f')
        if float_size == 4:
            dtype = np.dtype(np.float32)
        elif float_size == 8:
            dtype = np.dtype(np.float64)

        self.num_original_vectors = num_vectors
        self.trainables.vectors_ngrams = np.fromfile(file_handle, dtype=dtype, count=num_vectors * dim)
        self.trainables.vectors_ngrams = self.trainables.vectors_ngrams.reshape((num_vectors, dim))
        assert self.trainables.vectors_ngrams.shape == (self.trainables.bucket + len(self.vocabulary.vocab), self.vector_size), \
            'mismatch between actual weight matrix shape {} and expected shape {}'\
            .format(
                self.trainables.vectors_ngrams.shape, (self.trainables.bucket + len(self.wv.vocab), self.vector_size)
            )

        self.trainables.init_ngrams_post_load(self.file_name, vocabulary=self.vocabulary)
        self._clear_post_train()
        self._set_keyedvectors()

    def struct_unpack(self, file_handle, fmt):
        num_bytes = struct.calcsize(fmt)
        return struct.unpack(fmt, file_handle.read(num_bytes))

    def save(self, *args, **kwargs):
        """Save the model. This saved model can be loaded again using :func:`~gensim.models.fasttext.FastText.load`,
        which supports online training and getting vectors for out-of-vocabulary words.

        Parameters
        ----------
        fname : str
            Path to the file.

        """
        kwargs['ignore'] = kwargs.get('ignore', ['vectors_norm', 'vectors_vocab_norm', 'vectors_ngrams_norm'])
        super(FastText, self).save(*args, **kwargs)


class FastTextVocab(Word2VecVocab):
    def __init__(self, max_vocab_size=None, min_count=5, sample=1e-3, sorted_vocab=True, null_word=0,
                 min_n=3, max_n=6):
        super(FastTextVocab, self).__init__(
            max_vocab_size=max_vocab_size, min_count=min_count, sample=sample,
            sorted_vocab=sorted_vocab, null_word=null_word)
        self.min_n = min_n
        self.max_n = max_n
        self.ngrams_word = {}

    def prepare_vocab(self, update=False, keep_raw_vocab=False, trim_rule=None, min_count=None,
                      sample=None, dry_run=False):
        super(FastTextVocab, self).prepare_vocab(
            update=update, keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, min_count=min_count,
            sample=sample, dry_run=dry_run)
        self.build_ngrams(update=update)

    def build_ngrams(self, update=False):
        if not update:
            self.ngrams_word = {}
            for w, v in iteritems(self.vocab):
                self.ngrams_word[w] = compute_ngrams(w, self.min_n, self.max_n)
        else:
            for w, v in iteritems(self.vocab):
                self.ngrams_word[w] = compute_ngrams(w, self.min_n, self.max_n)


class FastTextTrainables(Word2VecTrainables):
    def __init__(self, vector_size=100, seed=1, alpha=0.025, min_alpha=0.0001, hs=0, negative=5,
                 hashfxn=hash, bucket=2000000):
        super(FastTextTrainables, self).__init__(
            vector_size=vector_size, seed=seed, alpha=alpha, min_alpha=min_alpha, hs=hs,
            negative=negative, hashfxn=hashfxn)
        self.bucket = bucket
        self.hash2index = {}
        self.vectors_vocab = None
        self.vectors_ngrams = None
        self.ngrams = {}

    def prepare_weights(self, update=False, vocabulary=None):
        super(FastTextTrainables, self).prepare_weights(update=update, vocabulary=vocabulary)
        self.init_ngrams_weights(update=update, vocabulary=vocabulary)

    def init_ngrams_weights(self, update=False, vocabulary=None):
        """Compute ngrams of all words present in vocabulary and stores vectors for only those ngrams.
        Vectors for other ngrams are initialized with a random uniform distribution in FastText.

        Parameters
        ----------
        update : bool
            If True, the new vocab words and their new ngrams word vectors are initialized
            with random uniform distribution and updated/added to the existing vocab word and ngram vectors.

        """
        if not update:
            self.ngrams = {}
            self.vectors_vocab = empty((len(vocabulary.vocab), self.vector_size), dtype=REAL)
            self.vectors_vocab_lockf = ones((len(vocabulary.vocab), self.vector_size), dtype=REAL)

            self.vectors_ngrams = empty((self.bucket, self.vector_size), dtype=REAL)
            self.vectors_ngrams_lockf = ones((self.bucket, self.vector_size), dtype=REAL)

            all_ngrams = []
            for w, ngrams in iteritems(vocabulary.ngrams_word):
                all_ngrams += ngrams

            all_ngrams = list(set(all_ngrams))
            self.num_ngram_vectors = len(all_ngrams)
            logger.info("Total number of ngrams is %d", len(all_ngrams))

            self.hash2index = {}
            ngram_indices = []
            new_hash_count = 0
            for i, ngram in enumerate(all_ngrams):
                ngram_hash = ft_hash(ngram) % self.bucket
                if ngram_hash in self.hash2index:
                    self.ngrams[ngram] = self.hash2index[ngram_hash]
                else:
                    ngram_indices.append(ngram_hash % self.bucket)
                    self.hash2index[ngram_hash] = new_hash_count
                    self.ngrams[ngram] = self.hash2index[ngram_hash]
                    new_hash_count = new_hash_count + 1

            self.vectors_ngrams = self.vectors_ngrams.take(ngram_indices, axis=0)
            self.vectors_ngrams_lockf = self.vectors_ngrams_lockf.take(ngram_indices, axis=0)
            self.reset_ngram_weights(vocabulary=vocabulary)
        else:
            new_ngrams = []
            for w, ngrams in iteritems(vocabulary.ngrams_word):
                new_ngrams += [ng for ng in ngrams if ng not in self.ngrams]

            new_ngrams = list(set(new_ngrams))
            self.num_ngram_vectors += len(new_ngrams)
            logger.info("Number of new ngrams is %d", len(new_ngrams))
            new_hash_count = 0
            for i, ngram in enumerate(new_ngrams):
                ngram_hash = ft_hash(ngram) % self.bucket
                if ngram_hash not in self.hash2index:
                    self.hash2index[ngram_hash] = new_hash_count + self.old_hash2index_len
                    self.ngrams[ngram] = self.hash2index[ngram_hash]
                    new_hash_count = new_hash_count + 1
                else:
                    self.ngrams[ngram] = self.hash2index[ngram_hash]

            rand_obj = np.random
            rand_obj.seed(self.seed)
            new_vocab_rows = rand_obj.uniform(
                -1.0 / self.vector_size, 1.0 / self.vector_size,
                (len(vocabulary.vocab) - vocabulary.old_vocab_len, self.vector_size)
            ).astype(REAL)
            new_vocab_lockf_rows = ones((len(vocabulary.vocab) - vocabulary.old_vocab_len,
                self.vector_size), dtype=REAL)
            new_ngram_rows = rand_obj.uniform(
                -1.0 / self.vector_size, 1.0 / self.vector_size,
                (len(self.hash2index) - self.old_hash2index_len, self.vector_size)
            ).astype(REAL)
            new_ngram_lockf_rows = ones(
                (len(self.hash2index) - self.old_hash2index_len,
                self.vector_size),
                dtype=REAL)

            self.vectors_vocab = vstack([self.vectors_vocab, new_vocab_rows])
            self.vectors_vocab_lockf = vstack([self.vectors_vocab_lockf, new_vocab_lockf_rows])
            self.vectors_ngrams = vstack([self.vectors_ngrams, new_ngram_rows])
            self.vectors_ngrams_lockf = vstack([self.vectors_ngrams_lockf, new_ngram_lockf_rows])

    def reset_ngram_weights(self, vocabulary=None):
        """Reset all projection weights to an initial (untrained) state,
        but keep the existing vocabulary and their ngrams.

        """
        rand_obj = np.random
        rand_obj.seed(self.seed)
        for index in range(len(vocabulary.vocab)):
            self.vectors_vocab[index] = rand_obj.uniform(
                -1.0 / self.vector_size, 1.0 / self.vector_size, self.vector_size
            ).astype(REAL)
        for index in range(len(self.hash2index)):
            self.vectors_ngrams[index] = rand_obj.uniform(
                -1.0 / self.vector_size, 1.0 / self.vector_size, self.vector_size
            ).astype(REAL)

    def get_vocab_word_vecs(self, vocabulary=None):
        """Calculate vectors for words in vocabulary and stores them in `wv.syn0`."""
        for w, v in vocabulary.vocab.items():
            word_vec = np.copy(self.vectors_vocab[v.index])
            ngrams = vocabulary.ngrams_word[w]
            ngram_weights = self.vectors_ngrams
            for ngram in ngrams:
                word_vec += ngram_weights[self.ngrams[ngram]]
            word_vec /= (len(ngrams) + 1)
            self.vectors[v.index] = word_vec

    def init_ngrams_post_load(self, file_name, vocabulary=None):  # put in trainables
        """
        Computes ngrams of all words present in vocabulary and stores vectors for only those ngrams.
        Vectors for other ngrams are initialized with a random uniform distribution in FastText. These
        vectors are discarded here to save space.

        """
        all_ngrams = []
        self.vectors = np.zeros((len(vocabulary.vocab), self.vector_size), dtype=REAL)

        for w, vocab in vocabulary.vocab.items():
            all_ngrams += compute_ngrams(w, vocabulary.min_n, vocabulary.max_n)
            self.vectors[vocab.index] += np.array(self.vectors_ngrams[vocab.index])

        all_ngrams = set(all_ngrams)
        # print all_ngrams
        self.num_ngram_vectors = len(all_ngrams)
        ngram_indices = []
        for i, ngram in enumerate(all_ngrams):
            ngram_hash = ft_hash(ngram)
            ngram_indices.append(len(vocabulary.vocab) + ngram_hash % self.bucket)
            self.ngrams[ngram] = i
        self.vectors_ngrams = self.vectors_ngrams.take(ngram_indices, axis=0)

        ngram_weights = self.vectors_ngrams

        logger.info(
            "loading weights for %s words for fastText model from %s",
            len(vocabulary.vocab), file_name
        )

        for w, vocab in vocabulary.vocab.items():
            word_ngrams = compute_ngrams(w, vocabulary.min_n, vocabulary.max_n)
            for word_ngram in word_ngrams:
                self.vectors[vocab.index] += np.array(ngram_weights[self.ngrams[word_ngram]])

            self.vectors[vocab.index] /= (len(word_ngrams) + 1)
        logger.info(
            "loaded %s weight matrix for fastText model from %s",
            self.vectors.shape, file_name
        )

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['vectors_norm', 'vectors_vocab_norm', 'vectors_ngrams_norm'])
        super(Word2VecTrainables, self).save(*args, **kwargs)
