#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <manneshiva@gmail.com>
# Copyright (C) 2018 RaRe Technologies s.r.o.
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Learn paragraph and document embeddings via the distributed memory and distributed bag of words models from
`Quoc Le and Tomas Mikolov: "Distributed Representations of Sentences and Documents"
<http://arxiv.org/pdf/1405.4053v2.pdf>`_.

The algorithms use either hierarchical softmax or negative sampling; see
`Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean: "Efficient Estimation of Word Representations in
Vector Space, in Proceedings of Workshop at ICLR, 2013" <https://arxiv.org/pdf/1301.3781.pdf>`_ and
`Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean: "Distributed Representations of Words
and Phrases and their Compositionality. In Proceedings of NIPS, 2013"
<https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf>`_.

For a usage example, see the `Doc2vec tutorial
<https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb>`_.

**Make sure you have a C compiler before installing Gensim, to use the optimized doc2vec routines** (70x speedup
compared to plain NumPy implementation <https://rare-technologies.com/parallelizing-word2vec-in-python/>`_).


Examples
--------

Initialize & train a model

>>> from gensim.test.utils import common_texts
>>> from gensim.models.doc2vec import Doc2Vec, TaggedDocument
>>>
>>> documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
>>> model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

Persist a model to disk

>>> from gensim.test.utils import get_tmpfile
>>>
>>> fname = get_tmpfile("my_doc2vec_model")
>>>
>>> model.save(fname)
>>> model = Doc2Vec.load(fname)  # you can continue training with the loaded model!

If you're finished training a model (=no more updates, only querying, reduce memory usage), you can do

>>> model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

Infer vector for new document

>>> vector = model.infer_vector(["system", "response"])

"""

import logging
import os
import warnings

try:
    from queue import Queue
except ImportError:
    from Queue import Queue  # noqa:F401

from collections import namedtuple, defaultdict
from timeit import default_timer

from numpy import zeros, float32 as REAL, empty, ones, \
    memmap as np_memmap, vstack, integer, dtype, sum as np_sum, add as np_add, repeat as np_repeat, concatenate


from gensim.utils import call_on_class_only
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.models.word2vec import Word2VecKeyedVectors, Word2VecVocab, Word2VecTrainables, train_cbow_pair,\
    train_sg_pair, train_batch_sg
from six.moves import xrange
from six import string_types, integer_types, itervalues
from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from gensim.models.keyedvectors import Doc2VecKeyedVectors
from types import GeneratorType
from gensim.utils import deprecated

logger = logging.getLogger(__name__)

try:
    from gensim.models.doc2vec_inner import train_document_dbow, train_document_dm, train_document_dm_concat
    from gensim.models.word2vec_inner import FAST_VERSION  # blas-adaptation shared from word2vec

except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1

    def train_document_dbow(model, doc_words, doctag_indexes, alpha, work=None,
                            train_words=False, learn_doctags=True, learn_words=True, learn_hidden=True,
                            word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
        """Update distributed bag of words model ("PV-DBOW") by training on a single document.

        Called internally from :meth:`~gensim.models.doc2vec.Doc2Vec.train` and
        :meth:`~gensim.models.doc2vec.Doc2Vec.infer_vector`.

        Notes
        -----
        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from :mod:`gensim.models.doc2vec_inner` instead.

        Parameters
        ----------
        model : :class:`~gensim.models.doc2vec.Doc2Vec`
            The model to train.
        doc_words : list of str
            The input document as a list of words to be used for training. Each word will be looked up in
            the model's vocabulary.
        doctag_indexes : list of int
            Indices into `doctag_vectors` used to obtain the tags of the document.
        alpha : float
            Learning rate.
        work : np.ndarray
            Private working memory for each worker.
        train_words : bool, optional
            Word vectors will be updated exactly as per Word2Vec skip-gram training only if **both**
            `learn_words` and `train_words` are set to True.
        learn_doctags : bool, optional
            Whether the tag vectors should be updated.
        learn_words : bool, optional
            Word vectors will be updated exactly as per Word2Vec skip-gram training only if **both**
            `learn_words` and `train_words` are set to True.
        learn_hidden : bool, optional
            Whether or not the weights of the hidden layer will be updated.
        word_vectors : object, optional
            UNUSED.
        word_locks : object, optional
            UNUSED.
        doctag_vectors : list of list of float, optional
            Vector representations of the tags. If None, these will be retrieved from the model.
        doctag_locks : list of float, optional
            The lock factors for each tag.

        Returns
        -------
        int
            Number of words in the input document.

        """
        if doctag_vectors is None:
            doctag_vectors = model.docvecs.doctag_syn0
        if doctag_locks is None:
            doctag_locks = model.docvecs.doctag_syn0_lockf

        if train_words and learn_words:
            train_batch_sg(model, [doc_words], alpha, work)
        for doctag_index in doctag_indexes:
            for word in doc_words:
                train_sg_pair(
                    model, word, doctag_index, alpha, learn_vectors=learn_doctags, learn_hidden=learn_hidden,
                    context_vectors=doctag_vectors, context_locks=doctag_locks
                )

        return len(doc_words)

    def train_document_dm(model, doc_words, doctag_indexes, alpha, work=None, neu1=None,
                          learn_doctags=True, learn_words=True, learn_hidden=True,
                          word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None):
        """Update distributed memory model ("PV-DM") by training on a single document.

        Called internally from :meth:`~gensim.models.doc2vec.Doc2Vec.train` and
        :meth:`~gensim.models.doc2vec.Doc2Vec.infer_vector`. This method implements
        the DM model with a projection (input) layer that is either the sum or mean of
        the context vectors, depending on the model's `dm_mean` configuration field.

        Notes
        -----
        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from :mod:`gensim.models.doc2vec_inner` instead.

        Parameters
        ----------
        model : :class:`~gensim.models.doc2vec.Doc2Vec`
            The model to train.
        doc_words : list of str
            The input document as a list of words to be used for training. Each word will be looked up in
            the model's vocabulary.
        doctag_indexes : list of int
            Indices into `doctag_vectors` used to obtain the tags of the document.
        alpha : float
            Learning rate.
        work : object
            UNUSED.
        neu1 : object
            UNUSED.
        learn_doctags : bool, optional
            Whether the tag vectors should be updated.
        learn_words : bool, optional
            Word vectors will be updated exactly as per Word2Vec skip-gram training only if **both**
            `learn_words` and `train_words` are set to True.
        learn_hidden : bool, optional
            Whether or not the weights of the hidden layer will be updated.
        word_vectors : iterable of list of float, optional
            Vector representations of each word in the model's vocabulary.
        word_locks : list of float, optional
            Lock factors for each word in the vocabulary.
        doctag_vectors : list of list of float, optional
            Vector representations of the tags. If None, these will be retrieved from the model.
        doctag_locks : list of float, optional
            The lock factors for each tag.

        Returns
        -------
        int
            Number of words in the input document that were actually used for training (they were found in the
            vocabulary and they were not discarded by negative sampling).

        """
        if word_vectors is None:
            word_vectors = model.wv.syn0
        if word_locks is None:
            word_locks = model.syn0_lockf
        if doctag_vectors is None:
            doctag_vectors = model.docvecs.doctag_syn0
        if doctag_locks is None:
            doctag_locks = model.docvecs.doctag_syn0_lockf

        word_vocabs = [model.wv.vocab[w] for w in doc_words if w in model.wv.vocab and
                       model.wv.vocab[w].sample_int > model.random.rand() * 2 ** 32]

        for pos, word in enumerate(word_vocabs):
            reduced_window = model.random.randint(model.window)  # `b` in the original doc2vec code
            start = max(0, pos - model.window + reduced_window)
            window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
            word2_indexes = [word2.index for pos2, word2 in window_pos if pos2 != pos]
            l1 = np_sum(word_vectors[word2_indexes], axis=0) + np_sum(doctag_vectors[doctag_indexes], axis=0)
            count = len(word2_indexes) + len(doctag_indexes)
            if model.cbow_mean and count > 1:
                l1 /= count
            neu1e = train_cbow_pair(model, word, word2_indexes, l1, alpha,
                                    learn_vectors=False, learn_hidden=learn_hidden)
            if not model.cbow_mean and count > 1:
                neu1e /= count
            if learn_doctags:
                for i in doctag_indexes:
                    doctag_vectors[i] += neu1e * doctag_locks[i]
            if learn_words:
                for i in word2_indexes:
                    word_vectors[i] += neu1e * word_locks[i]

        return len(word_vocabs)

    def train_document_dm_concat(model, doc_words, doctag_indexes, alpha, work=None, neu1=None, learn_doctags=True,
                                 learn_words=True, learn_hidden=True, word_vectors=None, word_locks=None,
                                 doctag_vectors=None, doctag_locks=None):
        """Update distributed memory model ("PV-DM") by training on a single document, using a
        concatenation of the context window word vectors (rather than a sum or average). This
        might be slower since the input at each batch will be significantly larger.

        Called internally from :meth:`~gensim.models.doc2vec.Doc2Vec.train` and
        :meth:`~gensim.models.doc2vec.Doc2Vec.infer_vector`.

        Notes
        -----
        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from :mod:`gensim.models.doc2vec_inner` instead.

        Parameters
        ----------
        model : :class:`~gensim.models.doc2vec.Doc2Vec`
            The model to train.
        doc_words : list of str
            The input document as a list of words to be used for training. Each word will be looked up in
            the model's vocabulary.
        doctag_indexes : list of int
            Indices into `doctag_vectors` used to obtain the tags of the document.
        alpha : float
            Learning rate.
        work : object
            UNUSED.
        neu1 : object
            UNUSED.
        learn_doctags : bool, optional
            Whether the tag vectors should be updated.
        learn_words : bool, optional
            Word vectors will be updated exactly as per Word2Vec skip-gram training only if **both**
            `learn_words` and `train_words` are set to True.
        learn_hidden : bool, optional
            Whether or not the weights of the hidden layer will be updated.
        word_vectors : iterable of list of float, optional
            Vector representations of each word in the model's vocabulary.
        word_locks : listf of float, optional
            Lock factors for each word in the vocabulary.
        doctag_vectors : list of list of float, optional
            Vector representations of the tags. If None, these will be retrieved from the model.
        doctag_locks : list of float, optional
            The lock factors for each tag.

        Returns
        -------
        int
            Number of words in the input document that were actually used for training (they were found in the
            vocabulary and they were not discarded by negative sampling).

        """
        if word_vectors is None:
            word_vectors = model.wv.syn0
        if word_locks is None:
            word_locks = model.syn0_lockf
        if doctag_vectors is None:
            doctag_vectors = model.docvecs.doctag_syn0
        if doctag_locks is None:
            doctag_locks = model.docvecs.doctag_syn0_lockf

        word_vocabs = [model.wv.vocab[w] for w in doc_words if w in model.wv.vocab and
                       model.wv.vocab[w].sample_int > model.random.rand() * 2 ** 32]
        doctag_len = len(doctag_indexes)
        if doctag_len != model.dm_tag_count:
            return 0  # skip doc without expected number of doctag(s) (TODO: warn/pad?)

        null_word = model.wv.vocab['\0']
        pre_pad_count = model.window
        post_pad_count = model.window
        padded_document_indexes = (
                (pre_pad_count * [null_word.index])  # pre-padding
                + [word.index for word in word_vocabs if word is not None]  # elide out-of-Vocabulary words
                + (post_pad_count * [null_word.index])  # post-padding
        )

        for pos in range(pre_pad_count, len(padded_document_indexes) - post_pad_count):
            word_context_indexes = (
                    padded_document_indexes[(pos - pre_pad_count): pos]  # preceding words
                    + padded_document_indexes[(pos + 1):(pos + 1 + post_pad_count)]  # following words
            )
            predict_word = model.wv.vocab[model.wv.index2word[padded_document_indexes[pos]]]
            # numpy advanced-indexing copies; concatenate, flatten to 1d
            l1 = concatenate((doctag_vectors[doctag_indexes], word_vectors[word_context_indexes])).ravel()
            neu1e = train_cbow_pair(model, predict_word, None, l1, alpha,
                                    learn_hidden=learn_hidden, learn_vectors=False)

            # filter by locks and shape for addition to source vectors
            e_locks = concatenate((doctag_locks[doctag_indexes], word_locks[word_context_indexes]))
            neu1e_r = (neu1e.reshape(-1, model.vector_size)
                       * np_repeat(e_locks, model.vector_size).reshape(-1, model.vector_size))

            if learn_doctags:
                np_add.at(doctag_vectors, doctag_indexes, neu1e_r[:doctag_len])
            if learn_words:
                np_add.at(word_vectors, word_context_indexes, neu1e_r[doctag_len:])

        return len(padded_document_indexes) - pre_pad_count - post_pad_count


class TaggedDocument(namedtuple('TaggedDocument', 'words tags')):
    """Represents a document along with a tag, input document format for :class:`~gensim.models.doc2vec.Doc2Vec`.

    A single document, made up of `words` (a list of unicode string tokens) and `tags` (a list of tokens).
    Tags may be one or more unicode string tokens, but typical practice (which will also be the most memory-efficient)
    is for the tags list to include a unique integer id as the only tag.

    Replaces "sentence as a list of words" from :class:`gensim.models.word2vec.Word2Vec`.

    """
    def __str__(self):
        """Human readable representation of the object's state, used for debugging.

        Returns
        -------
        str
           Human readable representation of the object's state (words and tags).

        """
        return '%s(%s, %s)' % (self.__class__.__name__, self.words, self.tags)


# for compatibility
@deprecated("Class will be removed in 4.0.0, use TaggedDocument instead")
class LabeledSentence(TaggedDocument):
    """Deprecated, use :class:`~gensim.models.doc2vec.TaggedDocument` instead."""
    pass


class Doctag(namedtuple('Doctag', 'offset, word_count, doc_count')):
    """A string document tag discovered during the initial vocabulary scan.
    The document-vector equivalent of a Vocab object.

    Will not be used if all presented document tags are ints.

    The offset is only the true index into the `doctags_syn0`/`doctags_syn0_lockf`
    if-and-only-if no raw-int tags were used.
    If any raw-int tags were used, string :class:`~gensim.models.doc2vec.Doctag` vectors begin at index
    `(max_rawint + 1)`, so the true index is `(rawint_index + 1 + offset)`.

    See Also
    --------
    :meth:`~gensim.models.keyedvectors.Doc2VecKeyedVectors._index_to_doctag`

    """
    __slots__ = ()

    def repeat(self, word_count):
        return self._replace(word_count=self.word_count + word_count, doc_count=self.doc_count + 1)


class Doc2Vec(BaseWordEmbeddingsModel):
    """Class for training, using and evaluating neural networks described in
    `Distributed Representations of Sentences and Documents <http://arxiv.org/abs/1405.4053v2>`_.

    Some important internal attributes are the following:

    Attributes
    ----------
    wv : :class:`~gensim.models.keyedvectors.Word2VecKeyedVectors`
        This object essentially contains the mapping between words and embeddings. After training, it can be used
        directly to query those embeddings in various ways. See the module level docstring for examples.

    docvecs : :class:`~gensim.models.keyedvectors.Doc2VecKeyedVectors`
        This object contains the paragraph vectors. Remember that the only difference between this model and
        :class:`~gensim.models.word2vec.Word2Vec` is that besides the word vectors we also include paragraph embeddings
        to capture the paragraph.

        In this way we can capture the difference between the same word used in a different context.
        For example we now have a different representation of the word "leaves" in the following two sentences ::

            1. Manos leaves the office every day at 18:00 to catch his train
            2. This season is called Fall, because leaves fall from the trees.

        In a plain :class:`~gensim.models.word2vec.Word2Vec` model the word would have exactly the same representation
        in both sentences, in :class:`~gensim.models.doc2vec.Doc2Vec` it will not.

    vocabulary : :class:`~gensim.models.doc2vec.Doc2VecVocab`
        This object represents the vocabulary (sometimes called Dictionary in gensim) of the model.
        Besides keeping track of all unique words, this object provides extra functionality, such as
        sorting words by frequency, or discarding extremely rare words.

    trainables : :class:`~gensim.models.doc2vec.Doc2VecTrainables`
        This object represents the inner shallow neural network used to train the embeddings. The semantics of the
        network differ slightly in the two available training modes (CBOW or SG) but you can think of it as a NN with
        a single projection and hidden layer which we train on the corpus. The weights are then used as our embeddings
        The only addition to the underlying NN used in :class:`~gensim.models.word2vec.Word2Vec` is that the input
        includes not only the word vectors of each word in the context, but also the paragraph vector.

    """
    def __init__(self, documents=None, dm_mean=None, dm=1, dbow_words=0, dm_concat=0, dm_tag_count=1,
                 docvecs=None, docvecs_mapfile=None, comment=None, trim_rule=None, callbacks=(), **kwargs):
        """

        Parameters
        ----------
        documents : iterable of list of :class:`~gensim.models.doc2vec.TaggedDocument`, optional
            Input corpus, can be simply a list of elements, but for larger corpora,consider an iterable that streams
            the documents directly from disk/network. If you don't supply `documents`, the model is
            left uninitialized -- use if you plan to initialize it in some other way.
        dm : {1,0}, optional
            Defines the training algorithm. If `dm=1`, 'distributed memory' (PV-DM) is used.
            Otherwise, `distributed bag of words` (PV-DBOW) is employed.
        size : int, optional
            Dimensionality of the feature vectors.
        window : int, optional
            The maximum distance between the current and predicted word within a sentence.
        alpha : float, optional
            The initial learning rate.
        min_alpha : float, optional
            Learning rate will linearly drop to `min_alpha` as training progresses.
        seed : int, optional
            Seed for the random number generator. Initial vectors for each word are seeded with a hash of
            the concatenation of word + `str(seed)`. Note that for a fully deterministically-reproducible run,
            you must also limit the model to a single worker thread (`workers=1`), to eliminate ordering jitter
            from OS thread scheduling.
            In Python 3, reproducibility between interpreter launches also requires use of the `PYTHONHASHSEED`
            environment variable to control hash randomization.
        min_count : int, optional
            Ignores all words with total frequency lower than this.
        max_vocab_size : int, optional
            Limits the RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
            Set to `None` for no limit.
        sample : float, optional
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        workers : int, optional
            Use these many worker threads to train the model (=faster training with multicore machines).
        iter : int, optional
            Number of iterations (epochs) over the corpus.
        hs : {1,0}, optional
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
        negative : int, optional
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        ns_exponent : float, optional
            The exponent used to shape the negative sampling distribution. A value of 1.0 samples exactly in proportion
            to the frequencies, 0.0 samples all words equally, while a negative value samples low-frequency words more
            than high-frequency words. The popular default value of 0.75 was chosen by the original Word2Vec paper.
            More recently, in https://arxiv.org/abs/1804.04212, Caselles-Dupré, Lesaint, & Royo-Letelier suggest that
            other values may perform better for recommendation applications.
        dm_mean : {1,0}, optional
            If 0 , use the sum of the context word vectors. If 1, use the mean.
            Only applies when `dm` is used in non-concatenative mode.
        dm_concat : {1,0}, optional
            If 1, use concatenation of context vectors rather than sum/average;
            Note concatenation results in a much-larger model, as the input
            is no longer the size of one (sampled or arithmetically combined) word vector, but the
            size of the tag(s) and all words in the context strung together.
        dm_tag_count : int, optional
            Expected constant number of document tags per document, when using
            dm_concat mode.
        dbow_words : {1,0}, optional
            If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW
            doc-vector training; If 0, only trains doc-vectors (faster).
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during current method call and is not stored as part
            of the model.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.

        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`, optional
            List of callbacks that need to be executed/run at specific stages during training.

        """
        if 'sentences' in kwargs:
            raise DeprecationWarning(
                "Parameter 'sentences' was renamed to 'documents', and will be removed in 4.0.0, "
                "use 'documents' instead."
            )

        if 'iter' in kwargs:
            warnings.warn("The parameter `iter` is deprecated, will be removed in 4.0.0, use `epochs` instead.")
            kwargs['epochs'] = kwargs['iter']

        if 'size' in kwargs:
            warnings.warn("The parameter `size` is deprecated, will be removed in 4.0.0, use `vector_size` instead.")
            kwargs['vector_size'] = kwargs['size']

        super(Doc2Vec, self).__init__(
            sg=(1 + dm) % 2,
            null_word=dm_concat,
            callbacks=callbacks,
            fast_version=FAST_VERSION,
            **kwargs)

        self.load = call_on_class_only

        if dm_mean is not None:
            self.cbow_mean = dm_mean

        self.dbow_words = int(dbow_words)
        self.dm_concat = int(dm_concat)
        self.dm_tag_count = int(dm_tag_count)

        kwargs['null_word'] = dm_concat
        vocabulary_keys = ['max_vocab_size', 'min_count', 'sample', 'sorted_vocab', 'null_word', 'ns_exponent']
        vocabulary_kwargs = dict((k, kwargs[k]) for k in vocabulary_keys if k in kwargs)
        self.vocabulary = Doc2VecVocab(**vocabulary_kwargs)

        trainables_keys = ['seed', 'hashfxn', 'window']
        trainables_kwargs = dict((k, kwargs[k]) for k in trainables_keys if k in kwargs)
        self.trainables = Doc2VecTrainables(
            dm=dm, dm_concat=dm_concat, dm_tag_count=dm_tag_count,
            vector_size=self.vector_size, **trainables_kwargs)

        self.wv = Word2VecKeyedVectors(self.vector_size)
        self.docvecs = docvecs or Doc2VecKeyedVectors(self.vector_size, docvecs_mapfile)

        self.comment = comment
        if documents is not None:
            if isinstance(documents, GeneratorType):
                raise TypeError("You can't pass a generator as the documents argument. Try an iterator.")
            self.build_vocab(documents, trim_rule=trim_rule)
            self.train(
                documents, total_examples=self.corpus_count, epochs=self.epochs,
                start_alpha=self.alpha, end_alpha=self.min_alpha, callbacks=callbacks)

    @property
    def dm(self):
        """Indicates whether 'distributed memory' (PV-DM) will be used, else 'distributed bag of words'
        (PV-DBOW) is used.

        """
        return not self.sg  # opposite of SG

    @property
    def dbow(self):
        """Indicates whether 'distributed bag of words' (PV-DBOW) will be used, else 'distributed memory'
        (PV-DM) is used.

        """
        return self.sg  # same as SG

    def _set_train_params(self, **kwargs):
        pass

    def _clear_post_train(self):
        """Alias for :meth:`~gensim.models.doc2vec.Doc2Vec.clear_sims`."""
        self.clear_sims()

    def clear_sims(self):
        """Resets the current word vectors. """
        self.wv.vectors_norm = None
        self.wv.vectors_docs_norm = None

    def reset_from(self, other_model):
        """Copy shareable data structures from another (possibly pre-trained) model.

        Parameters
        ----------
        other_model : :class:`~gensim.models.doc2vec.Doc2Vec`
            Other model whose internal data structures will be copied over to the current object.

        """
        self.wv.vocab = other_model.wv.vocab
        self.wv.index2word = other_model.wv.index2word
        self.vocabulary.cum_table = other_model.vocabulary.cum_table
        self.corpus_count = other_model.corpus_count
        self.docvecs.count = other_model.docvecs.count
        self.docvecs.doctags = other_model.docvecs.doctags
        self.docvecs.offset2doctag = other_model.docvecs.offset2doctag
        self.trainables.reset_weights(self.hs, self.negative, self.wv, self.docvecs)

    def _do_train_job(self, job, alpha, inits):
        """Train model using `job` data.

        Parameters
        ----------
        job : iterable of list of :class:`~gensim.models.doc2vec.TaggedDocument`
            The corpus chunk to be used for training this batch.
        alpha : float
            Learning rate to be used for training this batch.
        inits : (np.ndarray, np.ndarray)
            Each worker threads private work memory.

        Returns
        -------
        (int, int)
             2-tuple (effective word count after ignoring unknown words and sentence length trimming, total word count).

        """
        work, neu1 = inits
        tally = 0
        for doc in job:
            doctag_indexes = self.vocabulary.indexed_doctags(doc.tags, self.docvecs)
            doctag_vectors = self.docvecs.vectors_docs
            doctag_locks = self.trainables.vectors_docs_lockf
            if self.sg:
                tally += train_document_dbow(
                    self, doc.words, doctag_indexes, alpha, work, train_words=self.dbow_words,
                    doctag_vectors=doctag_vectors, doctag_locks=doctag_locks
                )
            elif self.dm_concat:
                tally += train_document_dm_concat(
                    self, doc.words, doctag_indexes, alpha, work, neu1,
                    doctag_vectors=doctag_vectors, doctag_locks=doctag_locks
                )
            else:
                tally += train_document_dm(
                    self, doc.words, doctag_indexes, alpha, work, neu1,
                    doctag_vectors=doctag_vectors, doctag_locks=doctag_locks
                )
        return tally, self._raw_word_count(job)

    def train(self, documents, total_examples=None, total_words=None,
              epochs=None, start_alpha=None, end_alpha=None,
              word_count=0, queue_factor=2, report_delay=1.0, callbacks=()):
        """Update the model's neural weights.

        To support linear learning-rate decay from (initial) `alpha` to `min_alpha`, and accurate
        progress-percentage logging, either `total_examples` (count of sentences) or `total_words` (count of
        raw words in sentences) **MUST** be provided. If `sentences` is the same corpus
        that was provided to :meth:`~gensim.models.word2vec.Word2Vec.build_vocab` earlier,
        you can simply use `total_examples=self.corpus_count`.

        To avoid common mistakes around the model's ability to do multiple training passes itself, an
        explicit `epochs` argument **MUST** be provided. In the common and recommended case
        where :meth:`~gensim.models.word2vec.Word2Vec.train` is only called once,
        you can set `epochs=self.iter`.

        Parameters
        ----------
        documents : iterable of list of :class:`~gensim.models.doc2vec.TaggedDocument`
            Can be simply a list of elements, but for larger corpora,consider an iterable that streams
            the documents directly from disk/network. If you don't supply `documents`, the model is
            left uninitialized -- use if you plan to initialize it in some other way.
        total_examples : int, optional
            Count of sentences.
        total_words : int, optional
            Count of raw words in documents.
        epochs : int, optional
            Number of iterations (epochs) over the corpus.
        start_alpha : float, optional
            Initial learning rate. If supplied, replaces the starting `alpha` from the constructor,
            for this one call to `train`.
            Use only if making multiple calls to `train`, when you want to manage the alpha learning-rate yourself
            (not recommended).
        end_alpha : float, optional
            Final learning rate. Drops linearly from `start_alpha`.
            If supplied, this replaces the final `min_alpha` from the constructor, for this one call to
            :meth:`~gensim.models.doc2vec.Doc2Vec.train`.
            Use only if making multiple calls to :meth:`~gensim.models.doc2vec.Doc2Vec.train`, when you want to manage
            the alpha learning-rate yourself (not recommended).
        word_count : int, optional
            Count of words already trained. Set this to 0 for the usual
            case of training on all words in sentences.
        queue_factor : int, optional
            Multiplier for size of queue (number of workers * queue_factor).
        report_delay : float, optional
            Seconds to wait before reporting progress.
        callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`, optional
            List of callbacks that need to be executed/run at specific stages during training.

        """
        super(Doc2Vec, self).train(
            documents, total_examples=total_examples, total_words=total_words,
            epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha, word_count=word_count,
            queue_factor=queue_factor, report_delay=report_delay, callbacks=callbacks)

    def _raw_word_count(self, job):
        """Get the number of words in a given job.

        Parameters
        ----------
        job : iterable of list of :class:`~gensim.models.doc2vec.TaggedDocument`
            Corpus chunk.

        Returns
        -------
        int
            Number of raw words in the corpus chunk.

        """
        return sum(len(sentence.words) for sentence in job)

    def estimated_lookup_memory(self):
        """Get estimated memory for tag lookup, 0 if using pure int tags.

        Returns
        -------
        int
            The estimated RAM required to look up a tag in bytes.

        """
        return 60 * len(self.docvecs.offset2doctag) + 140 * len(self.docvecs.doctags)

    def infer_vector(self, doc_words, alpha=0.1, min_alpha=0.0001, steps=5):
        """Infer a vector for given post-bulk training document.

        Notes
        -----
        Subsequent calls to this function may infer different representations for the same document.
        For a more stable representation, increase the number of steps to assert a stricket convergence.

        Parameters
        ----------
        doc_words : list of str
            A document for which the vector representation will be inferred.
        alpha : float, optional
            The initial learning rate.
        min_alpha : float, optional
            Learning rate will linearly drop to `min_alpha` as training progresses.
        steps : int, optional
            Number of times to train the new document. A higher value may slow down training, but it will result in more
            stable representations.

        Returns
        -------
        np.ndarray
            The inferred paragraph vector for the new document.

        """
        doctag_vectors, doctag_locks = self.trainables.get_doctag_trainables(doc_words, self.docvecs.vector_size)
        doctag_indexes = [0]
        work = zeros(self.trainables.layer1_size, dtype=REAL)
        if not self.sg:
            neu1 = matutils.zeros_aligned(self.trainables.layer1_size, dtype=REAL)

        alpha_delta = (alpha - min_alpha) / (steps - 1)

        for i in range(steps):
            if self.sg:
                train_document_dbow(
                    self, doc_words, doctag_indexes, alpha, work,
                    learn_words=False, learn_hidden=False, doctag_vectors=doctag_vectors, doctag_locks=doctag_locks
                )
            elif self.dm_concat:
                train_document_dm_concat(
                    self, doc_words, doctag_indexes, alpha, work, neu1,
                    learn_words=False, learn_hidden=False, doctag_vectors=doctag_vectors, doctag_locks=doctag_locks
                )
            else:
                train_document_dm(
                    self, doc_words, doctag_indexes, alpha, work, neu1,
                    learn_words=False, learn_hidden=False, doctag_vectors=doctag_vectors, doctag_locks=doctag_locks
                )
            alpha -= alpha_delta

        return doctag_vectors[0]

    def __getitem__(self, tag):
        """Get the vector representation of (possible multi-term) tag.

        Parameters
        ----------
        tag : {str, int, list of str, list of int}
            The tag (or tags) to be looked up in the model.

        Returns
        -------
        np.ndarray
            The vector representations of each tag as a matrix (will be 1D if `tag` was a single tag)

        """
        if isinstance(tag, string_types + integer_types + (integer,)):
            if tag not in self.wv.vocab:
                return self.docvecs[tag]
            return self.wv[tag]
        return vstack([self[i] for i in tag])

    def __str__(self):
        """Abbreviated name reflecting major configuration parameters.

        Returns
        -------
        str
            Human readable representation of the models internal state.

        """
        segments = []
        if self.comment:
            segments.append('"%s"' % self.comment)
        if self.sg:
            if self.dbow_words:
                segments.append('dbow+w')  # also training words
            else:
                segments.append('dbow')  # PV-DBOW (skip-gram-style)

        else:  # PV-DM...
            if self.dm_concat:
                segments.append('dm/c')  # ...with concatenative context layer
            else:
                if self.cbow_mean:
                    segments.append('dm/m')
                else:
                    segments.append('dm/s')
        segments.append('d%d' % self.docvecs.vector_size)  # dimensions
        if self.negative:
            segments.append('n%d' % self.negative)  # negative samples
        if self.hs:
            segments.append('hs')
        if not self.sg or (self.sg and self.dbow_words):
            segments.append('w%d' % self.window)  # window size, when relevant
        if self.vocabulary.min_count > 1:
            segments.append('mc%d' % self.vocabulary.min_count)
        if self.vocabulary.sample > 0:
            segments.append('s%g' % self.vocabulary.sample)
        if self.workers > 1:
            segments.append('t%d' % self.workers)
        return '%s(%s)' % (self.__class__.__name__, ','.join(segments))

    def delete_temporary_training_data(self, keep_doctags_vectors=True, keep_inference=True):
        """Discard parameters that are used in training and score. Use if you're sure you're done training a model.

        Parameters
        ----------
        keep_doctags_vectors : bool, optional
            Set to False if you don't want to save doctags vectors. In this case you will not be able to use
            :meth:`~gensim.models.keyedvectors.Doc2VecKeyedVectors.most_similar`,
            :meth:`~gensim.models.keyedvectors.Doc2VecKeyedVectors.similarity`, etc methods.
        keep_inference : bool, optional
            Set to False if you don't want to store parameters that are used for
            :meth:`~gensim.models.doc2vec.Doc2Vec.infer_vector` method.

        """
        if not keep_inference:
            if hasattr(self.trainables, 'syn1'):
                del self.trainables.syn1
            if hasattr(self.trainables, 'syn1neg'):
                del self.trainables.syn1neg
            if hasattr(self.trainables, 'vectors_lockf'):
                del self.trainables.vectors_lockf
        self.model_trimmed_post_training = True
        if self.docvecs and hasattr(self.docvecs, 'vectors_docs') and not keep_doctags_vectors:
            del self.docvecs.vectors_docs
        if self.docvecs and hasattr(self.trainables, 'vectors_docs_lockf'):
            del self.trainables.vectors_docs_lockf

    def save_word2vec_format(self, fname, doctag_vec=False, word_vec=True, prefix='*dt_', fvocab=None, binary=False):
        """Store the input-hidden weight matrix in the same format used by the original C word2vec-tool.

        Parameters
        ----------
        fname : str
            The file path used to save the vectors in.
        doctag_vec : bool, optional
            Indicates whether to store document vectors.
        word_vec : bool, optional
            Indicates whether to store word vectors.
        prefix : str, optional
            Uniquely identifies doctags from word vocab, and avoids collision in case of repeated string in doctag
            and word vocab.
        fvocab : str, optional
            Optional file path used to save the vocabulary.
        binary : bool, optional
            If True, the data wil be saved in binary word2vec format, otherwise - will be saved in plain text.

        """
        total_vec = len(self.wv.vocab) + len(self.docvecs)
        write_first_line = False
        # save word vectors
        if word_vec:
            if not doctag_vec:
                total_vec = len(self.wv.vocab)
            self.wv.save_word2vec_format(fname, fvocab, binary, total_vec)
        # save document vectors
        if doctag_vec:
            if not word_vec:
                total_vec = len(self.docvecs)
                write_first_line = True
            self.docvecs.save_word2vec_format(
                fname, prefix=prefix, fvocab=fvocab, total_vec=total_vec,
                binary=binary, write_first_line=write_first_line)

    def init_sims(self, replace=False):
        """Pre-compute L2-normalized vectors.

        Parameters
        ----------
        replace : bool
            If True - forget the original vectors and only keep the normalized ones to saved RAM (also you can't
            continue training if call it with `replace=True`).

        """
        self.docvecs.init_sims(replace=replace)

    @classmethod
    def load(cls, *args, **kwargs):
        """Load a previously saved :class:`~gensim.models.doc2vec.Doc2Vec` model.

        Parameters
        ----------
        fname : str
            Path to the saved file.
        *args : object
            Additional arguments, see `~gensim.models.base_any2vec.BaseWordEmbeddingsModel.load`.
        **kwargs : object
            Additional arguments, see `~gensim.models.base_any2vec.BaseWordEmbeddingsModel.load`.

        See Also
        --------
        :meth:`~gensim.models.doc2vec.Doc2Vec.save`
            Save :class:`~gensim.models.doc2vec.Doc2Vec` model.

        Returns
        -------
        :class:`~gensim.models.doc2vec.Doc2Vec`
            Loaded model.

        """
        try:
            return super(Doc2Vec, cls).load(*args, **kwargs)
        except AttributeError:
            logger.info('Model saved using code from earlier Gensim Version. Re-loading old model in a compatible way.')
            from gensim.models.deprecated.doc2vec import load_old_doc2vec
            return load_old_doc2vec(*args, **kwargs)

    def estimate_memory(self, vocab_size=None, report=None):
        """Estimate required memory for a model using current settings.

        Parameters
        ----------
        vocab_size : int, optional
            Number of raw words in the vocabulary.
        report : dict of (str, int), optional
            A dictionary from string representations of the **specific** model's memory consuming members
            to their size in bytes.

        Returns
        -------
        dict of (str, int), optional
            A dictionary from string representations of the model's memory consuming members to their size in bytes.
            Includes members from the base classes as well as weights and tag lookup memory estimation specific to the
            class.

        """
        report = report or {}
        report['doctag_lookup'] = self.estimated_lookup_memory()
        report['doctag_syn0'] = self.docvecs.count * self.vector_size * dtype(REAL).itemsize
        return super(Doc2Vec, self).estimate_memory(vocab_size, report=report)

    def build_vocab(self, documents, update=False, progress_per=10000, keep_raw_vocab=False, trim_rule=None, **kwargs):
        """Build vocabulary from a sequence of sentences (can be a once-only generator stream).

        Parameters
        ----------
        documents : iterable of list of :class:`~gensim.models.doc2vec.TaggedDocument`
            Can be simply a list of :class:`~gensim.models.doc2vec.TaggedDocument` elements, but for larger corpora,
            consider an iterable that streams the documents directly from disk/network.
            See :class:`~gensim.models.doc2vec.TaggedBrownCorpus` or :class:`~gensim.models.doc2vec.TaggedLineDocument`
        update : bool
            If true, the new words in `sentences` will be added to model's vocab.
        progress_per : int
            Indicates how many words to process before showing/updating the progress.
        keep_raw_vocab : bool
            If not true, delete the raw vocabulary after the scaling is done and free up RAM.
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during current method call and is not stored as part
            of the model.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.

        **kwargs
            Additional key word arguments passed to the internal vocabulary construction.

        """
        total_words, corpus_count = self.vocabulary.scan_vocab(
            documents, self.docvecs, progress_per=progress_per, trim_rule=trim_rule)
        self.corpus_count = corpus_count
        report_values = self.vocabulary.prepare_vocab(
            self.hs, self.negative, self.wv, update=update, keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule,
            **kwargs)

        report_values['memory'] = self.estimate_memory(vocab_size=report_values['num_retained_words'])
        self.trainables.prepare_weights(
            self.hs, self.negative, self.wv, self.docvecs, update=update)

    def build_vocab_from_freq(self, word_freq, keep_raw_vocab=False, corpus_count=None, trim_rule=None, update=False):
        """Build vocabulary from a dictionary of word frequencies.

        Build model vocabulary from a passed dictionary that contains a (word -> word count) mapping.
        Words must be of type unicode strings.

        Parameters
        ----------
        word_freq : dict of (str, int)
            Word <-> count mapping.
        keep_raw_vocab : bool, optional
            If not true, delete the raw vocabulary after the scaling is done and free up RAM.
        corpus_count : int, optional
            Even if no corpus is provided, this argument can set corpus_count explicitly.
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during
            :meth:`~gensim.models.doc2vec.Doc2Vec.build_vocab` and is not stored as part of the model.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.

        update : bool, optional
            If true, the new provided words in `word_freq` dict will be added to model's vocab.

        """
        logger.info("Processing provided word frequencies")
        # Instead of scanning text, this will assign provided word frequencies dictionary(word_freq)
        # to be directly the raw vocab
        raw_vocab = word_freq
        logger.info(
            "collected %i different raw word, with total frequency of %i",
            len(raw_vocab), sum(itervalues(raw_vocab))
        )

        # Since no sentences are provided, this is to control the corpus_count
        self.corpus_count = corpus_count or 0
        self.vocabulary.raw_vocab = raw_vocab

        # trim by min_count & precalculate downsampling
        report_values = self.vocabulary.prepare_vocab(
            self.hs, self.negative, self.wv, keep_raw_vocab=keep_raw_vocab,
            trim_rule=trim_rule, update=update)
        report_values['memory'] = self.estimate_memory(vocab_size=report_values['num_retained_words'])
        self.trainables.prepare_weights(
            self.hs, self.negative, self.wv, self.docvecs, update=update)


class Doc2VecVocab(Word2VecVocab):
    """Vocabulary used by :class:`~gensim.models.doc2vec.Doc2Vec`.

    This includes a mapping from words found in the corpus to their total frequency count.

    """
    def __init__(self, max_vocab_size=None, min_count=5, sample=1e-3, sorted_vocab=True, null_word=0, ns_exponent=0.75):
        """

        Parameters
        ----------
        max_vocab_size : int, optional
            Maximum number of words in the Vocabulary. Used to limit the RAM during vocabulary building;
            if there are more unique words than this, then prune the infrequent ones.
            Every 10 million word types need about 1GB of RAM, set to `None` for no limit.
        min_count : int
            Words with frequency lower than this limit will be discarded form the vocabulary.
        sample : float, optional
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        sorted_vocab : bool
            If True, sort the vocabulary by descending frequency before assigning word indexes.
        null_word : {0, 1}
            If True, a null pseudo-word will be created for padding when using concatenative L1 (run-of-words).
            This word is only ever input – never predicted – so count, huffman-point, etc doesn't matter.
        ns_exponent : float, optional
            The exponent used to shape the negative sampling distribution. A value of 1.0 samples exactly in proportion
            to the frequencies, 0.0 samples all words equally, while a negative value samples low-frequency words more
            than high-frequency words. The popular default value of 0.75 was chosen by the original Word2Vec paper.
            More recently, in https://arxiv.org/abs/1804.04212, Caselles-Dupré, Lesaint, & Royo-Letelier suggest that
            other values may perform better for recommendation applications.

        """
        super(Doc2VecVocab, self).__init__(
            max_vocab_size=max_vocab_size, min_count=min_count, sample=sample,
            sorted_vocab=sorted_vocab, null_word=null_word, ns_exponent=ns_exponent)

    def scan_vocab(self, documents, docvecs, progress_per=10000, trim_rule=None):
        """Create the models Vocabulary: A mapping from unique words in the corpus to their frequency count.

        Parameters
        ----------
        documents : iterable of :class:`~gensim.models.doc2vec.TaggedDocument`
            The tagged documents used to create the vocabulary. Their tags can be either str tokens or ints (faster).
        docvecs : list of :class:`~gensim.models.keyedvectors.Doc2VecKeyedVectors`
            The vector representations of the documents in our corpus. Each of them has a size == `vector_size`.
        progress_per : int
            Progress will be logged every `progress_per` documents.
        trim_rule : function, optional
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            The rule, if given, is only used to prune vocabulary during
            :meth:`~gensim.models.doc2vec.Doc2Vec.build_vocab` and is not stored as part of the model.

            The input parameters are of the following types:
                * `word` (str) - the word we are examining
                * `count` (int) - the word's frequency count in the corpus
                * `min_count` (int) - the minimum count threshold.

        Returns
        -------
        (int, int)
            Tuple of (Total words in the corpus, number of documents)

        """
        logger.info("collecting all words and their counts")
        document_no = -1
        total_words = 0
        min_reduce = 1
        interval_start = default_timer() - 0.00001  # guard against next sample being identical
        interval_count = 0
        checked_string_types = 0
        vocab = defaultdict(int)
        for document_no, document in enumerate(documents):
            if not checked_string_types:
                if isinstance(document.words, string_types):
                    logger.warning(
                        "Each 'words' should be a list of words (usually unicode strings). "
                        "First 'words' here is instead plain %s.",
                        type(document.words)
                    )
                checked_string_types += 1
            if document_no % progress_per == 0:
                interval_rate = (total_words - interval_count) / (default_timer() - interval_start)
                logger.info(
                    "PROGRESS: at example #%i, processed %i words (%i/s), %i word types, %i tags",
                    document_no, total_words, interval_rate, len(vocab), docvecs.count
                )
                interval_start = default_timer()
                interval_count = total_words
            document_length = len(document.words)

            for tag in document.tags:
                self.note_doctag(tag, document_no, document_length, docvecs)

            for word in document.words:
                vocab[word] += 1
            total_words += len(document.words)

            if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
                min_reduce += 1

        logger.info(
            "collected %i word types and %i unique tags from a corpus of %i examples and %i words",
            len(vocab), docvecs.count, document_no + 1, total_words
        )
        corpus_count = document_no + 1
        self.raw_vocab = vocab
        return total_words, corpus_count

    def note_doctag(self, key, document_no, document_length, docvecs):
        """Note a document tag during initial corpus scan, for correctly setting the keyedvectors size.

        Parameters
        ----------
        key : {int, str}
            The tag to be noted.
        document_no : int
            The document's index in `docvecs`. Unused.
        document_length : int
            The document's length in words.
        docvecs : list of :class:`~gensim.models.keyedvectors.Doc2VecKeyedVectors`
            Vector representations of the documents in the corpus. Each vector has size == `vector_size`

        """
        if isinstance(key, integer_types + (integer,)):
            docvecs.max_rawint = max(docvecs.max_rawint, key)
        else:
            if key in docvecs.doctags:
                docvecs.doctags[key] = docvecs.doctags[key].repeat(document_length)
            else:
                docvecs.doctags[key] = Doctag(len(docvecs.offset2doctag), document_length, 1)
                docvecs.offset2doctag.append(key)
        docvecs.count = docvecs.max_rawint + 1 + len(docvecs.offset2doctag)

    def indexed_doctags(self, doctag_tokens, docvecs):
        """Get the indexes and backing-arrays used in training examples.

        Parameters
        ----------
        doctag_tokens : list of {str, int}
            A list of tags for which we want the index.
        docvecs : list of :class:`~gensim.models.keyedvectors.Doc2VecKeyedVectors`
            Vector representations of the documents in the corpus. Each vector has size == `vector_size`

        Returns
        -------
        list of int
            Indices of the provided tag keys.

        """
        return [
            Doc2VecKeyedVectors._int_index(index, docvecs.doctags, docvecs.max_rawint)
            for index in doctag_tokens if self._tag_seen(index, docvecs)]

    def _tag_seen(self, index, docvecs):
        """Whether or not the tag exists in our Vocabulary.

        Parameters
        ----------
        index : {str, int}
            The tag to be checked.
        docvecs : :class:`~gensim.models.keyedvectors.Doc2VecKeyedVectors`
            Vector representations of the documents in the corpus. Each vector has size == `vector_size`

        Returns
        -------
        bool
            Whether or not the passed tag exists in our vocabulary.

        """
        if isinstance(index, integer_types + (integer,)):
            return index < docvecs.count
        else:
            return index in docvecs.doctags


class Doc2VecTrainables(Word2VecTrainables):
    """Represents the inner shallow neural network used to train :class:`~gensim.models.doc2vec.Doc2Vec`."""
    def __init__(self, dm=1, dm_concat=0, dm_tag_count=1, vector_size=100, seed=1, hashfxn=hash, window=5):
        super(Doc2VecTrainables, self).__init__(
            vector_size=vector_size, seed=seed, hashfxn=hashfxn)
        if dm and dm_concat:
            self.layer1_size = (dm_tag_count + (2 * window)) * vector_size
            logger.info("using concatenative %d-dimensional layer1", self.layer1_size)

    def prepare_weights(self, hs, negative, wv, docvecs, update=False):
        """Build tables and model weights based on final vocabulary settings."""
        # set initial input/projection and hidden weights
        if not update:
            self.reset_weights(hs, negative, wv, docvecs)
        else:
            self.update_weights(hs, negative, wv)

    def reset_weights(self, hs, negative, wv, docvecs, vocabulary=None):
        super(Doc2VecTrainables, self).reset_weights(hs, negative, wv)
        self.reset_doc_weights(docvecs)

    def reset_doc_weights(self, docvecs):
        length = max(len(docvecs.doctags), docvecs.count)
        if docvecs.mapfile_path:
            docvecs.vectors_docs = np_memmap(
                docvecs.mapfile_path + '.vectors_docs', dtype=REAL, mode='w+', shape=(length, docvecs.vector_size)
            )
            self.vectors_docs_lockf = np_memmap(
                docvecs.mapfile_path + '.vectors_docs_lockf', dtype=REAL, mode='w+', shape=(length,)
            )
            self.vectors_docs_lockf.fill(1.0)
        else:
            docvecs.vectors_docs = empty((length, docvecs.vector_size), dtype=REAL)
            self.vectors_docs_lockf = ones((length,), dtype=REAL)  # zeros suppress learning

        for i in xrange(length):
            # construct deterministic seed from index AND model seed
            seed = "%d %s" % (
                self.seed, Doc2VecKeyedVectors._index_to_doctag(i, docvecs.offset2doctag, docvecs.max_rawint))
            docvecs.vectors_docs[i] = self.seeded_vector(seed, docvecs.vector_size)

    def get_doctag_trainables(self, doc_words, vector_size):
        doctag_vectors = zeros((1, vector_size), dtype=REAL)
        doctag_vectors[0] = self.seeded_vector(' '.join(doc_words), vector_size)
        doctag_locks = ones(1, dtype=REAL)
        return doctag_vectors, doctag_locks


class TaggedBrownCorpus(object):
    """Reader for the `Brown corpus (part of NLTK data) <http://www.nltk.org/book/ch02.html#tab-brown-sources>`_."""

    def __init__(self, dirname):
        """

        Parameters
        ----------
        dirname : str
            Path to folder with Brown corpus.

        """
        self.dirname = dirname

    def __iter__(self):
        """Iterate through the corpus.

        Yields
        ------
        :class:`~gensim.models.doc2vec.TaggedDocument`
            Document from `source`.

        """
        for fname in os.listdir(self.dirname):
            fname = os.path.join(self.dirname, fname)
            if not os.path.isfile(fname):
                continue
            for item_no, line in enumerate(utils.smart_open(fname)):
                line = utils.to_unicode(line)
                # each file line is a single document in the Brown corpus
                # each token is WORD/POS_TAG
                token_tags = [t.split('/') for t in line.split() if len(t.split('/')) == 2]
                # ignore words with non-alphabetic tags like ",", "!" etc (punctuation, weird stuff)
                words = ["%s/%s" % (token.lower(), tag[:2]) for token, tag in token_tags if tag[:2].isalpha()]
                if not words:  # don't bother sending out empty documents
                    continue
                yield TaggedDocument(words, ['%s_SENT_%s' % (fname, item_no)])


class TaggedLineDocument(object):
    """Iterate over a file that contains sentences: one line = :class:`~gensim.models.doc2vec.TaggedDocument` object.

    Words are expected to be already preprocessed and separated by whitespace. Document tags are constructed
    automatically from the document line number (each document gets a unique integer tag).

    """
    def __init__(self, source):
        """

        Parameters
        ----------
        source : string or a file-like object
            Path to the file on disk, or an already-open file object (must support `seek(0)`).

        Examples
        --------
        >>> from gensim.test.utils import datapath
        >>> from gensim.models.doc2vec import TaggedLineDocument
        >>>
        >>> for document in TaggedLineDocument(datapath("head500.noblanks.cor")):
        ...     pass

        """
        self.source = source

    def __iter__(self):
        """Iterate through the lines in the source.

        Yields
        ------
        :class:`~gensim.models.doc2vec.TaggedDocument`
            Document from `source` specified in the constructor.

        """
        try:
            # Assume it is a file-like object and try treating it as such
            # Things that don't have seek will trigger an exception
            self.source.seek(0)
            for item_no, line in enumerate(self.source):
                yield TaggedDocument(utils.to_unicode(line).split(), [item_no])
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with utils.smart_open(self.source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [item_no])
