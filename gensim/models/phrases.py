#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Automatically detect common phrases -- multi-word expressions / word n-grams -- from a stream of sentences.

Inspired by:

* `Mikolov et al "Distributed Representations of Words and Phrases and their Compositionality"
  <https://arxiv.org/abs/1310.4546>`_
* `"Normalized (Pointwise) Mutual Information in Colocation Extraction" by Gerlof Bouma
  <https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf>`_


Examples
--------
>>> from gensim.test.utils import datapath
>>> from gensim.models.word2vec import Text8Corpus
>>> from gensim.models.phrases import Phrases, Phraser
>>>
>>> sentences = Text8Corpus(datapath('testcorpus.txt'))
>>> phrases = Phrases(sentences, min_count=1, threshold=1)  # train model
>>> phrases[[u'trees', u'graph', u'minors']]  # apply model to sentence
[u'trees_graph', u'minors']
>>>
>>> phrases.add_vocab([["hello", "world"], ["meow"]])  # update model with new sentences
>>>
>>> bigram = Phraser(phrases)  # construct faster model (this is only an wrapper)
>>> bigram[[u'trees', u'graph', u'minors']]  # apply model to sentence
[u'trees_graph', u'minors']
>>>
>>> for sent in bigram[sentences]:  # apply model to text corpus
...     pass

"""

import sys
import os
import logging
import warnings
from collections import defaultdict
import functools as ft
import itertools as it
from math import log
import pickle
import six

from six import iteritems, string_types, PY2, next

from gensim import utils, interfaces

if PY2:
    from inspect import getargspec
else:
    from inspect import getfullargspec as getargspec

logger = logging.getLogger(__name__)


def _is_single(obj):
    """Check whether `obj` is a single document or an entire corpus.

    Parameters
    ----------
    obj : object

    Return
    ------
    (bool, object)
        (is_single, new) tuple, where `new` yields the same sequence as `obj`.

    Notes
    -----
    `obj` is a single document if it is an iterable of strings. It is a corpus if it is an iterable of documents.

    """
    obj_iter = iter(obj)
    temp_iter = obj_iter
    try:
        peek = next(obj_iter)
        obj_iter = it.chain([peek], obj_iter)
    except StopIteration:
        # An empty object is a single document
        return True, obj
    if isinstance(peek, string_types):
        # It's a document, return the iterator
        return True, obj_iter
    if temp_iter is obj:
        # Checking for iterator to the object
        return False, obj_iter
    else:
        # If the first item isn't a string, assume obj is a corpus
        return False, obj


class SentenceAnalyzer(object):
    """Base util class for :class:`~gensim.models.phrases.Phrases` and :class:`~gensim.models.phrases.Phraser`."""
    def score_item(self, worda, wordb, components, scorer):
        """Get bi-gram score statistics.

        Parameters
        ----------
        worda : str
            First word of bi-gram.
        wordb : str
            Second word of bi-gram.
        components : generator
            Contain all phrases.
        scorer : function
            Scorer function, as given to :class:`~gensim.models.phrases.Phrases`.
            See :func:`~gensim.models.phrases.npmi_scorer` and :func:`~gensim.models.phrases.original_scorer`.

        Returns
        -------
        float
            Score for given bi-gram, if bi-gram not presented in dictionary - return -1.

        """
        vocab = self.vocab
        if worda in vocab and wordb in vocab:
            bigram = self.delimiter.join(components)
            if bigram in vocab:
                return scorer(
                    worda_count=float(vocab[worda]),
                    wordb_count=float(vocab[wordb]),
                    bigram_count=float(vocab[bigram]))
        return -1

    def analyze_sentence(self, sentence, threshold, common_terms, scorer):
        """Analyze a sentence, detecting any bigrams that should be concatenated.

        Parameters
        ----------
        sentence : iterable of str
            Token sequence representing the sentence to be analyzed.
        threshold : float
            The minimum score for a bigram to be taken into account.
        common_terms : list of object
            List of common terms, they have special treatment.
        scorer : function
            Scorer function, as given to :class:`~gensim.models.phrases.Phrases`.
            See :func:`~gensim.models.phrases.npmi_scorer` and :func:`~gensim.models.phrases.original_scorer`.

        Yields
        ------
        (str, score)
            If bi-gram detected, a tuple where the first element is a detect bigram, second its score.
            Otherwise, the first tuple element is a single word and second is None.

        """
        s = [utils.any2utf8(w) for w in sentence]
        # adding None is a trick that helps getting an automatic happy ending
        # as it won't be a common_word, nor score
        s.append(None)
        last_uncommon = None
        in_between = []
        for word in s:
            is_common = word in common_terms
            if not is_common and last_uncommon:
                chain = [last_uncommon] + in_between + [word]
                # test between last_uncommon
                score = self.score_item(
                    worda=last_uncommon,
                    wordb=word,
                    components=chain,
                    scorer=scorer,
                )
                if score > threshold:
                    yield (chain, score)
                    last_uncommon = None
                    in_between = []
                else:
                    # release words individually
                    for w in it.chain([last_uncommon], in_between):
                        yield (w, None)
                    in_between = []
                    last_uncommon = word
            elif not is_common:
                last_uncommon = word
            else:  # common term
                if last_uncommon:
                    # wait for uncommon resolution
                    in_between.append(word)
                else:
                    yield (word, None)


class PhrasesTransformation(interfaces.TransformationABC):
    """Base util class for :class:`~gensim.models.phrases.Phrases` and :class:`~gensim.models.phrases.Phraser`."""

    @classmethod
    def load(cls, *args, **kwargs):
        """Load a previously saved :class:`~gensim.models.phrases.Phrases` /
        :class:`~gensim.models.phrases.Phraser` class. Handles backwards compatibility from older
        :class:`~gensim.models.phrases.Phrases` / :class:`~gensim.models.phrases.Phraser`
        versions which did not support pluggable scoring functions.

        Parameters
        ----------
        args : object
            Sequence of arguments, see :class:`~gensim.utils.SaveLoad.load` for more information.
        kwargs : object
            Sequence of arguments, see :class:`~gensim.utils.SaveLoad.load` for more information.

        """
        model = super(PhrasesTransformation, cls).load(*args, **kwargs)
        # update older models
        # if no scoring parameter, use default scoring
        if not hasattr(model, 'scoring'):
            logger.info('older version of %s loaded without scoring function', cls.__name__)
            logger.info('setting pluggable scoring method to original_scorer for compatibility')
            model.scoring = original_scorer
        # if there is a scoring parameter, and it's a text value, load the proper scoring function
        if hasattr(model, 'scoring'):
            if isinstance(model.scoring, six.string_types):
                if model.scoring == 'default':
                    logger.info('older version of %s loaded with "default" scoring parameter', cls.__name__)
                    logger.info('setting scoring method to original_scorer pluggable scoring method for compatibility')
                    model.scoring = original_scorer
                elif model.scoring == 'npmi':
                    logger.info('older version of %s loaded with "npmi" scoring parameter', cls.__name__)
                    logger.info('setting scoring method to npmi_scorer pluggable scoring method for compatibility')
                    model.scoring = npmi_scorer
                else:
                    raise ValueError(
                        'failed to load %s model with unknown scoring setting %s' % (cls.__name__, model.scoring))
        # if there is non common_terms attribute, initialize
        if not hasattr(model, "common_terms"):
            logger.info('older version of %s loaded without common_terms attribute', cls.__name__)
            logger.info('setting common_terms to empty set')
            model.common_terms = frozenset()
        return model


class Phrases(SentenceAnalyzer, PhrasesTransformation):
    """Detect phrases based on collocation counts."""

    def __init__(self, sentences=None, min_count=5, threshold=10.0,
                 max_vocab_size=40000000, delimiter=b'_', progress_per=10000,
                 scoring='default', common_terms=frozenset()):
        """

        Parameters
        ----------
        sentences : iterable of list of str, optional
            The `sentences` iterable can be simply a list, but for larger corpora, consider a generator that streams
            the sentences directly from disk/network, See :class:`~gensim.models.word2vec.BrownCorpus`,
            :class:`~gensim.models.word2vec.Text8Corpus` or :class:`~gensim.models.word2vec.LineSentence`
            for such examples.
        min_count : float, optional
            Ignore all words and bigrams with total collected count lower than this value.
        threshold : float, optional
            Represent a score threshold for forming the phrases (higher means fewer phrases).
            A phrase of words `a` followed by `b` is accepted if the score of the phrase is greater than threshold.
            Hardly depends on concrete socring-function, see the `scoring` parameter.
        max_vocab_size : int, optional
            Maximum size (number of tokens) of the vocabulary. Used to control pruning of less common words,
            to keep memory under control. The default of 40M needs about 3.6GB of RAM. Increase/decrease
            `max_vocab_size` depending on how much available memory you have.
        delimiter : str, optional
            Glue character used to join collocation tokens, should be a byte string (e.g. b'_').
        scoring : {'default', 'npmi', function}, optional
            Specify how potential phrases are scored. `scoring` can be set with either a string that refers to a
            built-in scoring function, or with a function with the expected parameter names.
            Two built-in scoring functions are available by setting `scoring` to a string:

            #. "default" - :func:`~gensim.models.phrases.original_scorer`.
            #. "npmi" - :func:`~gensim.models.phrases.npmi_scorer`.
        common_terms : set of str, optional
            List of "stop words" that won't affect frequency count of expressions containing them.
            Allow to detect expressions like "bank_of_america" or "eye_of_the_beholder".

        Notes
        -----
        'npmi' is more robust when dealing with common words that form part of common bigrams, and
        ranges from -1 to 1, but is slower to calculate than the default. The default is the PMI-like scoring
        as described by `Mikolov et al "Distributed Representations of Words
        and Phrases and their Compositionality" <https://arxiv.org/abs/1310.4546>`_.

        To use a custom scoring function, pass in a function with the following signature:

        * worda_count - number of corpus occurrences in `sentences` of the first token in the bigram being scored
        * wordb_count - number of corpus occurrences in `sentences` of the second token in the bigram being scored
        * bigram_count - number of occurrences in `sentences` of the whole bigram
        * len_vocab - the number of unique tokens in `sentences`
        * min_count - the `min_count` setting of the Phrases class
        * corpus_word_count - the total number of tokens (non-unique) in `sentences`

        The scoring function **must accept all these parameters**, even if it doesn't use them in its scoring.
        The scoring function **must be pickleable**.

        """
        if min_count <= 0:
            raise ValueError("min_count should be at least 1")

        if threshold <= 0 and scoring == 'default':
            raise ValueError("threshold should be positive for default scoring")
        if scoring == 'npmi' and (threshold < -1 or threshold > 1):
            raise ValueError("threshold should be between -1 and 1 for npmi scoring")

        # set scoring based on string
        # intentially override the value of the scoring parameter rather than set self.scoring here,
        # to still run the check of scoring function parameters in the next code block

        if isinstance(scoring, six.string_types):
            if scoring == 'default':
                scoring = original_scorer
            elif scoring == 'npmi':
                scoring = npmi_scorer
            else:
                raise ValueError('unknown scoring method string %s specified' % (scoring))

        scoring_parameters = [
            'worda_count', 'wordb_count', 'bigram_count', 'len_vocab', 'min_count', 'corpus_word_count'
        ]
        if callable(scoring):
            if all(parameter in getargspec(scoring)[0] for parameter in scoring_parameters):
                self.scoring = scoring
            else:
                raise ValueError('scoring function missing expected parameters')

        self.min_count = min_count
        self.threshold = threshold
        self.max_vocab_size = max_vocab_size
        self.vocab = defaultdict(int)  # mapping between utf8 token => its count
        self.min_reduce = 1  # ignore any tokens with count smaller than this
        self.delimiter = delimiter
        self.progress_per = progress_per
        self.corpus_word_count = 0
        self.common_terms = frozenset(utils.any2utf8(w) for w in common_terms)

        # ensure picklability of custom scorer
        try:
            test_pickle = pickle.dumps(self.scoring)
            load_pickle = pickle.loads(test_pickle)
        except pickle.PickleError:
            raise pickle.PickleError('unable to pickle custom Phrases scoring function')
        finally:
            del(test_pickle)
            del(load_pickle)

        if sentences is not None:
            self.add_vocab(sentences)

    @classmethod
    def load(cls, *args, **kwargs):
        """Load a previously saved Phrases class.
        Handles backwards compatibility from older Phrases versions which did not support pluggable scoring functions.

        Parameters
        ----------
        args : object
            Sequence of arguments, see :class:`~gensim.utils.SaveLoad.load` for more information.
        kwargs : object
            Sequence of arguments, see :class:`~gensim.utils.SaveLoad.load` for more information.

        """
        model = super(Phrases, cls).load(*args, **kwargs)
        if not hasattr(model, 'corpus_word_count'):
            logger.info('older version of %s loaded without corpus_word_count', cls.__name__)
            logger.info('Setting it to 0, do not use it in your scoring function.')
            model.corpus_word_count = 0
        return model

    def __str__(self):
        """Get short string representation of this phrase detector."""
        return "%s<%i vocab, min_count=%s, threshold=%s, max_vocab_size=%s>" % (
            self.__class__.__name__, len(self.vocab), self.min_count,
            self.threshold, self.max_vocab_size
        )

    @staticmethod
    def learn_vocab(sentences, max_vocab_size, delimiter=b'_', progress_per=10000,
                    common_terms=frozenset()):
        """Collect unigram/bigram counts from the `sentences` iterable.

        Parameters
        ----------
        sentences : iterable of list of str
            The `sentences` iterable can be simply a list, but for larger corpora, consider a generator that streams
            the sentences directly from disk/network, See :class:`~gensim.models.word2vec.BrownCorpus`,
            :class:`~gensim.models.word2vec.Text8Corpus` or :class:`~gensim.models.word2vec.LineSentence`
            for such examples.
        max_vocab_size : int
            Maximum size (number of tokens) of the vocabulary. Used to control pruning of less common words,
            to keep memory under control. The default of 40M needs about 3.6GB of RAM. Increase/decrease
            `max_vocab_size` depending on how much available memory you have.
        delimiter : str, optional
            Glue character used to join collocation tokens, should be a byte string (e.g. b'_').
        progress_per : int
            Write logs every `progress_per` sentence.
        common_terms : set of str, optional
            List of "stop words" that won't affect frequency count of expressions containing them.
            Allow to detect expressions like "bank_of_america" or "eye_of_the_beholder".

        Return
        ------
        (int, dict of (str, int), int)
            Number of pruned words, counters for each word/bi-gram and total number of words.

        Example
        ----------
        >>> from gensim.test.utils import datapath
        >>> from gensim.models.word2vec import Text8Corpus
        >>> from gensim.models.phrases import Phrases
        >>>
        >>> sentences = Text8Corpus(datapath('testcorpus.txt'))
        >>> pruned_words, counters, total_words = Phrases.learn_vocab(sentences, 100)
        >>> (pruned_words, total_words)
        (1, 29)
        >>> counters['computer']
        2
        >>> counters['response_time']
        1

        """
        sentence_no = -1
        total_words = 0
        logger.info("collecting all words and their counts")
        vocab = defaultdict(int)
        min_reduce = 1
        for sentence_no, sentence in enumerate(sentences):
            if sentence_no % progress_per == 0:
                logger.info(
                    "PROGRESS: at sentence #%i, processed %i words and %i word types",
                    sentence_no, total_words, len(vocab),
                )
            s = [utils.any2utf8(w) for w in sentence]
            last_uncommon = None
            in_between = []
            for word in s:
                if word not in common_terms:
                    vocab[word] += 1
                    if last_uncommon is not None:
                        components = it.chain([last_uncommon], in_between, [word])
                        vocab[delimiter.join(components)] += 1
                    last_uncommon = word
                    in_between = []
                elif last_uncommon is not None:
                    in_between.append(word)
                total_words += 1

            if len(vocab) > max_vocab_size:
                utils.prune_vocab(vocab, min_reduce)
                min_reduce += 1

        logger.info(
            "collected %i word types from a corpus of %i words (unigram + bigrams) and %i sentences",
            len(vocab), total_words, sentence_no + 1
        )
        return min_reduce, vocab, total_words

    def add_vocab(self, sentences):
        """Update model with new `sentences`.

        Parameters
        ----------
        sentences : iterable of list of str
            Text corpus.

        Example
        -------
        >>> from gensim.test.utils import datapath
        >>> from gensim.models.word2vec import Text8Corpus
        >>> from gensim.models.phrases import Phrases
        >>> #Create corpus and use it for phrase detector
        >>> sentences = Text8Corpus(datapath('testcorpus.txt'))
        >>> phrases = Phrases(sentences)  # train model
        >>> assert len(phrases.vocab) == 37
        >>>
        >>> more_sentences = [
        ...    [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there'],
        ...    [u'machine', u'learning', u'can', u'be', u'new', u'york' , u'sometimes']
        ... ]
        >>>
        >>> phrases.add_vocab(more_sentences)  # add new sentences to model
        >>> assert len(phrases.vocab) == 60

        """
        # uses a separate vocab to collect the token counts from `sentences`.
        # this consumes more RAM than merging new sentences into `self.vocab`
        # directly, but gives the new sentences a fighting chance to collect
        # sufficient counts, before being pruned out by the (large) accummulated
        # counts collected in previous learn_vocab runs.
        min_reduce, vocab, total_words = self.learn_vocab(
            sentences, self.max_vocab_size, self.delimiter, self.progress_per, self.common_terms)

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
        """Get all phrases that appear in 'sentences' that pass the bigram threshold.

        Parameters
        ----------
        sentences : iterable of list of str
            Text corpus.
        out_delimiter : str, optional
            Delimiter used to "glue" together words that form a bigram phrase.
        as_tuples : bool, optional
            Yield `(tuple(words), score)` instead of `(out_delimiter.join(words), score)`?

        Yields
        ------
        ((str, str), float) **or** (str, float)
            Phrases detected in `sentences`. Return type depends on the `as_tuples` parameter.

        Example
        -------
        >>> from gensim.test.utils import datapath
        >>> from gensim.models.word2vec import Text8Corpus
        >>> from gensim.models.phrases import Phrases
        >>>
        >>> sentences = Text8Corpus(datapath('testcorpus.txt'))
        >>> phrases = Phrases(sentences, min_count=1, threshold=0.1)
        >>>
        >>> for phrase, score in phrases.export_phrases(sentences):
        ...     pass

        """
        analyze_sentence = ft.partial(
            self.analyze_sentence,
            threshold=self.threshold,
            common_terms=self.common_terms,
            scorer=ft.partial(
                self.scoring,
                len_vocab=float(len(self.vocab)),
                min_count=float(self.min_count),
                corpus_word_count=float(self.corpus_word_count),
            ),
        )
        for sentence in sentences:
            bigrams = analyze_sentence(sentence)
            # keeps only not None scores
            filtered = ((words, score) for words, score in bigrams if score is not None)
            for words, score in filtered:
                if as_tuples:
                    yield (tuple(words), score)
                else:
                    yield (out_delimiter.join(words), score)

    def __getitem__(self, sentence):
        """Convert the input tokens `sentence` into tokens where detected bigrams are joined by a selected delimiter.

        If `sentence` is an entire corpus (iterable of sentences rather than a single
        sentence), return an iterable that converts each of the corpus' sentences
        into phrases on the fly, one after another.

        Parameters
        ----------
        sentence : {list of str, iterable of list of str}
            Sentence or text corpus.

        Returns
        -------
        {list of str, :class:`gensim.interfaces.TransformedCorpus`}
            `sentence` with detected phrase bigrams merged together, or a streamed corpus of such sentences
            if the input was a corpus.

        Examples
        ----------
        >>> from gensim.test.utils import datapath
        >>> from gensim.models.word2vec import Text8Corpus
        >>> from gensim.models.phrases import Phrases, Phraser
        >>>
        >>> #Create corpus
        >>> sentences = Text8Corpus(datapath('testcorpus.txt'))
        >>>
        >>> #Train the detector with:
        >>> phrases = Phrases(sentences, min_count=1, threshold=1)
        >>> #Input is a list of unicode strings:
        >>> sent = [u'trees', u'graph', u'minors']
        >>> #Both of these tokens appear in corpus at least twice, and phrase score is higher, than treshold = 1:
        >>> print(phrases[sent])
        [u'trees_graph', u'minors']
        >>>
        >>> sentences = Text8Corpus(datapath('testcorpus.txt'))
        >>> phrases = Phrases(sentences, min_count=1, threshold=1)
        >>> phraser = Phraser(phrases)  # for speedup
        >>>
        >>> sent = [[u'trees', u'graph', u'minors'],[u'graph', u'minors']]
        >>> for phrase in phraser[sent]:
        ...     pass

        """
        warnings.warn("For a faster implementation, use the gensim.models.phrases.Phraser class")

        delimiter = self.delimiter  # delimiter used for lookup

        is_single, sentence = _is_single(sentence)
        if not is_single:
            # if the input is an entire corpus (rather than a single sentence),
            # return an iterable stream.
            return self._apply(sentence)

        delimiter = self.delimiter
        bigrams = self.analyze_sentence(
            sentence,
            threshold=self.threshold,
            common_terms=self.common_terms,
            scorer=ft.partial(
                self.scoring,
                len_vocab=float(len(self.vocab)),
                min_count=float(self.min_count),
                corpus_word_count=float(self.corpus_word_count),
            ),
        )
        new_s = []
        for words, score in bigrams:
            if score is not None:
                words = delimiter.join(words)
            new_s.append(words)

        return [utils.to_unicode(w) for w in new_s]


def original_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    """Bigram scoring function, based on the original `"Efficient Estimaton of Word Representations in Vector Space" by
    Mikolov <https://arxiv.org/pdf/1301.3781.pdf>`_.

    Parameters
    ----------
    worda_count : int
        Number of occurrences for first word.
    wordb_count : int
        Number of occurrences for second word.
    bigram_count : int
        Number of co-occurrences for phrase "worda_wordb".
    len_vocab : int
        Size of vocabulary.
    min_count: int
        Minimum score threshold.
    corpus_word_count : int
        Not used in this particular scoring technique.

    Notes
    -----
    Formula: :math:`\\frac{(worda\_count - min\_count) * len\_vocab }{ (worda\_count * wordb\_count)}`.

    """
    return (bigram_count - min_count) / worda_count / wordb_count * len_vocab


def npmi_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    """Calculation NPMI score based on `"Normalized (Pointwise) Mutual Information in Colocation Extraction"
    by Gerlof Bouma <https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf>`_.

    Parameters
    ----------
    worda_count : int
        Number of occurrences for first word.
    wordb_count : int
        Number of occurrences for second word.
    bigram_count : int
        Number of co-occurrences for phrase "worda_wordb".
    len_vocab : int
        Not used.
    min_count: int
        Not used.
    corpus_word_count : int
        Total number of words in the corpus.

    Notes
    -----
    Formula: :math:`\\frac{ln(prop(word_a, word_b) / (prop(word_a)*prop(word_b)))}{ -ln(prop(word_a, word_b)}`,
    where :math:`prob(word) = \\frac{word\_count}{corpus\_word\_count}`

    """
    pa = worda_count / corpus_word_count
    pb = wordb_count / corpus_word_count
    pab = bigram_count / corpus_word_count
    return log(pab / (pa * pb)) / -log(pab)


def pseudocorpus(source_vocab, sep, common_terms=frozenset()):
    """Feeds `source_vocab`'s compound keys back to it, to discover phrases.

    Parameters
    ----------
    source_vocab : iterable of list of str
        Tokens vocabulary.
    sep : str
        Separator element.
    common_terms : set, optional
        Immutable set of stopwords.

    Yields
    ------
    list of str
        Phrase.

    """
    for k in source_vocab:
        if sep not in k:
            continue
        unigrams = k.split(sep)
        for i in range(1, len(unigrams)):
            if unigrams[i - 1] not in common_terms:
                # do not join common terms
                cterms = list(it.takewhile(lambda w: w in common_terms, unigrams[i:]))
                tail = unigrams[i + len(cterms):]
                components = [sep.join(unigrams[:i])] + cterms
                if tail:
                    components.append(sep.join(tail))
                yield components


class Phraser(SentenceAnalyzer, PhrasesTransformation):
    """Minimal state & functionality exported from :class:`~gensim.models.phrases.Phrases`.

    The goal of this class is to cut down memory consumption of `Phrases`, by discarding model state
    not strictly needed for the bigram detection task.

    Use this instead of `Phrases` if you do not need to update the bigram statistics with new documents any more.

    """

    def __init__(self, phrases_model):
        """

        Parameters
        ----------
        phrases_model : :class:`~gensim.models.phrases.Phrases`
            Trained phrases instance.

        Notes
        -----
        After the one-time initialization, a :class:`~gensim.models.phrases.Phraser` will be much smaller and somewhat
        faster than using the full :class:`~gensim.models.phrases.Phrases` model.

        Examples
        --------
        >>> from gensim.test.utils import datapath
        >>> from gensim.models.word2vec import Text8Corpus
        >>> from gensim.models.phrases import Phrases, Phraser
        >>>
        >>> sentences = Text8Corpus(datapath('testcorpus.txt'))
        >>> phrases = Phrases(sentences, min_count=1, threshold=1)
        >>>
        >>> bigram = Phraser(phrases)
        >>> sent = [u'trees', u'graph', u'minors']
        >>> print(bigram[sent])
        [u'trees_graph', u'minors']

        """
        self.threshold = phrases_model.threshold
        self.min_count = phrases_model.min_count
        self.delimiter = phrases_model.delimiter
        self.scoring = phrases_model.scoring
        self.common_terms = phrases_model.common_terms
        corpus = self.pseudocorpus(phrases_model)
        self.phrasegrams = {}
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

    def pseudocorpus(self, phrases_model):
        """Alias for :func:`gensim.models.phrases.pseudocorpus`.

        Parameters
        ----------
        phrases_model : :class:`~gensim.models.phrases.Phrases`
            Phrases instance.

        Return
        ------
        generator
            Generator with phrases.

        """
        return pseudocorpus(phrases_model.vocab, phrases_model.delimiter, phrases_model.common_terms)

    def score_item(self, worda, wordb, components, scorer):
        """Score a bigram.

        Parameters
        ----------
        worda : str
            First word for comparison.
        wordb : str
            Second word for comparison.
        components : generator
            Contain phrases.
        scorer : {'default', 'npmi'}
            NOT USED.

        Returns
        -------
        float
            Score for given bi-gram, if bi-gram not presented in dictionary - return -1.

        """
        try:
            return self.phrasegrams[tuple(components)][1]
        except KeyError:
            return -1

    def __getitem__(self, sentence):
        """Convert the input sequence of tokens `sentence` into a sequence of tokens where adjacent
        tokens are replaced by a single token if they form a bigram collocation.

        Parameters
        ----------
        sentence : {list of str, iterable of list of str}
            Input sentence or a stream of sentences.

        Return
        ------
        {list of str, iterable of list of str}
            Sentence or sentences with phrase tokens joined by `self.delimiter` character.

        Examples
        ----------
        >>> from gensim.test.utils import datapath
        >>> from gensim.models.word2vec import Text8Corpus
        >>> from gensim.models.phrases import Phrases, Phraser
        >>>
        >>> sentences = Text8Corpus(datapath('testcorpus.txt'))  # Read corpus
        >>>
        >>> phrases = Phrases(sentences, min_count=1, threshold=1) # Train model
        >>> # Create a Phraser object to transform any sentence and turn 2 suitable tokens into 1 phrase
        >>> phraser_model = Phraser(phrases)
        >>>
        >>> sent = [u'trees', u'graph', u'minors']
        >>> print(phraser_model[sent])
        [u'trees_graph', u'minors']
        >>> sent = [[u'trees', u'graph', u'minors'],[u'graph', u'minors']]
        >>> for phrase in phraser_model[sent]:
        ...     print(phrase)
        [u'trees_graph', u'minors']
        [u'graph_minors']

        """
        is_single, sentence = _is_single(sentence)
        if not is_single:
            # if the input is an entire corpus (rather than a single sentence),
            # return an iterable stream.
            return self._apply(sentence)

        delimiter = self.delimiter
        bigrams = self.analyze_sentence(
            sentence,
            threshold=self.threshold,
            common_terms=self.common_terms,
            scorer=None)  # we will use our score_item function redefinition
        new_s = []
        for words, score in bigrams:
            if score is not None:
                words = delimiter.join(words)
            new_s.append(words)
        return [utils.to_unicode(w) for w in new_s]


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s", " ".join(sys.argv))

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    infile = sys.argv[1]

    from gensim.models import Phrases  # noqa:F811 for pickle
    from gensim.models.word2vec import Text8Corpus
    sentences = Text8Corpus(infile)

    # test_doc = LineSentence('test/test_data/testcorpus.txt')
    bigram = Phrases(sentences, min_count=5, threshold=100)
    for s in bigram[sentences]:
        print(utils.to_utf8(u' '.join(s)))
