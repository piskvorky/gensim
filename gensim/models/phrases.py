#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""
Automatically detect common phrases -- aka multi-word expressions, word n-gram collocations -- from
a stream of sentences.

Inspired by:

* `Mikolov, et. al: "Distributed Representations of Words and Phrases and their Compositionality"
  <https://arxiv.org/abs/1310.4546>`_
* `"Normalized (Pointwise) Mutual Information in Collocation Extraction" by Gerlof Bouma
  <https://svn.spraakdata.gu.se/repos/gerlof/pub/www/Docs/npmi-pfd.pdf>`_


Examples
--------
.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>> from gensim.models.word2vec import Text8Corpus
    >>> from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
    >>>
    >>> # Create training corpus. Must be a sequence of sentences (e.g. an iterable or a generator).
    >>> sentences = Text8Corpus(datapath('testcorpus.txt'))
    >>> # Each sentence must be a list of string tokens:
    >>> first_sentence = next(iter(sentences))
    >>> print(first_sentence[:10])
    ['computer', 'human', 'interface', 'computer', 'response', 'survey', 'system', 'time', 'user', 'interface']
    >>>
    >>> # Train a toy phrase model on our training corpus.
    >>> phrase_model = Phrases(sentences, min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)
    >>>
    >>> # Apply the trained phrases model to a new, unseen sentence.
    >>> new_sentence = ['trees', 'graph', 'minors']
    >>> phrase_model[new_sentence]
    ['trees_graph', 'minors']
    >>> # The toy model considered "trees graph" a single phrase => joined the two
    >>> # tokens into a single "phrase" token, using our selected `_` delimiter.
    >>>
    >>> # Apply the trained model to each sentence of a corpus, using the same [] syntax:
    >>> for sent in phrase_model[sentences]:
    ...     pass
    >>>
    >>> # Update the model with two new sentences on the fly.
    >>> phrase_model.add_vocab([["hello", "world"], ["meow"]])
    >>>
    >>> # Export the trained model = use less RAM, faster processing. Model updates no longer possible.
    >>> frozen_model = phrase_model.freeze()
    >>> # Apply the frozen model; same results as before:
    >>> frozen_model[new_sentence]
    ['trees_graph', 'minors']
    >>>
    >>> # Save / load models.
    >>> frozen_model.save("/tmp/my_phrase_model.pkl")
    >>> model_reloaded = Phrases.load("/tmp/my_phrase_model.pkl")
    >>> model_reloaded[['trees', 'graph', 'minors']]  # apply the reloaded model to a sentence
    ['trees_graph', 'minors']

"""

import logging
import itertools
from math import log
import pickle
from inspect import getfullargspec as getargspec
import time

from gensim import utils, interfaces


logger = logging.getLogger(__name__)

NEGATIVE_INFINITY = float('-inf')

# Words from this set are "ignored" during phrase detection:
# 1) Phrases may not start nor end with these words.
# 2) Phrases may include any number of these words inside.
ENGLISH_CONNECTOR_WORDS = frozenset(
    " a an the "  # articles; we never care about these in MWEs
    " for of with without at from to in on by "  # prepositions; incomplete on purpose, to minimize FNs
    " and or "  # conjunctions; incomplete on purpose, to minimize FNs
    .split()
)


def original_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    r"""Bigram scoring function, based on the original `Mikolov, et. al: "Distributed Representations
    of Words and Phrases and their Compositionality" <https://arxiv.org/abs/1310.4546>`_.

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
        Minimum collocation count threshold.
    corpus_word_count : int
        Not used in this particular scoring technique.

    Returns
    -------
    float
        Score for given phrase. Can be negative.

    Notes
    -----
    Formula: :math:`\frac{(bigram\_count - min\_count) * len\_vocab }{ (worda\_count * wordb\_count)}`.

    """
    denom = worda_count * wordb_count
    if denom == 0:
        return NEGATIVE_INFINITY
    return (bigram_count - min_count) / float(denom) * len_vocab


def npmi_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    r"""Calculation NPMI score based on `"Normalized (Pointwise) Mutual Information in Colocation Extraction"
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
        Ignore all bigrams with total collected count lower than this value.
    corpus_word_count : int
        Total number of words in the corpus.

    Returns
    -------
    float
        If bigram_count >= min_count, return the collocation score, in the range -1 to 1.
        Otherwise return -inf.

    Notes
    -----
    Formula: :math:`\frac{ln(prop(word_a, word_b) / (prop(word_a)*prop(word_b)))}{ -ln(prop(word_a, word_b)}`,
    where :math:`prob(word) = \frac{word\_count}{corpus\_word\_count}`

    """
    if bigram_count >= min_count:
        corpus_word_count = float(corpus_word_count)
        pa = worda_count / corpus_word_count
        pb = wordb_count / corpus_word_count
        pab = bigram_count / corpus_word_count
        try:
            return log(pab / (pa * pb)) / -log(pab)
        except ValueError:  # some of the counts were zero => never a phrase
            return NEGATIVE_INFINITY
    else:
        # Return -infinity to make sure that no phrases will be created
        # from bigrams less frequent than min_count.
        return NEGATIVE_INFINITY


def _is_single(obj):
    """Check whether `obj` is a single document or an entire corpus.

    Parameters
    ----------
    obj : object

    Return
    ------
    (bool, object)
        2-tuple ``(is_single_document, new_obj)`` tuple, where `new_obj`
        yields the same sequence as the original `obj`.

    Notes
    -----
    `obj` is a single document if it is an iterable of strings. It is a corpus if it is an iterable of documents.

    """
    obj_iter = iter(obj)
    temp_iter = obj_iter
    try:
        peek = next(obj_iter)
        obj_iter = itertools.chain([peek], obj_iter)
    except StopIteration:
        # An empty object is interpreted as a single document (not a corpus).
        return True, obj
    if isinstance(peek, str):
        # First item is a string => obj is a single document for sure.
        return True, obj_iter
    if temp_iter is obj:
        # An iterator / generator => interpret input as a corpus.
        return False, obj_iter
    # If the first item isn't a string, assume obj is an iterable corpus.
    return False, obj


class _PhrasesTransformation(interfaces.TransformationABC):
    """
    Abstract base class for :class:`~gensim.models.phrases.Phrases` and
    :class:`~gensim.models.phrases.FrozenPhrases`.

    """
    def __init__(self, connector_words):
        self.connector_words = frozenset(connector_words)

    def score_candidate(self, word_a, word_b, in_between):
        """Score a single phrase candidate.

        Returns
        -------
        (str, float)
            2-tuple of ``(delimiter-joined phrase, phrase score)`` for a phrase,
            or ``(None, None)`` if not a phrase.
        """
        raise NotImplementedError("ABC: override this method in child classes")

    def analyze_sentence(self, sentence):
        """Analyze a sentence, concatenating any detected phrases into a single token.

        Parameters
        ----------
        sentence : iterable of str
            Token sequence representing the sentence to be analyzed.

        Yields
        ------
        (str, {float, None})
            Iterate through the input sentence tokens and yield 2-tuples of:
            - ``(concatenated_phrase_tokens, score)`` for token sequences that form a phrase.
            - ``(word, None)`` if the token is not a part of a phrase.

        """
        start_token, in_between = None, []
        for word in sentence:
            if word not in self.connector_words:
                # The current word is a normal token, not a connector word, which means it's a potential
                # beginning (or end) of a phrase.
                if start_token:
                    # We're inside a potential phrase, of which this word is the end.
                    phrase, score = self.score_candidate(start_token, word, in_between)
                    if score is not None:
                        # Phrase detected!
                        yield phrase, score
                        start_token, in_between = None, []
                    else:
                        # Not a phrase after all. Dissolve the candidate's constituent tokens as individual words.
                        yield start_token, None
                        for w in in_between:
                            yield w, None
                        start_token, in_between = word, []  # new potential phrase starts here
                else:
                    # Not inside a phrase yet; start a new phrase candidate here.
                    start_token, in_between = word, []
            else:  # We're a connector word.
                if start_token:
                    # We're inside a potential phrase: add the connector word and keep growing the phrase.
                    in_between.append(word)
                else:
                    # Not inside a phrase: emit the connector word and move on.
                    yield word, None
        # Emit any non-phrase tokens at the end.
        if start_token:
            yield start_token, None
            for w in in_between:
                yield w, None

    def __getitem__(self, sentence):
        """Convert the input sequence of tokens ``sentence`` into a sequence of tokens where adjacent
        tokens are replaced by a single token if they form a bigram collocation.

        If `sentence` is an entire corpus (iterable of sentences rather than a single
        sentence), return an iterable that converts each of the corpus' sentences
        into phrases on the fly, one after another.

        Parameters
        ----------
        sentence : {list of str, iterable of list of str}
            Input sentence or a stream of sentences.

        Return
        ------
        {list of str, iterable of list of str}
            Sentence with phrase tokens joined by ``self.delimiter``, if input was a single sentence.
            A generator of such sentences if input was a corpus.

s        """
        is_single, sentence = _is_single(sentence)
        if not is_single:
            # If the input is an entire corpus (rather than a single sentence),
            # return an iterable stream.
            return self._apply(sentence)

        return [token for token, _ in self.analyze_sentence(sentence)]

    def find_phrases(self, sentences):
        """Get all unique phrases (multi-word expressions) that appear in ``sentences``, and their scores.

        Parameters
        ----------
        sentences : iterable of list of str
            Text corpus.

        Returns
        -------
        dict(str, float)
           Unique phrases found in ``sentences``, mapped to their scores.

        Example
        -------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>> from gensim.models.word2vec import Text8Corpus
            >>> from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
            >>>
            >>> sentences = Text8Corpus(datapath('testcorpus.txt'))
            >>> phrases = Phrases(sentences, min_count=1, threshold=0.1, connector_words=ENGLISH_CONNECTOR_WORDS)
            >>>
            >>> for phrase, score in phrases.find_phrases(sentences).items():
            ...     print(phrase, score)
        """
        result = {}
        for sentence in sentences:
            for phrase, score in self.analyze_sentence(sentence):
                if score is not None:
                    result[phrase] = score
        return result

    @classmethod
    def load(cls, *args, **kwargs):
        """Load a previously saved :class:`~gensim.models.phrases.Phrases` /
        :class:`~gensim.models.phrases.FrozenPhrases` model.

        Handles backwards compatibility from older versions which did not support pluggable scoring functions.

        Parameters
        ----------
        args : object
            See :class:`~gensim.utils.SaveLoad.load`.
        kwargs : object
            See :class:`~gensim.utils.SaveLoad.load`.

        """
        model = super(_PhrasesTransformation, cls).load(*args, **kwargs)

        # Upgrade FrozenPhrases
        try:
            phrasegrams = getattr(model, "phrasegrams", {})
            component, score = next(iter(phrasegrams.items()))
            if isinstance(score, tuple):
                # Value in phrasegrams used to be a tuple; keep only the 2nd tuple component = score.
                model.phrasegrams = {
                    str(model.delimiter.join(key), encoding='utf8'): val[1]
                    for key, val in phrasegrams.items()
                }
            elif isinstance(component, tuple):  # 3.8 => 4.0: phrasegram keys are strings, not tuples with bytestrings
                model.phrasegrams = {
                    str(model.delimiter.join(key), encoding='utf8'): val
                    for key, val in phrasegrams.items()
                }
        except StopIteration:
            # no phrasegrams, nothing to upgrade
            pass

        # If no scoring parameter, use default scoring.
        if not hasattr(model, 'scoring'):
            logger.warning('older version of %s loaded without scoring function', cls.__name__)
            logger.warning('setting pluggable scoring method to original_scorer for compatibility')
            model.scoring = original_scorer
        # If there is a scoring parameter, and it's a text value, load the proper scoring function.
        if hasattr(model, 'scoring'):
            if isinstance(model.scoring, str):
                if model.scoring == 'default':
                    logger.warning('older version of %s loaded with "default" scoring parameter', cls.__name__)
                    logger.warning('setting scoring method to original_scorer for compatibility')
                    model.scoring = original_scorer
                elif model.scoring == 'npmi':
                    logger.warning('older version of %s loaded with "npmi" scoring parameter', cls.__name__)
                    logger.warning('setting scoring method to npmi_scorer for compatibility')
                    model.scoring = npmi_scorer
                else:
                    raise ValueError(f'failed to load {cls.__name__} model, unknown scoring "{model.scoring}"')

        # common_terms didn't exist pre-3.?, and was renamed to connector in 4.0.0.
        if not hasattr(model, "connector_words"):
            if hasattr(model, "common_terms"):
                model.connector_words = model.common_terms
                del model.common_terms
            else:
                logger.warning('loaded older version of %s, setting connector_words to an empty set', cls.__name__)
                model.connector_words = frozenset()

        if not hasattr(model, 'corpus_word_count'):
            logger.warning('older version of %s loaded without corpus_word_count', cls.__name__)
            logger.warning('setting corpus_word_count to 0, do not use it in your scoring function')
            model.corpus_word_count = 0

        # Before 4.0.0, we stored strings as UTF8 bytes internally, to save RAM. Since 4.0.0, we use strings.
        if getattr(model, 'vocab', None):
            word = next(iter(model.vocab))  # get a random key – any key will do
            if not isinstance(word, str):
                logger.info("old version of %s loaded, upgrading %i words in memory", cls.__name__, len(model.vocab))
                logger.info("re-save the loaded model to avoid this upgrade in the future")
                vocab = {}
                for key, value in model.vocab.items():  # needs lots of extra RAM temporarily!
                    vocab[str(key, encoding='utf8')] = value
                model.vocab = vocab
        if not isinstance(model.delimiter, str):
            model.delimiter = str(model.delimiter, encoding='utf8')
        return model


class Phrases(_PhrasesTransformation):
    """Detect phrases based on collocation counts."""

    def __init__(
            self, sentences=None, min_count=5, threshold=10.0,
            max_vocab_size=40000000, delimiter='_', progress_per=10000,
            scoring='default', connector_words=frozenset(),
        ):
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
            Heavily depends on concrete scoring-function, see the `scoring` parameter.
        max_vocab_size : int, optional
            Maximum size (number of tokens) of the vocabulary. Used to control pruning of less common words,
            to keep memory under control. The default of 40M needs about 3.6GB of RAM. Increase/decrease
            `max_vocab_size` depending on how much available memory you have.
        delimiter : str, optional
            Glue character used to join collocation tokens.
        scoring : {'default', 'npmi', function}, optional
            Specify how potential phrases are scored. `scoring` can be set with either a string that refers to a
            built-in scoring function, or with a function with the expected parameter names.
            Two built-in scoring functions are available by setting `scoring` to a string:

            #. "default" - :func:`~gensim.models.phrases.original_scorer`.
            #. "npmi" - :func:`~gensim.models.phrases.npmi_scorer`.
        connector_words : set of str, optional
            Set of words that may be included within a phrase, without affecting its scoring.
            No phrase can start nor end with a connector word; a phrase may contain any number of
            connector words in the middle.

            **If your texts are in English, set** ``connector_words=phrases.ENGLISH_CONNECTOR_WORDS``.

            This will cause phrases to include common English articles, prepositions and
            conjuctions, such as `bank_of_america` or `eye_of_the_beholder`.

            For other languages or specific applications domains, use custom ``connector_words``
            that make sense there: ``connector_words=frozenset("der die das".split())`` etc.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>> from gensim.models.word2vec import Text8Corpus
            >>> from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
            >>>
            >>> # Load corpus and train a model.
            >>> sentences = Text8Corpus(datapath('testcorpus.txt'))
            >>> phrases = Phrases(sentences, min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)
            >>>
            >>> # Use the model to detect phrases in a new sentence.
            >>> sent = [u'trees', u'graph', u'minors']
            >>> print(phrases[sent])
            [u'trees_graph', u'minors']
            >>>
            >>> # Or transform multiple sentences at once.
            >>> sents = [[u'trees', u'graph', u'minors'], [u'graph', u'minors']]
            >>> for phrase in phrases[sents]:
            ...     print(phrase)
            [u'trees_graph', u'minors']
            [u'graph_minors']
            >>>
            >>> # Export a FrozenPhrases object that is more efficient but doesn't allow any more training.
            >>> frozen_phrases = phrases.freeze()
            >>> print(frozen_phrases[sent])
            [u'trees_graph', u'minors']

        Notes
        -----

        The ``scoring="npmi"`` is more robust when dealing with common words that form part of common bigrams, and
        ranges from -1 to 1, but is slower to calculate than the default ``scoring="default"``.
        The default is the PMI-like scoring as described in `Mikolov, et. al: "Distributed
        Representations of Words and Phrases and their Compositionality" <https://arxiv.org/abs/1310.4546>`_.

        To use your own custom ``scoring`` function, pass in a function with the following signature:

        * ``worda_count`` - number of corpus occurrences in `sentences` of the first token in the bigram being scored
        * ``wordb_count`` - number of corpus occurrences in `sentences` of the second token in the bigram being scored
        * ``bigram_count`` - number of occurrences in `sentences` of the whole bigram
        * ``len_vocab`` - the number of unique tokens in `sentences`
        * ``min_count`` - the `min_count` setting of the Phrases class
        * ``corpus_word_count`` - the total number of tokens (non-unique) in `sentences`

        The scoring function must accept all these parameters, even if it doesn't use them in its scoring.

        The scoring function **must be pickleable**.

        """
        super().__init__(connector_words=connector_words)
        if min_count <= 0:
            raise ValueError("min_count should be at least 1")

        if threshold <= 0 and scoring == 'default':
            raise ValueError("threshold should be positive for default scoring")
        if scoring == 'npmi' and (threshold < -1 or threshold > 1):
            raise ValueError("threshold should be between -1 and 1 for npmi scoring")

        # Set scoring based on string.
        # Intentially override the value of the scoring parameter rather than set self.scoring here,
        # to still run the check of scoring function parameters in the next code block.
        if isinstance(scoring, str):
            if scoring == 'default':
                scoring = original_scorer
            elif scoring == 'npmi':
                scoring = npmi_scorer
            else:
                raise ValueError(f'unknown scoring method string {scoring} specified')

        scoring_params = [
            'worda_count', 'wordb_count', 'bigram_count', 'len_vocab', 'min_count', 'corpus_word_count',
        ]
        if callable(scoring):
            missing = [param for param in scoring_params if param not in getargspec(scoring)[0]]
            if not missing:
                self.scoring = scoring
            else:
                raise ValueError(f'scoring function missing expected parameters {missing}')

        self.min_count = min_count
        self.threshold = threshold
        self.max_vocab_size = max_vocab_size
        self.vocab = {}  # mapping between token => its count
        self.min_reduce = 1  # ignore any tokens with count smaller than this
        self.delimiter = delimiter
        self.progress_per = progress_per
        self.corpus_word_count = 0

        # Ensure picklability of the scorer.
        try:
            pickle.loads(pickle.dumps(self.scoring))
        except pickle.PickleError:
            raise pickle.PickleError(f'Custom scoring function in {self.__class__.__name__} must be pickle-able')

        if sentences is not None:
            start = time.time()
            self.add_vocab(sentences)
            self.add_lifecycle_event("created", msg=f"built {self} in {time.time() - start:.2f}s")

    def __str__(self):
        return "%s<%i vocab, min_count=%s, threshold=%s, max_vocab_size=%s>" % (
            self.__class__.__name__, len(self.vocab), self.min_count,
            self.threshold, self.max_vocab_size,
        )

    @staticmethod
    def _learn_vocab(sentences, max_vocab_size, delimiter, connector_words, progress_per):
        """Collect unigram and bigram counts from the `sentences` iterable."""
        sentence_no, total_words, min_reduce = -1, 0, 1
        vocab = {}
        logger.info("collecting all words and their counts")
        for sentence_no, sentence in enumerate(sentences):
            if sentence_no % progress_per == 0:
                logger.info(
                    "PROGRESS: at sentence #%i, processed %i words and %i word types",
                    sentence_no, total_words, len(vocab),
                )
            start_token, in_between = None, []
            for word in sentence:
                if word not in connector_words:
                    vocab[word] = vocab.get(word, 0) + 1
                    if start_token is not None:
                        phrase_tokens = itertools.chain([start_token], in_between, [word])
                        joined_phrase_token = delimiter.join(phrase_tokens)
                        vocab[joined_phrase_token] = vocab.get(joined_phrase_token, 0) + 1
                    start_token, in_between = word, []  # treat word as both end of a phrase AND beginning of another
                elif start_token is not None:
                    in_between.append(word)
                total_words += 1

            if len(vocab) > max_vocab_size:
                utils.prune_vocab(vocab, min_reduce)
                min_reduce += 1

        logger.info(
            "collected %i token types (unigram + bigrams) from a corpus of %i words and %i sentences",
            len(vocab), total_words, sentence_no + 1,
        )
        return min_reduce, vocab, total_words

    def add_vocab(self, sentences):
        """Update model parameters with new `sentences`.

        Parameters
        ----------
        sentences : iterable of list of str
            Text corpus to update this model's parameters from.

        Example
        -------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>> from gensim.models.word2vec import Text8Corpus
            >>> from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
            >>>
            >>> # Train a phrase detector from a text corpus.
            >>> sentences = Text8Corpus(datapath('testcorpus.txt'))
            >>> phrases = Phrases(sentences, connector_words=ENGLISH_CONNECTOR_WORDS)  # train model
            >>> assert len(phrases.vocab) == 37
            >>>
            >>> more_sentences = [
            ...     [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there'],
            ...     [u'machine', u'learning', u'can', u'be', u'new', u'york', u'sometimes'],
            ... ]
            >>>
            >>> phrases.add_vocab(more_sentences)  # add new sentences to model
            >>> assert len(phrases.vocab) == 60

        """
        # Uses a separate vocab to collect the token counts from `sentences`.
        # This consumes more RAM than merging new sentences into `self.vocab`
        # directly, but gives the new sentences a fighting chance to collect
        # sufficient counts, before being pruned out by the (large) accumulated
        # counts collected in previous learn_vocab runs.
        min_reduce, vocab, total_words = self._learn_vocab(
            sentences, max_vocab_size=self.max_vocab_size, delimiter=self.delimiter,
            progress_per=self.progress_per, connector_words=self.connector_words,
        )

        self.corpus_word_count += total_words
        if self.vocab:
            logger.info("merging %i counts into %s", len(vocab), self)
            self.min_reduce = max(self.min_reduce, min_reduce)
            for word, count in vocab.items():
                self.vocab[word] = self.vocab.get(word, 0) + count
            if len(self.vocab) > self.max_vocab_size:
                utils.prune_vocab(self.vocab, self.min_reduce)
                self.min_reduce += 1
        else:
            # Optimization for a common case: the current vocab is empty, so apply
            # the new vocab directly, no need to double it in memory.
            self.vocab = vocab
        logger.info("merged %s", self)

    def score_candidate(self, word_a, word_b, in_between):
        # Micro optimization: check for quick early-out conditions, before the actual scoring.
        word_a_cnt = self.vocab.get(word_a, 0)
        if word_a_cnt <= 0:
            return None, None

        word_b_cnt = self.vocab.get(word_b, 0)
        if word_b_cnt <= 0:
            return None, None

        phrase = self.delimiter.join([word_a] + in_between + [word_b])
        # XXX: Why do we care about *all* phrase tokens? Why not just score the start+end bigram?
        phrase_cnt = self.vocab.get(phrase, 0)
        if phrase_cnt <= 0:
            return None, None

        score = self.scoring(
            worda_count=word_a_cnt, wordb_count=word_b_cnt, bigram_count=phrase_cnt,
            len_vocab=len(self.vocab), min_count=self.min_count, corpus_word_count=self.corpus_word_count,
        )
        if score <= self.threshold:
            return None, None

        return phrase, score

    def freeze(self):
        """
        Return an object that contains the bare minimum of information while still allowing
        phrase detection. See :class:`~gensim.models.phrases.FrozenPhrases`.

        Use this "frozen model" to dramatically reduce RAM footprint if you don't plan to
        make any further changes to your `Phrases` model.

        Returns
        -------
        :class:`~gensim.models.phrases.FrozenPhrases`
            Exported object that's smaller, faster, but doesn't support model updates.

        """
        return FrozenPhrases(self)

    def export_phrases(self):
        """Extract all found phrases.

        Returns
        ------
        dict(str, float)
            Mapping between phrases and their scores.

        """
        result, source_vocab = {}, self.vocab
        for token in source_vocab:
            unigrams = token.split(self.delimiter)
            if len(unigrams) < 2:
                continue  # no phrases here
            phrase, score = self.score_candidate(unigrams[0], unigrams[-1], unigrams[1:-1])
            if score is not None:
                result[phrase] = score
        return result


class FrozenPhrases(_PhrasesTransformation):
    """Minimal state & functionality exported from a trained :class:`~gensim.models.phrases.Phrases` model.

    The goal of this class is to cut down memory consumption of `Phrases`, by discarding model state
    not strictly needed for the phrase detection task.

    Use this instead of `Phrases` if you do not need to update the bigram statistics with new documents any more.

    """

    def __init__(self, phrases_model):
        """

        Parameters
        ----------
        phrases_model : :class:`~gensim.models.phrases.Phrases`
            Trained phrases instance, to extract all phrases from.

        Notes
        -----
        After the one-time initialization, a :class:`~gensim.models.phrases.FrozenPhrases` will be much
        smaller and faster than using the full :class:`~gensim.models.phrases.Phrases` model.

        Examples
        ----------
        .. sourcecode:: pycon

            >>> from gensim.test.utils import datapath
            >>> from gensim.models.word2vec import Text8Corpus
            >>> from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
            >>>
            >>> # Load corpus and train a model.
            >>> sentences = Text8Corpus(datapath('testcorpus.txt'))
            >>> phrases = Phrases(sentences, min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)
            >>>
            >>> # Export a FrozenPhrases object that is more efficient but doesn't allow further training.
            >>> frozen_phrases = phrases.freeze()
            >>> print(frozen_phrases[sent])
            [u'trees_graph', u'minors']

        """
        self.threshold = phrases_model.threshold
        self.min_count = phrases_model.min_count
        self.delimiter = phrases_model.delimiter
        self.scoring = phrases_model.scoring
        self.connector_words = phrases_model.connector_words
        logger.info('exporting phrases from %s', phrases_model)
        start = time.time()
        self.phrasegrams = phrases_model.export_phrases()
        self.add_lifecycle_event("created", msg=f"exported {self} from {phrases_model} in {time.time() - start:.2f}s")

    def __str__(self):
        return "%s<%i phrases, min_count=%s, threshold=%s>" % (
            self.__class__.__name__, len(self.phrasegrams), self.min_count, self.threshold,
        )

    def score_candidate(self, word_a, word_b, in_between):
        phrase = self.delimiter.join([word_a] + in_between + [word_b])
        score = self.phrasegrams.get(phrase, NEGATIVE_INFINITY)
        if score > self.threshold:
            return phrase, score
        return None, None


Phraser = FrozenPhrases  # alias for backward compatibility
