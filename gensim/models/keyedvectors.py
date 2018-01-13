#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Word vector storage and similarity look-ups.
Common code independent of the way the vectors are trained(Word2Vec, FastText, WordRank, VarEmbed etc)

The word vectors are considered read-only in this class.

Initialize the vectors by training e.g. Word2Vec::

>>> model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
>>> word_vectors = model.wv

Persist the word vectors to disk with::

>>> word_vectors.save(fname)
>>> word_vectors = KeyedVectors.load(fname)

The vectors can also be instantiated from an existing file on disk
in the original Google's word2vec C format as a KeyedVectors instance::

  >>> from gensim.models import KeyedVectors
  >>> word_vectors = KeyedVectors.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format
  >>> word_vectors = KeyedVectors.load_word2vec_format('/tmp/vectors.bin', binary=True)  # C binary format

You can perform various syntactic/semantic NLP word tasks with the vectors. Some of them
are already built-in::

  >>> word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
  [('queen', 0.50882536), ...]

  >>> word_vectors.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
  [('queen', 0.71382287), ...]

  >>> word_vectors.doesnt_match("breakfast cereal dinner lunch".split())
  'cereal'

  >>> word_vectors.similarity('woman', 'man')
  0.73723527

Correlation with human opinion on word similarity::

  >>> word_vectors.evaluate_word_pairs(os.path.join(module_path, 'test_data','wordsim353.tsv'))
  0.51, 0.62, 0.13

And on analogies::

  >>> word_vectors.accuracy(os.path.join(module_path, 'test_data', 'questions-words.txt'))

and so on.

"""
from __future__ import division  # py3 "true division"

import logging

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty  # noqa:F401

# If pyemd C extension is available, import it.
# If pyemd is attempted to be used, but isn't installed, ImportError will be raised in wmdistance
try:
    from pyemd import emd
    PYEMD_EXT = True
except ImportError:
    PYEMD_EXT = False

from numpy import dot, zeros, float32 as REAL, empty, memmap as np_memmap, \
    double, array, vstack, sqrt, newaxis, integer, \
    ndarray, sum as np_sum, prod, argmax, divide as np_divide
import numpy as np
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from six import string_types, integer_types
from six.moves import xrange, zip
from scipy import stats
from gensim.utils import deprecated
from gensim.models.utils_any2vec import save_word2vec_format, load_word2vec_format, compute_ngrams

logger = logging.getLogger(__name__)


class Vocab(object):
    """
    A single vocabulary item, used internally for collecting per-word frequency/sampling info,
    and for constructing binary trees (incl. both word leaves and inner nodes).

    """

    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))


class BaseKeyedVectors(utils.SaveLoad):

    def __init__(self):
        self.vectors = []
        self.vocab = {}
        self.vector_size = None
        self.index2entity = []

    def save(self, fname_or_handle, **kwargs):
        super(BaseKeyedVectors, self).save(fname_or_handle, **kwargs)

    @classmethod
    def load(cls, fname_or_handle, **kwargs):
        return super(BaseKeyedVectors, cls).load(fname_or_handle, **kwargs)

    def similarity(self, entity1, entity2):
        """Compute cosine similarity between entities, specified by string tag.
        """
        raise NotImplementedError()

    def most_similar(self, **kwargs):
        """Find the top-N most similar entities.
        Possibly have `positive` and `negative` list of entities in `**kwargs`.
        """
        return NotImplementedError()

    def distance(self, entity1, entity2):
        """Compute distance between vectors of two input entities, specified by string tag.
        """
        raise NotImplementedError()

    def distances(self, entity1, other_entities=()):
        """Compute distances from given entity (string tag) to all entities in `other_entity`.
        If `other_entities` is empty, return distance between `entity1` and all entities in vocab.
        """
        raise NotImplementedError()

    def get_vector(self, entity):
        """Accept a single entity as input, specified by string tag.
        Returns the entity's representations in vector space, as a 1D numpy array.
        """
        if entity in self.vocab:
            result = self.vectors[self.vocab[entity].index]
            result.setflags(write=False)
            return result
        else:
            raise KeyError("'%s' not in vocabulary" % entity)

    def __getitem__(self, entities):
        """
        Accept a single entity (string tag) or list of entities as input.

        If a single string or int, return designated tag's vector
        representation, as a 1D numpy array.

        If a list, return designated tags' vector representations as a
        2D numpy array: #tags x #vector_size.
        """
        if isinstance(entities, string_types):
            # allow calls like trained_model['office'], as a shorthand for trained_model[['office']]
            return self.get_vector(entities)

        return vstack([self.get_vector(entity) for entity in entities])

    def __contains__(self, entity):
        return entity in self.vocab

    def most_similar_to_given(self, entity1, entities_list):
        """Return the entity from entities_list most similar to entity1."""
        return entities_list[argmax([self.similarity(entity1, entity) for entity in entities_list])]

    def closer_than(self, entity1, entity2):
        """Returns all entities that are closer to `entity1` than `entity2` is to `entity1`."""
        all_distances = self.distances(entity1)
        e1_index = self.vocab[entity1].index
        e2_index = self.vocab[entity2].index
        closer_node_indices = np.where(all_distances < all_distances[e2_index])[0]
        return [self.index2entity[index] for index in closer_node_indices if index != e1_index]

    def rank(self, entity1, entity2):
        """Rank of the distance of `entity2` from `entity1`, in relation to distances of all entities from `entity1`."""
        return len(self.closer_than(entity1, entity2)) + 1


class WordEmbeddingsKeyedVectors(BaseKeyedVectors):
    """Class containing common methods for operations over word vectors.
    """
    def __init__(self):
        super(WordEmbeddingsKeyedVectors, self).__init__()
        self.vectors_norm = None
        self.index2word = []

    @property
    def index2entity(self):
        return self.index2word

    @index2entity.setter
    def index2entity(self, value):
        self.index2word = value

    @property
    def syn0(self):
        return self.vectors

    @syn0.setter
    def syn0(self, value):
        self.vectors = value

    @property
    def syn0norm(self):
        return self.vectors_norm

    @syn0norm.setter
    def syn0norm(self, value):
        self.vectors_norm = value

    def __contains__(self, word):
        return word in self.vocab

    def save(self, *args, **kwargs):
        """Saves the keyedvectors. This saved model can be loaded again using
        :func:`~gensim.models.*2vec.*2VecKeyedVectors.load` which supports
        operations on trained word vectors like `most_similar`.

        Parameters
        ----------
        fname : str
            Path to the file.

        Returns
        -------
        None

        """
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['vectors_norm'])
        super(WordEmbeddingsKeyedVectors, self).save(*args, **kwargs)

    def word_vec(self, word, use_norm=False):
        """
        Accept a single word as input.
        Returns the word's representations in vector space, as a 1D numpy array.

        If `use_norm` is True, returns the normalized word vector.

        Examples
        --------
        >>> trained_model['office']
        array([ -1.40128313e-02, ...])

        """
        if word in self.vocab:
            if use_norm:
                result = self.vectors_norm[self.vocab[word].index]
            else:
                result = self.vectors[self.vocab[word].index]

            result.setflags(write=False)
            return result
        else:
            raise KeyError("word '%s' not in vocabulary" % word)

    def get_vector(self, word):
        return self.word_vec(word)

    def words_closer_than(self, w1, w2):
        """
        Returns all words that are closer to `w1` than `w2` is to `w1`.

        Parameters
        ----------
        w1 : str
            Input word.
        w2 : str
            Input word.

        Returns
        -------
        list (str)
            List of words that are closer to `w1` than `w2` is to `w1`.

        Examples
        --------
        >>> model.words_closer_than('carnivore', 'mammal')
        ['dog', 'canine']

        """
        return super(WordEmbeddingsKeyedVectors, self).closer_than(w1, w2)

    def most_similar(self, positive=None, negative=None, topn=10, restrict_vocab=None, indexer=None):
        """
        Find the top-N most similar words. Positive words contribute positively towards the
        similarity, negative words negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given words and the vectors for each word in the model.
        The method corresponds to the `word-analogy` and `distance` scripts in the original
        word2vec implementation.

        Parameters
        ----------
        positive : :obj: `list` of :obj: `str`
            List of words that contribute positively.
        negative : :obj: `list` of :obj: `str`
            List of words that contribute negatively.
        topn : int
            Number of top-N similar words to return.
        restrict_vocab : int
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 word vectors in the vocabulary order. (This may be
            meaningful if you've sorted the vocabulary by descending frequency.)

        Returns
        -------
        :obj: `list` of :obj: `tuple`
            Returns a list of tuples (word, similarity)

        Examples
        --------
        >>> trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
        [('queen', 0.50882536), ...]

        """
        if positive is None:
            positive = []
        if negative is None:
            negative = []

        self.init_sims()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [
            (word, 1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in positive
        ]
        negative = [
            (word, -1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in negative
        ]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, ndarray):
                mean.append(weight * word)
            else:
                mean.append(weight * self.word_vec(word, use_norm=True))
                if word in self.vocab:
                    all_words.add(self.vocab[word].index)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        if indexer is not None:
            return indexer.most_similar(mean, topn)

        limited = self.vectors_norm if restrict_vocab is None else self.vectors_norm[:restrict_vocab]
        dists = dot(limited, mean)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]

    def similar_by_word(self, word, topn=10, restrict_vocab=None):
        """
        Find the top-N most similar words.

        Parameters
        ----------
        word : str
            Word
        topn : int
            Number of top-N similar words to return. If topn is False, similar_by_word returns
            the vector of similarity scores.
        restrict_vocab : int
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 word vectors in the vocabulary order. (This may be
            meaningful if you've sorted the vocabulary by descending frequency.)

        Returns
        -------
        :obj: `list` of :obj: `tuple`
            Returns a list of tuples (word, similarity)

        Example::

          >>> trained_model.similar_by_word('graph')
          [('user', 0.9999163150787354), ...]

        """
        return self.most_similar(positive=[word], topn=topn, restrict_vocab=restrict_vocab)

    def similar_by_vector(self, vector, topn=10, restrict_vocab=None):
        """
        Find the top-N most similar words by vector.

        Parameters
        ----------
        word : str
            Word
        topn : int
            Number of top-N similar words to return. If topn is False, similar_by_vector returns
            the vector of similarity scores.
        restrict_vocab : int
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 word vectors in the vocabulary order. (This may be
            meaningful if you've sorted the vocabulary by descending frequency.)

        Returns
        -------
        :obj: `list` of :obj: `tuple`
            Returns a list of tuples (word, similarity)

        """
        return self.most_similar(positive=[vector], topn=topn, restrict_vocab=restrict_vocab)

    def wmdistance(self, document1, document2):
        """
        Compute the Word Mover's Distance between two documents. When using this
        code, please consider citing the following papers:

        .. Ofir Pele and Michael Werman, "A linear time histogram metric for improved SIFT matching".
        .. Ofir Pele and Michael Werman, "Fast and robust earth mover's distances".
        .. Matt Kusner et al. "From Word Embeddings To Document Distances".

        Note that if one of the documents have no words that exist in the
        Word2Vec vocab, `float('inf')` (i.e. infinity) will be returned.

        This method only works if `pyemd` is installed (can be installed via pip, but requires a C compiler).

        Example:
            >>> # Train word2vec model.
            >>> model = Word2Vec(sentences)

            >>> # Some sentences to test.
            >>> sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
            >>> sentence_president = 'The president greets the press in Chicago'.lower().split()

            >>> # Remove their stopwords.
            >>> from nltk.corpus import stopwords
            >>> stopwords = nltk.corpus.stopwords.words('english')
            >>> sentence_obama = [w for w in sentence_obama if w not in stopwords]
            >>> sentence_president = [w for w in sentence_president if w not in stopwords]

            >>> # Compute WMD.
            >>> distance = model.wmdistance(sentence_obama, sentence_president)
        """

        if not PYEMD_EXT:
            raise ImportError("Please install pyemd Python package to compute WMD.")

        # Remove out-of-vocabulary words.
        len_pre_oov1 = len(document1)
        len_pre_oov2 = len(document2)
        document1 = [token for token in document1 if token in self]
        document2 = [token for token in document2 if token in self]
        diff1 = len_pre_oov1 - len(document1)
        diff2 = len_pre_oov2 - len(document2)
        if diff1 > 0 or diff2 > 0:
            logger.info('Removed %d and %d OOV words from document 1 and 2 (respectively).', diff1, diff2)

        if len(document1) == 0 or len(document2) == 0:
            logger.info(
                "At least one of the documents had no words that werein the vocabulary. "
                "Aborting (returning inf)."
            )
            return float('inf')

        dictionary = Dictionary(documents=[document1, document2])
        vocab_len = len(dictionary)

        if vocab_len == 1:
            # Both documents are composed by a single unique token
            return 0.0

        # Sets for faster look-up.
        docset1 = set(document1)
        docset2 = set(document2)

        # Compute distance matrix.
        distance_matrix = zeros((vocab_len, vocab_len), dtype=double)
        for i, t1 in dictionary.items():
            for j, t2 in dictionary.items():
                if t1 not in docset1 or t2 not in docset2:
                    continue
                # Compute Euclidean distance between word vectors.
                distance_matrix[i, j] = sqrt(np_sum((self[t1] - self[t2])**2))

        if np_sum(distance_matrix) == 0.0:
            # `emd` gets stuck if the distance matrix contains only zeros.
            logger.info('The distance matrix is all zeros. Aborting (returning inf).')
            return float('inf')

        def nbow(document):
            d = zeros(vocab_len, dtype=double)
            nbow = dictionary.doc2bow(document)  # Word frequencies.
            doc_len = len(document)
            for idx, freq in nbow:
                d[idx] = freq / float(doc_len)  # Normalized word frequencies.
            return d

        # Compute nBOW representation of documents.
        d1 = nbow(document1)
        d2 = nbow(document2)

        # Compute WMD.
        return emd(d1, d2, distance_matrix)

    def most_similar_cosmul(self, positive=None, negative=None, topn=10):
        """
        Find the top-N most similar words, using the multiplicative combination objective
        proposed by Omer Levy and Yoav Goldberg in [4]_. Positive words still contribute
        positively towards the similarity, negative words negatively, but with less
        susceptibility to one large distance dominating the calculation.

        In the common analogy-solving case, of two positive and one negative examples,
        this method is equivalent to the "3CosMul" objective (equation (4)) of Levy and Goldberg.

        Additional positive or negative examples contribute to the numerator or denominator,
        respectively – a potentially sensible but untested extension of the method. (With
        a single positive example, rankings will be the same as in the default most_similar.)

        Example::

          >>> trained_model.most_similar_cosmul(positive=['baghdad', 'england'], negative=['london'])
          [(u'iraq', 0.8488819003105164), ...]

        .. [4] Omer Levy and Yoav Goldberg. Linguistic Regularities in Sparse and Explicit Word Representations, 2014.

        """
        if positive is None:
            positive = []
        if negative is None:
            negative = []

        self.init_sims()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar_cosmul('dog'), as a shorthand for most_similar_cosmul(['dog'])
            positive = [positive]

        all_words = {
            self.vocab[word].index for word in positive + negative
            if not isinstance(word, ndarray) and word in self.vocab
            }

        positive = [
            self.word_vec(word, use_norm=True) if isinstance(word, string_types) else word
            for word in positive
        ]
        negative = [
            self.word_vec(word, use_norm=True) if isinstance(word, string_types) else word
            for word in negative
        ]

        if not positive:
            raise ValueError("cannot compute similarity with no input")

        # equation (4) of Levy & Goldberg "Linguistic Regularities...",
        # with distances shifted to [0,1] per footnote (7)
        pos_dists = [((1 + dot(self.vectors_norm, term)) / 2) for term in positive]
        neg_dists = [((1 + dot(self.vectors_norm, term)) / 2) for term in negative]
        dists = prod(pos_dists, axis=0) / (prod(neg_dists, axis=0) + 0.000001)

        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]

    def doesnt_match(self, words):
        """
        Which word from the given list doesn't go with the others?

        Parameters
        ----------
        words : :obj: `list` of :obj: `str`
            List of words

        Returns
        -------
        str
            The word further away from the mean of all words.

        Example
        -------
        >>> trained_model.doesnt_match("breakfast cereal dinner lunch".split())
        'cereal'

        """
        self.init_sims()

        used_words = [word for word in words if word in self]
        if len(used_words) != len(words):
            ignored_words = set(words) - set(used_words)
            logger.warning("vectors for words %s are not present in the model, ignoring these words", ignored_words)
        if not used_words:
            raise ValueError("cannot select a word from an empty list")
        vectors = vstack(self.word_vec(word, use_norm=True) for word in used_words).astype(REAL)
        mean = matutils.unitvec(vectors.mean(axis=0)).astype(REAL)
        dists = dot(vectors, mean)
        return sorted(zip(dists, used_words))[0][1]

    @staticmethod
    def cosine_similarities(vector_1, vectors_all):
        """
        Return cosine similarities between one vector and a set of other vectors.

        Parameters
        ----------
        vector_1 : numpy.array
            vector from which similarities are to be computed.
            expected shape (dim,)
        vectors_all : numpy.array
            for each row in vectors_all, distance from vector_1 is computed.
            expected shape (num_vectors, dim)

        Returns
        -------
        :obj: `numpy.array`
            Contains cosine distance between vector_1 and each row in vectors_all.
            shape (num_vectors,)

        """
        norm = np.linalg.norm(vector_1)
        all_norms = np.linalg.norm(vectors_all, axis=1)
        dot_products = dot(vectors_all, vector_1)
        similarities = dot_products / (norm * all_norms)
        return similarities

    def distances(self, word_or_vector, other_words=()):
        """
        Compute cosine distances from given word or vector to all words in `other_words`.
        If `other_words` is empty, return distance between `word_or_vectors` and all words in vocab.

        Parameters
        ----------
        word_or_vector : str or numpy.array
            Word or vector from which distances are to be computed.

        other_words : iterable(str) or None
            For each word in `other_words` distance from `word_or_vector` is computed.
            If None or empty, distance of `word_or_vector` from all words in vocab is computed (including itself).

        Returns
        -------
        numpy.array
            Array containing distances to all words in `other_words` from input `word_or_vector`,
            in the same order as `other_words`.

        Notes
        -----
        Raises KeyError if either `word_or_vector` or any word in `other_words` is absent from vocab.

        """
        if isinstance(word_or_vector, string_types):
            input_vector = self.word_vec(word_or_vector)
        else:
            input_vector = word_or_vector
        if not other_words:
            other_vectors = self.vectors
        else:
            other_indices = [self.vocab[word].index for word in other_words]
            other_vectors = self.vectors[other_indices]
        return 1 - self.cosine_similarities(input_vector, other_vectors)

    def distance(self, w1, w2):
        """
        Compute cosine distance between two words.

        Examples
        --------

        >>> trained_model.distance('woman', 'man')
        0.34

        >>> trained_model.distance('woman', 'woman')
        0.0

        """
        return 1 - self.similarity(w1, w2)

    def similarity(self, w1, w2):
        """
        Compute cosine similarity between two words.

        Examples
        --------

        >>> trained_model.similarity('woman', 'man')
        0.73723527

        >>> trained_model.similarity('woman', 'woman')
        1.0

        """
        return dot(matutils.unitvec(self[w1]), matutils.unitvec(self[w2]))

    def n_similarity(self, ws1, ws2):
        """
        Compute cosine similarity between two sets of words.

        Examples
        --------

        >>> trained_model.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])
        0.61540466561049689

        >>> trained_model.n_similarity(['restaurant', 'japanese'], ['japanese', 'restaurant'])
        1.0000000000000004

        >>> trained_model.n_similarity(['sushi'], ['restaurant']) == trained_model.similarity('sushi', 'restaurant')
        True

        """
        if not(len(ws1) and len(ws2)):
            raise ZeroDivisionError('At least one of the passed list is empty.')
        v1 = [self[word] for word in ws1]
        v2 = [self[word] for word in ws2]
        return dot(matutils.unitvec(array(v1).mean(axis=0)), matutils.unitvec(array(v2).mean(axis=0)))

    @staticmethod
    def log_accuracy(section):
        correct, incorrect = len(section['correct']), len(section['incorrect'])
        if correct + incorrect > 0:
            logger.info(
                "%s: %.1f%% (%i/%i)",
                section['section'], 100.0 * correct / (correct + incorrect), correct, correct + incorrect
            )

    def accuracy(self, questions, restrict_vocab=30000, most_similar=most_similar, case_insensitive=True):
        """
        Compute accuracy of the model. `questions` is a filename where lines are
        4-tuples of words, split into sections by ": SECTION NAME" lines.
        See questions-words.txt in
        https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip
        for an example.

        The accuracy is reported (=printed to log and returned as a list) for each
        section separately, plus there's one aggregate summary at the end.

        Use `restrict_vocab` to ignore all questions containing a word not in the first `restrict_vocab`
        words (default 30,000). This may be meaningful if you've sorted the vocabulary by descending frequency.
        In case `case_insensitive` is True, the first `restrict_vocab` words are taken first, and then
        case normalization is performed.

        Use `case_insensitive` to convert all words in questions and vocab to their uppercase form before
        evaluating the accuracy (default True). Useful in case of case-mismatch between training tokens
        and question words. In case of multiple case variants of a single word, the vector for the first
        occurrence (also the most frequent if vocabulary is sorted) is taken.

        This method corresponds to the `compute-accuracy` script of the original C word2vec.

        """
        ok_vocab = [(w, self.vocab[w]) for w in self.index2word[:restrict_vocab]]
        ok_vocab = {w.upper(): v for w, v in reversed(ok_vocab)} if case_insensitive else dict(ok_vocab)

        sections, section = [], None
        for line_no, line in enumerate(utils.smart_open(questions)):
            # TODO: use level3 BLAS (=evaluate multiple questions at once), for speed
            line = utils.to_unicode(line)
            if line.startswith(': '):
                # a new section starts => store the old section
                if section:
                    sections.append(section)
                    self.log_accuracy(section)
                section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
            else:
                if not section:
                    raise ValueError("missing section header before line #%i in %s" % (line_no, questions))
                try:
                    if case_insensitive:
                        a, b, c, expected = [word.upper() for word in line.split()]
                    else:
                        a, b, c, expected = [word for word in line.split()]
                except ValueError:
                    logger.info("skipping invalid line #%i in %s", line_no, questions)
                    continue
                if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                    logger.debug("skipping line #%i with OOV words: %s", line_no, line.strip())
                    continue

                original_vocab = self.vocab
                self.vocab = ok_vocab
                ignore = {a, b, c}  # input words to be ignored
                predicted = None
                # find the most likely prediction, ignoring OOV words and input words
                sims = most_similar(self, positive=[b, c], negative=[a], topn=False, restrict_vocab=restrict_vocab)
                self.vocab = original_vocab
                for index in matutils.argsort(sims, reverse=True):
                    predicted = self.index2word[index].upper() if case_insensitive else self.index2word[index]
                    if predicted in ok_vocab and predicted not in ignore:
                        if predicted != expected:
                            logger.debug("%s: expected %s, predicted %s", line.strip(), expected, predicted)
                        break
                if predicted == expected:
                    section['correct'].append((a, b, c, expected))
                else:
                    section['incorrect'].append((a, b, c, expected))
        if section:
            # store the last section, too
            sections.append(section)
            self.log_accuracy(section)

        total = {
            'section': 'total',
            'correct': sum((s['correct'] for s in sections), []),
            'incorrect': sum((s['incorrect'] for s in sections), []),
        }
        self.log_accuracy(total)
        sections.append(total)
        return sections

    @staticmethod
    def log_evaluate_word_pairs(pearson, spearman, oov, pairs):
        logger.info('Pearson correlation coefficient against %s: %.4f', pairs, pearson[0])
        logger.info('Spearman rank-order correlation coefficient against %s: %.4f', pairs, spearman[0])
        logger.info('Pairs with unknown words ratio: %.1f%%', oov)

    def evaluate_word_pairs(self, pairs, delimiter='\t', restrict_vocab=300000,
                            case_insensitive=True, dummy4unknown=False):
        """
        Compute correlation of the model with human similarity judgments. `pairs` is a filename of a dataset where
        lines are 3-tuples, each consisting of a word pair and a similarity value, separated by `delimiter`.
        An example dataset is included in Gensim (test/test_data/wordsim353.tsv). More datasets can be found at
        http://technion.ac.il/~ira.leviant/MultilingualVSMdata.html or https://www.cl.cam.ac.uk/~fh295/simlex.html.

        The model is evaluated using Pearson correlation coefficient and Spearman rank-order correlation coefficient
        between the similarities from the dataset and the similarities produced by the model itself.
        The results are printed to log and returned as a triple (pearson, spearman, ratio of pairs with unknown words).

        Use `restrict_vocab` to ignore all word pairs containing a word not in the first `restrict_vocab`
        words (default 300,000). This may be meaningful if you've sorted the vocabulary by descending frequency.
        If `case_insensitive` is True, the first `restrict_vocab` words are taken, and then case normalization
        is performed.

        Use `case_insensitive` to convert all words in the pairs and vocab to their uppercase form before
        evaluating the model (default True). Useful when you expect case-mismatch between training tokens
        and words pairs in the dataset. If there are multiple case variants of a single word, the vector for the first
        occurrence (also the most frequent if vocabulary is sorted) is taken.

        Use `dummy4unknown=True` to produce zero-valued similarities for pairs with out-of-vocabulary words.
        Otherwise (default False), these pairs are skipped entirely.
        """
        ok_vocab = [(w, self.vocab[w]) for w in self.index2word[:restrict_vocab]]
        ok_vocab = {w.upper(): v for w, v in reversed(ok_vocab)} if case_insensitive else dict(ok_vocab)

        similarity_gold = []
        similarity_model = []
        oov = 0

        original_vocab = self.vocab
        self.vocab = ok_vocab

        for line_no, line in enumerate(utils.smart_open(pairs)):
            line = utils.to_unicode(line)
            if line.startswith('#'):
                # May be a comment
                continue
            else:
                try:
                    if case_insensitive:
                        a, b, sim = [word.upper() for word in line.split(delimiter)]
                    else:
                        a, b, sim = [word for word in line.split(delimiter)]
                    sim = float(sim)
                except (ValueError, TypeError):
                    logger.info('skipping invalid line #%d in %s', line_no, pairs)
                    continue
                if a not in ok_vocab or b not in ok_vocab:
                    oov += 1
                    if dummy4unknown:
                        similarity_model.append(0.0)
                        similarity_gold.append(sim)
                        continue
                    else:
                        logger.debug('skipping line #%d with OOV words: %s', line_no, line.strip())
                        continue
                similarity_gold.append(sim)  # Similarity from the dataset
                similarity_model.append(self.similarity(a, b))  # Similarity from the model
        self.vocab = original_vocab
        spearman = stats.spearmanr(similarity_gold, similarity_model)
        pearson = stats.pearsonr(similarity_gold, similarity_model)
        oov_ratio = float(oov) / (len(similarity_gold) + oov) * 100

        logger.debug('Pearson correlation coefficient against %s: %f with p-value %f', pairs, pearson[0], pearson[1])
        logger.debug(
            'Spearman rank-order correlation coefficient against %s: %f with p-value %f',
            pairs, spearman[0], spearman[1]
        )
        logger.debug('Pairs with unknown words: %d', oov)
        self.log_evaluate_word_pairs(pearson, spearman, oov_ratio, pairs)
        return pearson, spearman, oov_ratio

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training** after doing a replace. The model becomes
        effectively read-only = you can call `most_similar`, `similarity` etc., but not `train`.

        """
        if getattr(self, 'vectors_norm', None) is None or replace:
            logger.info("precomputing L2-norms of word weight vectors")
            if replace:
                for i in xrange(self.vectors.shape[0]):
                    self.vectors[i, :] /= sqrt((self.vectors[i, :] ** 2).sum(-1))
                self.vectors_norm = self.vectors
            else:
                self.vectors_norm = (self.vectors / sqrt((self.vectors ** 2).sum(-1))[..., newaxis]).astype(REAL)


class Word2VecKeyedVectors(WordEmbeddingsKeyedVectors):
    """Class to contain vectors and vocab for the Word2Vec training class and other w2v methods not directly
    involved in training such as most_similar()
    """
    def save_word2vec_format(self, fname, fvocab=None, binary=False, total_vec=None):
        """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        Parameters
        ----------
        fname : str
            The file path used to save the vectors in
        fvocab : str
            Optional file path used to save the vocabulary
        binary : bool
            If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
        total_vec :  int
            Optional parameter to explicitly specify total no. of vectors
            (in case word vectors are appended with document vectors afterwards)

        Returns
        -------
        None

        """
        # from gensim.models.word2vec import save_word2vec_format
        save_word2vec_format(
            fname, self.vocab, self.vectors, fvocab=fvocab, binary=binary, total_vec=total_vec)

    @classmethod
    def load_word2vec_format(cls, fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict',
                             limit=None, datatype=REAL):
        """Load the input-hidden weight matrix from the original C word2vec-tool format.

        Note that the information stored in the file is incomplete (the binary tree is missing),
        so while you can query for word similarity etc., you cannot continue training
        with a model loaded this way.

        Parameters
        ----------
        fname : str
            The file path to the saved word2vec-format file.
        fvocab : str
                Optional file path to the vocabulary.Word counts are read from `fvocab` filename,
                if set (this is the file generated by `-save-vocab` flag of the original C tool).
        binary : bool
            If True, indicates whether the data is in binary word2vec format.
        encoding : str
            If you trained the C model using non-utf8 encoding for words, specify that
            encoding in `encoding`.
        unicode_errors : str
            default 'strict', is a string suitable to be passed as the `errors`
            argument to the unicode() (Python 2.x) or str() (Python 3.x) function. If your source
            file may include word tokens truncated in the middle of a multibyte unicode character
            (as is common from the original word2vec.c tool), 'ignore' or 'replace' may help.
        limit : int
            Sets a maximum number of word-vectors to read from the file. The default,
            None, means read all.
        datatype : :class: `numpy.float*`
            (Experimental) Can coerce dimensions to a non-default float type (such
            as np.float16) to save memory. (Such types may result in much slower bulk operations
            or incompatibility with optimized routines.)

        Returns
        -------
        :obj: `~gensim.models.word2vec.Wod2Vec`
            Returns the loaded model as an instance of :class: `~gensim.models.word2vec.Wod2Vec`.

        """
        # from gensim.models.word2vec import load_word2vec_format
        return load_word2vec_format(
            Word2VecKeyedVectors, fname, fvocab=fvocab, binary=binary, encoding=encoding, unicode_errors=unicode_errors,
            limit=limit, datatype=datatype)

    def get_keras_embedding(self, train_embeddings=False):
        """Return a Keras 'Embedding' layer with weights set as the Word2Vec model's learned word embeddings

        Parameter
        ---------
        train_embeddings : bool
            If False, the weights are frozen and stopped from being updated.
            If True, the weights can/will be further trained/updated.

        Return
        ------
        :obj: `keras.layers.Embedding`
            Embedding layer

        """
        try:
            from keras.layers import Embedding
        except ImportError:
            raise ImportError("Please install Keras to use this function")
        weights = self.vectors

        # set `trainable` as `False` to use the pretrained word embedding
        # No extra mem usage here as `Embedding` layer doesn't create any new matrix for weights
        layer = Embedding(
            input_dim=weights.shape[0], output_dim=weights.shape[1],
            weights=[weights], trainable=train_embeddings
        )
        return layer
KeyedVectors = Word2VecKeyedVectors  # alias for backward compatibility


class Doc2VecKeyedVectors(BaseKeyedVectors):
    def __init__(self):
        self.doctags = {}  # string -> Doctag (only filled if necessary)
        self.max_rawint = -1  # highest rawint-indexed doctag
        self.offset2doctag = []  # int offset-past-(max_rawint+1) -> String (only filled if necessary)
        self.count = 0
        self.vectors_docs = []
        self.mapfile_path = None

    @property
    def index2entity(self):
        return self.offset2doctag

    @index2entity.setter
    def index2entity(self, value):
        self.offset2doctag = value

    @property
    @deprecated("Attribute will be removed in 4.0.0, use docvecs.vectors_docs instead")
    def doctag_syn0(self):
        return self.vectors_docs

    @property
    @deprecated("Attribute will be removed in 4.0.0, use docvecs.vectors_docs_norm instead")
    def doctag_syn0norm(self):
        return self.vectors_docs_norm

    def __getitem__(self, index):
        """
        Accept a single key (int or string tag) or list of keys as input.

        If a single string or int, return designated tag's vector
        representation, as a 1D numpy array.

        If a list, return designated tags' vector representations as a
        2D numpy array: #tags x #vector_size.
        """
        if index in self:
            if isinstance(index, string_types + integer_types + (integer,)):
                return self.vectors_docs[self._int_index(index, self.doctags, self.max_rawint)]
            return vstack([self[i] for i in index])
        raise KeyError("tag '%s' not seen in training corpus/invalid" % index)

    def __contains__(self, index):
        if isinstance(index, integer_types + (integer,)):
            return index < self.count
        else:
            return index in self.doctags

    def __len__(self):
        return self.count

    def save(self, *args, **kwargs):
        """Saves the keyedvectors. This saved model can be loaded again using
        :func:`~gensim.models.doc2vec.Doc2VecKeyedVectors.load` which supports
        operations on trained document vectors like `most_similar`.

        Parameters
        ----------
        fname : str
            Path to the file.

        Returns
        -------
        None

        """
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['vectors_docs_norm'])
        super(Doc2VecKeyedVectors, self).save(*args, **kwargs)

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training or inference** after doing a replace.
        The model becomes effectively read-only = you can call `most_similar`, `similarity`
        etc., but not `train` or `infer_vector`.

        """
        if getattr(self, 'vectors_docs_norm', None) is None or replace:
            logger.info("precomputing L2-norms of doc weight vectors")
            if replace:
                for i in xrange(self.vectors_docs.shape[0]):
                    self.vectors_docs[i, :] /= sqrt((self.vectors_docs[i, :] ** 2).sum(-1))
                self.vectors_docs_norm = self.vectors_docs
            else:
                if self.mapfile_path:
                    self.vectors_docs_norm = np_memmap(
                        self.mapfile_path + '.vectors_docs_norm', dtype=REAL,
                        mode='w+', shape=self.vectors_docs.shape)
                else:
                    self.vectors_docs_norm = empty(self.vectors_docs.shape, dtype=REAL)
                np_divide(
                    self.vectors_docs, sqrt((self.vectors_docs ** 2).sum(-1))[..., newaxis], self.vectors_docs_norm)

    def most_similar(self, positive=None, negative=None, topn=10, clip_start=0, clip_end=None, indexer=None):
        """
        Find the top-N most similar docvecs known from training. Positive docs contribute
        positively towards the similarity, negative docs negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given docs. Docs may be specified as vectors, integer indexes
        of trained docvecs, or if the documents were originally presented with string tags,
        by the corresponding tags.

        The 'clip_start' and 'clip_end' allow limiting results to a particular contiguous
        range of the underlying `vectors_docs_norm` vectors. (This may be useful if the ordering
        there was chosen to be significant, such as more popular tag IDs in lower indexes.)

        Parameters
        ----------
        positive : :obj: `list`
            List of Docs specifed as vectors, integer indexes of trained docvecs or string tags
            that contribute positively.
        negative : :obj: `list`
            List of Docs specifed as vectors, integer indexes of trained docvecs or string tags
            that contribute negatively.
        topn : int
            Number of top-N similar docvecs to return.
        clip_start : int
            Start clipping index.
        clip_end : int
            End clipping index.

        Returns
        -------
        :obj: `list` of :obj: `tuple`
            Returns a list of tuples (doc, similarity)

        """
        if positive is None:
            positive = []
        if negative is None:
            negative = []

        self.init_sims()
        clip_end = clip_end or len(self.vectors_docs_norm)

        if isinstance(positive, string_types + integer_types + (integer,)) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each doc, if not already present; default to 1.0 for positive and -1.0 for negative docs
        positive = [
            (doc, 1.0) if isinstance(doc, string_types + integer_types + (ndarray, integer))
            else doc for doc in positive
        ]
        negative = [
            (doc, -1.0) if isinstance(doc, string_types + integer_types + (ndarray, integer))
            else doc for doc in negative
        ]

        # compute the weighted average of all docs
        all_docs, mean = set(), []
        for doc, weight in positive + negative:
            if isinstance(doc, ndarray):
                mean.append(weight * doc)
            elif doc in self.doctags or doc < self.count:
                mean.append(weight * self.vectors_docs_norm[self._int_index(doc, self.doctags, self.max_rawint)])
                all_docs.add(self._int_index(doc, self.doctags, self.max_rawint))
            else:
                raise KeyError("doc '%s' not in trained set" % doc)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        if indexer is not None:
            return indexer.most_similar(mean, topn)

        dists = dot(self.vectors_docs_norm[clip_start:clip_end], mean)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_docs), reverse=True)
        # ignore (don't return) docs from the input
        result = [
            (self._index_to_doctag(sim + clip_start, self.offset2doctag, self.max_rawint), float(dists[sim]))
            for sim in best
            if (sim + clip_start) not in all_docs
        ]
        return result[:topn]

    def doesnt_match(self, docs):
        """
        Which doc from the given list doesn't go with the others?

        (TODO: Accept vectors of out-of-training-set docs, as if from inference.)

        Parameters
        ----------
        docs : :obj: `list` of (str or int)
            List of seen documents specified by their corresponding string tags or integer indices.

        Returns
        -------
        str or int
            The document further away from the mean of all the documents.

        """
        self.init_sims()

        docs = [doc for doc in docs if doc in self.doctags or 0 <= doc < self.count]  # filter out unknowns
        logger.debug("using docs %s", docs)
        if not docs:
            raise ValueError("cannot select a doc from an empty list")
        vectors = vstack(
            self.vectors_docs_norm[self._int_index(doc, self.doctags, self.max_rawint)] for doc in docs).astype(REAL)
        mean = matutils.unitvec(vectors.mean(axis=0)).astype(REAL)
        dists = dot(vectors, mean)
        return sorted(zip(dists, docs))[0][1]

    def similarity(self, d1, d2):
        """
        Compute cosine similarity between two docvecs in the trained set, specified by int index or
        string tag. (TODO: Accept vectors of out-of-training-set docs, as if from inference.)

        Parameters
        ----------
        d1 : int or str
            Indicate the first document by it's string tag or integer index.
        d2 : int or str
            Indicate the second document by it's string tag or integer index.

        Returns
        -------
        float
            The cosine similarity between the vectors of the two documents.

        """
        return dot(matutils.unitvec(self[d1]), matutils.unitvec(self[d2]))

    def n_similarity(self, ds1, ds2):
        """
        Compute cosine similarity between two sets of docvecs from the trained set, specified by int
        index or string tag. (TODO: Accept vectors of out-of-training-set docs, as if from inference.)

        Parameters
        ----------
        ds1 : :obj: `list` of (str or int)
            Specify the first set of documents as a list of their integer indices or string tags.
        ds2 : :obj: `list` of (str or int)
            Specify the second set of documents as a list of their integer indices or string tags.

        Returns
        -------
        float
            The cosine similarity between the means of the documents in each of the two sets.

        """
        v1 = [self[doc] for doc in ds1]
        v2 = [self[doc] for doc in ds2]
        return dot(matutils.unitvec(array(v1).mean(axis=0)), matutils.unitvec(array(v2).mean(axis=0)))

    # required by base keyed vectors class
    def distances(self, d1, other_docs=()):
        """Compute distances from given document (string tag or int index) to all documents in `other_docs`.
        If `other_docs` is empty, return distance between `d1` and all documents seen during training.
        """
        input_vector = self[d1]
        if not other_docs:
            other_vectors = self.vectors_docs
        else:
            other_vectors = self[other_vectors]
        return 1 - WordEmbeddingsKeyedVectors.cosine_similarities(input_vector, other_vectors)

    def similarity_unseen_docs(self, model, doc_words1, doc_words2, alpha=0.1, min_alpha=0.0001, steps=5):
        """
        Compute cosine similarity between two post-bulk out of training documents.

        Parameters
        ----------
        model : :obj: `~gensim.models.doc2vec.Doc2Vec`
            An instance of a trained `Doc2Vec` model.
        doc_words1 : :obj: `list` of :obj: `str`
            The first document. Document should be a list of (word) tokens.
        doc_words2 : :obj: `list` of :obj: `str`
            The second document. Document should be a list of (word) tokens.
        alpha : float
            The initial learning rate.
        min_alpha : float
            Learning rate will linearly drop to `min_alpha` as training progresses.
        steps : int
            Number of times to train the new document.

        Returns
        -------
        float
            The cosine similarity between the unseen documents.

        """
        d1 = model.infer_vector(doc_words=doc_words1, alpha=alpha, min_alpha=min_alpha, steps=steps)
        d2 = model.infer_vector(doc_words=doc_words2, alpha=alpha, min_alpha=min_alpha, steps=steps)
        return dot(matutils.unitvec(d1), matutils.unitvec(d2))

    def save_word2vec_format(self, fname, prefix='*dt_', fvocab=None,
                             total_vec=None, binary=False, write_first_line=True):
        """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        Parameters
        ----------
        fname : str
            The file path used to save the vectors in.
        prefix : str
            Uniquely identifies doctags from word vocab, and avoids collision
            in case of repeated string in doctag and word vocab.
        fvocab : str
            Optional file path used to save the vocabulary
        binary : bool
            If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
        total_vec :  int
            Optional parameter to explicitly specify total no. of vectors
            (in case word vectors are appended with document vectors afterwards)
        write_first_line : bool
            Whether to print the first line in the file. Useful when saving doc-vectors after word-vectors.

        Returns
        -------
        None

        """
        total_vec = total_vec or len(self)
        with utils.smart_open(fname, 'ab') as fout:
            if write_first_line:
                logger.info("storing %sx%s projection weights into %s", total_vec, self.vectors_docs.shape[1], fname)
                fout.write(utils.to_utf8("%s %s\n" % (total_vec, self.vectors_docs.shape[1])))
            # store as in input order
            for i in range(len(self)):
                doctag = u"%s%s" % (prefix, self._index_to_doctag(i, self.offset2doctag, self.max_rawint))
                row = self.vectors_docs[i]
                if binary:
                    fout.write(utils.to_utf8(doctag) + b" " + row.tostring())
                else:
                    fout.write(utils.to_utf8("%s %s\n" % (doctag, ' '.join("%f" % val for val in row))))

    @staticmethod
    def _int_index(index, doctags, max_rawint):
        """Return int index for either string or int index"""
        if isinstance(index, integer_types + (integer,)):
            return index
        else:
            return max_rawint + 1 + doctags[index].offset

    @staticmethod
    def _index_to_doctag(i_index, offset2doctag, max_rawint):
        """Return string key for given i_index, if available. Otherwise return raw int doctag (same int)."""
        candidate_offset = i_index - max_rawint - 1
        if 0 <= candidate_offset < len(offset2doctag):
            return offset2doctag[candidate_offset]
        else:
            return i_index

    # for backward compatibility
    def index_to_doctag(self, i_index):
        """Return string key for given i_index, if available. Otherwise return raw int doctag (same int)."""
        candidate_offset = i_index - self.max_rawint - 1
        if 0 <= candidate_offset < len(self.offset2doctag):
            return self.ffset2doctag[candidate_offset]
        else:
            return i_index

    # for backward compatibility
    def int_index(index, doctags, max_rawint):
        """Return int index for either string or int index"""
        if isinstance(index, integer_types + (integer,)):
            return index
        else:
            return max_rawint + 1 + doctags[index].offset


class FastTextKeyedVectors(WordEmbeddingsKeyedVectors):
    """
    Class to contain vectors and vocab for the FastText training class and other methods not directly
    involved in training such as most_similar()
    """

    def __init__(self):
        super(FastTextKeyedVectors, self).__init__()
        self.vectors_vocab = None
        self.vectors_vocab_norm = None
        self.vectors_ngrams = None
        self.vectors_ngrams_norm = None
        self.ngrams = {}
        self.hash2index = {}
        self.ngrams_word = {}
        self.min_n = 0
        self.max_n = 0

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.vectors_vocab instead")
    def syn0_vocab(self):
        return self.vectors_vocab

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.vectors_vocab_norm instead")
    def syn0_vocab_norm(self):
        return self.vectors_vocab_norm

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.vectors_ngrams instead")
    def syn0_ngrams(self):
        return self.vectors_ngrams

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.vectors_ngrams_norm instead")
    def syn0_ngrams_norm(self):
        return self.vectors_ngrams_norm

    @property
    def num_ngram_vectors(self):
        return self.trainables.num_ngram_vectors

    def __contains__(self, word):
        """
        Check if `word` or any character ngrams in `word` are present in the vocabulary.
        A vector for the word is guaranteed to exist if `__contains__` returns True.
        """
        if word in self.vocab:
            return True
        else:
            # from gensim.models.fasttext import compute_ngrams
            char_ngrams = compute_ngrams(word, self.min_n, self.max_n)
            return any(ng in self.ngrams for ng in char_ngrams)

    def save(self, *args, **kwargs):
        """Saves the keyedvectors. This saved model can be loaded again using
        :func:`~gensim.models.fasttext.FastTextKeyedVectors.load` which supports
        getting vectors for out-of-vocabulary words.

        Parameters
        ----------
        fname : str
            Path to the file.

        Returns
        -------
        None

        """
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['vectors_norm', 'vectors_vocab_norm', 'vectors_ngrams_norm'])
        super(FastTextKeyedVectors, self).save(*args, **kwargs)

    def word_vec(self, word, use_norm=False):
        """
        Accept a single word as input.
        Returns the word's representations in vector space, as a 1D numpy array.

        If `use_norm` is True, returns the normalized word vector.

        """
        if word in self.vocab:
            return super(FastTextKeyedVectors, self).word_vec(word, use_norm)
        else:
            # from gensim.models.fasttext import compute_ngrams
            word_vec = np.zeros(self.vectors_ngrams.shape[1], dtype=np.float32)
            ngrams = compute_ngrams(word, self.min_n, self.max_n)
            ngrams = [ng for ng in ngrams if ng in self.ngrams]
            if use_norm:
                ngram_weights = self.vectors_ngrams_norm
            else:
                ngram_weights = self.vectors_ngrams
            for ngram in ngrams:
                word_vec += ngram_weights[self.ngrams[ngram]]
            if word_vec.any():
                return word_vec / len(ngrams)
            else:  # No ngrams of the word are present in self.ngrams
                raise KeyError('all ngrams for word %s absent from model' % word)

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training** after doing a replace. The model becomes
        effectively read-only = you can only call `most_similar`, `similarity` etc.

        """
        super(FastTextKeyedVectors, self).init_sims(replace)
        if getattr(self, 'vectors_ngrams_norm', None) is None or replace:
            logger.info("precomputing L2-norms of ngram weight vectors")
            if replace:
                for i in range(self.vectors_ngrams.shape[0]):
                    self.vectors_ngrams[i, :] /= sqrt((self.vectors_ngrams[i, :] ** 2).sum(-1))
                self.vectors_ngrams_norm = self.vectors_ngrams
            else:
                self.vectors_ngrams_norm = \
                    (self.vectors_ngrams / sqrt((self.vectors_ngrams ** 2).sum(-1))[..., newaxis]).astype(REAL)

    def save_word2vec_format(self, fname, fvocab=None, binary=False, total_vec=None):
        """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        Parameters
        ----------
        fname : str
            The file path used to save the vectors in.
        fvocab : str
            Optional file path used to save the vocabulary.
        binary : bool
            If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
        total_vec :  int
            Optional parameter to explicitly specify total no. of vectors
            (in case word vectors are appended with document vectors afterwards).

        Returns
        -------
        None

        """
        # from gensim.models.word2vec import save_word2vec_format
        save_word2vec_format(
            fname, self.vocab, self.vectors, fvocab=fvocab, binary=binary, total_vec=total_vec)
