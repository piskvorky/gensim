#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""Scikit learn interface for :class:`~gensim.models.tfidfmodel.TfidfModel`.

Follows scikit-learn API conventions to facilitate using gensim along with scikit-learn.

Examples
--------

>>> from gensim.test.utils import common_corpus, common_dictionary
>>> from gensim.sklearn_api import TfIdfTransformer
>>>
>>> # Transform the word counts inversely to their global frequency using the sklearn interface.
>>> model = TfIdfTransformer(dictionary=common_dictionary)
>>> weighted_corpus = model.fit_transform(common_corpus)
>>> weighted_corpus[0]
[(0, 0.57735026918962573), (1, 0.57735026918962573), (2, 0.57735026918962573)]

"""


from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from gensim.models import TfidfModel
import gensim


class TfIdfTransformer(TransformerMixin, BaseEstimator):
    """Base TfIdf module, wraps :class:`~gensim.models.tfidfmodel.TfidfModel`.

    For more information on the inner workings please take a look at
    the original class.

    """

    def __init__(self, id2word=None, dictionary=None, wlocal=gensim.utils.identity,
                 wglobal=gensim.models.tfidfmodel.df2idf, normalize=True, smartirs="ntc",
                 pivot=None, slope=0.65):
        """Sklearn wrapper for TfIdf model.

        Parameters
        ----------

        id2word : {dict, :class:`~gensim.corpora.Dictionary`}, optional
            Mapping from int id to word token, that was used for converting input data to bag of words format.
        dictionary : :class:`~gensim.corpora.Dictionary`, optional
            If specified it will be used to directly construct the inverse document frequency mapping.
        wlocals : function, optional
            Function for local weighting, default for `wlocal` is :func:`~gensim.utils.identity` which does nothing.
            Other options include :func:`math.sqrt`, :func:`math.log1p`, etc.
        wglobal : function, optional
            Function for global weighting, default is :func:`~gensim.models.tfidfmodel.df2idf`.
        normalize : bool, optional
            It dictates how the final transformed vectors will be normalized. `normalize=True` means set to unit length
            (default); `False` means don't normalize. You can also set `normalize` to your own function that accepts
            and returns a sparse vector.
        smartirs : str, optional
            SMART (System for the Mechanical Analysis and Retrieval of Text) Information Retrieval System,
            a mnemonic scheme for denoting tf-idf weighting variants in the vector space model.
            The mnemonic for representing a combination of weights takes the form XYZ,
            for example 'ntc', 'bpn' and so on, where the letters represents the term weighting of the document vector.

            Term frequency weighing:
                * `n` - natural,
                * `l` - logarithm,
                * `a` - augmented,
                * `b` - boolean,
                * `L` - log average.

            Document frequency weighting:
                * `n` - none,
                * `t` - idf,
                * `p` - prob idf.

            Document normalization:
                * `n` - none,
                * `c` - cosine.

            For more info, visit `"Wikipedia" <https://en.wikipedia.org/wiki/SMART_Information_Retrieval_System>`_.

        """
        self.gensim_model = None
        self.id2word = id2word
        self.dictionary = dictionary
        self.wlocal = wlocal
        self.wglobal = wglobal
        self.normalize = normalize
        self.smartirs = smartirs
        self.slope = slope
        self.pivot = pivot

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : iterable of iterable of (int, int)
            Input corpus

        Returns
        -------
        :class:`~gensim.sklearn_api.tfidf.TfIdfTransformer`
            The trained model.

        """
        self.gensim_model = TfidfModel(
            corpus=X, id2word=self.id2word, dictionary=self.dictionary, wlocal=self.wlocal,
            wglobal=self.wglobal, normalize=self.normalize, smartirs=self.smartirs,
            pivot=self.pivot, slope=self.slope
        )
        return self

    def transform(self, docs):
        """Get the transformed documents after multiplication with the tf-idf matrix.

        Parameters
        ----------
        docs: iterable of iterable of (int, int)
            Input corpus in BoW format.

        Returns
        -------
        iterable of list (int, float) 2-tuples.
            The BOW representation of each document. Will have  the same shape as `docs`.

        """
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # input as python lists
        if isinstance(docs[0], tuple):
            docs = [docs]
        return [self.gensim_model[doc] for doc in docs]
