from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.models.word2vec import MAX_WORDS_IN_BATCH
from numpy import vstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class SklD2VModel(BaseEstimator, TransformerMixin):

    _PARAMS = [
        'size', 'alpha', 'window', 'min_count',
        'max_vocab_size', 'sample', 'seed', 'workers',
        'min_alpha', 'hs', 'negative', 'hashfxn', 'iter',
        'trim_rule', 'sorted_vocab', 'batch_words', 'compute_loss',
        'dm_mean', 'dm', 'dbow_words', 'dm_concat',
        'dm_tag_count'
    ]

    def __init__(self, size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, sample=1e-3, seed=1, workers=3,
                 min_alpha=0.0001, hs=0, negative=5, hashfxn=hash, iter=5,
                 trim_rule=None, sorted_vocab=1,
                 batch_words=MAX_WORDS_IN_BATCH, compute_loss=False,
                 dm_mean=None, dm=1, dbow_words=0, dm_concat=0,
                 dm_tag_count=1):
        for param in self._PARAMS:
            setattr(self, param, locals()[param])

    def fit(self, X, y=None):
        """
        Train the model while manually decreasing the learning rate
         see https://rare-technologies.com/doc2vec-tutorial/#training
         for details

        :param X: an iterable of gensim.models.doc2vec.TaggedDocument
        :return: a trained SklD2VModel instance
        """
        # initialize model with doc2vec parameters
        params = {param: getattr(self, param, None) for param in self._PARAMS}
        self.gensim_model_ = Doc2Vec(**params)
        # learn the vocabulary from X
        self.gensim_model_.build_vocab(X)
        corpus_count = self.gensim_model_.corpus_count
        # train the model while manually controlling alpha
        alpha_step = (self.alpha - self.min_alpha) / self.iter
        for i in range(self.iter):
            self.gensim_model_.train(X, total_examples=corpus_count, epochs=1)
            self.gensim_model_.alpha -= alpha_step
            self.gensim_model_.min_alpha = self.gensim_model_.alpha
        return self

    def transform(self, X):
        """
        Transform TaggedDocument to their doc2vec representation

        :param X: an iterable of gensim.models.doc2vec.TaggedDocument
        :return: an array of shape (n_samples, n_features)
        """
        check_is_fitted(self, 'gensim_model_')
        return vstack([
            self.gensim_model_.infer_vector(
                x.words, self.alpha, self.min_alpha, self.iter)
            for x in X
        ])

    def fit_transform(self, X, y=None, **fit_params):
        """
        Train the model on TaggedDocument and
         then convert the latter to their doc2vec representation

        :param X: an iterable of gensim.models.doc2vec.TaggedDocument
        :return: an array of shape (n_samples, n_features)
        """
        self.fit(X)
        return self.gensim_model_.docvecs[range(len(X))]


class TaggedDocumentTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        """
        Take a list of tokenized documents, and tag them with their index

        :param X: an iterable of tokenized documents (i.e. list of strings)
        :return: an list of gensim.models.doc2vec.TaggedDocumentof, shape (n_samples)
        """
        return [
            TaggedDocument(words=list(x), tags=[i])
            for i, x in enumerate(X)
        ]
