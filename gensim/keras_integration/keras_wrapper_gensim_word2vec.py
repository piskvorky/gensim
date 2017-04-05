from keras.layers import Embedding
from gensim import models

class KerasWrapperWord2VecModel(models.Word2Vec):
    """
    Class to integrate Keras with Gensim's Word2Vec model
    """

    def __init__(
            self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
            max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
            trim_rule=None, sorted_vocab=1, batch_words=10000):

        """
        Keras wrapper for Word2Vec model. Class derived from gensim.model.Word2Vec.
        """
        self.sentences=sentences
        self.size=size
        self.alpha=alpha
        self.window=window
        self.min_count=min_count
        self.max_vocab_size=max_vocab_size
        self.sample=sample
        self.seed=seed
        self.workers=workers
        self.min_alpha=min_alpha
        self.sg=sg
        self.hs=hs
        self.negative=negative
        self.cbow_mean=cbow_mean
        self.hashfxn=hashfxn
        self.iter=iter
        self.null_word=null_word
        self.trim_rule=trim_rule
        self.sorted_vocab=sorted_vocab
        self.batch_words=batch_words

        models.Word2Vec.__init__(self, sentences=self.sentences, size=self.size, alpha=self.alpha, window=self.window, min_count=self.min_count,
            max_vocab_size=self.max_vocab_size, sample=self.sample, seed=self.seed, workers=self.workers, min_alpha=self.min_alpha,
            sg=self.sg, hs=self.hs, negative=self.negative, cbow_mean=self.cbow_mean, hashfxn=self.hashfxn, iter=self.iter, null_word=self.null_word,
            trim_rule=self.trim_rule, sorted_vocab=self.sorted_vocab, batch_words=self.batch_words)

    def get_embedding_layer(self):
        """
        Return a Keras 'Embedding' layer with weights set as our Word2Vec model's learned word embeddings
        """
        weights = self.wv.syn0
        layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights], trainable=False)      #set `trainable` as `False` to use the pretrained word embedding
        return layer
