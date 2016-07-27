import tensorflow as tf
from tensorflow.models.embedding.word2vec_optimized import Word2Vec
from gensim.models.word2vec import Word2Vec as GensimWord2Vec
from six import string_types
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Options(object):
    """Options class that doesn't use FLAGS"""

    def __init__(self, train_data=None, save_path=None, eval_data=None,
                 embedding_size=200, epochs_to_train=15, learning_rate=0.025,
                 num_neg_samples=25, batch_size=500, concurrent_steps=12,
                 window_size=5, min_count=5, subsample=1e-3):
        """
        train_data: Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.

        save_path: Directory to write the model.

        eval_data: Analogy questions.

        embedding_size: The embedding dimension size.

        epochs_to_train: Number of epochs to train. Each epoch processes the training data once

        learning_rate: Initial learning rate

        batch_size: Numbers of training examples each step processes

        concurrent_steps: The number of concurrent training steps.

        window_size: The number of worlds to predict to the left and right of the target word.

        min_count: The minimum number of word occurrences for it to be included in the vocabulary.

        subsample: Subsample threshold for word occurrence. Words that appear with higher frequency
                   will be randomly down-sampled. Set to 0 to disable.
        """

        if train_data is None or save_path is None:
            raise ValueError("train_data and save_path must be specified.")

        # Model options.

        # Embedding dimension.
        self.emb_dim = embedding_size

        # Training options.

        # The training text file.
        self.train_data = train_data

        # Number of negative samples per example.
        self.num_samples = num_neg_samples

        # The initial learning rate.
        self.learning_rate = learning_rate

        # Number of epochs to train. After these many epochs, the learning
        # rate decays linearly to zero and the training stops.
        self.epochs_to_train = epochs_to_train

        # Concurrent training steps.
        self.concurrent_steps = concurrent_steps

        # Number of examples for one training step.
        self.batch_size = batch_size

        # The number of words to predict to the left and right of the target word.
        self.window_size = window_size

        # The minimum number of word occurrences for it to be included in the
        # vocabulary.
        self.min_count = min_count

        # Subsampling threshold for word occurrence.
        self.subsample = subsample

        # Where to write out summaries.
        self.save_path = save_path

        # Eval options.

        # The text file for eval.
        self.eval_data = eval_data


class Vocab(object):

    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))


class TFWord2Vec(GensimWord2Vec):

    def __init__(self, train_data=None, save_path=None, eval_data=None,
                 embedding_size=200, epochs_to_train=15, learning_rate=0.025,
                 num_neg_samples=25, batch_size=500, concurrent_steps=12,
                 window_size=5, min_count=5, subsample=1e-3):

        self.options = Options(train_data=train_data, save_path=save_path, eval_data=eval_data,
                               embedding_size=embedding_size, epochs_to_train=epochs_to_train,
                               learning_rate=learning_rate, num_neg_samples=num_neg_samples,
                               batch_size=batch_size, concurrent_steps=concurrent_steps,
                               window_size=window_size, min_count=min_count, subsample=subsample)
        self.train()
        self.vocab = {}
        self.create_vocab()

    def train(self):
        with tf.Graph().as_default(), tf.Session() as session:
            self.model = Word2Vec(self.options, session)
            for _ in xrange(self.options.epochs_to_train):
                self.model.train()  # Process one epoch
                if self.options.eval_data is not None:
                    self.model.eval()  # Eval analogies.'''

            self.syn0 = self.model._w_in
            self.syn0norm = session.run(tf.nn.l2_normalize(self.model._w_in, 1))
            self.index2word = self.model._id2word

    def create_vocab(self):
        print self.options.vocab_words
        for word in self.options.vocab_words:
            self.vocab[word] = Vocab(index=self.model._word2id[word])

    def __getitem__(self, words):
        if isinstance(words, string_types):
            # allow calls like trained_model['office'], as a shorthand for trained_model[['office']]
            return self.syn0norm[self.model._word2id[words]]

        ids = [self.model._word2id[word] for word in words]
        return [self.syn0norm[id] for id in ids]
