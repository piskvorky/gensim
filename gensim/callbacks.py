#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class to build callbacks. Callbacks are used to apply custom functions over the model at specific points
during training (epoch start, batch end etc.). To implement a Callback, subclass :class:`~gensim.callbacks.Callback`,
look at the example below which creates a callback to save a training model
after each epoch:

>>> from gensim.test.utils import common_texts as sentences
>>> from gensim.callbacks import Callback
>>> from gensim.models import Word2Vec
>>> from gensim.test.utils import get_tmpfile
>>>
>>> class EpochSaver(Callback):
...     "Callback to save model after every epoch"
...     def __init__(self, path_prefix):
...         self.path_prefix = path_prefix
...         self.epoch = 0
...     def on_epoch_end(self, model):
...         output_path = '{}_epoch{}.model'.format(self.path_prefix, self.epoch)
...         print("Save model to {}".format(output_path))
...         model.save(output_path)
...         self.epoch += 1
...
>>>
>>> class EpochLogger(Callback):
...     "Callback to log information about training"
...     def __init__(self):
...         self.epoch = 0
...     def on_epoch_begin(self, model):
...         print("Epoch #{} start".format(self.epoch))
...     def on_epoch_end(self, model):
...         print("Epoch #{} end".format(self.epoch))
...         self.epoch += 1
...
>>> epoch_saver = EpochSaver(get_tmpfile("temporary_model"))
>>> epoch_logger = EpochLogger()
>>> w2v_model = Word2Vec(sentences, iter=5, size=10, min_count=0, seed=42, callbacks=[epoch_saver, epoch_logger])

"""


class Callback(object):
    """Base class used to build new callbacks."""

    def on_epoch_begin(self, model):
        """Method called on the start of epoch.

        Parameters
        ----------
        model : class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`
            Current model.

        """
        pass

    def on_epoch_end(self, model):
        """Method called on the end of epoch.

        Parameters
        ----------
        model

        Returns
        -------

        """
        pass

    def on_batch_begin(self, model):
        """Method called on the start of batch.

        Parameters
        ----------
        model : class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`
            Current model.

        """
        pass

    def on_batch_end(self, model):
        """Method called on the end of batch.

        Parameters
        ----------
        model : class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`
            Current model.

        """
        pass

    def on_train_begin(self, model):
        """Method called on the start of training process.

        Parameters
        ----------
        model : class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`
            Current model.

        """
        pass

    def on_train_end(self, model):
        """Method called on the end of training process.

        Parameters
        ----------
        model : class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`
            Current model.

        """
        pass
