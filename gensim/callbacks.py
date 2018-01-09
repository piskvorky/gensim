#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Abstract base class to build callbacks. Callbacks are used to apply custom functions over the model
at specific points during training (epoch start, batch end etc.). To implement a Callback, subclass
:class: ~gensim.callbacks.Callback, look at the example below which creates a callback to save a training model
after each epoch:

>>> from gensim.test.utils import common_texts as sentences
>>> from gensim.callbacks import Callback
>>> from gensim.models import word2vec
>>> class ModelEpochSaver(Callback):  # Callback to save model after every epoch
>>>     def __init__(self, path_prefix):
>>>         self.path_prefix = path_prefix
>>>     def on_epoch_end(self, model):
>>>         model.save('{}_epoch{}'.format(self.path_prefix, self.cur_epoch))
>>>         self.cur_epoch += 1
>>>     def on_train_begin(self, model):
>>>         self.cur_epoch = 0
>>> epoch_saver = ModelEpochSaver('axax')
>>> model = word2vec.Word2Vec(sentences, iter=5, size=10, min_count=0, seed=42, callbacks=[epoch_saver])

"""


class Callback(object):
    """Abstract base class used to build new callbacks."""

    def __init__(self):
        pass

    def on_epoch_begin(self, model):
        pass

    def on_epoch_end(self, model):
        pass

    def on_batch_begin(self, model):
        pass

    def on_batch_end(self, model):
        pass

    def on_train_begin(self, model):
        pass

    def on_train_end(self, model):
        pass
