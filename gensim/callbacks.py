#!/usr/bin/env python
# -*- coding: utf-8 -*-


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


class ModelEpochSaver(Callback):
    """Callback to save model after every epoch."""

    def __init__(self, path_prefix):
        self.path_prefix = path_prefix

    def on_epoch_end(self, model):
        model.save('{}_epoch{}'.format(self.path_prefix, self.cur_epoch))
        self.cur_epoch += 1

    def on_train_begin(self, model):
        self.cur_epoch = 0


class LossEpochHistory(Callback):
    """Callback to capture loss after every epoch. Currently only valid for Word2Vec"""

    def __init__(self):
        self.losses = []

    def on_epoch_end(self, model):
        self.losses.append(model.running_training_loss)

    def on_train_begin(self, model):
        model.compute_loss = True
