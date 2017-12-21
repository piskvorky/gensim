#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import uint32, uint8, zeros, array, random, float32 as REAL
import logging
import heapq
from six import itervalues
from gensim import matutils

logger = logging.getLogger(__name__)


def _get_job_params(model):
    """Return the paramter required for each batch."""
    if model.trainables.alpha > model.trainables.min_alpha_yet_reached:
        logger.warning("Effective 'alpha' higher than previous training cycles")
    return model.trainables.alpha


def _update_job_params(model, job_params, progress, cur_epoch):
    start_alpha = model.trainables.alpha
    end_alpha = model.trainables.min_alpha
    progress *= (float(cur_epoch + 1) / float(model.epochs))
    next_alpha = start_alpha - (start_alpha - end_alpha) * progress
    next_alpha = max(end_alpha, next_alpha)
    model.trainables.min_alpha_yet_reached = next_alpha
    return next_alpha


def _get_thread_working_mem(model):
    work = matutils.zeros_aligned(model.trainables.vector_size, dtype=REAL)  # per-thread private work memory
    neu1 = matutils.zeros_aligned(model.trainables.vector_size, dtype=REAL)
    return work, neu1


def _raw_word_count(job):
    """Return the number of words in a given job."""
    return sum(len(sentence) for sentence in job)


def _check_training_sanity(model, epochs=None, total_examples=None, total_words=None):
        if len(model.wv.vocab) > 0:
            model._set_params_from_kv()
        if model.model_trimmed_post_training:
            raise RuntimeError("Parameters for training were discarded using model_trimmed_post_training method")

        if not model.vocabulary.vocab:  # should be set by `build_vocab`
            raise RuntimeError("you must first build vocabulary before training the model")
        if not len(model.trainables.vectors):
            raise RuntimeError("you must initialize vectors before training the model")

        if not hasattr(model.vocabulary, 'corpus_count'):
            raise ValueError(
                "The number of examples in the training corpus is missing. "
                "Please make sure this is set inside `build_vocab` function."
                "Call the `build_vocab` function before calling `train`."
            )

        if total_words is None and total_examples is None:
            raise ValueError(
                "You must specify either total_examples or total_words, for proper job parameters updation"
                "and progress calculations. "
                "The usual value is total_examples=model.corpus_count."
            )
        if epochs is None:
            raise ValueError("You must specify an explict epochs count. The usual value is epochs=model.epochs.")
