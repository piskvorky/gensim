#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 RARE Technologies <info@rare-technologies.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Callbacks can be used to observe the training process.

Since training in huge corpora can be time consuming, we want to offer the users some insight
into the process, in real time. In this way, convergence issues
or other potential problems can be identified early in the process,
saving precious time and resources.

The metrics exposed through this module can be used to construct Callbacks, which will be called
at specific points in the training process, such as "epoch starts" or "epoch finished".
These metrics can be used to assess mod's convergence or correctness, for example
to save the model, visualize intermediate results, or anything else.

Usage examples
--------------

To implement a Callback, inherit from this base class and override one or more of its methods.

#. Create a callback to save the training model after each epoch:

>>> from gensim.test.utils import common_corpus, common_texts
>>> from gensim.models.callbacks import CallbackAny2Vec
>>> from gensim.models import Word2Vec
>>>
>>> class EpochSaver(CallbackAny2Vec):
...     '''Callback to save model after each epoch.'''
...
...     def __init__(self, path_prefix):
...         self.path_prefix = path_prefix
...         self.epoch = 0
...
...     def on_epoch_end(self, model):
...         output_path = '{}_epoch{}.model'.format(self.path_prefix, self.epoch)
...         print("Saving model to {}".format(output_path))
...         model.save(output_path)
...         self.epoch += 1
...

#. Create a callback to print progress information to the console:

>>> class EpochLogger(CallbackAny2Vec):
...     '''Callback to log information about training'''
...
...     def __init__(self):
...         self.epoch = 0
...
...     def on_epoch_begin(self, model):
...         print("Epoch #{} start".format(self.epoch))
...
...     def on_epoch_end(self, model):
...         print("Epoch #{} end".format(self.epoch))
...         self.epoch += 1
...

>>> epoch_logger = EpochLogger()

#. Bind the callbacks to a model before training it:

>>> w2v_model = Word2Vec(common_texts, iter=5, size=10, min_count=0, seed=42, callbacks=[epoch_logger])
Epoch #0 start
Epoch #0 end
Epoch #1 start
Epoch #1 end
Epoch #2 start
Epoch #2 end
Epoch #3 start
Epoch #3 end
Epoch #4 start
Epoch #4 end

#. Create and bind a callback to a topic model. This callback will log the perplexity metric in real time:

>>> from gensim.models.callbacks import PerplexityMetric
>>> from gensim.models.ldamodel import LdaModel
>>> from gensim.test.utils import common_texts
>>>
>>> # Log the perplexity score at the end of each epoch.
>>> perplexity_logger = PerplexityMetric(corpus=common_texts, logger='shell')
>>> lda = LdaModel(common_texts, num_topics=5, callbacks=[perplexity_logger])

"""

import gensim
import logging
import copy
import sys
import numpy as np

if sys.version_info[0] >= 3:
    from queue import Queue
else:
    from Queue import Queue

# Visdom is used for training stats visualization
try:
    from visdom import Visdom
    VISDOM_INSTALLED = True
except ImportError:
    VISDOM_INSTALLED = False


class Metric(object):
    """Base Metric class for topic model evaluation metrics.

    Concrete implementations include:

        * :class:`~gensim.models.callbacks.CoherenceMetric`
        * :class:`~gensim.models.callbacks.PerplexityMetric`
        * :class:`~gensim.models.callbacks.DiffMetric`
        * :class:`~gensim.models.callbacks.ConvergenceMetric`
    """
    def __str__(self):
        """Get a string representation of Metric class.

        Returns
        -------
        str
            Human readable representation of the metric.

        """
        if self.title is not None:
            return self.title
        else:
            return type(self).__name__[:-6]

    def set_parameters(self, **parameters):
        """Set the metric parameters.

        Parameters
        ----------
        **parameters
            Keyword arguments to override the object's internal attributes.

        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

    def get_value(self):
        """Get the metric's value at this point in time.

        Warnings
        --------
        The user **must** provide a concrete implementation for this method for every subclass of
        this class.

        See Also
        --------
        :meth:`gensim.models.callbacks.CoherenceMetric.get_value`
        :meth:`gensim.models.callbacks.PerplexityMetric.get_value`
        :meth:`gensim.models.callbacks.DiffMetric.get_value`
        :meth:`gensim.models.callbacks.ConvergenceMetric.get_value`

        Returns
        -------
        object
            The metric's type depends on what exactly it measures. In the simplest case it might
            be a real number corresponding to an error estimate. It could however be anything else
            that is useful to report or visualize.

        """
        raise NotImplementedError("Please provide an implementation for `get_value` in your subclass.")


class CoherenceMetric(Metric):
    """Metric class for coherence evaluation.

    See Also
    --------
    :class:`~gensim.models.coherencemodel.CoherenceModel`

    """

    def __init__(self, corpus=None, texts=None, dictionary=None, coherence=None,
                 window_size=None, topn=10, logger=None, viz_env=None, title=None):
        """

        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional
            Stream of document vectors or sparse matrix of shape (`num_terms`, `num_documents`).
        texts : list of char (str of length 1), optional
            Tokenized texts needed for coherence models that use sliding window based probability estimator.
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            Gensim dictionary mapping from integer IDs to words, needed to create corpus. If `model.id2word` is present,
            this is not needed. If both are provided, `dictionary` will be used.
        coherence : {'u_mass', 'c_v', 'c_uci', 'c_npmi'}, optional
            Coherence measure to be used. 'c_uci' is also known as 'c_pmi' in the literature.
            For 'u_mass', the corpus **MUST** be provided. If `texts` is provided, it will be converted
            to corpus using the dictionary. For 'c_v', 'c_uci' and 'c_npmi', `texts` **MUST** be provided.
            Corpus is not needed.
        window_size : int, optional
            Size of the window to be used for coherence measures using boolean
            sliding window as their probability estimator. For 'u_mass' this doesn't matter.
            If 'None', the default window sizes are used which are:

                * `c_v` - 110
                * `c_uci` - 10
                * `c_npmi` - 10
        topn : int, optional
            Number of top words to be extracted from each topic.
        logger : {'shell', 'visdom'}, optional
           Monitor training process using one of the available methods. 'shell' will print the coherence value in
           the active shell, while 'visdom' will visualize the coherence value with increasing epochs using the Visdom
           visualization framework.
        viz_env : object, optional
            Visdom environment to use for plotting the graph. Unused.
        title : str, optional
            Title of the graph plot in case `logger == 'visdom'`. Unused.

        """
        self.corpus = corpus
        self.dictionary = dictionary
        self.coherence = coherence
        self.texts = texts
        self.window_size = window_size
        self.topn = topn
        self.logger = logger
        self.viz_env = viz_env
        self.title = title

    def get_value(self, **kwargs):
        """Get the coherence score.

        Parameters
        ----------
        **kwargs
            Key word arguments to override the object's internal attributes.
            One of the following parameters are expected:

                * `model` - pre-trained topic model of type :class:`~gensim.models.ldamodel.LdaModel`, or one
                  of its wrappers, such as :class:`~gensim.models.wrappers.ldamallet.LdaMallet` or
                  :class:`~gensim.models.wrappers.ldavowpalwabbit.LdaVowpalWabbit`.
                * `topics` - list of tokenized topics.

        Returns
        -------
        float
            The coherence score.

        """
        # only one of the model or topic would be defined
        self.model = None
        self.topics = None
        super(CoherenceMetric, self).set_parameters(**kwargs)

        cm = gensim.models.CoherenceModel(
            model=self.model, topics=self.topics, texts=self.texts, corpus=self.corpus,
            dictionary=self.dictionary, window_size=self.window_size,
            coherence=self.coherence, topn=self.topn
        )

        return cm.get_coherence()


class PerplexityMetric(Metric):
    """Metric class for perplexity evaluation."""

    def __init__(self, corpus=None, logger=None, viz_env=None, title=None):
        """

        Parameters
        ----------
        corpus : {iterable of list of (int, float), scipy.sparse.csc}, optional
            Stream of document vectors or sparse matrix of shape (`num_terms`, `num_documents`).
        logger : {'shell', 'visdom'}, optional
           Monitor training process using one of the available methods. 'shell' will print the perplexity value in
           the active shell, while 'visdom' will visualize the coherence value with increasing epochs using the Visdom
           visualization framework.
        viz_env : object, optional
            Visdom environment to use for plotting the graph. Unused.
        title : str, optional
            Title of the graph plot in case `logger == 'visdom'`. Unused.

        """
        self.corpus = corpus
        self.logger = logger
        self.viz_env = viz_env
        self.title = title

    def get_value(self, **kwargs):
        """Get the coherence score.

        Parameters
        ----------
        **kwargs
            Key word arguments to override the object's internal attributes.
            A trained topic model is expected using the 'model' key. This can be of type
            :class:`~gensim.models.ldamodel.LdaModel`, or one of its wrappers, such as
            :class:`~gensim.models.wrappers.ldamallet.LdaMallet` or
            :class:`~gensim.models.wrapper.ldavowpalwabbit.LdaVowpalWabbit`.

        Returns
        -------
        float
            The perplexity score.

        """
        super(PerplexityMetric, self).set_parameters(**kwargs)
        corpus_words = sum(cnt for document in self.corpus for _, cnt in document)
        perwordbound = self.model.bound(self.corpus) / corpus_words
        return np.exp2(-perwordbound)


class DiffMetric(Metric):
    """Metric class for topic difference evaluation."""

    def __init__(self, distance="jaccard", num_words=100, n_ann_terms=10, diagonal=True,
                 annotation=False, normed=True, logger=None, viz_env=None, title=None):
        """

        Parameters
        ----------
        distance : {'kullback_leibler', 'hellinger', 'jaccard'}, optional
            Measure used to calculate difference between any topic pair.
        num_words : int, optional
            The number of most relevant words used if `distance == 'jaccard'`. Also used for annotating topics.
        n_ann_terms : int, optional
            Max number of words in intersection/symmetric difference between topics. Used for annotation.
        diagonal : bool, optional
            Whether we need the difference between identical topics (the diagonal of the difference matrix).
        annotation : bool, optional
            Whether the intersection or difference of words between two topics should be returned.
        normed : bool, optional
            Whether the matrix should be normalized or not.
        logger : {'shell', 'visdom'}, optional
           Monitor training process using one of the available methods. 'shell' will print the coherence value in
           the active shell, while 'visdom' will visualize the coherence value with increasing epochs using the Visdom
           visualization framework.
        viz_env : object, optional
            Visdom environment to use for plotting the graph. Unused.
        title : str, optional
            Title of the graph plot in case `logger == 'visdom'`. Unused.

        """
        self.distance = distance
        self.num_words = num_words
        self.n_ann_terms = n_ann_terms
        self.diagonal = diagonal
        self.annotation = annotation
        self.normed = normed
        self.logger = logger
        self.viz_env = viz_env
        self.title = title

    def get_value(self, **kwargs):
        """Get the difference between each pair of topics in two topic models.

        Parameters
        ----------
        **kwargs
            Key word arguments to override the object's internal attributes.
            Two models of type :class:`~gensim.models.ldamodelLdaModel` or its wrappers are expected using the keys
            `model` and `other_model`.

        Returns
        -------
        np.ndarray of shape (`model.num_topics`, `other_model.num_topics`)
            Matrix of differences between each pair of topics.
        np.ndarray of shape (`model.num_topics`, `other_model.num_topics`, 2), optional
            Annotation matrix where for each pair we include the word from the intersection of the two topics,
            and the word from the symmetric difference of the two topics. Only included if `annotation == True`.

        """
        super(DiffMetric, self).set_parameters(**kwargs)
        diff_diagonal, _ = self.model.diff(
            self.other_model, self.distance, self.num_words, self.n_ann_terms,
            self.diagonal, self.annotation, self.normed
        )
        return diff_diagonal


class ConvergenceMetric(Metric):
    """Metric class for convergence evaluation. """
    def __init__(self, distance="jaccard", num_words=100, n_ann_terms=10, diagonal=True,
                 annotation=False, normed=True, logger=None, viz_env=None, title=None):
        """

        Parameters
        ----------
        distance : {'kullback_leibler', 'hellinger', 'jaccard'}, optional
            Measure used to calculate difference between any topic pair.
        num_words : int, optional
            The number of most relevant words used if `distance == 'jaccard'`. Also used for annotating topics.
        n_ann_terms : int, optional
            Max number of words in intersection/symmetric difference between topics. Used for annotation.
        diagonal : bool, optional
            Whether we need the difference between identical topics (the diagonal of the difference matrix).
        annotation : bool, optional
            Whether the intersection or difference of words between two topics should be returned.
        normed : bool, optional
            Whether the matrix should be normalized or not.
        logger : {'shell', 'visdom'}, optional
           Monitor training process using one of the available methods. 'shell' will print the coherence value in
           the active shell, while 'visdom' will visualize the coherence value with increasing epochs using the Visdom
           visualization framework.
        viz_env : object, optional
            Visdom environment to use for plotting the graph. Unused.
        title : str, optional
            Title of the graph plot in case `logger == 'visdom'`. Unused.

       """
        self.distance = distance
        self.num_words = num_words
        self.n_ann_terms = n_ann_terms
        self.diagonal = diagonal
        self.annotation = annotation
        self.normed = normed
        self.logger = logger
        self.viz_env = viz_env
        self.title = title

    def get_value(self, **kwargs):
        """Get the sum of each element in the difference matrix between each pair of topics in two topic models.

        A small difference between the partially trained models produced by subsequent training iterations can indicate
        that the model has stopped significantly improving and has therefore converged to a local or global optimum.

        Parameters
        ----------
        **kwargs
            Key word arguments to override the object's internal attributes.
            Two models of type :class:`~gensim.models.ldamodelLdaModel` or its wrappers are expected using the keys
            `model` and `other_model`.

        Returns
        -------
        float
            The sum of the difference matrix between two trained topic models (usually the same model after two
            subsequent training iterations).

        """
        super(ConvergenceMetric, self).set_parameters(**kwargs)
        diff_diagonal, _ = self.model.diff(
            self.other_model, self.distance, self.num_words, self.n_ann_terms,
            self.diagonal, self.annotation, self.normed
        )
        return np.sum(diff_diagonal)


class Callback(object):
    """A class representing routines called reactively at specific phases during trained.

    These can be used to log or visualize the training progress using any of the metric scores developed before.
    The values are stored at the end of each training epoch. The following metric scores are currently available:

        * :class:`~gensim.models.callbacks.CoherenceMetric`
        * :class:`~gensim.models.callbacks.PerplexityMetric`
        * :class:`~gensim.models.callbacks.DiffMetric`
        * :class:`~gensim.models.callbacks.ConvergenceMetric`

    """
    def __init__(self, metrics):
        """

        Parameters
        ----------
        metrics : list of :class:`~gensim.models.callbacks.Metric`
            The list of metrics to be reported by the callback.

        """
        self.metrics = metrics

    def set_model(self, model):
        """Save the model instance and initialize any required variables which would be updated throughout training.

        Parameters
        ----------
        model : :class:`~gensim.models.basemodel.BaseTopicModel`
            The model for which the training will be reported (logged or visualized) by the callback.

        """
        self.model = model
        self.previous = None
        # check for any metric which need model state from previous epoch
        if any(isinstance(metric, (DiffMetric, ConvergenceMetric)) for metric in self.metrics):
            self.previous = copy.deepcopy(model)
            # store diff diagonals of previous epochs
            self.diff_mat = Queue()
        if any(metric.logger == "visdom" for metric in self.metrics):
            if not VISDOM_INSTALLED:
                raise ImportError("Please install Visdom for visualization")
            self.viz = Visdom()
            # store initial plot windows of every metric (same window will be updated with increasing epochs)
            self.windows = []
        if any(metric.logger == "shell" for metric in self.metrics):
            # set logger for current topic model
            self.log_type = logging.getLogger('gensim.models.ldamodel')

    def on_epoch_end(self, epoch, topics=None):
        """Report the current epoch's metric value.

        Called at the end of each training iteration.

        Parameters
        ----------
        epoch : int
            The epoch that just ended.
        topics : list of list of str, optional
            List of tokenized topics. This is required for the coherence metric.

        Returns
        -------
        dict of (str, object)
            Mapping from metric names to their values. The type of each value depends on the metric type,
            for example :class:`~gensim.models.callbacks.DiffMetric` computes a matrix while
            :class:`~gensim.models.callbacks.ConvergenceMetric` computes a float.

        """

        # stores current epoch's metric values
        current_metrics = {}

        # plot all metrics in current epoch
        for i, metric in enumerate(self.metrics):
            label = str(metric)
            value = metric.get_value(topics=topics, model=self.model, other_model=self.previous)

            current_metrics[label] = value

            if metric.logger == "visdom":
                if epoch == 0:
                    if value.ndim > 0:
                        diff_mat = np.array([value])
                        viz_metric = self.viz.heatmap(
                            X=diff_mat.T, env=metric.viz_env, opts=dict(xlabel='Epochs', ylabel=label, title=label)
                        )
                        # store current epoch's diff diagonal
                        self.diff_mat.put(diff_mat)
                        # saving initial plot window
                        self.windows.append(copy.deepcopy(viz_metric))
                    else:
                        viz_metric = self.viz.line(
                            Y=np.array([value]), X=np.array([epoch]), env=metric.viz_env,
                            opts=dict(xlabel='Epochs', ylabel=label, title=label)
                        )
                        # saving initial plot window
                        self.windows.append(copy.deepcopy(viz_metric))
                else:
                    if value.ndim > 0:
                        # concatenate with previous epoch's diff diagonals
                        diff_mat = np.concatenate((self.diff_mat.get(), np.array([value])))
                        self.viz.heatmap(
                            X=diff_mat.T, env=metric.viz_env, win=self.windows[i],
                            opts=dict(xlabel='Epochs', ylabel=label, title=label)
                        )
                        self.diff_mat.put(diff_mat)
                    else:
                        self.viz.updateTrace(
                            Y=np.array([value]), X=np.array([epoch]), env=metric.viz_env, win=self.windows[i]
                        )

            if metric.logger == "shell":
                statement = "".join(("Epoch ", str(epoch), ": ", label, " estimate: ", str(value)))
                self.log_type.info(statement)

        # check for any metric which need model state from previous epoch
        if isinstance(metric, (DiffMetric, ConvergenceMetric)):
            self.previous = copy.deepcopy(self.model)

        return current_metrics


class CallbackAny2Vec(object):
    """Base class to build callbacks for :class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`.

    Callbacks are used to apply custom functions over the model at specific points
    during training (epoch start, batch end etc.). This is a base class and its purpose is to be inherited by
    custom Callbacks that implement one or more of its methods (depending on the point during training where they
    want some action to be taken).

    See examples at the module level docstring for how to define your own callbacks by inheriting  from this class.

    """

    def on_epoch_begin(self, model):
        """Method called at the start of each epoch.

        Parameters
        ----------
        model : class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`
            Current model.

        """
        pass

    def on_epoch_end(self, model):
        """Method called at the end of each epoch.

        Parameters
        ----------
        model : class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`
            Current model.

        """
        pass

    def on_batch_begin(self, model):
        """Method called at the start of each batch.

        Parameters
        ----------
        model : class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`
            Current model.

        """
        pass

    def on_batch_end(self, model):
        """Method called at the end of each batch.

        Parameters
        ----------
        model : class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`
            Current model.

        """
        pass

    def on_train_begin(self, model):
        """Method called at the start of the training process.

        Parameters
        ----------
        model : class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`
            Current model.

        """
        pass

    def on_train_end(self, model):
        """Method called at the end of the training process.

        Parameters
        ----------
        model : class:`~gensim.models.base_any2vec.BaseWordEmbeddingsModel`
            Current model.

        """
        pass
