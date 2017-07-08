import gensim
import logging
import copy
import numpy as np
from queue import Queue 

# Visdom is used for training stats visualization
try:
    from visdom import Visdom
    VISDOM_INSTALLED = True
except ImportError:
    VISDOM_INSTALLED = False


class Metric(object):
    def __init__(self):
        pass

    def get_value(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)


class CoherenceMetric(Metric):
    def __init__(self, corpus=None, texts=None, dictionary=None, coherence=None, window_size=None, topn=None, logger=None, viz_env=None, title=None):
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
        # only one of the model or topic would be defined
        self.model = None
        self.topics = None
        super(CoherenceMetric, self).get_value(**kwargs)
        cm = gensim.models.CoherenceModel(self.model, self.topics, self.texts, self.corpus, self.dictionary, self.window_size, self.coherence, self.topn)
        return cm.get_coherence()


class PerplexityMetric(Metric):
    def __init__(self, corpus=None, logger=None, viz_env=None, title=None):
        self.corpus = corpus
        self.logger = logger
        self.viz_env = viz_env
        self.title = title

    def get_value(self, **kwargs):
        super(PerplexityMetric, self).get_value(**kwargs)
        corpus_words = sum(cnt for document in self.corpus for _, cnt in document)
        perwordbound = self.model.bound(self.corpus) / corpus_words
        return np.exp2(-perwordbound)

        
class DiffMetric(Metric):
    def __init__(self, distance="jaccard", num_words=100, n_ann_terms=10, normed=True, logger=None, viz_env=None, title=None):
        self.distance = distance
        self.num_words = num_words
        self.n_ann_terms = n_ann_terms
        self.normed = normed
        self.logger = logger
        self.viz_env = viz_env
        self.title = title

    def get_value(self, **kwargs):
        super(DiffMetric, self).get_value(**kwargs)
        diff_matrix, _ = self.model.diff(self.other_model, self.distance, self.num_words, self.n_ann_terms, self.normed)
        return np.diagonal(diff_matrix)


class ConvergenceMetric(Metric):
    def __init__(self, distance="jaccard", num_words=100, n_ann_terms=10, normed=True, logger=None, viz_env=None, title=None):
        self.distance = distance
        self.num_words = num_words
        self.n_ann_terms = n_ann_terms
        self.normed = normed
        self.logger = logger
        self.viz_env = viz_env
        self.title = title

    def get_value(self, **kwargs):
        super(ConvergenceMetric, self).get_value(**kwargs)
        diff_matrix, _ = self.model.diff(self.other_model, self.distance, self.num_words, self.n_ann_terms, self.normed)
        return np.sum(np.diagonal(diff_matrix))
        

class Callback(object):
    def __init__(self, metrics):
        # list of metrics to be plot
        self.metrics = metrics

    def set_model(self, model):
        self.model = model
        self.previous = None
        # check for any metric which need model state from previous epoch
        if any(isinstance(metric, (DiffMetric, ConvergenceMetric)) for metric in self.metrics):
            self.previous = copy.deepcopy(model)
            # store diff diagnols of previous epochs
            self.diff_mat = Queue()
        if any(metric.logger=="visdom" for metric in self.metrics):
            if not VISDOM_INSTALLED:
                raise ImportError("Please install Visdom for visualization")
            self.viz = Visdom()
            # store initial plot windows of every metric (same window will be updated with increasing epochs) 
            self.windows = []
        if any(metric.logger=="shell" for metric in self.metrics):
            # set logger for current topic model
            model_type = type(self.model).__name__
            self.log_type = logging.getLogger(model_type)

    def on_epoch_end(self, epoch, topics=None):
        # plot all metrics in current epoch
        for i, metric in enumerate(self.metrics):
            value = metric.get_value(topics=topics, model=self.model, other_model=self.previous)
            metric_label = type(metric).__name__[:-6]
            # check for any metric which need model state from previous epoch
            if isinstance(metric, (DiffMetric, ConvergenceMetric)):
                self.previous = copy.deepcopy(self.model)

            if metric.logger=="visdom":
                if epoch==0:
                    if value.ndim>0:
                        diff_mat = np.array([value])
                        viz_metric = self.viz.heatmap(X=diff_mat.T, env=metric.viz_env, opts=dict(xlabel='Epochs', ylabel=metric_label, title=metric.title))
                        # store current epoch's diff diagonal
                        self.diff_mat.put(diff_mat)
                        # saving initial plot window
                        self.windows.append(copy.deepcopy(viz_metric)) 
                    else:
                        viz_metric = self.viz.line(Y=np.array([value]), X=np.array([epoch]), env=metric.viz_env, opts=dict(xlabel='Epochs', ylabel=metric_label, title=metric.title))
                        # saving initial plot window
                        self.windows.append(copy.deepcopy(viz_metric))                   
                else:
                    if value.ndim>0:
                        # concatenate with previous epoch's diff diagonals
                        diff_mat = np.concatenate((self.diff_mat.get(), np.array([value])))
                        self.viz.heatmap(X=diff_mat.T, env=metric.viz_env, win=self.windows[i], opts=dict(xlabel='Epochs', ylabel=metric_label, title=metric.title))
                        self.diff_mat.put(diff_mat)
                    else:
                        self.viz.updateTrace(Y=np.array([value]), X=np.array([epoch]), env=metric.viz_env, win=self.windows[i])
                        
            if metric.logger=="shell":
                statement = " ".join(("Epoch:", epoch, metric_label, "estimate:", str(value)))
                self.log_type.info(statement)

