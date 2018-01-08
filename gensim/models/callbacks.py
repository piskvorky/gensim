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
    """
    Base Metric class for topic model evaluation metrics
    """
    def __str__(self):
        """
        Return a string representation of Metric class
        """
        if self.title is not None:
            return self.title
        else:
            return type(self).__name__[:-6]

    def set_parameters(self, **parameters):
        """
        Set the parameters
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

    def get_value(self):
        pass


class CoherenceMetric(Metric):
    """
    Metric class for coherence evaluation
    """
    def __init__(self, corpus=None, texts=None, dictionary=None, coherence=None,
                 window_size=None, topn=10, logger=None, viz_env=None, title=None):
        """
        Args:
            corpus : Gensim document corpus.
            texts : Tokenized texts. Needed for coherence models that use sliding window based probability estimator,
            dictionary : Gensim dictionary mapping of id word to create corpus. If model.id2word is present,
                this is not needed. If both are provided, dictionary will be used.
            window_size : Is the size of the window to be used for coherence measures using boolean
                sliding window as their probability estimator. For 'u_mass' this doesn't matter.
                If left 'None' the default window sizes are used which are:

                    'c_v' : 110
                    'c_uci' : 10
                    'c_npmi' : 10

            coherence : Coherence measure to be used. Supported values are:
                'u_mass'
                'c_v'
                'c_uci' also popularly known as c_pmi
                'c_npmi'
                For 'u_mass' corpus should be provided. If texts is provided, it will be converted
                to corpus using the dictionary. For 'c_v', 'c_uci' and 'c_npmi' texts should be provided.
                Corpus is not needed.
            topn : Integer corresponding to the number of top words to be extracted from each topic.
            logger : Monitor training process using:
                        "shell" : print coherence value in shell
                        "visdom" : visualize coherence value with increasing epochs in Visdom visualization framework
            viz_env : Visdom environment to use for plotting the graph
            title : title of the graph plot
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
        """
        Args:
            model : Pre-trained topic model. Should be provided if topics is not provided.
                    Currently supports LdaModel, LdaMallet wrapper and LdaVowpalWabbit wrapper. Use 'topics'
                    parameter to plug in an as yet unsupported model.
            topics : List of tokenized topics.
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
    """
    Metric class for perplexity evaluation
    """
    def __init__(self, corpus=None, logger=None, viz_env=None, title=None):
        """
        Args:
            corpus : Gensim document corpus
            logger : Monitor training process using:
                        "shell" : print coherence value in shell
                        "visdom" : visualize coherence value with increasing epochs in Visdom visualization framework
            viz_env : Visdom environment to use for plotting the graph
            title : title of the graph plot
        """
        self.corpus = corpus
        self.logger = logger
        self.viz_env = viz_env
        self.title = title

    def get_value(self, **kwargs):
        """
        Args:
            model : Trained topic model
        """
        super(PerplexityMetric, self).set_parameters(**kwargs)
        corpus_words = sum(cnt for document in self.corpus for _, cnt in document)
        perwordbound = self.model.bound(self.corpus) / corpus_words
        return np.exp2(-perwordbound)


class DiffMetric(Metric):
    """
    Metric class for topic difference evaluation
    """
    def __init__(self, distance="jaccard", num_words=100, n_ann_terms=10, diagonal=True,
                 annotation=False, normed=True, logger=None, viz_env=None, title=None):
        """
        Args:
            distance : measure used to calculate difference between any topic pair. Available values:
                `kullback_leibler`
                `hellinger`
                `jaccard`
            num_words : is quantity of most relevant words that used if distance == `jaccard` (also used for annotation)
            n_ann_terms : max quantity of words in intersection/symmetric difference
                          between topics (used for annotation)
            diagonal : difference between  identical topic no.s
            annotation : intersection or difference of words between topics
            normed (bool) : If `true`, matrix/array Z will be normalized
            logger : Monitor training process using:
                        "shell" : print coherence value in shell
                        "visdom" : visualize coherence value with increasing epochs in Visdom visualization framework
            viz_env : Visdom environment to use for plotting the graph
            title : title of the graph plot
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
        """
        Args:
            model : Trained topic model
            other_model : second topic model instance to calculate the difference from
        """
        super(DiffMetric, self).set_parameters(**kwargs)
        diff_diagonal, _ = self.model.diff(
            self.other_model, self.distance, self.num_words, self.n_ann_terms,
            self.diagonal, self.annotation, self.normed
        )
        return diff_diagonal


class ConvergenceMetric(Metric):
    """
    Metric class for convergence evaluation
    """
    def __init__(self, distance="jaccard", num_words=100, n_ann_terms=10, diagonal=True,
                 annotation=False, normed=True, logger=None, viz_env=None, title=None):
        """
        Args:
            distance : measure used to calculate difference between any topic pair. Available values:
                `kullback_leibler`
                `hellinger`
                `jaccard`
            num_words : is quantity of most relevant words that used if distance == `jaccard` (also used for annotation)
            n_ann_terms : max quantity of words in intersection/symmetric difference
                          between topics (used for annotation)
            diagonal : difference between  identical topic no.s
            annotation : intersection or difference of words between topics
            normed (bool) : If `true`, matrix/array Z will be normalized
            logger : Monitor training process using:
                        "shell" : print coherence value in shell
                        "visdom" : visualize coherence value with increasing epochs in Visdom visualization framework
            viz_env : Visdom environment to use for plotting the graph
            title : title of the graph plot
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
        """
        Args:
            model : Trained topic model
            other_model : second topic model instance to calculate the difference from
        """
        super(ConvergenceMetric, self).set_parameters(**kwargs)
        diff_diagonal, _ = self.model.diff(
            self.other_model, self.distance, self.num_words, self.n_ann_terms,
            self.diagonal, self.annotation, self.normed
        )
        return np.sum(diff_diagonal)


class Callback(object):
    """
    Used to log/visualize the evaluation metrics during training. The values are stored at the end of each epoch.
    """
    def __init__(self, metrics):
        """
        Args:
            metrics : a list of callbacks. Possible values:
                "CoherenceMetric"
                "PerplexityMetric"
                "DiffMetric"
                "ConvergenceMetric"
        """
        # list of metrics to be plot
        self.metrics = metrics

    def set_model(self, model):
        """
        Save the model instance and initialize any required variables which would be updated throughout training
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
        """
        Log or visualize current epoch's metric value

        Args:
            epoch : current epoch no.
            topics : topic distribution from current epoch (required for coherence of unsupported topic models)
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
