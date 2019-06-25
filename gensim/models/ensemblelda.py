#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors: Tobias B <github.com/sezanzeb>, Alex Loosley <aloosley@alumni.brown.edu>,
# Stephan Sahm <stephan.sahm@gmx.de>, Alex Salles <alex.salles@gmail.com>, Data Reply Munich

"""Ensemble Latent Dirichlet Allocation (LDA), a method of ensembling multiple gensim topic models. They are clustered
using DBSCAN, for which nearby topic mixtures - which are mixtures of two or more true topics and therefore undesired -
are being used to support topics in becoming cores, but not vice versa.

Usage examples
--------------

Train an ensemble of LdaModels using a Gensim corpus

.. sourcecode:: pycon

    >>> from gensim.test.utils import common_texts
    >>> from gensim.corpora.dictionary import Dictionary
    >>> from gensim.models import EnsembleLda
    >>>
    >>> # Create a corpus from a list of texts
    >>> common_dictionary = Dictionary(common_texts)
    >>> common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
    >>>
    >>> # Train the model on the corpus. corpus has to be provided as a
    >>> # keyword argument, as they are passed through to the children.
    >>> elda = EnsembleLda(corpus=common_corpus, id2word=common_dictionary, num_topics=10, num_models=4)

Save a model to disk, or reload a pre-trained model

.. sourcecode:: pycon

    >>> from gensim.test.utils import datapath
    >>>
    >>> # Save model to disk.
    >>> temp_file = datapath("model")
    >>> elda.save(temp_file)
    >>>
    >>> # Load a potentially pretrained model from disk.
    >>> elda = EnsembleLda.load(temp_file)

Query, the model using new, unseen documents

.. sourcecode:: pycon

    >>> # Create a new corpus, made of previously unseen documents.
    >>> other_texts = [
    ...     ['computer', 'time', 'graph'],
    ...     ['survey', 'response', 'eps'],
    ...     ['human', 'system', 'computer']
    ... ]
    >>> other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]
    >>>
    >>> unseen_doc = other_corpus[0]
    >>> vector = elda[unseen_doc]  # get topic probability distribution for a document

Increase the ensemble size by adding a new model. Make sure it uses the same dictionary

.. sourcecode:: pycon

    >>> from gensim.models import LdaModel
    >>> elda.add_model(LdaModel(common_corpus, id2word=common_dictionary, num_topics=10))
    >>> elda.recluster()
    >>> vector = elda[unseen_doc]

To optimize the ensemble for your specific case, the children can be clustered again using
different hyperparameters

.. sourcecode:: pycon

    >>> elda.recluster(eps=0.2)

References
----------
.. [1] REHUREK, Radim and Sojka, PETR, 2010, Software framework for topic
       modelling with large corpora. In : THE LREC 2010 WORKSHOP ON NEW
       CHALLENGES FOR NLP FRAMEWORKS [online]. Msida : University of Malta.
       2010. p. 45-50. Available from:
       http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.695.4595

"""

import logging
import os
from multiprocessing import Process, Pipe, ProcessError
import numpy as np
from scipy.spatial.distance import cosine
from gensim import utils
from gensim.models import ldamodel, ldamulticore, basemodel

logger = logging.getLogger(__name__)


class EnsembleLda():
    """Ensemble Latent Dirichlet Allocation (LDA), a method of ensembling multiple gensim topic models.
    They are clustered using a variation of DBSCAN, for which nearby topic mixtures - which are mixtures
    of two or more true topics and therefore undesired - are being used to support topics in becoming
    cores, but not vice versa.

    """
    def __init__(self, topic_model_kind="lda", num_models=3,
                 min_cores=None,  # default value from generate_stable_topics()
                 epsilon=0.1, ensemble_workers=1, memory_friendly_ttda=True,
                 min_samples=None, masking_method="mass", masking_threshold=None,
                 distance_workers=1, random_state=None, **gensim_kw_args):
        """

        Parameters
        ----------
        topic_model_kind : str, topic model, optional
            Examples:
                'ldamulticore' (recommended), 'lda' (default),
        ensemble_workers : number, optional
            Spawns that many processes and distributes the models from the ensemble to those as evenly as possible.
            num_models should be a multiple of ensemble_workers.

            Setting it to 0 or 1 will both use the nonmultiprocessing version. Default:1
                gensim.models.ldamodel, gensim.models.ldamulticore
        num_models : int, optional
            How many LDA models to train in this ensemble.
            Default: 3
        min_cores : int, optional
            Minimum cores a cluster of topics has to contain so that it is recognized as stable topic.
        epsilon : float, optional
            Defaults to 0.1. Epsilon for the cbdbscan clustering that generates the stable topics.
        ensemble_workers : int, optional
            Spawns that many processes and distributes the models from the ensemble to those as evenly as possible.
            num_models should be a multiple of ensemble_workers.

            Setting it to 0 or 1 will both use the nonmultiprocessing version. Default:1
        memory_friendly_ttda : boolean, optional
            If True, the models in the ensemble are deleted after training and only the total topic word distribution
            is kept to save memory.

            Defaults to True. When False, trained models are stored in a list in self.tms, and no models that are not
            of a gensim model type can be added to this ensemble using the add_model function.

            If False, any topic term matrix can be supllied to add_model.
        min_samples : int, optional
            Required int of nearby topics for a topic to be considered as 'core' in the CBDBSCAN clustering.
        masking_method : {'mass', 'rank}, optional
            One of "mass" (default) or "rank" (faster).
        masking_threshold : float, optional
            Default: None, which uses 0.11 for masking_method "rank", and 0.95 for "mass".
        distance_workers : int, optional
            When distance_workers is None, it defaults to os.cpu_count() for maximum performance. Default is 1, which
            is not multiprocessed. Set to > 1 to enable multiprocessing.
        **gensim_kw_args
            Parameters for each gensim model (e.g. :py:class:`gensim.models.LdaModel`) in the ensemble.

        """
        # Set random state
        # nps max random state of 2**32 - 1 is too large for windows:
        self._MAX_RANDOM_STATE = np.iinfo(np.int32).max

        if "id2word" not in gensim_kw_args:
            gensim_kw_args["id2word"] = None
        if "corpus" not in gensim_kw_args:
            gensim_kw_args["corpus"] = None

        # dictionary. modified version of from gensim/models/ldamodel.py
        # error messages are copied from the original gensim module [1]:
        # will create a fake dict if no dict is provided.
        if gensim_kw_args["id2word"] is None and not gensim_kw_args["corpus"] is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            gensim_kw_args["id2word"] = utils.dict_from_corpus(gensim_kw_args["corpus"])
        if gensim_kw_args["id2word"] is None and gensim_kw_args["corpus"] is None:
            raise ValueError("at least one of corpus/id2word must be specified, to establish "
                             "input space dimensionality. Corpus should be provided using the "
                             "`corpus` keyword argument.")

        if isinstance(topic_model_kind, ldamodel.LdaModel):
            self.topic_model_kind = topic_model_kind
        else:
            kinds = {
                "lda": ldamodel.LdaModel,
                "ldamulticore": ldamulticore.LdaMulticore
            }
            if topic_model_kind not in kinds:
                raise ValueError(
                    "topic_model_kind should be one of 'lda', 'ldamulticode' or a model inheriting from LdaModel")
            self.topic_model_kind = kinds[topic_model_kind]

        # Store some of the parameters
        self.num_models = num_models
        self.gensim_kw_args = gensim_kw_args

        # some settings affecting performance
        self.memory_friendly_ttda = memory_friendly_ttda
        self.distance_workers = distance_workers

        # this will provide the gensim api to the ensemble basically
        self.classic_model_representation = None

        # the ensembles state
        self.random_state = utils.get_random_state(random_state)
        self.sstats_sum = 0
        self.eta = None
        self.tms = []
        # initialize empty topic term distribution array
        self.ttda = np.empty((0, len(gensim_kw_args["id2word"])))
        self.asymmetric_distance_matrix_outdated = True

        # in case the model will not train due to some
        # parameters, stop here and don't train.
        if num_models <= 0:
            return
        if ("corpus" not in gensim_kw_args):
            return
        if gensim_kw_args["corpus"] is None:
            return
        if "iterations" in gensim_kw_args and gensim_kw_args["iterations"] <= 0:
            return
        if "passes" in gensim_kw_args and gensim_kw_args["passes"] <= 0:
            return

        logger.info("Generating {} topic models...".format(num_models))
        # training
        if ensemble_workers > 1:
            # multiprocessed
            self.generate_topic_models_multiproc(num_models, ensemble_workers)
        else:
            # singlecore
            self.generate_topic_models(num_models)

        self.generate_asymmetric_distance_matrix(workers=self.distance_workers,
                                                 threshold=masking_threshold,
                                                 method=masking_method)
        self.generate_topic_clusters(epsilon, min_samples)
        self.generate_stable_topics(min_cores)

        # create model that can provide the usual gensim api to the stable topics from the ensemble
        self.generate_gensim_representation()

    def convert_to_memory_friendly(self):
        """removes the stored gensim models and only keeps their ttdas"""
        self.tms = []
        self.memory_friendly_ttda = True

    def generate_gensim_representation(self):
        """creates a gensim-model from the stable topics

        The prior for the gensim model, eta, Can be anything, (a parameter in object constructor) and won't influence
        get_topics(). Note that different eta means different log_perplexity (the default is the same as the gensim
        default eta, which is for example [0.5, 0.5, 0.5]).

        Note, that when the clustering of the topics was changed using add_model, etc., and when no topics were
        detected because of that, the old classic gensim model will stay in place.

        Returns
        -------
        LdaModel
            A Gensim LDA Model classic_model_representation for which:
            classic_model_representation.get_topics() == self.get_topics()

        """

        logger.info("Generating classic gensim model representation based on results from the ensemble")

        number_of_words = self.sstats_sum
        # if number_of_words should be wrong for some fantastic funny reason
        # that makes you want to peel your skin off, recreate it (takes a while):
        if number_of_words == 0 and "corpus" in self.gensim_kw_args and not self.gensim_kw_args["corpus"] is None:
            for document in self.gensim_kw_args["corpus"]:
                for token in document:
                    number_of_words += token[1]
            self.sstats_sum = number_of_words
        assert number_of_words != 0

        stable_topics = self.get_topics()

        num_stable_topics = len(stable_topics)

        if num_stable_topics == 0:
            logger.error("The model did not detect any stable topic. You can try to adjust epsilon: "
                         "recluster(eps=...)")
            return

        # create a new gensim model
        params = self.gensim_kw_args.copy()
        params["eta"] = self.eta
        params["num_topics"] = num_stable_topics
        # adjust params in a way that no training happens
        params["iterations"] = 0  # no training
        params["passes"] = 0  # no training

        classic_model_representation = self.topic_model_kind(**params)

        # when eta was None, use what gensim generates as default eta for the following tasks:
        eta = classic_model_representation.eta

        # the following is important for the denormalization
        # to generate the proper sstats for the new gensim model:
        # transform to dimensionality of stable_topics. axis=1 is summed
        eta_sum = 0
        if int == type(eta) or float == type(eta):
            eta_sum = [eta * len(stable_topics[0])] * num_stable_topics
        else:
            if len(eta.shape) == 1:  # [e1, e2, e3]
                eta_sum = [[eta.sum()]] * num_stable_topics
            if len(eta.shape) > 1:  # [[e11, e12, ...], [e21, e22, ...], ...]
                eta_sum = np.array(eta.sum(axis=1)[:, None])

        # the factor, that will be used when get_topics() is used, for normalization
        # will never change, because the sum for eta as well as the sum for sstats is constant.
        # Therefore predicting normalization_factor becomes super easy.
        # corpus is a mapping of id to occurences

        # so one can also easily calculate the
        # right sstats, so that get_topics() will return the stable topics no
        # matter eta.

        normalization_factor = np.array([[number_of_words / num_stable_topics]] * num_stable_topics) + eta_sum

        sstats = stable_topics * normalization_factor
        sstats -= eta

        classic_model_representation.state.sstats = sstats.astype(np.float32)
        # fix expElogbeta.
        classic_model_representation.sync_state()

        self.classic_model_representation = classic_model_representation

        return classic_model_representation

    def add_model(self, target, num_new_models=None):
        """Adds the ttda of another model to the ensemble this way, multiple topic models can be connected to an
        ensemble.

        Make sure that all the models use the exact same dictionary/idword mapping.

        In order to generate new stable topics afterwards, use
            self.generate_asymmetric_distance_matrix()
            self.recluster()

        The ttda of another ensemble can also be used, in that case set num_new_models to the num_models parameter
        of the ensemble, that means the number of classic models in the ensemble that generated the ttda. This is
        important, because that information is used to estimate "min_samples" for generate_topic_clusters.

        If you trained this ensemble in the past with a certain Dictionary that you want to reuse for other
        models, you can get it from: self.id2word.

        Parameters
        ----------
        target : {see description}
            1. A single EnsembleLda object
            2. List of EnsembleLda objects
            3. A single Gensim topic model (e.g. (:py:class:`gensim.models.LdaModel`)
            4. List of Gensim topic models

            if memory_friendly_ttda is True, target can also be:
            5. topic-term-distribution-array

            example: [[0.1, 0.1, 0.8], [...], ...]

            [topic1, topic2, ...]
            with topic being an array of probabilities:
            [token1, token2, ...]

            token probabilities in a single topic sum to one, therefore, all the words sum to len(ttda)

        num_new_models : integer, optional
            the model keeps track of how many models were used in this ensemble. Set higher if ttda contained topics
            from more than one model. Default: None, which takes care of it automatically.

            If target is a 2D-array of float values, it assumes 1.

            If the ensemble has memory_friendly_ttda set to False, then it will always use the number of models in
            the target parameter.

        """
        # If the model has never seen a ttda before, initialize.
        # If it has, append.

        # Be flexible. Can be a single element or a list of elements
        # make sure it's a numpy array
        if not isinstance(target, (np.ndarray, list)):
            target = np.array([target])
        else:
            target = np.array(target)
            assert len(target) > 0

        if self.memory_friendly_ttda:
            # for memory friendly models/ttdas, append the ttdas to itself

            detected_num_models = 0

            ttda = []

            # 1. ttda array, because that's the only accepted input that contains numbers
            if isinstance(target.dtype.type(), (np.number, float)):
                ttda = target
                detected_num_models = 1

            # 2. list of ensemblelda objects
            elif isinstance(target[0], type(self)):
                ttda = np.concatenate([ensemble.ttda for ensemble in target], axis=0)
                detected_num_models = sum([ensemble.num_models for ensemble in target])

            # 3. list of gensim models
            elif isinstance(target[0], basemodel.BaseTopicModel):
                ttda = np.concatenate([model.get_topics() for model in target], axis=0)
                detected_num_models = len(target)

            # unknown
            else:
                raise ValueError("target is of unknown type or a list of unknown types: {}".format(type(target[0])))

            # new models were added, increase num_models
            # if the user didn't provide a custon numer to use
            if num_new_models is None:
                self.num_models += detected_num_models
            else:
                self.num_models += num_new_models

        else:  # memory unfriendly ensembles

            ttda = []

            # 1. ttda array
            if isinstance(target.dtype.type(), (np.number, float)):
                raise ValueError('ttda arrays cannot be added to ensembles, for which memory_friendly_ttda=False, '
                                 'you can call convert_to_memory_friendly, but it will discard the stored gensim'
                                 'models and only keep the relevant topic term distributions from them.')

            # 2. list of ensembles
            elif isinstance(target[0], type(self)):
                for ensemble in target:
                    self.tms += ensemble.tms
                ttda = np.concatenate([ensemble.ttda for ensemble in target], axis=0)

            # 3. list of gensim models
            elif isinstance(target[0], basemodel.BaseTopicModel):
                self.tms += target.tolist()
                ttda = np.concatenate([model.get_topics() for model in target], axis=0)

            # unknown
            else:
                raise ValueError("target is of unknown type or a list of unknown types: {}".format(type(target[0])))

            # in this case, len(self.tms) should
            # always match self.num_models
            if num_new_models is not None and num_new_models + self.num_models != len(self.tms):
                logger.info('num_new_models will be ignored. num_models should match the number of '
                            'stored models for a memory unfriendly ensemble')
            self.num_models = len(self.tms)

        logger.info("Ensemble contains {} models and {} topics now".format(self.num_models, len(self.ttda)))

        if self.ttda.shape[1] != ttda.shape[1]:
            raise ValueError(("target ttda dimensions do not match. Topics must be {} but was"
                                "{} elements large").format(self.ttda.shape[-1], ttda.shape[-1]))
        self.ttda = np.append(self.ttda, ttda, axis=0)

        # tell recluster that the distance matrix needs to be regenerated
        self.asymmetric_distance_matrix_outdated = True

    def save(self, fname):
        """Save the ensemble to a file.

        Parameters
        ----------
        fname : str
            Path to the system file where the model will be persisted.

        """

        logger.info("saving %s object to %s", self.__class__.__name__, fname)

        utils.pickle(self, fname)

    @staticmethod
    def load(fname):
        """Load a previously stored ensemble from disk.

        Parameters
        ----------
        fname : str
            Path to file that contains the needed object.

        Returns
        -------
            A previously saved ensembleLda object

        """

        # message copied from [1]
        logger.info("loading %s object from %s", EnsembleLda.__name__, fname)

        eLDA = utils.unpickle(fname)

        return eLDA

    def generate_topic_models_multiproc(self, num_models, ensemble_workers):
        """Will make the ensemble multiprocess, which results in a speedup on multicore machines. Results from the
        processes will be piped to the parent and concatenated.

        Parameters
        ----------
        num_models : int
            how many models to train in the ensemble
        ensemble_workers : int
            into how many processes to split the models will be set to max(workers, num_models), to avoid workers that
            are supposed to train 0 models.

            to get maximum performance, set to the number of your cores, if non-parallelized models are being used in
            the ensemble (LdaModel).

            For LdaMulticore, the performance gain is small and gets larger for a significantly smaller corpus.
            In that case, ensemble_workers=2 can be used.

        """

        # the way random_states is handled needs to prevent getting different results when multiprocessing is on,
        # or getting the same results in every lda children. so it is solved by generating a list of state seeds before
        # multiprocessing is started.
        random_states = [self.random_state.randint(self._MAX_RANDOM_STATE) for _ in range(num_models)]

        # each worker has to work on at least one model.
        # Don't spawn idle workers:
        workers = min(ensemble_workers, num_models)

        # create worker processes:
        # from what I know this is basically forking with a jump to a target function in each child
        # so modifying the ensemble object will not modify the one in the parent because of no shared memory
        processes = []
        pipes = []
        num_models_unhandled = num_models  # how many more models need to be trained by workers?

        for i in range(workers):
            try:
                parentConn, childConn = Pipe()
                num_subprocess_models = 0
                if i == workers - 1:  # i is a index, hence -1
                    # is this the last worker that needs to be created?
                    # then task that worker with all the remaining models
                    num_subprocess_models = num_models_unhandled
                else:
                    num_subprocess_models = int(num_models_unhandled / (workers - i))

                # get the chunk from the random states that is meant to be for those models
                random_states_for_worker = random_states[-num_models_unhandled:][:num_subprocess_models]

                p = Process(target=self.generate_topic_models,
                            args=(num_subprocess_models, random_states_for_worker, childConn))

                processes += [p]
                pipes += [(parentConn, childConn)]
                p.start()

                num_models_unhandled -= num_subprocess_models

            except ProcessError:
                logger.error("could not start process {}".format(i))
                # close all pipes
                for p in pipes:
                    p[1].close()
                    p[0].close()
                # end all processes
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                    del p
                # stop
                raise

        # aggregate results
        # will also block until workers are finished
        for p in pipes:
            answer = p[0].recv()  # [0], because that is the parentConn
            p[0].close()
            # this does basically the same as the generate_topic_models function (concatenate all the ttdas):
            if not self.memory_friendly_ttda:
                self.tms += answer
                ttda = np.concatenate([model.get_topics() for model in answer])
            else:
                ttda = answer
            self.ttda = np.concatenate([self.ttda, ttda])

        # end all processes
        for p in processes:
            p.terminate()

    def generate_topic_models(self, num_models, random_states=None, pipe=None):
        """Will train the topic models, that form the ensemble.

        Parameters
        ----------
        num_models : int
            number of models to be generated
        random_states : list
            list of numbers or np.random.RandomState objects. Will be autogenerated based on the ensembles
            RandomState if None (default).
        pipe : multiprocessing.pipe
            Default None. If provided, will send the trained models over this pipe. If memory friendly, it will only
            send the ttda.

        """

        if pipe is not None:
            logger.info("Spawned worker to generate {} topic models".format(num_models))

        if random_states is None:
            random_states = [self.random_state.randint(self._MAX_RANDOM_STATE) for _ in range(num_models)]

        assert len(random_states) == num_models

        kwArgs = self.gensim_kw_args.copy()

        tm = None  # remember one of the topic models from the following
        # loop, in order to collect some properties from it afterwards.

        for i in range(num_models):

            kwArgs["random_state"] = random_states[i]

            tm = self.topic_model_kind(**kwArgs)

            # adds the lambda (that is the unnormalized get_topics) to ttda, which is
            # a list of all those lambdas
            self.ttda = np.concatenate([self.ttda, tm.get_topics()])

            # only saves the model if it is not "memory friendly"
            if not self.memory_friendly_ttda:
                self.tms += [tm]

        # use one of the tms to get some info that will be needed later
        self.sstats_sum = tm.state.sstats.sum()
        self.eta = tm.eta

        if pipe is not None:
            # send the ttda that is in the child/workers version of the memory into the pipe
            # available, after generate_topic_models has been called in the worker
            if self.memory_friendly_ttda:
                # remember that this code is inside the worker processes memory,
                # so self.ttda is the ttda of only a chunk of models
                pipe.send(self.ttda)
            else:
                pipe.send(self.tms)

            pipe.close()

    def asymmetric_distance_matrix_worker(self, worker_id, ttdas_sent, n_ttdas, pipe, threshold, method):
        """ worker, that computes the distance to all other nodes
        from a chunk of nodes. https://stackoverflow.com/a/1743350
        """

        logger.info("Spawned worker to generate {} rows of the asymmetric distance matrix".format(n_ttdas))
        # the chunk of ttda that's going to be calculated:
        ttda1 = self.ttda[ttdas_sent:ttdas_sent + n_ttdas]
        distance_chunk = self.calculate_asymmetric_distance_matrix_chunk(ttda1=ttda1, ttda2=self.ttda,
                                                                         threshold=threshold,
                                                                         start_index=ttdas_sent,
                                                                         method=method)
        pipe.send((worker_id, distance_chunk))  # remember that this code is inside the workers memory
        pipe.close()

    def generate_asymmetric_distance_matrix(self, threshold=None, workers=1, method="mass"):
        """Makes the pairwise distance matrix for all the ttdas from the ensemble.

        Returns the asymmetric pairwise distance matrix that is used in the DBSCAN clustering.

        Afterwards, the model needs to be reclustered for this generated matrix to take effect.

        Parameters
        ----------
        threshold : float, optional
            if threshold is None and method == "rank":
                threshold = 0.11
            if threshold is None and method == "mass":
                threshold = 0.95

            threshold keeps by default 95% of the largest terms by mass. Except the "fast" parameter is "rank", then
            it just selects that many of the largest terms.
        workers : number, optional
            when workers is None, it defaults to os.cpu_count() for maximum performance. Default is 1, which is not
            multiprocessed. Set to > 1 to enable multiprocessing.

        """

        # matrix is up to date afterwards
        self.asymmetric_distance_matrix_outdated = False

        if threshold is None:
            threshold = {"mass": 0.95, "rank": 0.11}[method]

        logger.info("Generating a {} x {} asymmetric distance matrix...".format(len(self.ttda), len(self.ttda)))

        # singlecore:
        if workers is not None and workers <= 1:
            self.asymmetric_distance_matrix = self.calculate_asymmetric_distance_matrix_chunk(ttda1=self.ttda,
                                                                                              ttda2=self.ttda,
                                                                                              threshold=threshold,
                                                                                              start_index=0,
                                                                                              method=method)
            return self.asymmetric_distance_matrix

        # multicore:

        # best performance on 2-core machine: 2 workers
        if workers is None:
            workers = os.cpu_count()

        # create worker processes:
        processes = []
        pipes = []
        ttdas_sent = 0

        for i in range(workers):

            try:
                parentConn, childConn = Pipe()

                # figure out how many ttdas to send to the worker
                # 9 ttdas to 4 workers: 2 2 2 3
                n_ttdas = 0
                if i == workers - 1:  # i is a index, hence -1
                    # is this the last worker that needs to be created?
                    # then task that worker with all the remaining models
                    n_ttdas = len(self.ttda) - ttdas_sent
                else:
                    n_ttdas = int((len(self.ttda) - ttdas_sent) / (workers - i))

                p = Process(target=self.asymmetric_distance_matrix_worker,
                            args=(i, ttdas_sent, n_ttdas, childConn, threshold, method))
                ttdas_sent += n_ttdas

                processes += [p]
                pipes += [(parentConn, childConn)]
                p.start()

            except ProcessError:
                logger.error("could not start process {}".format(i))
                # close all pipes
                for p in pipes:
                    p[1].close()
                    p[0].close()
                # end all processes
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                    del p
                raise

        distances = []
        # note, that the following loop maintains order in how the ttda will be concatenated
        # which is very important. Ordering in ttda has to be the same as when using only one process
        for p in pipes:
            answer = p[0].recv()  # [0], because that is the parentConn
            p[0].close()  # child conn will be closed from inside the worker
            # this does basically the same as the generate_topic_models function (concatenate all the ttdas):
            distances += [answer[1]]

        # end all processes
        for p in processes:
            p.terminate()

        self.asymmetric_distance_matrix = np.concatenate(distances)

        return self.asymmetric_distance_matrix

    def calculate_asymmetric_distance_matrix_chunk(self, ttda1, ttda2, threshold, start_index, method):
        """Iterates over ttda1 and calculates the
        distance to every ttda in tttda2.

        Parameters
        ----------
        ttda1 and ttda2: 2D arrays of floats
            Two ttda matrices that are going to be used for distance calculation. Each row in ttda corresponds to one
            topic. Each cell in the resulting matrix corresponds to the distance between a topic pair.
        threshold : float, optional
            threshold defaults to: {"mass": 0.95, "rank": 0.11}, depending on the selected method
        start_index : int
            this function might be used in multiprocessing, so start_index has to be set as ttda1 is a chunk of the
            complete ttda in that case. start_index would be 0 if ttda1 == self.ttda. When self.ttda is split into two
            pieces, each 100 ttdas long, then start_index should be be 100. default is 0
        method : {'mass', 'rank}, optional
            method can be "mass" for the original masking method or "rank" for a faster masking method that selects
            by rank of largest elements in the topic word distribution, to determine which tokens are relevant for the
            topic.

        Returns
        -------
        2D Numpy.numpy.ndarray of floats
        """

        if method not in ["mass", "rank"]:
            raise ValueError("method {} unknown".format(method))

        # select masking method:
        def mass_masking(a):
            """original masking method. returns a new binary mask"""
            sorted_a = np.sort(a)[::-1]
            largest_mass = sorted_a.cumsum() < threshold
            smallest_valid = sorted_a[largest_mass][-1]
            return a >= smallest_valid

        def rank_masking(a):
            """faster masking method. returns a new binary mask"""
            return a > np.sort(a)[::-1][int(len(a) * threshold)]

        create_mask = {"mass": mass_masking, "rank": rank_masking}[method]

        # some help to find a better threshold by useful log messages
        avg_mask_size = 0

        # initialize the distance matrix. ndarray is faster than zeros
        distances = np.ndarray((len(ttda1), len(ttda2)))

        # now iterate over each topic
        for i in range(len(ttda1)):

            # create mask from a, that removes noise from a and keeps the largest terms
            a = ttda1[i]
            mask = create_mask(a)
            a_masked = a[mask]

            avg_mask_size += mask.sum()

            # now look at every possible pair for topic a:
            for j in range(len(ttda2)):

                # distance to itself is 0
                if i + start_index == j:
                    distances[i][j] = 0
                    continue

                # now mask b based on a, which will force the shape of a onto b
                b_masked = ttda2[j][mask]

                distance = 0
                # is the masked b just some empty stuff? Then forget about it, no similarity, distance is 1
                # (not 2 because that correspondsto negative values. The maximum distance is 1 here)
                # don't normalize b_masked, otherwise the following threshold will never work:
                if b_masked.sum() <= 0.05:
                    distance = 1
                else:
                    # if there is indeed some non-noise stuff in the masked topic b, look at
                    # how similar the two topics are. note that normalizing is not needed for cosine distance,
                    # as it only looks at the angle between vectors
                    distance = cosine(a_masked, b_masked)

                distances[i][j] = distance

        avg_mask_size = avg_mask_size / ttda1.shape[0] / ttda1.shape[1]
        percent = round(100 * avg_mask_size, 1)
        if avg_mask_size > 0.75:
            # if the masks covered more than 75% of tokens on average,
            # print a warning. info otherwise.
            # The default mass setting of 0.95 makes uniform true masks on the opinosis corpus.
            logger.warning('The given threshold of {} covered on average '
                '{}% of tokens, which might be too much'.format(threshold, percent))
        else:
            logger.info('The given threshold of {} covered on average {}% of tokens'.format(threshold, percent))

        return distances

    def generate_topic_clusters(self, eps=0.1, min_samples=None):
        """Runs the DBSCAN algorithm on all the detected topics from the models in the ensemble and labels them with
        label-indices.

        The final approval and generation of stable topics is done in generate_stable_topics().

        Parameters
        ----------
        eps : float
            eps is 0.1 by default.
        min_samples : int
            min_samples is int(self.num_models / 2)
        """

        if min_samples is None:
            min_samples = int(self.num_models / 2)

        logger.info("Fitting the clustering model")

        self.cluster_model = CBDBSCAN(eps=eps, min_samples=min_samples)
        self.cluster_model.fit(self.asymmetric_distance_matrix)

    def generate_stable_topics(self, min_cores=None):
        """generates stable topics out of the clusters. This function is the last step that has to be done in the
        ensemble.

        Stable topics can be retreived afterwards using get_topics().

        Parameters
        ----------
        min_cores : int
            how many cores a cluster has to have, to be treated as stable topic. That means, how many topics
            that look similar have to be present, so that the average topic in those is used as stable topic.

            defaults to min_cores = min(3, max(1, int(self.num_models /4 +1)))

        """

        logger.info("Generating stable topics")

        # min_cores being 0 makes no sense. there has to be a core for a cluster
        # or there is no cluster
        if min_cores == 0:
            min_cores = 1

        if min_cores is None:
            # min_cores is a number between 1 and 3, depending on the number of models
            min_cores = min(3, max(1, int(self.num_models / 4 + 1)))

        results = self.cluster_model.results

        # first, group all the learned cores based on the label,
        # which was assigned in the cluster_model. The result is a
        # dict of {group: [topic, ...]}
        grouped_by_labels = {}
        for topic in results:
            if topic["is_core"]:
                topic = topic.copy()

                # counts how many different labels a core has as parents
                topic["amount_parent_labels"] = len(topic["parent_labels"])

                label = topic["label"]
                if label not in grouped_by_labels:
                    grouped_by_labels[label] = []
                grouped_by_labels[label].append(topic)

        # then aggregate their amount_parent_labels by maxing and parent_labels by concatenating
        # the result is sorted by amount_parent_labels and stored in sorted_clusters
        sorted_clusters = []
        for label, group in grouped_by_labels.items():
            amount_parent_labels = 0
            parent_labels = []  # will be a list of sets
            for topic in group:
                amount_parent_labels = max(topic["amount_parent_labels"], amount_parent_labels)
                parent_labels.append(topic["parent_labels"])
            # - here, nan means "not yet evaluated"
            # - removing empty sets from parent_labels
            sorted_clusters.append({
                "amount_parent_labels": amount_parent_labels,
                "parent_labels": [x for x in parent_labels if len(x) > 0],
                "is_valid": np.nan,
                "label": label
            })

        # start with the most significant core.
        sorted_clusters = sorted(sorted_clusters,
            key=lambda cluster: (
                cluster["amount_parent_labels"],
                cluster["label"]  # makes sorting deterministic
            ), reverse=True)

        def remove_from_all_sets(label):
            """removes a label from every set in "parent_labels" for
            each core in sorted_clusters."""
            for a in sorted_clusters:
                if label in a["parent_labels"]:
                    a["parent_labels"].remove(label)

        # APPLYING THE RULES 1 to 3
        # iterate over the cluster labels, see which clusters/labels
        # are valid to cause the creation of a stable topic
        for i, cluster in enumerate(sorted_clusters):
            label = cluster["label"]
            # label is iterating over 0, 1, 3, ...

            # 1. rule - remove if the cores in the cluster have no parents
            if cluster["amount_parent_labels"] == 0:
                # label is NOT VALID
                cluster["is_valid"] = False
                remove_from_all_sets(label)

            # 2. rule - remove if it has less than min_cores as parents
            # (parents are always also cores)
            if len(cluster["parent_labels"]) < min_cores:
                cluster["is_valid"] = False
                remove_from_all_sets(label)

            # 3. checking for "easy_valid"s
            # checks if the core has at least min_cores of cores with the only label as itself
            if cluster["amount_parent_labels"] >= 1:
                if sum(map(lambda x: x == {label}, cluster["parent_labels"])) >= min_cores:
                    cluster["is_valid"] = True

        # Reaplying the rule 3
        # this happens when we have a close relationship among 2 or more clusters
        for cluster in [cluster for cluster in sorted_clusters if np.isnan(cluster["is_valid"])]:

            # this checks for parent_labels, which are also modified in this function
            # hence order will influence the result. it starts with the most significant
            # label (hence "sorted_clusters")
            if sum(map(lambda x: x == {label}, cluster["parent_labels"])) >= min_cores:
                cluster["is_valid"] = True
            else:
                cluster["is_valid"] = False
                # removes a label from every set in parent_labels for each core
                for a in sorted_clusters:
                    if label in a["parent_labels"]:
                        a["parent_labels"].remove(label)

        # list of all the label numbers that are valid
        valid_labels = np.array([cluster["label"] for cluster in sorted_clusters if cluster["is_valid"]])

        for topic in results:
            topic["valid_parents"] = {label for label in topic["parent_labels"] if label in valid_labels}

        def validate_core(core):
            """Core is a dict of {is_core, valid_parents, labels} among others.
            If not a core returns False. Only cores with the valid_parents as
            its own label can be a valid core. Returns True if that is the case,
            False otherwise"""
            if not core["is_core"]:
                return False
            else:
                ret = core["valid_parents"] == {core["label"]}
                return ret

        # keeping only VALID cores
        valid_core_mask = np.vectorize(validate_core)(results)
        valid_topics = self.ttda[valid_core_mask]
        topic_labels = np.array([topic["label"] for topic in results])[valid_core_mask]
        unique_labels = np.unique(topic_labels)

        num_stable_topics = len(unique_labels)
        stable_topics = np.empty((num_stable_topics, len(self.id2word)))

        # for each cluster
        for l, label in enumerate(unique_labels):
            # mean of all the topics that are of that cluster
            topics_of_cluster = np.array([topic for t, topic in enumerate(valid_topics) if topic_labels[t] == label])
            stable_topics[l] = topics_of_cluster.mean(axis=0)

        self.sorted_clusters = sorted_clusters
        self.stable_topics = stable_topics

    def recluster(self, eps=0.1, min_samples=None, min_cores=None):
        """Runs the CBDBSCAN algorithm on all the detected topics from the children of the ensemble.

        Generates stable topics out of the clusters afterwards.

        Finally, stable topics can be retreived using get_topics().

        Parameters
        ----------
        eps : float
            epsilon for the CBDBSCAN algorithm, having the same meaning as in classic DBSCAN clustering.
            default: 0.1
        min_samples : int
            The minimum number of sampels in the neighborhood of a topic to be considered a core in CBDBSCAN.
            default: int(self.num_models / 2)
        min_cores : int
            how many cores a cluster has to have, to be treated as stable topic. That means, how many topics
            that look similar have to be present, so that the average topic in those is used as stable topic.
            default: min(3, max(1, int(self.num_models /4 +1)))

        """
        # if new models were added to the ensemble, the distance matrix needs to be generated again
        if self.asymmetric_distance_matrix_outdated:
            logger.info("asymmetric distance matrix is outdated due to add_model")
            self.generate_asymmetric_distance_matrix()

        self.generate_topic_clusters(eps, min_samples)
        self.generate_stable_topics(min_cores)
        self.generate_gensim_representation()

    # GENSIM API
    # to make using the ensemble in place of a gensim model as easy as possible

    def get_topics(self):
        return self.stable_topics

    def has_gensim_representation(self):
        """checks if stable topics area vailable and if the internal
        gensim representation exists. If not, raises errors"""
        if self.classic_model_representation is None:
            if len(self.stable_topics) == 0:
                raise ValueError("no stable topic was detected")
            else:
                raise ValueError("use generate_gensim_representation() first")

    def __getitem__(self, i):
        """see :py:class:`gensim.models.LdaModel`"""
        self.has_gensim_representation()
        return self.classic_model_representation[i]

    def inference(self, *posargs, **kwArgs):
        """see :py:class:`gensim.models.LdaModel`"""
        self.has_gensim_representation()
        return self.classic_model_representation.inference(*posargs, **kwArgs)

    def log_perplexity(self, *posargs, **kwArgs):
        """see :py:class:`gensim.models.LdaModel`"""
        self.has_gensim_representation()
        return self.classic_model_representation.log_perplexity(*posargs, **kwArgs)

    def print_topics(self, *posargs, **kwArgs):
        """see :py:class:`gensim.models.LdaModel`"""
        self.has_gensim_representation()
        return self.classic_model_representation.print_topics(*posargs, **kwArgs)

    @property
    def id2word(self):
        return self.gensim_kw_args["id2word"]


class CBDBSCAN():

    def __init__(self, eps=0.1, min_samples=4):

        self.eps = eps
        self.min_samples = min_samples

    def fit(self, amatrix):

        self.next_label = 0

        results = [{
            "is_core": False,
            "parent_labels": set(),
            "parent_ids": set(),
            "num_samples": 0,
            "label": None
        } for _ in range(len(amatrix))]

        tmp_amatrix = amatrix.copy()

        # to avoid problem about comparing the topic with itself
        np.fill_diagonal(tmp_amatrix, 1)

        min_distance_per_topic = [(distance, index) for index, distance in enumerate(tmp_amatrix.min(axis=1))]
        min_distance_per_topic_sorted = sorted(min_distance_per_topic, key=lambda x: x)
        ordered_min_similarity = [index for distance, index in min_distance_per_topic_sorted]

        num_topics = len(amatrix)

        def scan_topic(topic_index, parent_id=None, current_label=None, parent_neighbors=None):

            # count how many neighbors
            # check which indices (arange 0 to num_topics) is closer than eps
            neighbors = np.arange(num_topics)[tmp_amatrix[topic_index] < self.eps]
            num_samples = len(neighbors)

            # If the number of neighbors of a topic is large enough,
            # it is considered a core.
            # This also takes neighbors that already are identified as core in count.
            if num_samples >= self.min_samples:
                # This topic is a core!
                results[topic_index]["is_core"] = True
                results[topic_index]["num_samples"] = num_samples

                # if current_label is none, then this is the first core
                # of a new cluster (hence next_label is used)
                if current_label is None:
                    # next_label is initialized with 0 in
                    # fit() for the first cluster
                    current_label = self.next_label
                    self.next_label += 1

                else:
                    # In case the core has a parent, that means
                    # it has a label (= the parents label).
                    # Check the distance between the
                    # new core and the parent

                    # parent neighbors is the list of neighbors of parent_id
                    # (the topic_index that called this function recursively)
                    all_members_of_current_cluster = list(parent_neighbors)
                    all_members_of_current_cluster.append(parent_id)

                    # look if 25% of the members of the current cluster are also
                    # close to the current/new topic...

                    # example: (topic_index, 0), (topic_index, 2), (topic_index, ...)
                    all_members_of_current_cluster_ix = np.ix_([topic_index], all_members_of_current_cluster)
                    # use the result of the previous step to index the matrix and see if those distances
                    # are smaller then epsilon. relations_to_the_cluster is a boolean array, True for close elements
                    relations_to_the_cluster = tmp_amatrix[all_members_of_current_cluster_ix] < self.eps

                    # if less than 25% of the elements are close, then the topic index in question is not a
                    # core of the current_label, but rather the core of a new cluster
                    if relations_to_the_cluster[0].mean() < 0.25:
                        # start new cluster by changing current_label
                        current_label = self.next_label
                        self.next_label += 1

                results[topic_index]["label"] = current_label

                for neighbor in neighbors:

                    if results[neighbor]["label"] is None:
                        ordered_min_similarity.remove(neighbor)
                        # try to extend the cluster into the direction
                        # of the neighbor
                        scan_topic(neighbor, parent_id=topic_index,
                                   current_label=current_label,
                                   parent_neighbors=neighbors)

                    results[neighbor]["parent_ids"].add(topic_index)
                    results[neighbor]["parent_labels"].add(current_label)

            else:
                # this topic is not a core!
                if current_label is None:
                    results[topic_index]["label"] = -1
                else:
                    results[topic_index]["label"] = current_label

        # elements are going to be removed from that array in scan_topic, do until it is empty
        while len(ordered_min_similarity) != 0:
            next_topic_index = ordered_min_similarity.pop(0)
            scan_topic(next_topic_index)

        self.results = results
