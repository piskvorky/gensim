#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors: Tobias Brigl <github.com/sezanzeb>, Alex Loosley <aloosley@alumni.brown.edu>,
# Stephan Sahm <stephan.sahm@gmx.de>, Alex Salles <alex.salles@gmail.com>, Data Reply Munich

"""Ensemble Latent Dirichlet Allocation (eLDA), a method of training a topic model ensemble and extracting
reliable topics that are consistently learned accross the ensemble.  eLDA has the added benefit that the user
does not need to know the exact number of topics the topic model should extract ahead of time.  For more details
read our paper (https://www.hip70890b.de/machine_learning/ensemble_LDA/).

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
.. [1] REHUREK, Radim and Sojka, PETR, 2010, Software framework for topic modelling with large corpora. In : THE LREC
       2010 WORKSHOP ON NEW CHALLENGES FOR NLP FRAMEWORKS [online]. Msida : University of Malta. 2010. p. 45-50.
       Available from: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.695.4595

.. [2] BRIGL, Tobias, 2019, Extracting Reliable Topics using Ensemble Latent Dirichlet Allocation [Bachelor Thesis].
       Ingolstadt: Technische Hochschule. Available from: https://www.hip70890b.de/machine_learning/ensemble_LDA/

Citation
--------
At the moment, there is no paper associated to ensemble LDA but we are working on publicly releasing Tobi Brigl's
bachelor thesis on the topic.  In the meantime, please include a mention of us (Brigl, T.; Salles, A.; Loosley, A) and
a link to this file (https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/ensemblelda.py) if you use
this work in a published or open-source project.

Other Notes
-----------
.. The adjectives stable and reliable (topics) are used somewhat interchangeably throughout the doc strings and
comments.

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
    """Ensemble Latent Dirichlet Allocation (eLDA), a method of training a topic model ensemble and extracting
    reliable topics that are consistently learned accross the ensemble.  eLDA has the added benefit that the user
    does not need to know the exact number of topics the topic model should extract ahead of time [2].

    """
    def __init__(self, topic_model_class="lda", num_models=3,
                 min_cores=None,  # default value from _generate_stable_topics()
                 epsilon=0.1, ensemble_workers=1, memory_friendly_ttda=True,
                 min_samples=None, masking_method="mass", masking_threshold=None,
                 distance_workers=1, random_state=None, **gensim_kw_args):
        """

        Parameters
        ----------
        topic_model_class : str, topic model, optional
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
            Defaults to 0.1. Epsilon for the CBDBSCAN clustering that generates the stable topics.
        ensemble_workers : int, optional
            Spawns that many processes and distributes the models from the ensemble to those as evenly as possible.
            num_models should be a multiple of ensemble_workers.

            Setting it to 0 or 1 will both use the nonmultiprocessing version. Default:1
        memory_friendly_ttda : boolean, optional
            If True, the models in the ensemble are deleted after training and only a concatenation of each model's
            topic term distribution (called ttda) is kept to save memory.

            Defaults to True. When False, trained models are stored in a list in self.tms, and no models that are not
            of a gensim model type can be added to this ensemble using the add_model function.

            If False, any topic term matrix can be suplied to add_model.
        min_samples : int, optional
            Required int of nearby topics for a topic to be considered as 'core' in the CBDBSCAN clustering.
        masking_method : {'mass', 'rank'}, optional
            Choose one of "mass" (default) or "rank" (percentile, faster).

            For clustering, distances between topic-term distributions are asymmetric.  In particular, the distance
            (technically a divergence) from distribution A to B is more of a measure of if A is contained in B.  At a
            high level, this involves using distribution A to mask distribution B and then calculating the cosine
            distance between the two.  The masking can be done in two ways:
                mass: forms mask by taking the top ranked terms until their cumulative mass reaches the
                    `masking_threshold`
                rank: forms mask by taking the top ranked terms (by mass) until the `masking_threshold` is reached.
                    For example, a ranking threshold of 0.11 means the top 0.11 terms by weight are used to form a mask.
        masking_threshold : float, optional
            Default: None, which uses 0.95 for "mass", and 0.11 for masking_method "rank".  In general, too small a mask
            threshold leads to inaccurate calculations (no signal) and too big a mask leads to noisy distance
            calculations.  Defaults are often a good sweet spot for this hyperparameter.
        distance_workers : int, optional
            When distance_workers is None, it defaults to os.cpu_count() for maximum performance. Default is 1, which
            is not multiprocessed. Set to > 1 to enable multiprocessing.
        **gensim_kw_args
            Parameters for each gensim model (e.g. :py:class:`gensim.models.LdaModel`) in the ensemble.

        """
        # INTERNAL PARAMETERS
        # Set random state
        # nps max random state of 2**32 - 1 is too large for windows:
        self._MAX_RANDOM_STATE = np.iinfo(np.int32).max

        # _COSINE_DISTANCE_CALCULATION_THRESHOLD is used so that cosine distance calculations can be sped up by skipping
        # distance calculations for highly masked topic-term distributions
        self._COSINE_DISTANCE_CALCULATION_THRESHOLD = 0.05

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
            raise ValueError(
                "at least one of corpus/id2word must be specified, to establish "
                "input space dimensionality. Corpus should be provided using the "
                "`corpus` keyword argument.")

        if type(topic_model_class) == type and issubclass(topic_model_class, ldamodel.LdaModel):
            self.topic_model_class = topic_model_class
        else:
            kinds = {
                "lda": ldamodel.LdaModel,
                "ldamulticore": ldamulticore.LdaMulticore
            }
            if topic_model_class not in kinds:
                raise ValueError(
                    "topic_model_class should be one of 'lda', 'ldamulticode' or a model inheriting from LdaModel")
            self.topic_model_class = kinds[topic_model_class]

        self.num_models = num_models
        self.gensim_kw_args = gensim_kw_args

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

        if gensim_kw_args.get("corpus") is None:
            return
        if "iterations" in gensim_kw_args and gensim_kw_args["iterations"] <= 0:
            return
        if "passes" in gensim_kw_args and gensim_kw_args["passes"] <= 0:
            return

        logger.info("Generating {} topic models...".format(num_models))

        if ensemble_workers > 1:
            self._generate_topic_models_multiproc(num_models, ensemble_workers)
        else:
            # singlecore
            self._generate_topic_models(num_models)

        self._generate_asymmetric_distance_matrix(
            workers=self.distance_workers,
            threshold=masking_threshold,
            method=masking_method)
        self._generate_topic_clusters(epsilon, min_samples)
        self._generate_stable_topics(min_cores)

        # create model that can provide the usual gensim api to the stable topics from the ensemble
        self.generate_gensim_representation()

    def convert_to_memory_friendly(self):
        """removes the stored gensim models and only keeps their ttdas"""
        self.tms = []
        self.memory_friendly_ttda = True

    def generate_gensim_representation(self):
        """Creates a gensim model from the stable topics.

        The returned representation is an Gensim LdaModel (:py:class:`gensim.models.LdaModel`) that has been
        instantiated with an A-priori belief on word probability, eta, that represents the topic-term distributions of
        any stable topics the were found by clustering over the ensemble of topic distributions.

        When no stable topics have been detected, None is returned.

        Returns
        -------
        :py:class:`gensim.models.LdaModel`
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
            self.classic_model_representation = None
            return

        # create a new gensim model
        params = self.gensim_kw_args.copy()
        params["eta"] = self.eta
        params["num_topics"] = num_stable_topics
        # adjust params in a way that no training happens
        params["iterations"] = 0  # no training
        params["passes"] = 0  # no training

        classic_model_representation = self.topic_model_class(**params)

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
            self._generate_asymmetric_distance_matrix()
            self.recluster()

        The ttda of another ensemble can also be used, in that case set num_new_models to the num_models parameter
        of the ensemble, that means the number of classic models in the ensemble that generated the ttda. This is
        important, because that information is used to estimate "min_samples" for _generate_topic_clusters.

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

    def _generate_topic_models_multiproc(self, num_models, ensemble_workers):
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

                p = Process(target=self._generate_topic_models,
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
            # this does basically the same as the _generate_topic_models function (concatenate all the ttdas):
            if not self.memory_friendly_ttda:
                self.tms += answer
                ttda = np.concatenate([model.get_topics() for model in answer])
            else:
                ttda = answer
            self.ttda = np.concatenate([self.ttda, ttda])

        # end all processes
        for p in processes:
            p.terminate()

    def _generate_topic_models(self, num_models, random_states=None, pipe=None):
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

            tm = self.topic_model_class(**kwArgs)

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
            # available, after _generate_topic_models has been called in the worker
            if self.memory_friendly_ttda:
                # remember that this code is inside the worker processes memory,
                # so self.ttda is the ttda of only a chunk of models
                pipe.send(self.ttda)
            else:
                pipe.send(self.tms)

            pipe.close()

    def _asymmetric_distance_matrix_worker(self, worker_id, ttdas_sent, n_ttdas, pipe, threshold, method):
        """ Worker that computes the distance to all other nodes
        from a chunk of nodes. https://stackoverflow.com/a/1743350
        """

        logger.info("Spawned worker to generate {} rows of the asymmetric distance matrix".format(n_ttdas))
        # the chunk of ttda that's going to be calculated:
        ttda1 = self.ttda[ttdas_sent:ttdas_sent + n_ttdas]
        distance_chunk = self._calculate_asymmetric_distance_matrix_chunk(
            ttda1=ttda1, ttda2=self.ttda, threshold=threshold, start_index=ttdas_sent, method=method)
        pipe.send((worker_id, distance_chunk))  # remember that this code is inside the workers memory
        pipe.close()

    def _generate_asymmetric_distance_matrix(self, threshold=None, workers=1, method="mass"):
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

        # singlecore
        if workers is not None and workers <= 1:
            self.asymmetric_distance_matrix = self._calculate_asymmetric_distance_matrix_chunk(
                ttda1=self.ttda, ttda2=self.ttda, threshold=threshold, start_index=0, method=method)
            return self.asymmetric_distance_matrix

        # else, if workers > 1 use multiprocessing
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

                p = Process(target=self._asymmetric_distance_matrix_worker,
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
            # this does basically the same as the _generate_topic_models function (concatenate all the ttdas):
            distances += [answer[1]]

        # end all processes
        for p in processes:
            p.terminate()

        self.asymmetric_distance_matrix = np.concatenate(distances)

        return self.asymmetric_distance_matrix

    def _calculate_asymmetric_distance_matrix_chunk(self, ttda1, ttda2, threshold, start_index, method):
        """Internal method calculating an (asymmetric) distance from each topic term distribution in ttda1 to each topic
        term distribution in ttda2 and returns a matrix of distances, size len(ttda1) by len(ttda2).

        This is an internal method and should not be used for two general ttda arrays.

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
            by rank of largest elements in the topic term distribution, to determine which tokens are relevant for the
            topic.

        Returns
        -------
        2D Numpy.numpy.ndarray of floats
            Asymmetric distance matrix of size len(ttda1) by len(ttda2).
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
        for ttd1_idx, ttd1 in enumerate(ttda1):

            # create mask from ttd1 that removes noise from a and keeps the largest terms
            mask = create_mask(ttd1)
            ttd1_masked = ttd1[mask]

            avg_mask_size += mask.sum()

            # now look at every possible pair for topic a:
            for ttd2_idx, ttd2 in enumerate(ttda2):

                # distance to itself is 0
                if ttd1_idx + start_index == ttd2_idx:
                    distances[ttd1_idx][ttd2_idx] = 0
                    continue

                # now mask b based on a, which will force the shape of a onto b
                ttd2_masked = ttd2[mask]

                # Smart distance calculation avoids calculating cosine distance for highly masked topic-term
                # distributions that will have distance values near 1.
                if ttd2_masked.sum() <= self._COSINE_DISTANCE_CALCULATION_THRESHOLD:
                    distance = 1
                else:
                    distance = cosine(ttd1_masked, ttd2_masked)

                distances[ttd1_idx][ttd2_idx] = distance

        percent = round(100 * avg_mask_size / ttda1.shape[0] / ttda1.shape[1], 1)
        logger.info('The given threshold of {} covered on average {}% of tokens'.format(threshold, percent))

        return distances

    def _generate_topic_clusters(self, eps=0.1, min_samples=None):
        """Runs the DBSCAN algorithm on all the detected topics from the models in the ensemble and labels them with
        label-indices.

        The final approval and generation of stable topics is done in _generate_stable_topics().

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

    def _validate_core(self, core):
        """If core is not a core returns False. Only cores with the valid_parents as its own label can be a valid core.
        Returns True if that is the case, False otherwise.

        Parameters
        ----------
        core : {'is_core', 'valid_parents', 'labels'}
            eps is 0.1 by default.
        """
        return core["is_core"] and (core["valid_parents"] == {core["label"]})

    def _group_by_labels(self, results):
        """Groups all the learned cores by their label, which was assigned in the cluster_model

        Parameters
        ----------
        results : {list of {'is_core', 'parent_labels', 'num_samples', 'label'}}
            After calling .fit on a CBDBSCAN model, the results can be retrieved from it by accessing the .results
            member, which can be used as the argument to this function. It's a list of infos gathered during
            the clustering step and each element in the list corresponds to a single topic.

        Returns
        -------
            dict of (int, list of {'is_core', 'amount_parent_labels', 'parent_labels'})
            A mapping of the label to a list of topics that belong to that particular label. Also adds
            a new member to each topic called amount_parent_labels, which is the number of parent_labels of that
            topic.
        """
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
        return grouped_by_labels

    def _aggregate_topics(self, grouped_by_labels):
        """Aggregates the topics to a list of clusters

        Parameters
        ----------
        grouped_by_labels : dict of (int, list of {'is_core', 'amount_parent_labels', 'parent_labels'})
            The return value of _group_by_labels. A mapping of the label to a list of each topic which belongs to the
            label.

        Returns
        -------
            list of {'amount_parent_labels', 'parent_labels', 'label'}
            amount_parent_labels is the max number of parent labels among each topic of a given cluster. label refers
            to the label identifier of the cluster. parent_labels is a concatenated list of the parent_labels sets of
            each topic. Its sorted by amount_parent_labels in descending order. There is one single element for each
            cluster.
        """
        sorted_clusters = []

        for label, group in grouped_by_labels.items():
            amount_parent_labels = 0
            parent_labels = []  # will be a list of sets

            for topic in group:
                amount_parent_labels = max(topic["amount_parent_labels"], amount_parent_labels)
                parent_labels.append(topic["parent_labels"])

            sorted_clusters.append({
                "amount_parent_labels": amount_parent_labels,
                "parent_labels": [x for x in parent_labels if len(x) > 0],
                "label": label
            })

        # start with the most significant core.
        sorted_clusters = sorted(sorted_clusters,
            key=lambda cluster: (
                cluster["amount_parent_labels"],
                cluster["label"]  # makes sorting deterministic
            ), reverse=True)

        return sorted_clusters

    def _remove_from_all_sets(self, label, clusters):
        """removes a label from every set in "parent_labels" for
        each core in clusters"""
        for a in clusters:
            if label in a["parent_labels"]:
                a["parent_labels"].remove(label)

    def _is_easy_valid_cluster(self, label, cluster, min_cores):
        """checks if the cluster has at least min_cores of cores with the only parent-label as itself,
        meaning no other cluster influenced those cores, hence the cluster in question is easy valid."""
        return sum(map(lambda x: x == {label}, cluster["parent_labels"])) >= min_cores

    def _generate_stable_topics(self, min_cores=None):
        """generates stable topics out of the clusters. This function is the last step that has to be done in the
        ensemble.

        Stable topics can be retrieved afterwards using get_topics().

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

        grouped_by_labels = self._group_by_labels(results)

        sorted_clusters = self._aggregate_topics(grouped_by_labels)

        # Iterate over the cluster labels, see which clusters/labels
        # are valid to trigger the creation of a stable topic
        for i, cluster in enumerate(sorted_clusters):
            # here, None means "not yet evaluated"
            cluster["is_valid"] = None

            label = cluster["label"]
            # label is iterating over 0, 1, 3, ...

            # 1. rule - remove if the cores in the cluster have no parents
            if cluster["amount_parent_labels"] == 0:
                cluster["is_valid"] = False
                self._remove_from_all_sets(label, sorted_clusters)

            # 2. rule - remove if it has less than min_cores as parents
            # (parents are always also cores)
            if len(cluster["parent_labels"]) < min_cores:
                cluster["is_valid"] = False
                self._remove_from_all_sets(label, sorted_clusters)

            # 3. checking for "easy_valid"s
            # checks if the cluster has at least min_cores of cores with the only label as itself
            if cluster["amount_parent_labels"] >= 1:
                if self._is_easy_valid_cluster(label, cluster, min_cores):
                    cluster["is_valid"] = True

        # Reapplying the rule 3
        # this happens when we have a close relationship among 2 or more clusters
        for cluster in [cluster for cluster in sorted_clusters if cluster["is_valid"] is None]:
            # This modifies parent labels
            # which is important in _is_easy_valid_cluster, so the result depends on where to start.
            # That's why sorted_clusters is sorted, so it starts with the most significant cluster.
            if self._is_easy_valid_cluster(label, cluster, min_cores):
                cluster["is_valid"] = True
            else:
                cluster["is_valid"] = False
                # removes a label from every set in parent_labels for each core.
                for a in sorted_clusters:
                    if label in a["parent_labels"]:
                        a["parent_labels"].remove(label)

        # list of all the label numbers that are valid
        valid_labels = np.array([cluster["label"] for cluster in sorted_clusters if cluster["is_valid"]])

        for topic in results:
            topic["valid_parents"] = {label for label in topic["parent_labels"] if label in valid_labels}

        # keeping only VALID cores
        valid_core_mask = np.vectorize(self._validate_core)(results)
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
        """Runs the CBDBSCAN algorithm on all the topics in the ensemble and uses resulting clusters to identify
        reliable topics.

        Stable topics can be retrieved using get_topics().

        Parameters
        ----------
        eps : float
            epsilon for the CBDBSCAN algorithm, having the same meaning as in classic DBSCAN clustering.
            default: 0.1
        min_samples : int
            The minimum number of samples in the neighborhood of a topic to be considered a core in CBDBSCAN.
            default: int(self.num_models / 2)
        min_cores : int
            how many cores a cluster has to have, to be treated as stable topic. That means, how many topics
            that look similar have to be present, so that the average topic in those is used as stable topic.
            default: min(3, max(1, int(self.num_models /4 +1)))

        """
        # if new models were added to the ensemble, the distance matrix needs to be generated again
        if self.asymmetric_distance_matrix_outdated:
            logger.info("asymmetric distance matrix is outdated due to add_model")
            self._generate_asymmetric_distance_matrix()

        # Run CBDBSCAN to get topic clusters:
        self._generate_topic_clusters(eps, min_samples)

        # Interpret the results of CBDBSCAN to identify reliable topics:
        self._generate_stable_topics(min_cores)

        # Create gensim LdaModel representation of topic model with reliable topics (can be used for inference):
        self.generate_gensim_representation()

    # GENSIM API
    # to make using the ensemble in place of a gensim model as easy as possible

    def get_topics(self):
        return self.stable_topics

    def _has_gensim_representation(self):
        """checks if stable topics area vailable and if the internal
        gensim representation exists. If not, raises errors"""
        if self.classic_model_representation is None:
            if len(self.stable_topics) == 0:
                raise ValueError("no stable topic was detected")
            else:
                raise ValueError("use generate_gensim_representation() first")

    def __getitem__(self, i):
        """see :py:class:`gensim.models.LdaModel`"""
        self._has_gensim_representation()
        return self.classic_model_representation[i]

    def inference(self, *posargs, **kwargs):
        """see :py:class:`gensim.models.LdaModel`"""
        self._has_gensim_representation()
        return self.classic_model_representation.inference(*posargs, **kwargs)

    def log_perplexity(self, *posargs, **kwargs):
        """see :py:class:`gensim.models.LdaModel`"""
        self._has_gensim_representation()
        return self.classic_model_representation.log_perplexity(*posargs, **kwargs)

    def print_topics(self, *posargs, **kwargs):
        """see :py:class:`gensim.models.LdaModel`"""
        self._has_gensim_representation()
        return self.classic_model_representation.print_topics(*posargs, **kwargs)

    @property
    def id2word(self):
        return self.gensim_kw_args["id2word"]


class CBDBSCAN():
    """A Variation of the DBSCAN algorithm called Checkback DBSCAN (CBDBSCAN).  The algorithm works based on
    DBSCAN-like parameters `eps` and `min_samples` that respectively define how far a "nearby" point is, and the
    minimum number of nearby points needed to label a candidate datapoint a core of a cluster.
    (See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).

    The algorithm works as follows:
    1. (A)symmetric distance matrix provided at fit-time.
    For the sake of example below, assume the there are only three topics (amatrix contains distances with dim 3x3),
        T_1, T_2, and T_3:
    2. Start by scanning a candidate topic
        (e.g. T_1)
    3. Check which topics are nearby using `self.eps` as a threshold and call them neighbours
        (e.g. assume T_2, and T_3 are nearby and become neighbours)
    4. If there are more neighbours than `self.min_samples`, the candidate topic becomes a core candidate for a cluster
        (e.g. if `min_samples`<=2 then T_1 becomes the first core of a cluster)
    5. CheckBack (CB) Step: Check if at least 25% of other cluster cores are nearby to core candidate
        (e.g. at this stage in the example, the cluster is empty, so checkback passes)
    6. Recursively scan the next nearby topic (e.g. scan T_2) - repeat step 2.-6.

    The CB step has the effect that it enforces cluster compactness and allows the model to avoid creating clusters for
    unreliable topics made of a composition of multiple reliable topics (something that occurs often LDA models that is
    one cause of unreliable topics).

    Parameters
    ----------
    eps : float
        epsilon for the CBDBSCAN algorithm, having the same meaning as in classic DBSCAN clustering.
        default: 0.1
    min_samples : int
        The minimum number of samples in the neighborhood of a topic to be considered a core in CBDBSCAN.
    """

    def __init__(self, eps, min_samples):
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

        amatrix_copy = amatrix.copy()

        # to avoid problem about comparing the topic with itself
        np.fill_diagonal(amatrix_copy, 1)

        min_distance_per_topic = [(distance, index) for index, distance in enumerate(amatrix_copy.min(axis=1))]
        min_distance_per_topic_sorted = sorted(min_distance_per_topic, key=lambda x: x)
        ordered_min_similarity = [index for distance, index in min_distance_per_topic_sorted]

        def scan_topic(topic_index, parent_id=None, current_label=None, parent_neighbors=None):
            # count how many neighbor_indices
            # check which indices is closer than eps
            neighbouring_topic_indices = np.array(
                [
                    candidate_topic_index
                    for candidate_topic_index, is_neighbour in enumerate(amatrix_copy[topic_index] < self.eps)
                    if is_neighbour
                ]
            )
            num_neighbouring_topics = len(neighbouring_topic_indices)

            # If the number of neighbor_indices of a topic is large enough,
            # it is considered a core.
            # This also takes neighbor_indices that already are identified as core in count.
            if num_neighbouring_topics >= self.min_samples:
                # This topic is a core!
                results[topic_index]["is_core"] = True
                results[topic_index]["num_samples"] = num_neighbouring_topics

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

                    # parent neighbor_indices is the list of neighbor_indices of parent_id
                    # (the topic_index that called this function recursively)
                    all_members_of_current_cluster = list(parent_neighbors)
                    all_members_of_current_cluster.append(parent_id)

                    # look if 25% of the members of the current cluster are also
                    # close to the current/new topic...

                    # example: (topic_index, 0), (topic_index, 2), (topic_index, ...)
                    all_members_of_current_cluster_ix = np.ix_([topic_index], all_members_of_current_cluster)
                    # use the result of the previous step to index the matrix and see if those distances
                    # are smaller then epsilon. relations_to_the_cluster is a boolean array, True for close elements
                    relations_to_the_cluster = amatrix_copy[all_members_of_current_cluster_ix] < self.eps

                    # if less than 25% of the elements are close, then the topic index in question is not a
                    # core of the current_label, but rather the core of a new cluster
                    if relations_to_the_cluster[0].mean() < 0.25:
                        # start new cluster by changing current_label
                        current_label = self.next_label
                        self.next_label += 1

                results[topic_index]["label"] = current_label

                for neighbouring_topic_index in neighbouring_topic_indices:

                    if results[neighbouring_topic_index]["label"] is None:
                        ordered_min_similarity.remove(neighbouring_topic_index)
                        # try to extend the cluster into the direction
                        # of the neighbor
                        scan_topic(
                            neighbouring_topic_index, parent_id=topic_index,
                            current_label=current_label,
                            parent_neighbors=neighbouring_topic_indices
                        )

                    results[neighbouring_topic_index]["parent_ids"].add(topic_index)
                    results[neighbouring_topic_index]["parent_labels"].add(current_label)

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
