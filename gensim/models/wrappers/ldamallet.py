#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Python wrapper for Latent Dirichlet Allocation (LDA) from MALLET, the Java topic modelling
toolkit [1]_.

This module allows both LDA model estimation from a training corpus and inference of topic
distribution on new, unseen documents, using an (optimized version of) collapsed
gibbs sampling from MALLET.

MALLET's LDA training requires O(#corpus_words) of memory, keeping the entire corpus in RAM.
If you find yourself running out of memory, either decrease the `workers` constructor
parameter, or use `LdaModel` which needs only O(1) memory.

The wrapped model can NOT be updated with new documents for online training -- use gensim's `LdaModel` for that.

Example:

>>> model = gensim.models.wrappers.LdaMallet('/Users/kofola/mallet-2.0.7/bin/mallet', corpus=my_corpus, num_topics=20, id2word=dictionary)
>>> print model[my_vector]  # print LDA topics of a document

.. [1] http://mallet.cs.umass.edu/

"""


import logging
import random
import tempfile
import os

import numpy

import xml.etree.ElementTree as et
import zipfile

from six import iteritems
from smart_open import smart_open

from gensim import utils, matutils
from gensim.utils import check_output
from gensim.models.ldamodel import LdaModel

logger = logging.getLogger(__name__)


class LdaMallet(utils.SaveLoad):
    """
    Class for LDA training using MALLET. Communication between MALLET and Python
    takes place by passing around data files on disk and calling Java with subprocess.call().

    """
    def __init__(self, mallet_path, corpus=None, num_topics=100, alpha=50, id2word=None, workers=4, prefix=None,
                 optimize_interval=0, iterations=1000, topic_threshold=0.0):
        """
        `mallet_path` is path to the mallet executable, e.g. `/home/kofola/mallet-2.0.7/bin/mallet`.

        `corpus` is a gensim corpus, aka a stream of sparse document vectors.

        `id2word` is a mapping between tokens ids and token.

        `workers` is the number of threads, for parallel training.

        `prefix` is the string prefix under which all data files will be stored; default: system temp + random filename prefix.

        `optimize_interval` optimize hyperparameters every N iterations (sometimes leads to Java exception; 0 to switch off hyperparameter optimization).

        `iterations` is the number of sampling iterations.

        `topic_threshold` is the threshold of the probability above which we consider a topic. This is basically for sparse topic distribution.

        """
        self.mallet_path = mallet_path
        self.id2word = id2word
        if self.id2word is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        else:
            self.num_terms = 0 if not self.id2word else 1 + max(self.id2word.keys())
        if self.num_terms == 0:
            raise ValueError("cannot compute LDA over an empty collection (no terms)")
        self.num_topics = num_topics
        self.topic_threshold=topic_threshold
        self.alpha = alpha
        if prefix is None:
            rand_prefix = hex(random.randint(0, 0xffffff))[2:] + '_'
            prefix = os.path.join(tempfile.gettempdir(), rand_prefix)
        self.prefix = prefix
        self.workers = workers
        self.optimize_interval = optimize_interval
        self.iterations = iterations
        if corpus is not None:
            self.train(corpus)

    def finferencer(self):
        return self.prefix + 'inferencer.mallet'

    def ftopickeys(self):
        return self.prefix + 'topickeys.txt'

    def fstate(self):
        return self.prefix + 'state.mallet.gz'

    def fdoctopics(self):
        return self.prefix + 'doctopics.txt'

    def fcorpustxt(self):
        return self.prefix + 'corpus.txt'

    def fcorpusmallet(self):
        return self.prefix + 'corpus.mallet'

    def fwordweights(self):
        return self.prefix + 'wordweights.txt'

    def corpus2mallet(self, corpus, file_like):
        """
        Write out `corpus` in a file format that MALLET understands: one document per line:

          document id[SPACE]label (not used)[SPACE]whitespace delimited utf8-encoded tokens[NEWLINE]
        """
        for docno, doc in enumerate(corpus):
            if self.id2word:
                tokens = sum(([self.id2word[tokenid]] * int(cnt) for tokenid, cnt in doc), [])
            else:
                tokens = sum(([str(tokenid)] * int(cnt) for tokenid, cnt in doc), [])
            file_like.write(utils.to_utf8("%s 0 %s\n" % (docno, ' '.join(tokens))))

    def convert_input(self, corpus, infer=False, serialize_corpus=True):
        """
        Serialize documents (lists of unicode tokens) to a temporary text file,
        then convert that text file to MALLET format `outfile`.

        """
        if serialize_corpus:
            logger.info("serializing temporary corpus to %s", self.fcorpustxt())
            with smart_open(self.fcorpustxt(), 'wb') as fout:
                self.corpus2mallet(corpus, fout)

        # convert the text file above into MALLET's internal format
        cmd = self.mallet_path + " import-file --preserve-case --keep-sequence --remove-stopwords --token-regex '\S+' --input %s --output %s"
        if infer:
            cmd += ' --use-pipe-from ' + self.fcorpusmallet()
            cmd = cmd % (self.fcorpustxt(), self.fcorpusmallet() + '.infer')
        else:
            cmd = cmd % (self.fcorpustxt(), self.fcorpusmallet())
        logger.info("converting temporary corpus to MALLET format with %s", cmd)
        check_output(cmd, shell=True)

    def train(self, corpus):
        self.convert_input(corpus, infer=False)
        cmd = self.mallet_path + " train-topics --input %s --num-topics %s  --alpha %s --optimize-interval %s "\
            "--num-threads %s --output-state %s --output-doc-topics %s --output-topic-keys %s "\
            "--num-iterations %s --inferencer-filename %s --doc-topics-threshold %s"
        cmd = cmd % (
            self.fcorpusmallet(), self.num_topics, self.alpha, self.optimize_interval, self.workers,
            self.fstate(), self.fdoctopics(), self.ftopickeys(), self.iterations, self.finferencer(), self.topic_threshold)
        # NOTE "--keep-sequence-bigrams" / "--use-ngrams true" poorer results + runs out of memory
        logger.info("training MALLET LDA with %s", cmd)
        check_output(cmd, shell=True)
        self.word_topics = self.load_word_topics()
        # NOTE - we are still keeping the wordtopics variable to not break backward compatibility. 
        # word_topics has replaced wordtopics throughout the code; wordtopics just stores the values of word_topics when train is called.
        self.wordtopics = self.word_topics

    def __getitem__(self, bow, iterations=100):
        is_corpus, corpus = utils.is_corpus(bow)
        if not is_corpus:
            # query is a single document => make a corpus out of it
            bow = [bow]

        self.convert_input(bow, infer=True)
        cmd = self.mallet_path + " infer-topics --input %s --inferencer %s --output-doc-topics %s --num-iterations %s --doc-topics-threshold %s"
        cmd = cmd % (self.fcorpusmallet() + '.infer', self.finferencer(), self.fdoctopics() + '.infer', iterations, self.topic_threshold)
        logger.info("inferring topics with MALLET LDA '%s'", cmd)
        check_output(cmd, shell=True)
        result = list(self.read_doctopics(self.fdoctopics() + '.infer'))
        return result if is_corpus else result[0]

    def load_word_topics(self):
        logger.info("loading assigned topics from %s", self.fstate())
        word_topics = numpy.zeros((self.num_topics, self.num_terms), dtype=numpy.float32)
        if hasattr(self.id2word, 'token2id'):
            word2id = self.id2word.token2id
        else:
            word2id = dict((v, k) for k, v in iteritems(self.id2word))

        with utils.smart_open(self.fstate()) as fin:
            _ = next(fin)  # header
            self.alpha = numpy.array([float(val) for val in next(fin).split()[2:]])
            assert len(self.alpha) == self.num_topics, "mismatch between MALLET vs. requested topics"
            _ = next(fin)  # beta
            for lineno, line in enumerate(fin):
                line = utils.to_unicode(line)
                doc, source, pos, typeindex, token, topic = line.split(" ")
                if token not in word2id:
                    continue
                tokenid = word2id[token]
                word_topics[int(topic), tokenid] += 1.0
        return word_topics

    def print_topics(self, num_topics=10, num_words=10):
        return self.show_topics(num_topics, num_words, log=True)

    def load_document_topics(self):
        """
        Return an iterator over the topic distribution of training corpus, by reading
        the doctopics.txt generated during training.
        """
        return self.read_doctopics(self.fdoctopics())

    def show_topics(self, num_topics=10, num_words=10, log=False, formatted=True):
        """
        Print the `num_words` most probable words for `num_topics` number of topics.
        Set `num_topics=-1` to print all topics.

        Set `formatted=True` to return the topics as a list of strings, or `False` as lists of (weight, word) pairs.

        """
        if num_topics < 0 or num_topics >= self.num_topics:
            num_topics = self.num_topics
            chosen_topics = range(num_topics)
        else:
            num_topics = min(num_topics, self.num_topics)
            sort_alpha = self.alpha + 0.0001 * numpy.random.rand(len(self.alpha)) # add a little random jitter, to randomize results around the same alpha
            sorted_topics = list(matutils.argsort(sort_alpha))
            chosen_topics = sorted_topics[ : num_topics//2] + sorted_topics[-num_topics//2 : ]
        shown = []
        for i in chosen_topics:
            if formatted:
                topic = self.print_topic(i, num_words=num_words)
            else:
                topic = self.show_topic(i, num_words=num_words)
            shown.append(topic)
            if log:
                logger.info("topic #%i (%.3f): %s", i, self.alpha[i], topic)
        return shown

    def show_topic(self, topicid, num_words=10):
        if self.word_topics is None:
            logger.warn("Run train or load_word_topics before showing topics.")
        topic = self.word_topics[topicid]
        topic = topic / topic.sum()  # normalize to probability dist
        bestn = matutils.argsort(topic, num_words, reverse=True)
        beststr = [(topic[id], self.id2word[id]) for id in bestn]
        return beststr

    def print_topic(self, topicid, num_words=10):
        return ' + '.join(['%.3f*%s' % v for v in self.show_topic(topicid, num_words)])


    def get_version(self, direc_path):
        """"

        function to return the version of `mallet`

        """
        try:
            """
            Check version of mallet via jar file
            """
            archive = zipfile.ZipFile(direc_path, 'r')
            if u'cc/mallet/regression/' not in archive.namelist():     
                return '2.0.7'
            else:
                return '2.0.8RC3'
        except Exception:
            
            xml_path = direc_path.split("bin")[0]
            try:
                doc = et.parse(xml_path + "pom.xml").getroot()
                namespace = doc.tag[:doc.tag.index('}') + 1]
                return doc.find(namespace + 'version').text.split("-")[0]
            except Exception:
                return "Can't parse pom.xml version file"
        


    def read_doctopics(self, fname, eps=1e-6, renorm=True):
        """
        Yield document topic vectors from MALLET's "doc-topics" format, as sparse gensim vectors.

        """
        mallet_version = self.get_version(self.mallet_path)
        with utils.smart_open(fname) as fin:
            for lineno, line in enumerate(fin):
                if lineno == 0 and line.startswith(b"#doc "):
                    continue  # skip the header line if it exists

                parts = line.split()[2:]  # skip "doc" and "source" columns

                # the MALLET doctopic format changed in 2.0.8 to exclude the id,
                # this handles the file differently dependent on the pattern
                if len(parts) == 2 * self.num_topics:
                    doc = [(id_, weight)
                           for id_, weight in zip(map(int, parts[::2]),
                                                  map(float, parts[1::2]))
                           if abs(weight) > eps]
                elif len(parts) == self.num_topics and mallet_version != '2.0.7':
                    doc = [(id_, weight)
                           for id_, weight in enumerate(map(float, parts))
                           if abs(weight) > eps]
                else:
                    if mallet_version == "2.0.7":
                        """

                            1   1   0   1.0780612802674239  30.005575655428533364   2   0.005575655428533364    1   0.005575655428533364    
                            2   2   0   0.9184413079632608  40.009062076892971008   3   0.009062076892971008    2   0.009062076892971008    1   0.009062076892971008
                            In the above example there is a mix of the above if and elif statement. There are neither `2*num_topics` nor `num_topics` elements.
                            It has 2 formats 40.009062076892971008 and 0   1.0780612802674239 which cannot be handled by above if elif.
                            Also, there are some topics are missing(meaning that the topic is not there) which is another reason why the above if elif
                            fails even when the `mallet` produces the right results

                        """
                        count = 0
                        doc = []
                        if len(parts) > 0:
                            while count < len(parts):
                                """ 
                                if section is to deal with formats of type 2 0.034
                                so if count reaches index of 2 and since int(2) == float(2) so if block is executed
                                now  there is one extra element afer 2, so count + 1 access should not give an error

                                else section handles  formats of type 20.034
                                now count is there on index of 20.034 since float(20.034) != int(20.034) so else block
                                is executed 

                                """
                                if float(parts[count]) == int(parts[count]):
                                    if float(parts[count + 1]) > eps:
                                        doc.append((int(parts[count]), float(parts[count + 1])))
                                    count += 2
                                else:
                                    if float(parts[count]) - int(parts[count]) > eps:
                                        doc.append((int(parts[count]) % 10, float(parts[count]) - int(parts[count])))
                                    count += 1
                    else:
                        raise RuntimeError("invalid doc topics format at line %i in %s" % (lineno + 1, fname))

                if renorm:
                    # explicitly normalize weights to sum up to 1.0, just to be sure...
                    total_weight = float(sum([weight for _, weight in doc]))
                    if total_weight:
                        doc = [(id_, float(weight) / total_weight) for id_, weight in doc]
                yield doc


def malletmodel2ldamodel(mallet_model, gamma_threshold=0.001, iterations=50):
    """
    Function to convert mallet model to gensim LdaModel. This works by copying the
    training model weights (alpha, beta...) from a trained mallet model into the
    gensim model.

    Args:
    ----
    mallet_model : Trained mallet model
    gamma_threshold : To be used for inference in the new LdaModel.
    iterations : number of iterations to be used for inference in the new LdaModel.

    Returns:
    -------
    model_gensim : LdaModel instance; copied gensim LdaModel
    """
    model_gensim = LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha, iterations=iterations,
        gamma_threshold=gamma_threshold)
    model_gensim.expElogbeta[:] = mallet_model.wordtopics
    return model_gensim
