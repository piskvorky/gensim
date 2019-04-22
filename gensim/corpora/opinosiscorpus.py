#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Tobias B <github.com/sezanzeb>

# Opinosis Corpus Source:
# title:         Opinosis: a graph-based approach to abstractive summarization of highly redundant opinions
# authors:       Ganesan, Kavita and Zhai, ChengXiang and Han, Jiawei
# booktitle:     Proceedings of the 23rd International Conference on Computational Linguistics
# pages:         340-348
# year:          2010
# organization:  Association for Computational Linguistics
# http://kavita-ganesan.com/opinosis-opinion-dataset/

import os
import re
from gensim.corpora import Dictionary
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import STOPWORDS


class OpinosisCorpus():
    """Creates a corpus and dictionary from the opinosis dataset:

    http://kavita-ganesan.com/opinosis-opinion-dataset/

    This data is organized in folders, each folder containing a few short docs.

    Data can be obtained quickly using the following commands in bash:

        mkdir opinosis && cd opinosis
        wget https://github.com/kavgan/opinosis/raw/master/OpinosisDataset1.0_0.zip
        unzip OpinosisDataset1.0_0.zip

    corpus and dictionary can be accessed by using the .corpus and .id2word members

    """

    def __init__(self, path):
        """

        Parameters
        ----------
        path : string
            Path to the extracted zip file. If 'summaries-gold' is in a folder
            called 'opinosis', then the Path parameter would be 'opinosis',
            either relative to you current working directory or absolute.

        """

        # citation
        path = os.path.join(path, "summaries-gold")
        dictionary = Dictionary()
        corpus = []
        stemmer = PorterStemmer()

        for directory, b, filenames in os.walk(path):
            # iterates over folders, so in this
            # scope a new topic is prepared

            # root directory?
            if len(filenames) == 0:
                continue

            # folder = directory.split(os.sep)[-1]
            # print("processing folder ", "'"+folder+"'", "...")

            # now get the corpus/documents
            for filename in filenames:

                filepath = directory + os.sep + filename
                # write down the document and the topicId and split into train and testdata
                with open(filepath) as file:

                    doc = file.read()
                    # overwrite dictionary, add corpus to test or train afterwards
                    # the following function takes an array of documents, so wrap it in square braces
                    processed = [
                        stemmer.stem(token) for token in re.findall(r'\w+', doc.lower())
                        if token not in STOPWORDS
                    ]

                    dictionary.add_documents([processed])
                    corpus += [dictionary.doc2bow(processed)]

        # and return the results the same way the other corpus generating functions do
        self.corpus = corpus
        self.id2word = dictionary
