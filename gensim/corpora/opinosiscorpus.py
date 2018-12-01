#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Tobias B <github.com/sezanzeb>

import os
from pathlib import Path
import re
from gensim.corpora import Dictionary
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


class OpinosisCorpus():
    """Creates a corpus and dictionary from the opinosis dataset:

    http://kavita-ganesan.com/opinosis-opinion-dataset/#.WyF_JNWxW00

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
        print("data source:")
        print("title:\t\tOpinosis: a graph-based approach to abstractive summarization of highly redundant opinions")
        print("authors:\tGanesan, Kavita and Zhai, ChengXiang and Han, Jiawei")
        print("booktitle:\tProceedings of the 23rd International Conference on Computational Linguistics")
        print("pages:\t\t340-348")
        print("year:\t\t2010")
        print("organization:\tAssociation for Computational Linguistics")

        path = Path(path).joinpath("summaries-gold")
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
                        stemmer.stem(token) for token in re.findall('\w+', doc.lower())
                        if token not in stopwords.words('english')
                    ]

                    dictionary.add_documents([processed])
                    corpus += [dictionary.doc2bow(processed)]

        # and return the results the same way the other corpus generating functions do
        self.corpus = corpus
        self.id2word = dictionary
