#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Tobias B <proxima@sezanzeb.de>
# Copyright (C) 2021 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""Creates a corpus and dictionary from the Opinosis dataset.

References
----------
.. [1] Ganesan, Kavita and Zhai, ChengXiang and Han, Jiawei. Opinosis: a graph-based approach to abstractive
       summarization of highly redundant opinions [online]. In : Proceedings of the 23rd International Conference on
       Computational Linguistics. 2010. p. 340-348. Available from: https://kavita-ganesan.com/opinosis/
"""

import os
import re
from gensim.corpora import Dictionary
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import STOPWORDS


class OpinosisCorpus:
    """Creates a corpus and dictionary from the Opinosis dataset.

    http://kavita-ganesan.com/opinosis-opinion-dataset/

    This data is organized in folders, each folder containing a few short docs.

    Data can be obtained quickly using the following commands in bash:

        mkdir opinosis && cd opinosis
        wget https://github.com/kavgan/opinosis/raw/master/OpinosisDataset1.0_0.zip
        unzip OpinosisDataset1.0_0.zip

    corpus and dictionary can be accessed by using the .corpus and .id2word members

    """

    def __init__(self, path):
        """Load the downloaded corpus.

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
            # each subdirectory of path is one collection of reviews to a specific product
            # now get the corpus/documents
            for filename in filenames:
                filepath = directory + os.sep + filename
                # write down the document and the topicId and split into train and testdata
                with open(filepath) as file:
                    doc = file.read()

                preprocessed_doc = [
                    stemmer.stem(token) for token in re.findall(r'\w+', doc.lower())
                    if token not in STOPWORDS
                ]

                dictionary.add_documents([preprocessed_doc])
                corpus += [dictionary.doc2bow(preprocessed_doc)]

        # and return the results the same way the other corpus generating functions do
        self.corpus = corpus
        self.id2word = dictionary
