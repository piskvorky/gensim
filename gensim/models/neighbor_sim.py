#!/usr/bin/env python
# -*- coding: utf-8 -*-
from doc2vec import Doc2Vec
from word2vec import Word2Vec

from annoy import AnnoyIndex

"""
Using 3rd party k-NN libraries to improve performance for method DocvecsArray.most_similar().

More information can be found at: https://github.com/piskvorky/gensim/wiki/Ideas-&-Feature-proposals#integrate-a-fast-k-nn-library

Implemented by: Anh Le (anhldbk@gmail.com)

Website: http://bigsonata.com

Release 1: November 2015
"""


class NeighborIndexer(object):
    """
    Base class for k-NN libraries
    """

    def __init__(self, model=None, **kwargs):
        """
        Create a new indexer
        :param model:   Instance of Doc2Vec or Word2Vec
        :param kwargs:  Additional named parameters
        """
        if model is None:
            raise Exception("Invalid model parameter. Please provide model with an instance of Doc2Vec or Word2Vec")
        if type(model) is Doc2Vec:
            self._init_doc2vec_(model)
        elif type(model) is Word2Vec:
            self._init_word2vec_(model)
        else:
            raise Exception("Invalid model parameter. Please provide model with an instance of Doc2Vec or Word2Vec")
        self.start_indexing()

    def _init_doc2vec_(self, model):
        docvecs = model.docvecs
        docvecs.init_sims()
        size = len(docvecs.doctag_syn0norm)

        for i in range(size):
            self.add_item(docvecs.offset2doctag[i], docvecs.doctag_syn0norm[i])

    def _init_word2vec_(self, model):
        # raise Exception("Not supported at the moment")
        model.init_sims()
        size = len(model.syn0norm)

        for i in range(size):
            self.add_item(model.index2word[i], model.syn0norm[i])

    def get_item(self, label):
        """
        Get an item's vector by its label
        :param label: The label
        :return: The item's vector if have. Otherwise, returns None
        """
        pass

    def add_item(self, label, vector):
        """
        Add an item to index
        :param label: the label of the item (must be unique).
        :param vector: the item's vector.
        """
        pass

    def start_indexing(self):
        """
        Start the indexing operation. This method is supposed to run after all items are added
        It may take time to finish.
        """
        pass

    def get_nearest_items(self, vector, top_n=10):
        """
        Get nearest items for an item described by its vector
        :param	vector: The vector
        :param	top_n: Number of nearest items to get
        :return	A list of tuples (label, similarity) for the nearest ones
        """
        pass

    def save(self, file_name):
        """
        Dump internal data into disk
        :param file_name:   Output file path
        :return: Returns true on success. Otherwise, returns false
        """
        pass

    def load(self, file_name):
        """
        Load previously dumped info
        :param file_name:  Input file paht
        :return: Returns true on success. Otherwise, returns false
        """
        pass


class AnnoyIndexer(NeighborIndexer):
    """
    An Annoy-based indexer
    """

    def __init__(self, model=None, **kwargs):
        """
        Constructor
        :param features_num: Number of features
        """
        if "features_num" not in kwargs:
            raise Exception("Not enough parameter")
        self.features_num = kwargs["features_num"]

        if "tree_size" not in kwargs:
            raise Exception("Not enough parameter")
        self.tree_size = kwargs["tree_size"]

        self.annoy_indexer = AnnoyIndex(self.features_num)  # Length of item vector that will be indexed
        self.index_label_dict = {}
        self.label_index_dict = {}
        self.id = 0

        # call parent constructor
        super(AnnoyIndexer, self).__init__(model, **kwargs)

    def add_item(self, label, vector):
        """
        Add an item to index
        :param label: is the label of the item.
        :param vector: is the item's vector.
        """
        self.index_label_dict[self.id] = label
        self.label_index_dict[label] = self.id
        self.annoy_indexer.add_item(self.id, vector)
        self.id += 1

    def get_item(self, label):
        """
        Get an item's vector by its label
        :param label: The label
        :return: The item's vector if have. Otherwise, returns None
        """
        if label not in self.label_index_dict:
            return None
        return self.annoy_indexer.get_item_vector(self.label_index_dict[label])

    def start_indexing(self):
        """
        Start the indexing operation. This method is supposed to run after all items are added
        It may take time to finish
        """
        self.annoy_indexer.build(self.tree_size)

    def get_nearest_items(self, vector, top_n=10):
        """
        Get nearest items for an item described by its vector
        :param	vector: The vector
        :param	top_n: Number of nearest items to get
        :return	A list of tuples (label, similarity) for the nearest ones
        """
        nearest = self.annoy_indexer.get_nns_by_vector(vector, top_n, include_distances=True)
        size = len(nearest[0])

        # By default, Annoy uses Cosine Distance
        # Reference: http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.spatial.distance.cosine.html
        # To get the similarity, we use the formula: similarity = 1 - cosine_distance/2
        result = []
        for index in range(size):
            label = self.index_label_dict[nearest[0][index]]
            similarity = 1 - nearest[1][index] / 2
            result += [(label, similarity)]

        return result

    def save(self, file_name):
        """
        Dump internal data into disk
        :param file_name:   Output file path
        :return: Returns true on success. Otherwise, returns false
        """
        try:
            self.annoy_indexer.save(file_name)
            return True
        except Exception as e:
            return False

    def load(self, file_name):
        """
        Load previously dumped info
        :param file_name:  Input file paht
        :return: Returns true on success. Otherwise, returns false
        """
        try:
            self.annoy_indexer.load(file_name)
            return True
        except Exception as e:
            return False