#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

try:
    from annoy import AnnoyIndex
except ImportError:
    raise ImportError("Annoy has not been installed, if you wish to use the annoy indexer, please run `pip install annoy`")


class SimilarityIndex(object):

    @classmethod
    def build_from_word2vec(cls, model, num_trees):
        """Build an Annoy index using word vectors from a Word2Vec model"""

        model.init_sims()
        return cls._build_from_model(model.syn0norm, model.index2word, model.vector_size, num_trees)

    @classmethod
    def build_from_doc2vec(cls, model, num_trees):
        """Build an Annoy index using document vectors from a Doc2Vec model"""

        docvecs = model.docvecs
        docvecs.init_sims()
        labels = [docvecs.index_to_doctag(i) for i in range(0, docvecs.count)]
        return cls._build_from_model(docvecs.doctag_syn0norm, labels, model.vector_size, num_trees)

    @classmethod
    def _build_from_model(cls, vectors, labels, num_features, num_trees):
        index = AnnoyIndex(num_features)

        for i in range(len(vectors)):
            vector = vectors[i]
            index.add_item(i, vector)

        index.build(num_trees)
        return SimilarityIndex(index, labels)

    def __init__(self, index, labels):
        self.index = index
        self.labels = labels

    def most_similar(self, vector, num_neighbors):
        """Find the top-N most similar items"""

        ids, distances = self.index.get_nns_by_vector(
            vector, num_neighbors, include_distances=True)

        result = []
        for i in range(len(ids)):
            label = self.labels[ids[i]]
            similarity = 1 - distances[i] / 2
            result += [(label, similarity)]

        return result
