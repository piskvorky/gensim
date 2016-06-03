#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from annoy import AnnoyIndex


class SimilarityIndex(object):

    @classmethod
    def build_from_word2vec(cls, model, num_trees):
        """Build an Annoy index using word vectors from a Word2Vec model"""

        model.init_sims()
        return cls._build_from_model(model.syn0norm, model.index2word, model.vector_size, num_trees)

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
