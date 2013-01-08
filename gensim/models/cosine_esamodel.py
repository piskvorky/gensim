#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 23.11.2012

@author: karsten

Modified ESA model to use consine similarity instead of sums

For details on the ESA model see: 
"Computing semantic relatedness using wikipedia-based explicit semantic analysis"
by Gabrilovich, E. and Markovitch, S. 
in "Proceedings of the 20th international joint conference on artificial intelligence"
'''

from collections import defaultdict
from gensim.similarities import Similarity, MatrixSimilarity
from itertools import izip
import logging
import itertools
from kmedoids import KMedoids 
import math
import numpy as np
import scipy
from selectkbest import iSelectKBest, if_classif

from gensim import interfaces, matutils, utils, similarities


logger = logging.getLogger('gensim.models.esamodel')

class DocumentTitles(object):
    '''
    Loads a list of document titles form a text file.
    Each line is considered to be a title.
    '''
    
    def __init__(self):
        self.document_titles = []
    
    @classmethod
    def load(cls, file_path):
        logger.info("Loading concept titles from %s" % file_path)
        
        result = DocumentTitles()
        with open(file_path, "r") as file:
            for line in file:
                doc_title = line.strip("\n").decode("UTF-8")
                result.document_titles.append(doc_title)
        
        logger.info("Loaded %d concept titles." % len(result.document_titles))
               
        return result
    
    def append(self, value):
        self.document_titles.append(value)
    
    def __iter__(self):
        for title in self.document_titles: yield title
        
    def __getitem__(self, key):
        return self.document_titles[key]
    
    def __len__(self):
        return len(self.document_titles)

class CosineEsaModel(interfaces.TransformationABC):
    """
    The cosine ESA, cESA, model is a modified version of the Explicit Semantic Analysis
    model.
    
    ESA uses the product of a document and a concept to calculate their similarity.
    The cESA model uses the cosine similarity.
    
    Originally ESA uses all Wikipedia concepts which have a certain amount of
    incoming inter-Wikipedia links. Denpending on the settings one is still left
    with well over 1 million concepts. To reduce this number cESA uses feature
    selection.
    """
    def __init__(self, corpus, document_titles, 
                 test_corpus, test_corpus_targets, num_test_corpus,
                 num_best_features = 1000,
                 num_features = None,
                 tmp_path = 'complete_similarity'):
        """
        The similarity between a document and each document of the corpus is
        the feature created.
        
        The corpus is filtered for significant features.
        
        Parameters
        ----------
        test_corpus : The test corpus is used to select features. 
                      All documents in this corous should be classified.
        test_corpus_targets : The target classes of each document in the 
                              test corpus.
        num_test_corpus : Number of documents in the test corpus.
        document_titles : give the names of each concept (doc) in corpus.
        num_features : gives the number of features of corpus
        """
        
        if num_features is None:
            logger.info("scanning corpus to determine the number of features")
            num_features = 1 + utils.get_max_id(corpus)
            
        self.num_features = num_features
        
        #create similarity index of complete corpus
        complete_similarity_index = Similarity(output_prefix = tmp_path,
                                                    corpus = corpus,
                                                    num_features = self.num_features)
        
        #reduce concept count by feature selection
        self.selector = iSelectKBest(if_classif, k = num_best_features)
        
        #transform each document of test_corpus
        logger.info("Test corpus of %d documents..." % num_test_corpus)
            
        transformed_test_corpus = (complete_similarity_index[doc]
                                   for doc 
                                   in test_corpus)

        logger.info("Select best features...")
        X_y = izip(transformed_test_corpus, test_corpus_targets)
        self.selector.fit(X_y, len(document_titles))
        
        logger.info("Done selecting.")
        
        #reduce similarity index
        selected_documents = [doc 
                              for doc, mask 
                              in itertools.izip(corpus, self.selector.get_support())
                              if mask]
        self.similarity_index = MatrixSimilarity(corpus = selected_documents,
                                                 num_features = self.num_features)
        
        #reduce document titles
        self.document_titles = DocumentTitles()
        for doc_title, mask in itertools.izip(document_titles, self.selector.get_support()):
            if mask:
                self.document_titles.append(doc_title)
  
        #print doc titles
        for title in self.document_titles:
            logger.debug("%s" % title)
            

    def __str__(self):
        return " \n".join(self.document_titles)
    
    def get_concept_titles(self, doc_vec):
        '''
        Converts ids from document vector to concept titles.
        '''
        return [(self.document_titles[concept_id], weight) 
                for concept_id, weight in doc_vec]

    def __getitem__(self, bow, eps=1e-12):
        """
        Return esa representation of the input vector and/or corpus.
        
        bow should already be weights, e.g. with TF-IDF
        """
        # if the input vector is in fact a corpus, return a transformed corpus 
        # as a result
        is_corpus, bow = utils.is_corpus(bow)
        if is_corpus:
            return self._apply(bow)

        #use similarity index to calculate similarity with each vector of corpus
        vector = self.similarity_index[bow]
        
        #consine similarity is in [-1, 1] shift and scale to make it [0, 1]
        vector += 1
        vector /= 2

        #normalize
        vector = matutils.unitvec(vector)

        # make sure there are no explicit zeroes in the vector (must be sparse)
        vector = [(concept_id, weight) 
                  for concept_id, weight 
                  in enumerate(vector) 
                  if abs(weight) > eps]
        return vector
    
    def save(self, fname):
        '''
        See MatrixSimilarity.save()
        '''
        logger.info("storing %s object to %s and %s" % (self.__class__.__name__, 
                                                        fname, 
                                                        fname + '.index'))
        # first, remove the similarity index from self., so it doesn't get pickled
        sim = self.similarity_index
        del self.similarity_index
        try:
            sim.save(fname + ".index")
            utils.pickle(self, fname) # store index-less object
        finally:
            self.similarity_index = sim
        

    @classmethod
    def load(cls, fname):
        """
        Load a previously saved object from file (also see `save`).
        """
        logger.info("loading %s object from %s and %s" % (cls.__name__, 
                                                          fname, 
                                                          fname + ".index"))
        result = utils.unpickle(fname)
        result.similarity_index = MatrixSimilarity.load(fname + ".index")
        return result
#endclass CosineEsaModel
