
# computing dependencies
import numpy as np
from numpy import float64
from scipy.spatial import distance
from operator import itemgetter
from pyemd import emd

# sklearn dependencies
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import check_array
from sklearn.preprocessing import normalize
from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics import euclidean_distances

# gensim dependencies
from gensim.models import Word2Vec


class FastKNN():
    def __init__(self, docs, n_neighbours = 5, n_jobs = 1):
        """
        Parameters
        ----------

        docs : array of string documents
        example : 
        docs = ["Obama speaks to the media in Illinois","The President addresses the press in Chicago"]
        
        n_neighbours : No. of nearest documents to be used for nearest neighbour query.
        Dafault value is 5
        
        n_jobs : No. of jobs to run for computing  Word Centroid Distance and Relaxed Word Moving Distance
        Default value is 1
        """
        
        self.n_neighbours = n_neighbours
        self.n_jobs = n_jobs

        common_vocab = self._create_embedding(docs) # common_vocab contains words both in word2vec model and the count vectorizer feature names

        self._create_bow(docs, common_vocab)
        self.embedding_distances = euclidean_distances(self.word_embedding)
        print self.word_embedding.shape
        print len(self.embedding_distances[0])
        
    #         super(FastKNN, self).__init__(n_neighbours = n_neighbours, n_jobs = n_jobs)
        
    def _create_bow(self,  docs, common_vocabulary):
        """
        This function 
        1) creates a normalised Bag of Words of the given docs (self.docs).
        2) creates a mapping from doc index to the doc itself (self.id2doc), so that closes documents are recovered 
           from the input docs.

        Parameters
        ----------

        docs : input docs 

        common_vocabulary : tokens for the count vectorizer, so that words in word embedding are present in BOW vector
        """
        self.vectorizer = CountVectorizer(vocabulary = common_vocabulary, dtype = np.double)
        self.docs = normalize(check_array([doc.toarray().ravel() for doc in self.vectorizer.transform(docs)]), norm='l1') # BOW : Normalised Bag Of Words representation of doc
        self.id2doc = dict([(idx, doc) for idx, doc in enumerate(self.docs)]) # creates doc_index mapping of docs to docs
        
    def _create_embedding(self, docs):
        """
        This function creates a word2vec model over the given data and creates an embedding matrix (self.word_embedding)
        of dimension (num of docs X dimension of word2vec vector)

        It returns the tokens for BOW

        Parameters
        ----------

        docs : Input docs conveted to their BOW representation.(no. of docs X vocabulary size)
        """
        vectorizer = CountVectorizer(stop_words = "english").fit(docs)
        tokenizer = vectorizer.build_tokenizer() # sklearn tokenizer to break the sentence into tokens
        
        word_embedding_model = Word2Vec([tokenizer(doc) for doc in docs], min_count = 1) # creating word2vec model from the given data
        model_vocab = word_embedding_model.vocab
        common_vocab = [word for word in vectorizer.get_feature_names() if word in model_vocab]
        self.word_embedding = check_array([word_embedding_model[word] for word in common_vocab])  # representation of tokens in model space : no of docs X word2vec vector size
        self.word_embedding_distance = euclidean_distances(self.word_embedding)
        # np.fill_diagonal(self.word_embedding_distance, float('inf'))
    #         print "fdsd", self.word_embedding.shape
        return common_vocab

    def _pairwise_wcd_dist_row(self, test_doc):
        """
        This function returns the sorted word centroid distances of all given docs with respect to query doc.
        
        Parameters
        ----------

        test_doc : a normalised BOW representation of the query doc.
        """        
        wcd_distances = Parallel(n_jobs = self.n_jobs)(
                                delayed(self.get_wcd)(test_doc, self.docs[doc_id], doc_id)
                                for doc_id in range(len(self.docs)))

        return sorted(dict(wcd_distances).items(), key = itemgetter(1))

    def get_wcd(self, test_doc, doc, doc_id):
    #         print self.word_embedding.shape
    #         print doc1.shape
    #         print doc2.shape
        """
        This function calculates the Word Centroid Distance between docs
        
        Parameters
        ----------

        test_doc : Normalised BOW representaion of test doc .
        doc : Normalised BOW representaion of stored doc .
        doc_id : id of the stored doc
        """
        return (doc_id, distance.euclidean(np.dot(np.transpose(self.word_embedding), np.transpose(test_doc)), np.dot(np.transpose(self.word_embedding), doc)))
    
    def get_rwmd(self, test_doc, docs, doc_id):
        """
        This function calculates and returns the Relaxed Word Movers Distance (RWMD).

        Parameters:
        ----------

        test_doc : Normalised BOW representaion of test doc .
        doc : Normalised BOW representaion of stored doc .
        doc_id : id of the stored doc
        """
        nonzeros_test_doc = np.nonzero(test_doc)[0]
        nonzeros_train_doc = np.nonzero(docs[doc_id])[0]
        union_bow = np.union1d(nonzeros_test_doc, nonzeros_train_doc)
        w2v_distances = self.word_embedding_distance[np.ix_(union_bow, union_bow)]
        np.fill_diagonal(w2v_distances, float('inf'))
        dist1 = np.dot(np.transpose(test_doc)[union_bow][:,0], np.min(w2v_distances, axis = 1))
        dist2 = np.dot(np.transpose(docs[doc_id][union_bow]), np.min(w2v_distances, axis = 1))
        return (doc_id, max(dist1, dist2))

    # def get_wmd():
    #     wcd_distances = 
    def get_wmd(self, test_doc, docs, doc_id):
        """
        This function calculates the wmd distances between 2 documents using emd function in pyemd

        Parameters:
        -----------

        test_doc : Normalised BOW representaion of test doc .
        doc : Normalised BOW representaion of stored doc .
        doc_id : id of the stored doc
        """
        nonzeros_test_doc = np.nonzero(test_doc)[0]
        nonzeros_train_doc = np.nonzero(docs[doc_id])[0]
        union_bow = np.union1d(nonzeros_test_doc, nonzeros_train_doc)
        # print union_bow.shape
        # print test_doc.shape
        # print docs[doc_id].shape
        # print self.word_embedding_distance[np.ix_(union_bow, union_bow)].shape
        return (doc_id,emd(float64(np.transpose(test_doc))[union_bow].ravel(), float64(docs)[doc_id][union_bow].ravel(), float64(self.word_embedding_distance[np.ix_(union_bow, union_bow)])))

    def prune(self, wmd_dist_lst):
        """
        This function helps to place the new neighbor document at its right place.

        Parameters:
        -----------

        wmd_dist_lst : list of wmd distances of n_neighbours from the test document
        """
        for doc_idx in range(self.n_neighbours-2,-1,-1):
            if wmd_dist_lst[doc_idx+1][1] < wmd_dist_lst[doc_idx][1]:
                tmp = wmd_dist_lst[doc_idx] 
                wmd_dist_lst[doc_idx] = wmd_dist_lst[doc_idx+1] 
                wmd_dist_lst[doc_idx+1]  = tmp
        return wmd_dist_lst

    def __getitem__(self, doc):
        """
        This function queries for n nearest neighbours for the given doc
        
        Parameters:
        ----------

        doc : Sentence to query in string form as follows:
                "Obama speaks to the media in Illinois"
        """
    #         print np.array(self.vectorizer.transform([doc]))
        doc = normalize(np.array(self.vectorizer.transform([doc]).toarray().ravel()), norm='l1')
        # wcd_dists = self._pairwise_wcd_dist_row(doc)[:self.n_neighbours]
        wcd_dists = self._pairwise_wcd_dist_row(doc)

        wmd_distances = Parallel(n_jobs = self.n_jobs)(
                                delayed(self.get_wmd)(doc, self.docs, wcd_dists[doc_idx][0])
                                for doc_idx in range(self.n_neighbours))
        
        for i in range(self.n_neighbours, len(wcd_dists)):
            rwmd_dist = self.get_rwmd(doc, self.docs, wcd_dists[i][0])
            if wmd_distances[-1][1]>rwmd_dist[1]:
                wmd_dist = self.get_wmd(doc, self.docs, wcd_dists[i][0])
                if wmd_dist[1] < wmd_distances[-1][1]:
                    wmd_distances[-1] = wmd_dist
                    wmd_distances = self.prune(wmd_distances)    
        print wmd_distances

        

        
    
#     def fit(self, X, y):
#         X = check_array(X, accept_sparse='csr', copy=True)
#         X = normalize(X, norm='l1', copy=False)
#         super(FastKNN, self).fit(X,y)
    
#     def predict(self, X):
#         X = check_array(X, accept_sparse='csr', copy=True)
#         X = normalize(X, norm='l1', copy=False)
#         wcd_pait_dist = self._wcd_distance(X)
#         return super(FastKNN, self).predict(dist)
    