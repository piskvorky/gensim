
# computing dependencies
import numpy as np
from scipy.spatial import distance
from operator import itemgetter

# sklearn dependencies
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import check_array
from sklearn.preprocessing import normalize
from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.externals.joblib import Parallel, delayed

# gensim dependencies
from gensim.models import Word2Vec


class FastKNN():
    def __init__(self, docs, n_neighbours = 5, n_jobs = 1):
        """
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
        
#         super(FastKNN, self).__init__(n_neighbours = n_neighbours, n_jobs = n_jobs)
        
    def _create_bow(self,  docs, common_vocabulary):
        """
        This function 
        1) creates a normalised Bag of Words of the given docs (self.docs).
        2) creates a mapping from doc index to the doc itself (self.id2doc), so that closes documents are recovered 
           from the input docs.

        docs : input docs 

        common_vocabulary : tokens for the count vectorizer, so that words in word embedding are present in BOW vector
        """
        self.vectorizer = CountVectorizer(vocabulary = valid_vocabulary, dtype = np.double)
        self.docs = normalize(check_array([doc.toarray().ravel() for doc in self.vectorizer.transform(docs)]), norm='l1') # BOW : Normalised Bag Of Words representation of doc
        self.id2doc = dict([(idx, doc) for idx, doc in enumerate(self.docs)]) # creates doc_index mapping of docs to docs
        
    def _create_embedding(self, docs):
        """
        This function creates a word2vec model over the given data and creates an embedding matrix (self.word_embedding)
        of dimension (num of docs X dimension of word2vec vector)

        It returns the tokens for BOW

        docs : Input docs conveted to their BOW representation.(no. of docs X vocabulary size)
        """
        vectorizer = CountVectorizer(stop_words = "english").fit(docs)
        tokenizer = vectorizer.build_tokenizer() # sklearn tokenizer to break the sentence into tokens
        
        word_embedding_model = Word2Vec([tokenizer(doc) for doc in docs], min_count = 1) # creating word2vec model from the given data
        model_vocab = word_embedding_model.vocab
        common_vocab = [word for word in vectorizer.get_feature_names() if word in model_vocab]
        self.word_embedding = check_array([word_embedding_model[word] for word in common_vocab])  # representation of tokens in model space : no of docs X word2vec vector size
#         print "fdsd", self.word_embedding.shape
        return common_vocab
    
    def _pairwise_wcd_dist_row(self, test_doc):
        """
        This function returns the sorted word centroid distances of all given docs with respect to query doc.
        
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
        
        test_doc : Normalised BOW representaion of test doc .
        doc : Normalised BOW representaion of stored doc .
        doc_id : id of the stored doc
        """
        return (doc_id,distance.euclidean(np.dot(np.transpose(self.word_embedding), np.transpose(test_doc)), np.dot(np.transpose(self.word_embedding), doc)))

    def __getitem__(self, doc):
        """
        This function queries for n nearest neighbours for the given doc
        
        doc : Sentence to query in string form as follows:
                "Obama speaks to the media in Illinois"
        """
#         print np.array(self.vectorizer.transform([doc]))
        doc = normalize(np.array(self.vectorizer.transform([doc]).toarray().ravel()), norm='l1')
        wcd_dists = self._pairwise_wcd_dist_row(doc)[:self.n_neighbours]
        # wcd_dists = self._pairwise_wcd_dist_row(doc)
        return wcd_dists
        
#     def fit(self, X, y):
#         X = check_array(X, accept_sparse='csr', copy=True)
#         X = normalize(X, norm='l1', copy=False)
#         super(FastKNN, self).fit(X,y)
    
#     def predict(self, X):
#         X = check_array(X, accept_sparse='csr', copy=True)
#         X = normalize(X, norm='l1', copy=False)
#         wcd_pait_dist = self._wcd_distance(X)
#         return super(FastKNN, self).predict(dist)
    
    
        