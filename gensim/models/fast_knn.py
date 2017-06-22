
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

        embed_dict = _preprocess(docs)
        self.word_embedding = check_array(embed_dict.values())

        common_vocab = embed_dict.keys() # common_vocab contains words both in word2vec model and the count vectorizer feature names
        self._create_bow(docs, common_vocab)
        
        
    #         super(FastKNN, self).__init__(n_neighbours = n_neighbours, n_jobs = n_jobs)
    def _preprocess(self, dataset_data):
        """
        This function creates a word2vec model from the google news data.
        The vector size of eacg word vector is 300
        It
            removes any kind of punctuation and stopwords according to @wkusner's list.
            creates an embedding for words specific to a dataset.
            returns an embedding dictionary.

        Parameters:
        -----------

        dataset_data : Input Docs - It is  a list of documents

        """
        word_vectors_model = Word2Vec.load_word2vec_format(
                                        "./GoogleNews-vectors-negative300.bin.gz",
                                        binary=True
                                            )
        punctuation_marks = string.punctuation

        for doc_idx in range(len(dataset_data)):
            for punctuation in punctuation_marks:
                dataset_data[doc_idx] = dataset_data[doc_idx].replace(punctuation,"")
            dataset_data[doc_idx] = dataset_data[doc_idx].replace("\n"," ")
            
        stopwords = open("stopwords.txt", "r").readlines()
        sopwords = set([word.strip() for word in stopwords])

        vocab = [] # vocabulary array of given data
        embedding = {} # word2vec embedding of vocab

        [vocab.extend(line.strip().lower().split("\t")[1].split(" ")) for line in dataset_data]
        # get the word embeddings of given docs vocab
        for word_idx in range(len(vocab)):
            if vocab[word_idx] in stopwords:
                del vocab[word_idx] # stopwords are not to be a part of Bag Of Words
            else:
                try:
                    embedding[vocab[word_idx]] = word_vectors_mode[vocab[word_idx]]
                except Exception:
                    continue
        return embedding
    
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
        Here misplaced wmd distance is the last element  of the sorted wmd_dist_lst. So, we traverse
        back from 2nd last element (self.n_neighbours-2) all the way upto  1st element  to keep the list sorted
        in ascending order.

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

    
    def check_distance_order(self, doc):
        """
        This function plots the distance of a test document from all the docs in the reserve

        Parameters:
        -----------

        
        doc : Sentence to query in string form as follows:
                "Obama speaks to the media in Illinois"
        """
        doc = normalize(np.array(self.vectorizer.transform([doc]).toarray().ravel()), norm='l1')
        # wcd_dists = self._pairwise_wcd_dist_row(doc)[:self.n_neighbours]

        #plotting the WCD distances
        wcd_dists = self._pairwise_wcd_dist_row(doc)
        doc_ids = [pair[0] for pair in wcd_dists]
        doc_dists = [pair[1] for pair in wcd_dists]
        plt.scatter(doc_ids, doc_dists)
#         plt.show()

        # plotting the WMD distances with red colored dots
        wmd_distances = Parallel(n_jobs = self.n_jobs)(
                                delayed(self.get_wmd)(doc, self.docs, wcd_dists[doc_idx][0])
                                for doc_idx in range(20))
        
        doc_wmd_ids = [pair[0] for pair in wmd_distances]
        doc_wmd_dists = [pair[1] for pair in wmd_distances]
        plt.scatter(doc_wmd_ids, doc_wmd_dists, color = "r")

        #plotting the RWMD distances with green colored dots
        rwmd_distances = Parallel(n_jobs = self.n_jobs)(
                                delayed(self.get_rwmd)(doc, self.docs, wcd_dists[doc_idx][0])
                                for doc_idx in range(20))
        doc_rwmd_ids = [pair[0] for pair in rwmd_distances]
        doc_rwmd_dists = [pair[1] for pair in rwmd_distances]
        plt.scatter(doc_rwmd_ids, doc_rwmd_dists, color = "g")
            

        
    
#     def fit(self, X, y):
#         X = check_array(X, accept_sparse='csr', copy=True)
#         X = normalize(X, norm='l1', copy=False)
#         super(FastKNN, self).fit(X,y)
    
#     def predict(self, X):
#         X = check_array(X, accept_sparse='csr', copy=True)
#         X = normalize(X, norm='l1', copy=False)
#         wcd_pait_dist = self._wcd_distance(X)
#         return super(FastKNN, self).predict(dist)
    