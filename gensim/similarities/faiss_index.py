from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
try:
    import faiss
except ImportError:
    raise ImportError("Faiss has not been installed")


class FaissIndexer(object):
    def __init__(self, model=None, nlist=None):
        self.quantizer = None
        self.index = None
        self.d = None
        self.labels = None
        self.vectors = None
        self.model = model
        self.nlist = nlist
        self.nprobe = None
        if model and nlist:
            if isinstance(self.model, Doc2Vec):
                self.build_from_doc2vec()
            elif isinstance(self.model, Word2Vec):
                self.build_from_word2vec()
            else:
                raise ValueError("Only Word2Vec or Doc2Vec models can be used")

    def build_from_doc2vec(self):
        docvecs = self.model.docvecs
        docvecs.init_sims()
        labels = [docvecs.index_to_doctag(i) for i in range(0, docvecs.count)]
        return self._build_from_model(docvecs.doctag_syn0norm, labels,
                                      self.model.vector_size)

    def build_from_word2vec(self):
        self.model.init_sims()
        return self._build_from_model(self.model.wv.syn0norm,
                                      self.model.wv.index2word,
                                      self.model.vector_size)

    def _build_from_model(self, vectors, labels, num_features):
        self.vectors = vectors
        self.d = num_features
        self.quantizer = faiss.IndexFlatL2(num_features)
        self.labels = labels

    def most_similar_l2(self, query, num_neighbors, nprobe=1):
        self.nprobe = nprobe
        self.index = faiss.IndexIVFFlat(self.quantizer, self.d, self.nlist,
                                        faiss.METRIC_L2)
        self.index.train(self.vectors)
        self.index.add(self.vectors)
        query_vec = self.model.wv.syn0norm[self.labels.index(query)]
        query_vec = query_vec.reshape((1, query_vec.shape[0]))
        self.index.nprobe = nprobe
        distances, indices = self.index.search(query_vec, num_neighbors)
        list_similar = []
        for ind, distance in zip(indices, distances):
            for ind_i, dist_j in zip(ind, distance):
                list_similar.append([self.labels[ind_i], dist_j])
        return list_similar

    def most_similar_dot_product(self, query, num_neighbors, nprobe=1):
        self.nprobe = nprobe
        self.index = faiss.IndexIVFFlat(self.quantizer, self.d, self.nlist)
        self.index.train(self.vectors)
        self.index.add(self.vectors)
        query_vec = self.model.wv.syn0norm[self.labels.index(query)]
        query_vec = query_vec.reshape((1, query_vec.shape[0]))
        self.index.nprobe = nprobe
        distances, indices = self.index.search(query_vec, num_neighbors)
        list_similar = []
        for ind, dot_product in zip(indices, distances):
            for ind_i, dot_product_j in zip(ind, dot_product):
                list_similar.append([self.labels[ind_i], dot_product_j])
        return list_similar
