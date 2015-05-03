
import math
import gensim.corpora


class BM25(object):

    def __init__(self, corpus, dictionary):
        self.D = len(corpus)
        self.avgdl = sum(map(lambda x: float(len(x)), corpus)) / self.D
        self.docs = corpus
        self.f = []
        self.df = {}
        self.idf = {}
        self.k1 = 1.5
        self.b = 0.75
        self.init()

    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                if not word in tmp:
                    tmp[word] = 0
                tmp[word] += 1
            self.f.append(tmp)
            for k, v in tmp.items():
                if k not in self.df:
                    self.df[k] = 0
                self.df[k] += 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D-v+0.5)-math.log(v+0.5)

    def sim(self, doc, index, average_idf):
        EPSILON = 0.05 * average_idf
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else EPSILON
            score += (idf*self.f[index][word]*(self.k1+1)
                      / (self.f[index][word]+self.k1*(1-self.b+self.b*self.D
                                                  / self.avgdl)))
        return score

    def simall(self, doc, average_idf):
        scores = []
        for index in xrange(self.D):
            score = self.sim(doc, index, average_idf)
            scores.append(score)
        return scores

def bm25_weights(corpus, dictionary):
    bm25 = BM25(corpus, dictionary)
    average_idf = sum(map(lambda k: bm25.idf[k] + 0.00 ,bm25.idf.keys())) / len(bm25.idf.keys())
    weights = []
    for doc in corpus:
        scores = bm25.simall(doc, average_idf)
        weights.append(scores)
    return weights
