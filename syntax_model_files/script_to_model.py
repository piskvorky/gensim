# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 04:54:50 2016

@author: robert
"""

import pickle

# use nltk.download() to download stopwords corpus if not yet




# Iterable to be passed to word2vec class as sentences.
# reads sentences one by one from picke dump

class SentIterable(object):
    
# correct path to parsed corpus dump HERE

    dumpPath = '/run/media/robert/1TB-1/linuxfolder/pythonworks/corporaDump'
   # dumpPath = '/run/media/robert/1TB-1/linuxfolder/pythonworks/miniCorp'
    dumpOpened = open(dumpPath, 'rb')
#    pickleDump = open(dumpPath, 'rb')
    sentCount = 10000000000
    
# self.sentCount to be inherited (got?) while creating context 86468867
#minicorpus length 3410461
    def __iter__(self):
        try:
            for sent in range(self.sentCount):
                yield pickle.load(self.dumpOpened)
        except EOFError:
            print('Pickler Done')
            self.dumpOpened = open(self.dumpPath, 'rb')
            


class ContIterable(object):
    
# correct path to parsed corpus dump HERE

    dumpPath = '/run/media/robert/1TB-1/linuxfolder/pythonworks/contDumpFinal'
  #  dumpPath = '/run/media/robert/1TB-1/linuxfolder/pythonworks/miniCont'
    dumpOpened = open(dumpPath, 'rb')
#    pickleDump = open(dumpPath, 'rb')
    sentCount = 10000000000
    
# self.sentCount to be inherited (got?) while creating context 98366448
#minicorpus length 3410461
    def __iter__(self):
        try:
            for sent in range(self.sentCount):
                yield pickle.load(self.dumpOpened)
        except EOFError:
            print('Pickler for synt Done')
            self.dumpOpened = open(self.dumpPath, 'rb')            






import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)



num_features = 500    # Word vector dimensionality                      
min_word_count = 100   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 2         # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words


#from gensim.models import word2vec_synt
#print("Training model...")
#model = word2vec_synt.Word2Vec(SentIterable(), workers=num_workers, \
#            size=num_features, min_count = min_word_count, \
#            window = context, sample = downsampling, sg = 1, hs = 1, negative = 0,
#            synt_cash = ContIterable())

sentences = ContIterable()
from gensim.models import word2vec_synt
print("Training model...")
model = word2vec_synt.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling, sg = 1)
