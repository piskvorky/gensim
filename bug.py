from gensim.models.fasttext import FastText
from gensim.utils import tokenize
from gensim.test.utils import datapath
import smart_open

import logging
logging.basicConfig(level=logging.INFO)

path = datapath('alldata-id-10.txt')
with smart_open.smart_open(path, 'r', encoding='utf-8') as fin:
    sentences = [l.rstrip() for l in fin]
model = FastText(sentences=sentences, sg=1, hs=1, min_n=4, max_n=6)
print(model)
