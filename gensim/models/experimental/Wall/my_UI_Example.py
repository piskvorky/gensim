import sys
import os

sys.path.append(os.path.join('..'))

import csv
import re
from drmm_tks import DRMM_TKS

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

# Fill the below with wherever you have your WikiQACorpus Folder
wikiqa_data_path = os.path.join(
    '..', 'data', 'WikiQACorpus', 'WikiQA-train.tsv')


def preprocess_sent(sent):
    """Utility function to lower, strip and tokenize each sentence

    Replace this function if you want to handle preprocessing differently"""
    return re.sub("[^a-zA-Z0-9]", " ", sent.strip().lower()).split()


# Defining some consants for .tsv reading
QUESTION_ID_INDEX = 0
QUESTION_INDEX = 1
ANSWER_INDEX = 5
LABEL_INDEX = 6

with open(wikiqa_data_path, encoding='utf8') as tsv_file:
    tsv_reader = csv.reader(tsv_file, delimiter='\t')
    data_rows = []
    for row in tsv_reader:
        data_rows.append(row)


document_group = []
label_group = []

n_relevant_docs = 0
n_filtered_docs = 0

queries = []
docs = []
labels = []

for i, line in enumerate(data_rows[1:], start=1):
    if i < len(data_rows) - 1:  # check if out of bounds might occur
        if data_rows[i][QUESTION_ID_INDEX] == data_rows[i + 1][QUESTION_ID_INDEX]:
            document_group.append(preprocess_sent(data_rows[i][ANSWER_INDEX]))
            label_group.append(int(data_rows[i][LABEL_INDEX]))
            n_relevant_docs += int(data_rows[i][LABEL_INDEX])
        else:
            document_group.append(preprocess_sent(data_rows[i][ANSWER_INDEX]))
            label_group.append(int(data_rows[i][LABEL_INDEX]))

            n_relevant_docs += int(data_rows[i][LABEL_INDEX])

            if n_relevant_docs > 0:
                docs.append(document_group)
                labels.append(label_group)
                queries.append(preprocess_sent(data_rows[i][QUESTION_INDEX]))
            else:
                n_filtered_docs += 1

            n_relevant_docs = 0
            document_group = []
            label_group = []

    else:
        # If we are on the last line
        document_group.append(preprocess_sent(data_rows[i][ANSWER_INDEX]))
        label_group.append(int(data_rows[i][LABEL_INDEX]))
        n_relevant_docs += int(data_rows[i][LABEL_INDEX])

        if n_relevant_docs > 0:
            docs.append(document_group)
            labels.append(label_group)
            queries.append(preprocess_sent(data_rows[i][QUESTION_INDEX]))
        else:
            n_filtered_docs += 1
            n_relevant_docs = 0




train_queries, test_queries = queries[:int(len(queries)*0.1)], queries[int(len(queries)*0.1):]
train_docs, test_docs = docs[:int(len(docs)*0.1)], docs[int(len(docs)*0.1):]
train_labels, test_labels = labels[:int(len(labels)*0.1)], labels[int(len(labels)*0.1):]


print(len(train_queries), len(test_queries))
print(len(train_docs), len(test_docs))
print(len(train_labels), len(test_labels))


word_embedding_path = os.path.join('..', 'data', 'glove.6B.50d.txt')

from collections import Iterable

glove_file = word_embedding_path
tmp_file = get_tmpfile("tmp_word2vec.txt")
embedding_vocab_size, embedding_dim = glove2word2vec(
    glove_file, tmp_file)
kv_model = KeyedVectors.load_word2vec_format(tmp_file)

# Train the model
drmm_tks_model = DRMM_TKS(train_queries, train_docs, train_labels, word_embedding=kv_model,
                          epochs=1, validation_data=[test_queries, test_docs, test_labels])


queries = ["how are glacier caves formed ?".split()]
docs = ["A partly submerged glacier cave on Perito Moreno Glacier".split(),
        "A glacier cave is a cave formed within the ice of a glacier".split()]


print(drmm_tks_model.predict(queries, docs))

drmm_tks_model.save(fname='mera_model.h5')
