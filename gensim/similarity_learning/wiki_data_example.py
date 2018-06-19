import os
import csv
import re
from gensim.similarity_learning import DRMM_TKS_Model
from pprint import pprint
from sklearn.preprocessing import normalize
from gensim.similarity_learning import ValidationCallback
from gensim.similarity_learning.preprocessing import ListGenerator

def preprocess_sent(sent):
    return re.sub("[^a-zA-Z0-9]", " ", sent.strip().lower()).split()

# Defining some consants for .tsv reading
QUESTION_ID_INDEX = 0
QUESTION_INDEX = 1
ANSWER_INDEX = 5
LABEL_INDEX = 6

wikiqa_data_path = os.path.join('data', 'WikiQACorpus', 'WikiQA-train.tsv')

with open(wikiqa_data_path, encoding='utf8') as tsv_file:
    tsv_reader = csv.reader(tsv_file, delimiter='\t')
    data_rows = []
    for row in tsv_reader:
        data_rows.append(row)

print(data_rows[:3])
print(len(data_rows[1:]))
asd

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

train_queries, test_queries = queries[:int(len(queries)*0.8)], queries[int(len(queries)*0.8): ]
train_docs, test_docs = docs[:int(len(docs)*0.8)], docs[int(len(docs)*0.8):]
train_labels, test_labels = labels[:int(len(labels)*0.8)], labels[int(len(labels)*0.8):]

# Train the model
drmm_tks_model = DRMM_TKS_Model(train_queries, train_docs, train_labels, word_embedding_path=os.path.join('data', 'glove.6B.50d.txt'),
                                    epochs=1, validation_data=[test_queries, test_docs, test_labels])


# Test queries

correct_labels = [[0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, 1, 1]]

# predictied_labels = drmm_tks_model.predict(queries=test_q, docs=test_d)
pprint(test_l)
pprint(predictied_labels)
