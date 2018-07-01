import sys
import os
from pprint import pprint
sys.path.append(os.path.join('..'))

import csv
import re
from drmm_tks import DRMM_TKS


class MyIter:
    def __init__(self, iter_type, fpath):
        self.type_translator = {'query': 0, 'doc': 1, 'label': 2}
        self.iter_type = iter_type
        with open(fpath, encoding='utf8') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t')
            self.data_rows = []
            for row in tsv_reader:
                self.data_rows.append(row)

    def preprocess_sent(self, sent):
        """Utility function to lower, strip and tokenize each sentence

        Replace this function if you want to handle preprocessing differently"""
        return re.sub("[^a-zA-Z0-9]", " ", sent.strip().lower()).split()

    def __iter__(self):
        # Defining some consants for .tsv reading
        QUESTION_ID_INDEX = 0
        QUESTION_INDEX = 1
        ANSWER_INDEX = 5
        LABEL_INDEX = 6

        document_group = []
        label_group = []

        n_relevant_docs = 0
        n_filtered_docs = 0

        queries = []
        docs = []
        labels = []

        for i, line in enumerate(self.data_rows[1:], start=1):
            if i < len(self.data_rows) - 1:  # check if out of bounds might occur
                if self.data_rows[i][QUESTION_ID_INDEX] == self.data_rows[i + 1][QUESTION_ID_INDEX]:
                    document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                    label_group.append(int(self.data_rows[i][LABEL_INDEX]))
                    n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])
                else:
                    document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                    label_group.append(int(self.data_rows[i][LABEL_INDEX]))

                    n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])

                    if n_relevant_docs > 0:
                        docs.append(document_group)
                        labels.append(label_group)
                        queries.append(self.preprocess_sent(self.data_rows[i][QUESTION_INDEX]))

                        yield [queries[-1], document_group, label_group][self.type_translator[self.iter_type]]
                    else:
                        n_filtered_docs += 1

                    n_relevant_docs = 0
                    document_group = []
                    label_group = []

            else:
                # If we are on the last line
                document_group.append(self.preprocess_sent(self.data_rows[i][ANSWER_INDEX]))
                label_group.append(int(self.data_rows[i][LABEL_INDEX]))
                n_relevant_docs += int(self.data_rows[i][LABEL_INDEX])

                if n_relevant_docs > 0:
                    docs.append(document_group)
                    labels.append(label_group)
                    queries.append(self.preprocess_sent(self.data_rows[i][QUESTION_INDEX]))
                    yield [queries[-1], document_group, label_group][self.type_translator[self.iter_type]]
                else:
                    n_filtered_docs += 1
                    n_relevant_docs = 0

q_iterable = MyIter('query', os.path.join('..', 'data', 'WikiQACorpus', 'WikiQA-train.tsv'))
d_iterable = MyIter('doc', os.path.join('..', 'data', 'WikiQACorpus', 'WikiQA-train.tsv'))
l_iterable = MyIter('label', os.path.join('..', 'data', 'WikiQACorpus', 'WikiQA-train.tsv'))

word_embedding_path = os.path.join('..', 'data', 'glove.6B.50d.txt')


# Train the model
drmm_tks_model = DRMM_TKS(q_iterable, d_iterable, l_iterable, word_embedding=word_embedding_path,
                          epochs=1, validation_data=None)#[test_queries, test_docs, test_labels])


queries = ["how are glacier caves formed ?".split()]
docs = ["A partly submerged glacier cave on Perito Moreno Glacier".split(),
        "A glacier cave is a cave formed within the ice of a glacier".split()]


drmm_tks_model.predict(queries, docs)

drmm_tks_model.save(fname='mera_model.h5')
