import sys
import os
from pprint import pprint
import csv
import re
from drmm import DRMM
import gensim.downloader as api

class MyWikiIterable:
    def __init__(self, iter_type, fpath):
        self.type_translator = {'query': 0, 'doc': 1, 'label': 2}
        self.iter_type = iter_type
        with open(fpath, encoding='utf8') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
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

q_iterable = MyWikiIterable('query', os.path.join('experimental_data', 'WikiQACorpus', 'WikiQA-train.tsv'))
d_iterable = MyWikiIterable('doc', os.path.join('experimental_data', 'WikiQACorpus', 'WikiQA-train.tsv'))
l_iterable = MyWikiIterable('label', os.path.join('experimental_data', 'WikiQACorpus', 'WikiQA-train.tsv'))

q_val_iterable = MyWikiIterable('query', os.path.join( 'experimental_data', 'WikiQACorpus', 'WikiQA-dev.tsv'))
d_val_iterable = MyWikiIterable('doc', os.path.join('experimental_data', 'WikiQACorpus', 'WikiQA-dev.tsv'))
l_val_iterable = MyWikiIterable('label', os.path.join('experimental_data', 'WikiQACorpus', 'WikiQA-dev.tsv'))

q_test_iterable = MyWikiIterable('query', os.path.join( 'experimental_data', 'WikiQACorpus', 'WikiQA-test.tsv'))
d_test_iterable = MyWikiIterable('doc', os.path.join('experimental_data', 'WikiQACorpus', 'WikiQA-test.tsv'))
l_test_iterable = MyWikiIterable('label', os.path.join('experimental_data', 'WikiQACorpus', 'WikiQA-test.tsv'))

kv_model = api.load('glove-wiki-gigaword-50')

# Train the model
drmm_tks_model = DRMM(
                    queries=q_iterable, docs=d_iterable, labels=l_iterable, word_embedding=kv_model, epochs=20,
                    hist_size=60, validation_data=None, text_maxlen=100
                )

drmm_tks_model.save('mera_drmm_')

print('Test set results')
drmm_tks_model.evaluate(q_test_iterable, d_test_iterable, l_test_iterable)