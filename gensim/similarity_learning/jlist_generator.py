import pandas as pd
import numpy as np
from collections import Counter
import logging
import re
import six
from sklearn.utils import shuffle

logger = logging.getLogger(__name__)

class JListGenerator:
    def __init__(self, file_path, word_embedding_path=None, text_maxlen=140, train_word2index=None,
                additional_word2index=None, oov_handle_method="ignore", zero_word_index=None, train_pad_word_index=None):
        if file_path is not None:
            with open(file_path, mode='rU', encoding='utf8') as f:
                self.df = pd.read_csv(f, sep='\t')
        else:
            raise NotImplementedError()

        self.queries = list(self.df['Question'])
        self.documents = list(self.df['Sentence'])
        self.relations = list(self.df['Label'])
        self.relations = [int(r) for r in self.relations]

        self.text_maxlen = text_maxlen
        self.word_embedding_path = word_embedding_path
        self.word2index, self.index2word = {}, {}
        self.word_counter = Counter()
        self.corpus = self.queries + self.documents


        if train_word2index is not None:
            if zero_word_index is None:
                raise ValueError("zero_word_index cannot be None. \
                    Please set it to the value of the zero embedding in the Embedding Matrix")
            if train_pad_word_index is None:
                raise ValueError("Pad word index cannot be None")

            self.pad_word_index = train_pad_word_index
            self.build_vocab_from_dict(train_word2index=train_word2index, additional_word2index=additional_word2index,
                                        oov_handle_method=oov_handle_method, zero_word_index=zero_word_index)
        else:
            self.build_vocab()



    def preprocess(self, sentence):
        """Preprocess an input string to allow only alphabets and numbers

        Parameters:
        ----------
        sentence: str
            The sentence to preprocess
        """
        return re.sub("[^a-zA-Z0-9]", " ", sentence.lower())

    def make_indexed(self, sentence):
        """Returns the indexed version of the sentence based on self.word2index
        in the form of a list

        Parameters:
        -----------
        sentenceL str
            The sentence to be indexed
        """
        indexed_sentence = [self.word2index[word] for word in sentence.split()]
        padded_sentence = indexed_sentence + [self.pad_word_index]*(self.text_maxlen - len(indexed_sentence))
        if len(padded_sentence) != self.text_maxlen:
            print(sentence)
            raise ValueError("length of sentence: %d exceeds the text_maxlen: %d. Consider \
                increasing text_maxlen or a possible fault in the dataset" %
                              (len(padded_sentence), self.text_maxlen))
        return  np.array(padded_sentence)

    def build_vocab_from_dict(self, train_word2index, additional_word2index,
                              oov_handle_method, zero_word_index):
        """Building vocab from the train vocab dicts
        This way, indexes from train set will be maintained and the Embedding Layer won't
        go OOV"""

        logger.info("Getting List Vocab from given vocab")
        train_word2index = dict(train_word2index, **additional_word2index)
        for sentence in self.corpus:
            sentence = self.preprocess(sentence)
            for word in sentence.split():
                if word in train_word2index:
                    self.word2index[word] = train_word2index[word]
                    self.index2word[train_word2index[word]] = word
                else:
                    if oov_handle_method == "ignore":
                        # Map the word to a zero
                        self.word2index[word] = zero_word_index
                    else:
                        raise ValueError("Unknown OOV Handling method %s" % oov_handle_method)
        logger.info("Vocab Transfer complete. Method used for OOV is %s" % oov_handle_method)


    def build_vocab(self):
        """Indexes all the words and makes an embedding_matrix which
        can be fed directly into an Embedding layer"""

        logger.info("Starting Vocab Build")
        for sentence in self.corpus:
            sentence = self.preprocess(sentence)
            self.word_counter.update(sentence.split())
        for i, word in enumerate(self.word_counter.keys()):
            self.word2index[word] = i
            self.index2word[i] = word
        self.vocab_size = len(self.word2index)
        logger.info("Vocab Build Complete")

        logger.info("Building embedding index using pretrained word embeddings")
        embeddings_index = {}
        with open(self.word_embedding_path) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        self.embedding_dim = six.next(six.itervalues(embeddings_index)).shape[0]
        self.embedding_matrix = np.zeros((len(self.word2index) + 1, self.embedding_dim))

        for word, i in self.word2index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector
        logger.info("Embedding index build complete")

        self.pad_word_index = self.vocab_size
        logger.info("Pad word has been set to %")

    def get_data(self):
        self.query_grouped_data = []
        for Question, Answer in self.df.groupby('QuestionID').apply(dict).items():

            document_group = []
            for q, d, l in zip(Answer['Question'], Answer['Sentence'], Answer['Label']):
                document_group.append([self.make_indexed(self.preprocess(q)),
                                        self.make_indexed(self.preprocess(d)), int(l)])
            self.query_grouped_data.append(document_group)
        return self.query_grouped_data

    def get_data_long(self):
        X1 = []
        X2 = []
        y = []
        doc_lengths = []
        for Question, Answer in self.df.groupby('QuestionID').apply(dict).items():
            n_docs = 0
            for q, d, l in zip(Answer['Question'], Answer['Sentence'], Answer['Label']):
                X1.append(self.make_indexed(self.preprocess(q)))
                X2.append(self.make_indexed(self.preprocess(d)))
                y.append(int(l))
                n_docs += 1
            doc_lengths.append(n_docs)
        return {"X1": X1, "X2": X2, "y": y, "doc_lengths": doc_lengths}