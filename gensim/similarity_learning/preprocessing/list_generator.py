import pandas as pd
import numpy as np
from collections import Counter
import logging
import re

logger = logging.getLogger(__name__)


class ListGenerator:
    """Class to generate the data for validation and testing in a list format.
    Givem a dataset, this object will provide lists to be ranked by a trained model

    List format is given through the `get_list_data` funciton which returns a dict
    with the list format:
    {"X1": X1, "X2": X2, "y": y, "doc_lengths": doc_lengths}

    Example Usage:
    -------------
    list_generator = JListGenerator(test_file_path, text_maxlen=200)
    list_data = list_generator.get_list_data
    predictions = model.predict(x={"query": list_data["X1"], "doc": list_data["X2"]})
    """

    def __init__(self, file_path, text_maxlen, train_word2index, additional_word2index,
                 zero_word_index, train_pad_word_index, oov_handle_method="ignore"):
        """
        file_path: str
        text_maxlen: int
            The maximum size of a given sentence/query
            If the sentence is smaller, `train_pad_word_index` is used to fill
            the remaining words
        train_word2index: dict
            This dict holds the word2index of the train set. It can be used
            to translate the validation/dev set to have the same vocabulary as the
            training word2index
        additional_word2index: dict
            This dict holds the word2index of the words in the embedding matrix of
            the trained model which aren't there in the train set. It can be used
            to translate the validation/dev set to have the same vocabulary as the
            training word2index
        zero_word_index: int
            The index in the training set for words which should be ignored.
            The embedding matrix has one row set aside for these words and has the
            value of all zeros.
            This is used if `oov_handle_method` is "ignore"
        train_pad_word_index: int
            The index of the pad word used while training.
            It is used to fill the sentence vector if it is smaller than `text_maxlen`
        oov_handle_method: str
            The method to be used to handle out of vocabulary words.
            Current options:
            - ignore : Make all out-of-vocabulary words zero vectors, i.e., ignore them
        """
        if file_path is not None:
            with open(file_path, mode='rU', encoding='utf8') as f:
                self.df = pd.read_csv(f, sep='\t')
        else:
            raise ValueError("file path cannot be None")

        self.queries = list(self.df['Question'])
        self.documents = list(self.df['Sentence'])
        self.relations = list(self.df['Label'])
        self.relations = [int(r) for r in self.relations]
        self.text_maxlen = text_maxlen
        self.word2index, self.index2word = {}, {}
        self.word_counter = Counter()
        self.corpus = self.queries + self.documents

        self.pad_word_index = train_pad_word_index
        self.build_vocab_from_dict(train_word2index=train_word2index,
                                   additional_word2index=additional_word2index,
                                   oov_handle_method=oov_handle_method, zero_word_index=zero_word_index)

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

        Raises:
        -------
        ValueError: if length of sentence exceeds the text_maxlen
        """
        indexed_sentence = [self.word2index[word] for word in sentence.split()]
        padded_sentence = indexed_sentence + \
            [self.pad_word_index] * (self.text_maxlen - len(indexed_sentence))
        if len(padded_sentence) != self.text_maxlen:
            raise ValueError("For sentence: %s\n, the length of sentence: %d exceeds the text_maxlen: %d. Consider \
                increasing text_maxlen or a possible fault in the dataset" %
                             (sentence, len(padded_sentence), self.text_maxlen))
        return np.array(padded_sentence)

    def build_vocab_from_dict(self, train_word2index, additional_word2index,
                              oov_handle_method, zero_word_index):
        """Building vocab from the train vocab dicts. This way, indexes from train
        set will be maintained and the Embedding Layer will get the same index
        for the same word in train and valid/test set"""
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
                        raise ValueError(
                            "Unknown OOV Handling method %s" % oov_handle_method)
        logger.info("Vocab Transfer complete. Method used for OOV is %s" %
                    oov_handle_method)

    def get_list_data(self):
        """Provides the data in the form a dict which has the following Key Value Pairs:
            "X1": The query list.
                  shape: (n_samples, text_maxlen)
            "X2": The document list
                  shape: (n_samples, text_maxlen)
            "y":  The length of the document
                  shape: (n_samples, 1)
            "doc_lengths": The length of each document
                  shape: (n_docs, 1)
                  This will be used later when evaluating the results on a per query basis.

            Example:
            The dataset:
            q1, d1, 0
            q1, d2, 1
            q1, d3, 0
            q1, d4, 0

            q2, d5, 0
            q2, d6, 1
            q2, d7, 0

            ...

            will become:
            X1 : [q1, q1, q1, q1,   q2, q2, q2, ...]
            X2 : [d1, d2, d3, d4,   d5, d6, d7, ...]
            y :  [0,   1, 0,  0,    0,   1, 0,  ...]
            doc_lengths : [4, 3, ...]

            ""

        """
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
        return {"X1": np.array(X1), "X2": np.array(X2), "y": np.array(y), "doc_lengths": doc_lengths}
