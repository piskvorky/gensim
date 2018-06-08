from collections import Counter
import pandas as pd
import numpy as np
import logging
import re
import random
import six
from sklearn.utils import shuffle

logger = logging.getLogger(__name__)


"""
This file contains the WikiQAExtractor, WikiQA_DRMM_TKS_Extractor and QuoraQPExtractor which are utility classes to
extract data from the data files.
"""


class WikiQA_DRMM_TKS_Extractor:
    """Class to extract data from the WikiQA dataset and provide it in a streamable format for training
    It will provide a generator for training which will have int-indexed data in the form of:
    query, positive document, negative document

    The generator can be directly used with a keras model like:
    model.fit_generator(DRRM_TKS_generator)

    Example Usage:
    -------------
    wikiqa = WikiQAExtractor('data/WikiQACorpus/WikiQA-train.tsv')
    gen  = wikiqa.get_batch_generator(32)

    How it works:
    ------------
    The __init__ function will call:
    1. build_vocab: which will index all the words and set up and embedding matrix from the pretrained
                    Glove vectors whose path is provided in `word_embedding_path`
    2. get_pair_list: which will for a list of all the queries and documents in the format of
                    (query, positive document, negative document)

    When the `get_batch_generator` function is called by the user to get the iterable generator, it provides
    a generator calls `get_batch` which provides the batch.
    """

    def __init__(self, file_path, word_embedding_path=None, text_maxlen=100, keep_full_embedding=False):
        """Initializes the extractor

        Parameters:
        -----------
        file_path: str
            path to the WikiQA-xxxx.tsv
            where xxxx could be train, test or valid

        word_embedding_path: str
            path to the Glove vectors which have the embeddings in a .txt format

        embedding_dim: int
            The size of the vectors in the above Glove vectors

        text_maxlen: int
            The maximum possible length of a query or a document
            This is used for padding.

        keep_full_embedding: boolean
            Whether the full embedding should be built or only the words in the dataset's vocab
            This becomes important for checking validation and test sets
        """

        if file_path is not None:
            with open(file_path, encoding='utf8') as f:
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
        self.keep_full_embedding = keep_full_embedding
        self.additional_word2index = {}
        self.build_vocab()
        self.pair_list = self.get_pair_list()

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
        logger.info("Vocab Size is %d" % self.vocab_size)

        logger.info("Building embedding index using pretrained word embeddings")
        embeddings_index = {}
        with open(self.word_embedding_path) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        self.embedding_dim = six.next(
            six.itervalues(embeddings_index)).shape[0]
        embedding_vocab_size = len(embeddings_index)

        logger.info("The embeddings_index built from the given file has %d words of %d dimensions" %
                    (embedding_vocab_size, self.embedding_dim))

        if self.keep_full_embedding:
            self.embedding_matrix = np.zeros(
                (self.vocab_size + 1, self.embedding_dim))  # one for ignore vec
        else:
            # one for pad, one for ignore vec
            self.embedding_matrix = np.zeros(
                (self.vocab_size + 2, self.embedding_dim))

        # We add 1 for the padding word
        logger.info("Embedding Matrix for Embedding Layer has shape %s " %
                    str(self.embedding_matrix.shape))

        n_non_embedding_words = 0
        for word, i in self.word2index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector
            else:
                n_non_embedding_words += 1

        logger.info("There are %d words not in the embeddings. Setting them to zero" %
                    n_non_embedding_words)

        if self.keep_full_embedding:
            logger.info(
                "Adding additional dimensions from the embedding file to embedding matrix")
            i = self.vocab_size
            extra_embeddings = []
            for word in embeddings_index.keys():
                if word not in self.word2index:
                    # Stack the new word's vector and index it
                    extra_embeddings.append(embeddings_index.get(word))
                    # We also need to keep an additional indexing of these
                    # words
                    self.additional_word2index[word] = i
                    i += 1

            self.embedding_matrix = np.vstack([self.embedding_matrix, np.array(extra_embeddings),
                                               np.random.random((1, self.embedding_dim)), np.zeros((1, self.embedding_dim))])

            # Last word is kept as the pad word
            # Here that is the last word in the embedding matrix
            self.pad_word_index = i
            self.zero_word_index = i + 1

        else:
            self.pad_word_index = self.vocab_size
            self.zero_word_index = self.vocab_size + 1

        logger.info("Embedding Matrix now has shape %s" %
                    str(self.embedding_matrix.shape))
        logger.info("Pad word has been set to index %d" % self.pad_word_index)
        logger.info("Embedding index build complete")

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
        return [self.word2index[word] for word in sentence.split()]

    def get_data(self, filter_queries=True):
        """Provides the data in the format:
        (query, document, relevance)
        where query: list of ints representing the words in the query sentence
              document: list of ints representing the words in the document sentence
              relevance: the relevance score of the two in terms of 1 or 0

        Parameters:
        ----------
        filter_queries: boolean
            Whether queries which have no relevant answer should be kept
            if True, no answer queries are dropped
        """

        self.data = []
        for Question, Answer in self.df.groupby('QuestionID').apply(dict).items():
            n_relevant_docs = 0
            document_group = []
            for q, d, l in zip(Answer['Question'], Answer['Sentence'], Answer['Label']):
                document_group.append([self.make_indexed(self.preprocess(q)),
                                       self.make_indexed(self.preprocess(d)), l])
                n_relevant_docs += l  # CHECK

            n_filtered = 0
            if filter_queries:  # Only add the document group if it has relevant documents
                if n_relevant_docs > 0:
                    self.data.append(document_group)
                else:
                    n_filtered += 1
            else:
                self.data.append(document_group)
        logger.info("%d queries were filtered" % n_filtered)
        logger.info("There are a total of %d queries" % len(self.data))
        return self.data

    def get_pair_list(self):
        """Returns a list with query document pairs in the format
        (query, positive_doc, negative_doc)

        Example output:
        --------------
        [(q1, d+, d-), (q2, d+, d-), (q3, d+, d-), ..., (qn, d+, d-)]

        wher each query or document is a list of ints
        """

        # call get_data to make sure self.data has been generated
        self.get_data()
        pair_list = []
        for document_group in self.data:
            document_group = sorted(
                document_group, key=lambda x: x[2], reverse=True)
            for item in document_group:
                if item[2] == 1:
                    for new_item in document_group:
                        if new_item[2] == 0:
                            pair_list.append((item[0], item[1], new_item[1]))

        logger.info("There are %d pairs to train on. So, a total of %d datapoints" % (
            len(pair_list), len(pair_list) * 2))
        return shuffle(pair_list)

    def get_full_batch(self):
        """Generator to provide a batch of training sample of size batch_size
        The batch provided is actually twice the size and will have alternate positive
        and negative examples. CHECK does alternation even matter?

        So, if batch_size is 16
        the returned batch will be of size 32 like:
        [positive_example, negative_example, positive_example, ...]

        TOCHECK : maybe this behaviour shouldn't be like this

        Parameters:
        ----------
        batch_size: int
            provides of batches of that size
        """

        num_samples = len(self.pair_list)
        X1 = np.zeros((num_samples * 2, self.text_maxlen))
        X2 = np.zeros((num_samples * 2, self.text_maxlen))
        y = np.zeros((num_samples * 2, 1))

        X1[:] = self.pad_word_index
        X2[:] = self.pad_word_index
        y[::2] = 1

        for i, (query, pos_doc, neg_doc) in enumerate(self.pair_list):
            query_len = min(self.text_maxlen, len(query))
            pos_doc_len = min(self.text_maxlen, len(pos_doc))
            neg_doc_len = min(self.text_maxlen, len(neg_doc))

            X1[i * 2, :query_len] = query[:query_len]
            X2[i * 2, :pos_doc_len] = pos_doc[:pos_doc_len]
            X1[i * 2 + 1, :query_len] = query[:query_len]
            X2[i * 2 + 1, :neg_doc_len] = neg_doc[:neg_doc_len]

        return X1, X2, y

    def get_batch(self, batch_size=32):
        """Function to provide a batch of training sample of size batch_size
        This function is called by the get_batch_generator function to provide
        individual batches

        The batch provided is actually twice the size and will have alternate positive
        and negative examples.
        So, if batch_size is 16
        the returned batch will be of size 32 like:
        [positive_example, negative_example, positive_example, ...]

        TOCHECK : maybe this behaviour shouldn't be like this

        Parameters:
        ----------
        batch_size: int
            provides of batches of that size
        """

        # Initialize the query, doc and relevance arrays to zero
        X1 = np.zeros((batch_size * 2, self.text_maxlen), dtype=np.int32)
        X1_len = np.zeros((batch_size * 2,), dtype=np.int32)

        X2 = np.zeros((batch_size * 2, self.text_maxlen), dtype=np.int32)
        X2_len = np.zeros((batch_size * 2,), dtype=np.int32)

        Y = np.zeros((batch_size * 2,), dtype=np.int32)

        # set alternate relevances to 1, starting with 1
        Y[::2] = 1

        # set all the values to pad words
        X1[:] = self.pad_word_index
        X2[:] = self.pad_word_index

        for i in range(batch_size):
            query, pos_doc, neg_doc = random.choice(self.pair_list)

            query_len = min(self.text_maxlen, len(query))
            pos_doc_len = min(self.text_maxlen, len(pos_doc))
            neg_doc_len = min(self.text_maxlen, len(neg_doc))

            X1[i * 2, :query_len], X1_len[i * 2] = query[:query_len], query_len
            X2[i * 2, :pos_doc_len], X2_len[i *
                                            2] = pos_doc[:pos_doc_len], pos_doc_len
            X1[i * 2 + 1, :query_len], X1_len[i *
                                              2 + 1] = query[:query_len], query_len
            X2[i * 2 + 1, :neg_doc_len], X2_len[i * 2 +
                                                1] = neg_doc[:neg_doc_len], neg_doc_len

        return X1, X1_len, X2, X2_len, Y

    def jbatch_gen(self, batch_size=32):
        """Generator to provide a batch of training sample of size batch_size
        The batch provided is actually twice the size and will have alternate positive
        and negative examples. CHECK does alternation even matter?

        So, if batch_size is 16
        the returned batch will be of size 32 like:
        [positive_example, negative_example, positive_example, ...]

        TOCHECK : maybe this behaviour shouldn't be like this

        Parameters:
        ----------
        batch_size: int
            provides of batches of that size
        """

        while True:
            # Initialize the query, doc and relevance arrays to zero
            X1 = np.zeros((batch_size * 2, self.text_maxlen), dtype=np.int32)
            X1_len = np.zeros((batch_size * 2,), dtype=np.int32)

            X2 = np.zeros((batch_size * 2, self.text_maxlen), dtype=np.int32)
            X2_len = np.zeros((batch_size * 2,), dtype=np.int32)

            Y = np.zeros((batch_size * 2,), dtype=np.int32)

            # set alternate relevances to 1, starting with 1
            Y[::2] = 1

            # set all the values to pad words
            X1[:] = self.pad_word_index
            X2[:] = self.pad_word_index

            num_samples = len(self.pair_list)

            for offset in range(0, num_samples, batch_size):
                batch_sample = self.pair_list[offset: offset + batch_size]
                for i, (query, pos_doc, neg_doc) in enumerate(batch_sample):
                    query_len = min(self.text_maxlen, len(query))
                    pos_doc_len = min(self.text_maxlen, len(pos_doc))
                    neg_doc_len = min(self.text_maxlen, len(neg_doc))

                    X1[i * 2, :query_len], X1_len[i *
                                                  2] = query[:query_len], query_len
                    X2[i * 2, :pos_doc_len], X2_len[i *
                                                    2] = pos_doc[:pos_doc_len], pos_doc_len
                    X1[i * 2 + 1, :query_len], X1_len[i *
                                                      2 + 1] = query[:query_len], query_len
                    X2[i * 2 + 1, :neg_doc_len], X2_len[i * 2 +
                                                        1] = neg_doc[:neg_doc_len], neg_doc_len

            yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len}, Y)

    def get_batch_generator(self, batch_size):
        """Acts as a generator which provides a batch of batch_size
        Returns the batch like a dict since the model takes in 2 inputs
        1. query
        2. doc

        Parameter:
        ---------
        batch_size: int
            size of the batch"""
        while True:
            X1, X1_len, X2, X2_len, Y = self.get_batch(batch_size)
            yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len}, Y)


class WikiQAExtractor:
    """[WIP]Class to extract data from the WikiQA dataset
    It provides data in a shape which makes it easier to train neural networks
    Usage:
    =====
    wiki_extractor = WikiQAExtractor("path/to/file")
    train_data = wiki_extractor.get_data()
    the format provided is : [ [
                                [query1, doc1, label_1_1],
                                [query1, doc2, label_1_2],
                                .
                                .
                               ],
                               [
                                [query2, doc1, label_2_1],
                                [query2, doc2, label_2_2],
                                .
                                .
                                ],
                                .
                                .
                                .
                               [
                                [query_n, doc1, label_n_1],
                                [query_n, doc2, label_n_2],
                               ]
                             ]
    """

    def __init__(self, file_path, embedding_path=None):
        if file_path is not None:
            with open(file_path) as f:
                self.df = pd.read_csv(f, sep='\t')
        else:
            raise NotImplementedError()

        self.queries = list(self.df.iloc[:, 0])
        self.documents = list(self.df.iloc[:, 1])
        self.relations = list(self.df.iloc[:, 2])

        # TODO add 300k vector for all permutes
        # TODO add option for using word embeddings

        self.word2int, self.int2word = {}, {}
        self.tri2index, self.index2tri = {}, {}

        self.word_counter = Counter()
        self.triletter_counter = Counter()

        self.corpus = self.queries + self.documents
        self.triletter_corpus = []

        self.build_vocab()

    def preprocess(self, sentence):
        return re.sub("[^a-zA-Z0-9]", " ", sentence.lower())

    def get_preprocessed_corpus(self):
        preprocessed_corpus = []
        for sent in self.corpus:
            preprocessed_corpus.append(self.preprocess(sent))

        return preprocessed_corpus

    def build_vocab(self):
        logger.info("Starting Vocab Build")

        for sentence in self.corpus:
            sentence = self.preprocess(sentence)
            self.word_counter.update(sentence.split())

            # update triletter scanning
            tri_sentence = []
            for word in sentence.split():
                tri_word = []
                word = '#' + word + '#'

                for offset in range(0, len(word))[:-2]:
                    triletter = word[offset: offset + 3]
                    tri_word.append(triletter)
                    self.triletter_counter.update([triletter])

                tri_sentence.append(tri_word)

            self.triletter_corpus.append(tri_sentence)

        for i, word in enumerate(self.word_counter.keys()):
            self.word2int[word] = i
            self.int2word[i] = word

        for i, triletter in enumerate(self.triletter_counter.keys()):
            self.tri2index[triletter] = i
            self.index2tri[i] = triletter

        self.vocab_size = len(self.tri2index)
        logger.info("Vocab Build Complete")

    def sent2triletter_indexed_sent(self, sentence):
        """Converts a sentence to a triletter sentence
        Parameters
        ==========
        sentence:
            A list of words
        """
        triletter_sentence = []
        for word in sentence:
            tri_word = []
            word = '#' + word + '#'

            for offset in range(len(word))[:-2]:
                tri_word.append(self.tri2index[word[offset: offset + 3]])
                # TODO The above code might cause issues for out of vocabulary words
                # This branch has an error and needs to be fixed
                # logger.info("Found a tri not in dict: %s" % word[offset: offset + 3])
                # for adding unseen charaacter trigrams
                # self.tri2index[word[offset: offset + 3]] = self.vocab_size
                # self.index2tri[self.vocab_size] = word[offset: offset + 3]
                # self.vocab_size += 1
                # tri_word.append(self.tri2index[word[offset: offset + 3]])
            triletter_sentence.append(tri_word)

        return triletter_sentence

    def get_term_vector(self, sentence):
        """Converts a sentence into its term vector to be pushed into the neural network
        Parameters
        ==========
        sentence:
            A list of words
        Example:
        ========
        >> get_term_vector("how are glaciers formed ?".split())
        [1, 0, 23, 53, ..., 12]
        """
        # TODO check if vocab has been built

        indexed_triletter_sentence = self.sent2triletter_indexed_sent(sentence)

        # TODO vocab size may change due to unseen tris
        # bad coding practice?
        # this line has to be after the above line
        vector = np.zeros(self.vocab_size)

        for word in indexed_triletter_sentence:
            for triletter in word:
                vector[triletter] += 1
        return vector.reshape(self.vocab_size,)

    def make_indexed_corpus(self):
        """Returns an indexed corpus instead of a word corpus
        """
        logger.info('making indexed triletter corpus')

        self.indexed_triletter_corpus = []
        for sentence in self.triletter_corpus:
            indexed_tri_sentence = []
            for word in sentence:
                indexed_tri_word = []
                for triletter in word:
                    indexed_tri_word.append(self.tri2indexed[triletter])
                indexed_tri_sentence.append(indexed_tri_word)
            self.indexed_triletter_corpus.append(indexed_tri_sentence)

        logger.info('indexed triletter corpus made')
        return self.indexed_triletter_corpus

    def get_X_y(self, batch_size=32):

        # TODO Implement batch sizing

        queries = []
        docs = []
        labels = []

        for Question, Answer in self.df.groupby('QuestionID').apply(dict).items():
            for q, d, l in zip(Answer['Question'], Answer['Sentence'], Answer['Label']):
                queries.append(self.get_term_vector(self.preprocess(q)))
                docs.append(self.get_term_vector(self.preprocess(d)))
                labels.append(l)

        queries = np.array(queries)
        docs = np.array(docs)
        labels = np.array(labels)
        return queries, docs, labels

    def get_data(self):
        self.questions = []
        for Question, Answer in self.df.groupby('QuestionID').apply(dict).items():

            document_group = []
            for q, d, l in zip(Answer['Question'], Answer['Sentence'], Answer['Label']):
                document_group.append(
                    [self.preprocess(q), self.preprocess(d), l])

            self.questions.append(document_group)
        return self.questions


class QuoraQPExtractor:
    """[WIP]Class to extract data from the Quora Duplicate question pairs dataset
    It provides data in a shapes which makes it easier to train neural networks

    Usage:
    =====
    quoraqp = QuoraQPExtractor("path/to/file")
    X_train, y_train = quoraqp.get_data()

    X_train consists of the question pairs
    """

    def __init__(self, file_path, embedding_path=None):
        if file_path is not None:
            with open(file_path) as f:
                self.df = pd.read_csv(f, sep='\t')
                self.q1 = list(self.df.iloc[:, 3])
                self.q2 = list(self.df.iloc[:, 4])
                self.isDuplicate = list(self.df.iloc[:, 5])
        else:
            raise NotImplementedError()

        self.word2index = {}
        self.index2word = {}

        self.corpus = self.q1 + self.q2
        self.word_counter = Counter()

        self.build_vocab()

    def preprocess(self, sentence):
        return re.sub("[^a-zA-Z0-9]", " ", sentence.lower())

    def build_vocab(self):
        logger.info("Starting Vocab Build")

        for sentence in self.corpus:
            sentence = self.preprocess(sentence)
            self.word_counter.update(sentence.split())

        for i, word in enumerate(self.word_counter.keys()):
            self.word2index[word] = i
            self.index2word[i] = word

        self.vocab_size = len(self.word2index)

        logger.info("Vocab Build Complete")

    def get_preprocessed_corpus(self):
        preprocessed_corpus = []
        for sent in self.corpus:
            preprocessed_corpus.append(self.preprocess(sent))

        return preprocessed_corpus

    def get_data(self):

        question_pairs = []
        labels = []

        for Question1, Question2, label in zip(self.q1, self.q2, self.isDuplicate):
            question_pairs.append(
                [self.preprocess(Question1), self.preprocess(Question2)])
            labels.append(int(label))

        return question_pairs, labels
