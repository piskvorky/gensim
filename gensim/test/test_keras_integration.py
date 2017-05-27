import unittest
import os
import numpy as np
from gensim.models import word2vec

try:
    from sklearn.datasets import fetch_20newsgroups
except:
    raise unittest.SkipTest("Test requires sklearn to be installed, which is not available")

try:
    import keras
    from keras.engine import Input
    from keras.models import Model
    from keras.layers import merge
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils.np_utils import to_categorical
    from keras.layers import Dense, Flatten
    from keras.layers import Conv1D, MaxPooling1D
except:
    raise unittest.SkipTest("Test requires Keras to be installed, which is not available")

sentences = [
    ['human', 'interface', 'computer'],
    ['survey', 'user', 'computer', 'system', 'response', 'time'],
    ['eps', 'user', 'interface', 'system'],
    ['system', 'human', 'system', 'eps'],
    ['user', 'response', 'time'],
    ['trees'],
    ['graph', 'trees'],
    ['graph', 'minors', 'trees'],
    ['graph', 'minors', 'survey']
]

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


class TestKerasWord2VecWrapper(unittest.TestCase):
    def setUp(self):
        self.model_cos_sim = word2vec.Word2Vec(sentences, size=100, min_count=1, hs=1)
        # self.model_twenty_ng = word2vec.Word2Vec(word2vec.LineSentence(datapath('20_newsgroup_keras_w2v_data.txt')), min_count=1)
        self.model_twenty_ng = word2vec.Word2Vec(min_count=1)

    def testWord2VecTraining(self):
        """
        Test word2vec training.
        """
        model = self.model_cos_sim
        self.assertTrue(model.wv.syn0.shape == (len(model.wv.vocab), 100))
        self.assertTrue(model.syn1.shape == (len(model.wv.vocab), 100))
        sims = model.most_similar('graph', topn=10)
        # self.assertTrue(sims[0][0] == 'trees', sims)  # most similar

        # test querying for "most similar" by vector
        graph_vector = model.wv.syn0norm[model.wv.vocab['graph'].index]
        sims2 = model.most_similar(positive=[graph_vector], topn=11)
        sims2 = [(w, sim) for w, sim in sims2 if w != 'graph']  # ignore 'graph' itself
        self.assertEqual(sims, sims2)

    def testEmbeddingLayerCosineSim(self):
        """
        Test Keras 'Embedding' layer returned by 'get_embedding_layer' function for a simple word similarity task.
        """
        keras_w2v_model = self.model_cos_sim
        keras_w2v_model_wv = keras_w2v_model.wv

        embedding_layer = keras_w2v_model_wv.get_embedding_layer()

        input_a = Input(shape=(1,), dtype='int32', name='input_a')
        input_b = Input(shape=(1,), dtype='int32', name='input_b')
        embedding_a = embedding_layer(input_a)
        embedding_b = embedding_layer(input_b)
        similarity = merge([embedding_a, embedding_b], mode='cos', dot_axes=2)

        model = Model(input=[input_a, input_b], output=similarity)
        model.compile(optimizer='sgd', loss='mse')

        word_a = 'graph'
        word_b = 'trees'
        output = model.predict([np.asarray([keras_w2v_model.wv.vocab[word_a].index]), np.asarray([keras_w2v_model.wv.vocab[word_b].index])])    # probability of the two words occuring together
        self.assertTrue(type(output[0][0][0][0]) == np.float32)     # verify that  a float is returned

    def testEmbeddingLayer20NewsGroup(self):
        """
        Test Keras 'Embedding' layer returned by 'get_embedding_layer' function for a smaller version of the 20NewsGroup classification problem.
        """
        MAX_SEQUENCE_LENGTH = 1000

        # Prepare text samples and their labels

        # Processing text dataset
        texts = []  # list of text samples
        texts_w2v = []  # used to train the word embeddings
        labels = []  # list of label ids

        data = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'comp.graphics', 'sci.space'])
        for index in range(len(data)):
            label_id = data.target[index]
            file_data = data.data[index]
            i = file_data.find('\n\n')  # skip header
            if i > 0:
                file_data = file_data[i:]
            try:
                curr_str = str(file_data)
                sentence_list = curr_str.split('\n')
                for sentence in sentence_list:
                    sentence = (sentence.strip()).lower()
                    texts.append(sentence)
                    texts_w2v.append(sentence.split(' '))
                    labels.append(label_id)
            except:
                None

        # Vectorize the text samples into a 2D integer tensor
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)

        # word_index = tokenizer.word_index
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        labels = to_categorical(np.asarray(labels))

        x_train = data
        y_train = labels

        # prepare the embedding layer using the wrapper
        Keras_w2v = self.model_twenty_ng
        Keras_w2v.build_vocab(texts_w2v)
        Keras_w2v.train(texts, total_examples=Keras_w2v.corpus_count, epochs=Keras_w2v.iter)
        Keras_w2v_wv = Keras_w2v.wv
        embedding_layer = Keras_w2v_wv.get_embedding_layer()

        # create a 1D convnet to solve our classification task
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Conv1D(128, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(35)(x)  # global max pooling
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(y_train.shape[1], activation='softmax')(x)

        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
        fit_ret_val = model.fit(x_train, y_train, epochs=2)

        # verify the type of the object returned after training
        self.assertTrue(type(fit_ret_val) == keras.callbacks.History)  # value returned is a `History` instance. Its `history` attribute contains all information collected during training.


if __name__ == '__main__':
    unittest.main()
