import unittest
import numpy as np
from gensim.models import word2vec

try:
    from sklearn.datasets import fetch_20newsgroups
except ImportError:
    raise unittest.SkipTest("Test requires sklearn to be installed, which is not available")

try:
    import keras
    from keras.engine import Input
    from keras.models import Model
    from keras.layers.merge import dot
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils.np_utils import to_categorical
    from keras.layers import Dense, Flatten
    from keras.layers import Conv1D, MaxPooling1D
except ImportError:
    raise unittest.SkipTest("Test requires Keras to be installed, which is not available")

from gensim.test.utils import common_texts


class TestKerasWord2VecWrapper(unittest.TestCase):
    def setUp(self):
        self.model_cos_sim = word2vec.Word2Vec(common_texts, size=100, min_count=1, hs=1)
        self.model_twenty_ng = word2vec.Word2Vec(min_count=1)

    def testWord2VecTraining(self):
        """
        Test word2vec training.
        """
        model = self.model_cos_sim
        self.assertTrue(model.wv.vectors.shape == (len(model.wv.vocab), 100))
        self.assertTrue(model.trainables.syn1.shape == (len(model.wv.vocab), 100))
        sims = model.wv.most_similar('graph', topn=10)
        # self.assertTrue(sims[0][0] == 'trees', sims)  # most similar

        # test querying for "most similar" by vector
        graph_vector = model.wv.vectors_norm[model.wv.vocab['graph'].index]
        sims2 = model.wv.most_similar(positive=[graph_vector], topn=11)
        sims2 = [(w, sim) for w, sim in sims2 if w != 'graph']  # ignore 'graph' itself
        self.assertEqual(sims, sims2)

    def testEmbeddingLayerCosineSim(self):
        """
        Test Keras 'Embedding' layer returned by 'get_embedding_layer' function for a simple word similarity task.
        """
        keras_w2v_model = self.model_cos_sim
        keras_w2v_model_wv = keras_w2v_model.wv

        embedding_layer = keras_w2v_model_wv.get_keras_embedding()

        input_a = Input(shape=(1,), dtype='int32', name='input_a')
        input_b = Input(shape=(1,), dtype='int32', name='input_b')
        embedding_a = embedding_layer(input_a)
        embedding_b = embedding_layer(input_b)
        similarity = dot([embedding_a, embedding_b], axes=2, normalize=True)

        model = Model(input=[input_a, input_b], output=similarity)
        model.compile(optimizer='sgd', loss='mse')

        word_a = 'graph'
        word_b = 'trees'
        output = model.predict([
            np.asarray([keras_w2v_model.wv.vocab[word_a].index]),
            np.asarray([keras_w2v_model.wv.vocab[word_b].index])
        ])
        # output is the cosine distance between the two words (as a similarity measure)

        self.assertTrue(type(output[0][0][0]) == np.float32)     # verify that  a float is returned

    def testEmbeddingLayer20NewsGroup(self):
        """
        Test Keras 'Embedding' layer returned by 'get_embedding_layer' function
        for a smaller version of the 20NewsGroup classification problem.
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
            except Exception:
                pass

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
        keras_w2v = self.model_twenty_ng
        keras_w2v.build_vocab(texts_w2v)
        keras_w2v.train(texts, total_examples=keras_w2v.corpus_count, epochs=keras_w2v.epochs)
        keras_w2v_wv = keras_w2v.wv
        embedding_layer = keras_w2v_wv.get_keras_embedding()

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
        fit_ret_val = model.fit(x_train, y_train, epochs=1)

        # verify the type of the object returned after training
        # value returned is a `History` instance.
        # Its `history` attribute contains all information collected during training.
        self.assertTrue(type(fit_ret_val) == keras.callbacks.History)


if __name__ == '__main__':
    unittest.main()
