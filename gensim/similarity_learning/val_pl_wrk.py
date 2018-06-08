from sl_vocab import WikiQA_DRMM_TKS_Extractor
from drmm_tks import DRMM_TKS
from custom_losses import rank_hinge_loss
import os
import argparse
from custom_callbacks import SLCallback
import logging
import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding
from jlist_generator import JListGenerator
from metrics import mean_ndcg
from metrics import mapk

"""Jboy's test script to make sure training and validation can happen

Example Usage:
$ python val_pl_wrk.py --wikiqa_folder_path ./data/WikiQACorpus/ --word_embedding_path evaluation_scripts/glove.6B.50d.txt  # noqa
"""

if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--wikiqa_folder_path', help='path to the the folder with WikiQACorpus')
    parser.add_argument('--word_embedding_path', help='the path to the glove word embedding .txt')
    args = parser.parse_args()

    # Raise an error if params aren't passed
    if not (args.wikiqa_folder_path or args.word_embedding_path):
        parser.error('Please specify --wikiqa_folder_path and --word_embedding_path')

    train_file_path = os.path.join(args.wikiqa_folder_path, 'WikiQA-train.tsv')
    word_embedding_path = args.word_embedding_path

    wikiqa_train = WikiQA_DRMM_TKS_Extractor(file_path=train_file_path, word_embedding_path=word_embedding_path,
                                             keep_full_embedding=True, text_maxlen=140)

    print(wikiqa_train.vocab_size)
    print(wikiqa_train.pad_word_index)
    print(wikiqa_train.embedding_matrix.shape)

    test_file_path = os.path.join(args.wikiqa_folder_path, 'WikiQA-test.tsv')

    jlist_gen = JListGenerator(test_file_path, text_maxlen=wikiqa_train.text_maxlen,
        train_word2index=wikiqa_train.word2index, additional_word2index=wikiqa_train.additional_word2index,
        oov_handle_method="ignore", zero_word_index=wikiqa_train.zero_word_index, train_pad_word_index=wikiqa_train.pad_word_index)


    X1_train, X2_train, y_train = wikiqa_train.get_full_batch()
    drmm_tks = DRMM_TKS(embedding=wikiqa_train.embedding_matrix, vocab_size=wikiqa_train.embedding_matrix.shape[0],
                        text_maxlen=wikiqa_train.text_maxlen)

    model = drmm_tks.get_model()
    model.summary()

    optimizer = 'adadelta'
    loss = 'mse'


    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.fit(x={"query": X1_train, "doc": X2_train}, y=y_train, batch_size=64, verbose=1, epochs=10)

    data = jlist_gen.get_data_long()
    X1 = np.array(data["X1"])

    X2 = np.array(data["X2"])
    y = data["y"]
    doc_lengths = data["doc_lengths"]


    predictions = model.predict(x={"query": X1, "doc": X1})

    Y_predicted = []
    Y_true = []

    offset = 0

    for doc_size in doc_lengths:
        Y_predicted.append(predictions[offset: offset + doc_size])
        Y_true.append(y[offset: offset + doc_size])
        offset += doc_size

    print("MAP :", mapk(Y_true, Y_predicted))
    for k in [1, 3, 5, 10, 20]:
        print("nDCG@", str(k), " : ", mean_ndcg(Y_true, Y_predicted, k=k))





    


    