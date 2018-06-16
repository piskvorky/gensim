# from gensim.similarity_learning.models import DRMM
from gensim.similarity_learning.preprocessing import WikiQA_DRMM_TKS_Extractor
import os
import logging
import argparse
from keras.optimizers import get
from gensim.similarity_learning import rank_hinge_loss
from keras.losses import hinge
from gensim.similarity_learning.preprocessing import ListGenerator

logger = logging.getLogger(__name__)

"""Proof of Concept/Example script to demonstrate the trianing of DSSM model
Note: This is just training currently. Validation and Testing currently missing

Example Usage:
    $  python drmm_example.py --wikiqa_folder_path ./data/WikiQACorpus/ --word_embedding_path data/glove.6B.50d.txt
"""

if __name__ == '__main__':

    logging.basicConfig(
            format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
            level=logging.INFO
        )

    parser = argparse.ArgumentParser()
    parser.add_argument('--wikiqa_folder_path',
                        help='path to the the folder with WikiQACorpus')
    parser.add_argument('--word_embedding_path',
                        help='path to the Glove embedding .txt')
    args = parser.parse_args()

    # Raise an error if params aren't passed
    if not (args.wikiqa_folder_path or args.word_embedding_path):
        parser.error('Please specify --wikiqa_folder_path and --word_embedding_path')

    datapath = args.wikiqa_folder_path
    word_embedding_path = args.word_embedding_path

    train_file_path = os.path.join(datapath, 'WikiQA-train.tsv')
    test_file_path = os.path.join(datapath, 'WikiQA-test.tsv')
    dev_file_path = os.path.join(datapath, 'WikiQA-dev.tsv')

    hist_size = 60

    wikiqa_train = WikiQA_DRMM_TKS_Extractor(file_path=train_file_path, word_embedding_path=word_embedding_path,
                                             keep_full_embedding=True, text_maxlen=140, hist_size=hist_size, normalize_embeddings=True)

    data = wikiqa_train.get_data()

    print(data)


    # dev_list_gen = ListGenerator(dev_file_path, text_maxlen=wikiqa_train.text_maxlen,
    #                              train_word2index=wikiqa_train.word2index,
    #                              additional_word2index=wikiqa_train.additional_word2index,
    #                              oov_handle_method="ignore", zero_word_index=wikiqa_train.zero_word_index,
    #                              train_pad_word_index=wikiqa_train.pad_word_index, hist_size=hist_size, embedding_matrix=wikiqa_train.embedding_matrix)

    # test_list_gen = ListGenerator(test_file_path, text_maxlen=wikiqa_train.text_maxlen,
    #                               train_word2index=wikiqa_train.word2index,
    #                               additional_word2index=wikiqa_train.additional_word2index,
    #                               oov_handle_method="ignore", zero_word_index=wikiqa_train.zero_word_index,
    #                               train_pad_word_index=wikiqa_train.pad_word_index, hist_size=hist_size,
    #                               embedding_matrix=wikiqa_train.embedding_matrix)


    # data = test_list_gen.get_list_data()
    # X1 = data["X1"]
    # X2 = data["X2"]
    # y = data["y"]
    # doc_lengths = data["doc_lengths"]

    # print(X2, X2.shape)


    # X1_train, X2_train, y_train = wikiqa_train.get_full_batch()
    # # print(X1_train, X1_train.shape)
    # # print(X2_train, X2_train.shape)

    # drmm = DRMM(wikiqa_train.text_maxlen, wikiqa_train.vocab_size, wikiqa_train.embedding_matrix)
    # model = drmm.get_model()

    # model.summary()

    # optimizer = 'adadelta'
    # optimizer = 'adam'
    # optimizer = optimizers.get(optimizer)
    # K.set_value(optimizer.lr, 0.0001)

    # # either one can be selected. Currently, the choice is manual.
    # loss = rank_hinge_loss
    # loss = hinge
    # loss = 'mse'
    # # validation_data = dev_list_gen.get_list_data()

    # model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    # model.fit(x={"query": X1_train, "doc": X2_train}, y=y_train, batch_size=5,
    #           verbose=1, epochs=1, shuffle=True)

    # data = test_list_gen.get_list_data()
    # X1 = data["X1"]
    # X2 = data["X2"]
    # y = data["y"]
    # doc_lengths = data["doc_lengths"]

    # predictions = model.predict(x={"query": X1, "doc": X2})

    # Y_pred = []
    # Y_true = []
    # offset = 0

    # for doc_size in doc_lengths:
    #     Y_pred.append(predictions[offset: offset + doc_size])
    #     Y_true.append(y[offset: offset + doc_size])
    #     offset += doc_size

