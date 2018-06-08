from jlist_generator import JListGenerator
from sl_vocab import WikiQA_DRMM_TKS_Extractor
import os
import argparse
import numpy as np
from pprint import pprint
from drmm_tks import DRMM_TKS
from custom_losses import rank_hinge_loss
"""
Test script for JListGen

Example Usage:
$ python jlist_test.py --wikiqa_folder_path ./data/WikiQACorpus/ --word_embedding_path evaluation_scripts/glove.6B.50d.txt  # noqa
"""


def get_model(wikiqa):	
    drmm_tks = DRMM_TKS(embedding=wikiqa.embedding_matrix, vocab_size=wikiqa.vocab_size + 1,
                        text_maxlen=wikiqa.text_maxlen)
    model = drmm_tks.get_model()

    optimizer = 'adadelta'
    loss = rank_hinge_loss
    display_interval = 100

    model.compile(optimizer=optimizer, loss=rank_hinge_loss, metrics=['accuracy'])

    generator_function = wikiqa.get_batch_generator(batch_size=32)
    history = model.fit_generator(generator_function, steps_per_epoch=display_interval, epochs=1,
                                  shuffle=False, verbose=1)
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--wikiqa_folder_path', help='path to the the folder with WikiQACorpus')
    parser.add_argument('--word_embedding_path', help='the path to the glove word embedding .txt')
    args = parser.parse_args()

    # Raise an error if params aren't passed
    if not (args.wikiqa_folder_path or args.word_embedding_path):
        parser.error('Please specify --wikiqa_folder_path and --word_embedding_path')

    train_file_path = os.path.join(args.wikiqa_folder_path, 'WikiQA-train.tsv')
    word_embedding_path = args.word_embedding_path

    list_generator = JListGenerator(train_file_path, word_embedding_path, text_maxlen=200)

    wikiqa = WikiQA_DRMM_TKS_Extractor(file_path=train_file_path, word_embedding_path=word_embedding_path, text_maxlen=200)
    model = get_model(wikiqa)
    print(model)

    data = list_generator.get_data_long()
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
    	offset += doc_size

