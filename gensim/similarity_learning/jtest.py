from sl_vocab import WikiQA_DRMM_TKS_Extractor
from drmm_tks import DRMM_TKS
from custom_losses import rank_hinge_loss
import os
import argparse
from custom_callbacks import SLCallback

"""Jboy's test script to make sure training is happening properly and the model validation takes place!

Example Usage:
$ python jtest.py --wikiqa_folder_path ./data/WikiQACorpus/ --word_embedding_path evaluation_scripts/glove.6B.50d.txt  # noqa
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--wikiqa_folder_path', help='path to the the folder with WikiQACorpus')
    parser.add_argument('--word_embedding_path', help='the path to the glove word embedding .txt')
    args = parser.parse_args()

    # Raise an error if params aren't passed
    if not (args.wikiqa_folder_path or args.word_embedding_path):
        parser.error('Please specify --wikiqa_folder_path and --word_embedding_path')

    train_file_path = os.path.join(args.wikiqa_folder_path, 'WikiQA-train.tsv')
    valid_file_path = os.path.join(args.wikiqa_folder_path, 'WikiQA-dev.tsv')
    word_embedding_path = args.word_embedding_path

    batch_size = 32

    wikiqa_train = WikiQA_DRMM_TKS_Extractor(file_path=train_file_path, word_embedding_path=word_embedding_path)
    training_generator = wikiqa_train.jbatch_gen(batch_size)

    wikiqa_valid = WikiQA_DRMM_TKS_Extractor(file_path=valid_file_path, word_embedding_path=word_embedding_path)
    validation_data = wikiqa_valid.get_full_batch(batch_size)

    drmm_tks = DRMM_TKS(embedding=wikiqa_train.embedding_matrix, vocab_size=wikiqa_train.vocab_size + 1,
                        text_maxlen=wikiqa_train.text_maxlen)
    model = drmm_tks.get_model()
    optimizer = 'adadelta'
    loss = 'mse'

    steps_per_epoch = 64

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    model.fit_generator(generator=training_generator, steps_per_epoch=steps_per_epoch, epochs=10,
                        shuffle=False, verbose=1, callbacks=[SLCallback(validation_data)])