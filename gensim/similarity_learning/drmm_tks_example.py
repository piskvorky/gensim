from sl_vocab import WikiQA_DRMM_TKS_Extractor
from drmm_tks import DRMM_TKS
from custom_losses import rank_hinge_loss
import os
import argparse


"""Proof of Concept/Example script to demonstrate the trianing of DRMM_TKS model
Note: This is just training currently. Validation and Testing currently missing

Example Usage:
$ python drmm_tks_example.py --wikiqa_folder_path ./data/WikiQACorpus/ --word_embedding_path evaluation_scripts/glove.6B.50d.txt  # noqa
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--wikiqa_folder_path', help='path to the the folder with WikiQACorpus')
    parser.add_argument('--word_embedding_path', help='the path to the glove word embedding .txt')
    args = parser.parse_args()

    # Raise an error if params aren't passed
    if not (args.wikiqa_folder_path or args.word_embedding_path):
        parser.error('Please specify --wikiqa_folder_path and --word_embedding_path')

    file_path = os.path.join(args.wikiqa_folder_path, 'WikiQA-train.tsv')
    word_embedding_path = os.path.join('evaluation_scripts', 'glove.6B.50d.txt')

    wikiqa = WikiQA_DRMM_TKS_Extractor(file_path=file_path, word_embedding_path=word_embedding_path)

    drmm_tks = DRMM_TKS(embedding=wikiqa.embedding_matrix, vocab_size=wikiqa.vocab_size + 1,
                        text_maxlen=wikiqa.text_maxlen)

    model = drmm_tks.get_model()

    optimizer = 'adadelta'
    loss = rank_hinge_loss
    display_interval = 100

    model.compile(optimizer=optimizer, loss=rank_hinge_loss, metrics=['accuracy'])

    generator_function = wikiqa.get_batch_generator(batch_size=32)
    history = model.fit_generator(generator_function, steps_per_epoch=display_interval, epochs=10,
                                  shuffle=False, verbose=1)
