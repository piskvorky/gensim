from gensim.similarity_learning import DSSM
from gensim.similarity_learning import WikiQAExtractor
import os
import logging
import argparse

logger = logging.getLogger(__name__)

"""Proof of Concept/Example script to demonstrate the trianing of DSSM model
Note: This is just training currently. Validation and Testing currently missing

Example Usage:
$ python dssm_example.py --wikiqa_folder_path ./data/WikiQACorpus/
"""

if __name__ == '__main__':

    logging.basicConfig(
            format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
            level=logging.INFO
        )

    parser = argparse.ArgumentParser()
    parser.add_argument('--wikiqa_folder_path',
                        help='path to the the folder with WikiQACorpus')
    args = parser.parse_args()

    # Raise an error if params aren't passed
    if not (args.wikiqa_folder_path):
        parser.error('Please specify --wikiqa_folder_path')

    wikiqa_path = os.path.join(args.wikiqa_folder_path, "WikiQA-train.tsv")
    wiki_extractor = WikiQAExtractor(wikiqa_path)

    queries, docs, labels = wiki_extractor.get_X_y(batch_size=32)

    logger.info("There are a total of %d query, document pairs in the dataset" % len(queries))

    dssm = DSSM(vocab_size=wiki_extractor.vocab_size)
    dssm.train(queries, docs, labels, epochs=2)

    term_vec1 = wiki_extractor.get_term_vector("how a water pump works".split())
    term_vec2 = wiki_extractor.get_term_vector("Pumps operate by some mechanism ( typically reciprocating or rotary ) ,\
     and consume energy to perform mechanical work by moving the fluid".split())

    # TODO currently doesn't take the shapes. Needs fix.
    # print(dssm.predict(term_vec1, term_vec2))
