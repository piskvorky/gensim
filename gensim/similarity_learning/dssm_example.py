from gensim.similarity_learning import DSSM
from gensim.similarity_learning import WikiQAExtractor
import os
import logging

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    logging.basicConfig(
            format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
            level=logging.INFO
        )

    wiki_extractor = WikiQAExtractor(os.path.join("data", "WikiQACorpus", "WikiQA-train.tsv"))

    queries, docs, labels = wiki_extractor.get_X_y(batch_size=32)

    logger.info("There are a total of %d query, document pairs in the dataset" % len(queries))

    dssm = DSSM(vocab_size=wiki_extractor.vocab_size)
    dssm.train(queries, docs, labels, epochs=2)

    term_vec1 = wiki_extractor.get_term_vector("how a water pump works".split())
    term_vec2 = wiki_extractor.get_term_vector("Pumps operate by some mechanism ( typically reciprocating or rotary ) ,\
     and consume energy to perform mechanical work by moving the fluid".split())

    # TODO currently doesn't take the shapes. Needs fix.
    # print(dssm.predict(term_vec1, term_vec2))
