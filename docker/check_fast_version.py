import sys

try:
    from gensim.models.word2vec_inner import train_batch_sg, train_batch_cbow
    from gensim.models.word2vec_inner import score_sentence_sg, score_sentence_cbow
    from gensim.models.word2vec_inner import FAST_VERSION, MAX_WORDS_IN_BATCH

    print('FAST_VERSION ok ! Retrieved with value ', FAST_VERSION)
    sys.exit()
except ImportError:
    print('Failed... fall back to plain numpy (20-80x slower training than the above)')
    sys.exit(-1)
