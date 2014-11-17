import gensim
import os
from gensim import corpora
from gensim import utils

class DtmCorpus(corpora.textcorpus.TextCorpus):
        def get_texts(self):
            return self.input

        def __len__(self):
            return len(self.input)


if __name__ == '__main__':
    corpus, time_seq = utils.unpickle('gensim/test/test_data/dtm_test')

    dtm_home = os.environ.get('DTM_HOME', "C:/Users/Artyom/SkyDrive/TopicModels/dtm-master/")
    dtm_path = os.path.join(dtm_home, 'bin', 'dtm') if dtm_home else None

    model = gensim.models.DtmModel(dtm_path, corpus, time_seq, num_topics=2, id2word=corpus.dictionary)
    topics = model.show_topics(topics=2, times=2, topn=10)
