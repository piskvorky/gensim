from __future__ import unicode_literals
from __future__ import print_function

import logging
import codecs

import gensim.downloader as api
from gensim.parsing.preprocessing import preprocess_string


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    data = api.load('wiki-english-20171001')

    fout = codecs.open('gensim-enwiki.txt', 'w', encoding='utf8')
    for i, article in enumerate(data):
        for section in article['section_texts']:
            fout.write(' '.join(preprocess_string(section)) + '\n')

        if (i + 1) % 10000 == 0:
            logger.info('Processed {} articles.'.format(i + 1))

        i += 1

    fout.close()