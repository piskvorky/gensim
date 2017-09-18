import logging
import unittest

from gensim.corpora.stateful_pool import StatefulProcessor, StatefulProcessingPool


class _TextTokenizer(StatefulProcessor):
    def process(self, text):
        return text.split()


class TestStatefulPool(unittest.TestCase):

    def test_simple_tokenize_example(self):
        pool = StatefulProcessingPool(4, processor_class=_TextTokenizer)
        texts = ['this is some test text for multiprocessing'] * 10
        expected = [text.split() for text in texts]

        results = pool.imap(texts)
        self.assertEquals(list(results), expected)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
