#!/usr/bin/env python
# encoding: utf-8
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated test to reproduce the results of Mihalcea and Tarau (2004).

Mihalcea and Tarau (2004) introduces the TextRank summarization algorithm.
As a validation of the gensim implementation we reproduced its results
in this test.

"""

import os.path
import logging
import unittest

from gensim import utils
from gensim.corpora import Dictionary
from gensim.summarization import summarize, summarize_corpus, keywords, mz_keywords


class TestSummarizationTest(unittest.TestCase):

    def test_text_summarization(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.smart_open(os.path.join(pre_path, "mihalcea_tarau.txt"), mode="r") as f:
            text = f.read()

        # Makes a summary of the text.
        generated_summary = summarize(text)

        # To be compared to the method reference.
        with utils.smart_open(os.path.join(pre_path, "mihalcea_tarau.summ.txt"), mode="r") as f:
            summary = f.read()

        self.assertEqual(generated_summary, summary)

    def test_corpus_summarization(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.smart_open(os.path.join(pre_path, "mihalcea_tarau.txt"), mode="r") as f:
            text = f.read()

        # Generate the corpus.
        sentences = text.split("\n")
        tokens = [sentence.split() for sentence in sentences]
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(sentence_tokens) for sentence_tokens in tokens]

        # Extract the most important documents.
        selected_documents = summarize_corpus(corpus)

        # They are compared to the method reference.
        with utils.smart_open(os.path.join(pre_path, "mihalcea_tarau.summ.txt"), mode="r") as f:
            summary = f.read()
            summary = summary.split('\n')

        # Each sentence in the document selection has to be in the model summary.
        for doc_number, document in enumerate(selected_documents):
            # Retrieves all words from the document.
            words = [dictionary[token_id] for (token_id, count) in document]

            # Asserts that all of them are in a sentence from the model reference.
            self.assertTrue(any(all(word in sentence for word in words)) for sentence in summary)

    def test_summary_from_unrelated_sentences(self):
        # Tests that the summarization of a text with unrelated sentences does not raise an exception.
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.smart_open(os.path.join(pre_path, "testsummarization_unrelated.txt"), mode="r") as f:
            text = f.read()

        generated_summary = summarize(text)

        self.assertNotEqual(generated_summary, None)

    def test_text_summarization_raises_exception_on_short_input_text(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.smart_open(os.path.join(pre_path, "testsummarization_unrelated.txt"), mode="r") as f:
            text = f.read()

        # Keeps the first 8 sentences to make the text shorter.
        text = "\n".join(text.split('\n')[:8])

        self.assertTrue(summarize(text) is not None)

    def test_corpus_summarization_raises_exception_on_short_input_text(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.smart_open(os.path.join(pre_path, "testsummarization_unrelated.txt"), mode="r") as f:
            text = f.read()

        # Keeps the first 8 sentences to make the text shorter.
        sentences = text.split('\n')[:8]

        # Generate the corpus.
        tokens = [sentence.split() for sentence in sentences]
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(sentence_tokens) for sentence_tokens in tokens]

        self.assertTrue(summarize_corpus(corpus) is not None)

    def test_empty_text_summarization_none(self):
        self.assertTrue(summarize("") is None)

    def test_empty_corpus_summarization_is_none(self):
        self.assertTrue(summarize_corpus([]) is None)

    def test_corpus_summarization_ratio(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.smart_open(os.path.join(pre_path, "mihalcea_tarau.txt"), mode="r") as f:
            text = f.read()

        # Generate the corpus.
        sentences = text.split('\n')
        tokens = [sentence.split() for sentence in sentences]
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(sentence_tokens) for sentence_tokens in tokens]

        # Makes summaries of the text using different ratio parameters.
        for x in range(1, 10):
            ratio = x / float(10)
            selected_docs = summarize_corpus(corpus, ratio=ratio)
            expected_summary_length = int(len(corpus) * ratio)

            self.assertEqual(len(selected_docs), expected_summary_length)

    def test_keywords_runs(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.smart_open(os.path.join(pre_path, "mihalcea_tarau.txt")) as f:
            text = f.read()

        kwds = keywords(text)
        self.assertTrue(len(kwds.splitlines()))

        kwds_u = keywords(utils.to_unicode(text))
        self.assertTrue(len(kwds_u.splitlines()))

        kwds_lst = keywords(text, split=True)
        self.assertTrue(len(kwds_lst))
        
    def test_mz_keywords(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.smart_open(os.path.join(pre_path, "head500.noblanks.cor")) as f:
            text = f.read()

        kwds = mz_keywords(text)
        self.assertTrue(len(kwds.splitlines()))

        kwds_u = mz_keywords(utils.to_unicode(text))
        self.assertTrue(len(kwds_u.splitlines()))

        kwds_lst = mz_keywords(text, split=True)
        self.assertTrue(len(kwds_lst))
        

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
