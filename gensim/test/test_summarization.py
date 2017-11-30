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

    def _get_text_from_test_data(self, file):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')
        with utils.smart_open(os.path.join(pre_path, file), mode="r") as f:
            return f.read()

    def test_text_summarization(self):
        text = self._get_text_from_test_data("mihalcea_tarau.txt")

        # Makes a summary of the text.
        generated_summary = summarize(text)

        # To be compared to the method reference.
        summary = self._get_text_from_test_data("mihalcea_tarau.summ.txt")

        self.assertEqual(generated_summary, summary)

    def test_corpus_summarization(self):
        text = self._get_text_from_test_data("mihalcea_tarau.txt")

        # Generate the corpus.
        sentences = text.split("\n")
        tokens = [sentence.split() for sentence in sentences]
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(sentence_tokens) for sentence_tokens in tokens]

        # Extract the most important documents.
        selected_documents = summarize_corpus(corpus)

        # They are compared to the method reference.
        summary = self._get_text_from_test_data("mihalcea_tarau.summ.txt")
        summary = summary.split('\n')

        # Each sentence in the document selection has to be in the model summary.
        for doc_number, document in enumerate(selected_documents):
            # Retrieves all words from the document.
            words = [dictionary[token_id] for (token_id, count) in document]

            # Asserts that all of them are in a sentence from the model reference.
            self.assertTrue(any(all(word in sentence for word in words)) for sentence in summary)

    def test_summary_from_unrelated_sentences(self):
        # Tests that the summarization of a text with unrelated sentences is not empty string.
        text = self._get_text_from_test_data("testsummarization_unrelated.txt")
        generated_summary = summarize(text)
        self.assertNotEqual(generated_summary, u"")

    def test_text_summarization_on_short_input_text_is_empty_string(self):
        text = self._get_text_from_test_data("testsummarization_unrelated.txt")

        # Keeps the first 8 sentences to make the text shorter.
        text = "\n".join(text.split('\n')[:8])

        self.assertNotEqual(summarize(text), u"")

    def test_text_summarization_raises_exception_on_single_input_sentence(self):
        text = self._get_text_from_test_data("testsummarization_unrelated.txt")

        # Keeps the first sentence only.
        text = text.split('\n')[0]

        self.assertRaises(ValueError, summarize, text)

    def test_corpus_summarization_is_not_empty_list_on_short_input_text(self):
        text = self._get_text_from_test_data("testsummarization_unrelated.txt")

        # Keeps the first 8 sentences to make the text shorter.
        sentences = text.split('\n')[:8]

        # Generate the corpus.
        tokens = [sentence.split() for sentence in sentences]
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(sentence_tokens) for sentence_tokens in tokens]

        self.assertNotEqual(summarize_corpus(corpus), [])

    def test_empty_text_summarization_is_empty_string(self):
        self.assertEqual(summarize(""), u"")

    def test_empty_text_summarization_with_split_is_empty_list(self):
        self.assertEqual(summarize("", split=True), [])

    def test_empty_corpus_summarization_is_empty_list(self):
        self.assertEqual(summarize_corpus([]), [])

    def test_corpus_summarization_ratio(self):
        text = self._get_text_from_test_data("mihalcea_tarau.txt")

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

    def test_repeated_keywords(self):
        text = self._get_text_from_test_data("testrepeatedkeywords.txt")

        kwds = keywords(text)
        self.assertTrue(len(kwds.splitlines()))

        kwds_u = keywords(utils.to_unicode(text))
        self.assertTrue(len(kwds_u.splitlines()))

        kwds_lst = keywords(text, split=True)
        self.assertTrue(len(kwds_lst))

    def test_keywords_runs(self):
        text = self._get_text_from_test_data("mihalcea_tarau.txt")

        kwds = keywords(text)
        self.assertTrue(len(kwds.splitlines()))

        kwds_u = keywords(utils.to_unicode(text))
        self.assertTrue(len(kwds_u.splitlines()))

        kwds_lst = keywords(text, split=True)
        self.assertTrue(len(kwds_lst))

    def test_mz_keywords(self):
        pre_path = os.path.join(os.path.dirname(__file__), 'test_data')

        with utils.smart_open(os.path.join(pre_path, "head500.noblanks.cor")) as f:
            text = utils.to_unicode(f.read())
        text = u' '.join(text.split()[:10240])
        kwds = mz_keywords(text)
        self.assertTrue(kwds.startswith('autism'))
        self.assertTrue(kwds.endswith('uk'))
        self.assertTrue(len(kwds.splitlines()))

        kwds_lst = mz_keywords(text, split=True)
        self.assertTrue(len(kwds_lst))
        # Automatic thresholding selects words with n_blocks / n_blocks+1
        # bits of entropy. For this text, n_blocks=10
        n_blocks = 10.
        kwds_auto = mz_keywords(text, scores=True, weighted=False, threshold='auto')
        self.assertTrue(kwds_auto[-1][1] > (n_blocks / n_blocks + 1.))

    def test_low_distinct_words_corpus_summarization_is_empty_list(self):
        text = self._get_text_from_test_data("testlowdistinctwords.txt")

        # Generate the corpus.
        sentences = text.split("\n")
        tokens = [sentence.split() for sentence in sentences]
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(sentence_tokens) for sentence_tokens in tokens]

        self.assertEqual(summarize_corpus(corpus), [])

    def test_low_distinct_words_summarization_is_empty_string(self):
        text = self._get_text_from_test_data("testlowdistinctwords.txt")
        self.assertEqual(summarize(text), u"")

    def test_low_distinct_words_summarization_with_split_is_empty_list(self):
        text = self._get_text_from_test_data("testlowdistinctwords.txt")
        self.assertEqual(summarize(text, split=True), [])


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
