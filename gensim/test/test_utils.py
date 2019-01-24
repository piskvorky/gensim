#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking various utils functions.
"""
from __future__ import unicode_literals

import sys
import logging
import unittest

import numpy as np
from six import iteritems

from gensim import utils
from gensim.test.utils import datapath, get_tmpfile

import gensim.models.utils_any2vec

import smart_open

DISABLE_CYTHON_TESTS = getattr(gensim.models.utils_any2vec, 'FAST_VERSION', None) == -1


class TestIsCorpus(unittest.TestCase):
    def test_None(self):
        # test None
        result = utils.is_corpus(None)
        expected = (False, None)
        self.assertEqual(expected, result)

    def test_simple_lists_of_tuples(self):
        # test list words

        # one document, one word
        potentialCorpus = [[(0, 4.)]]
        result = utils.is_corpus(potentialCorpus)
        expected = (True, potentialCorpus)
        self.assertEqual(expected, result)

        # one document, several words
        potentialCorpus = [[(0, 4.), (1, 2.)]]
        result = utils.is_corpus(potentialCorpus)
        expected = (True, potentialCorpus)
        self.assertEqual(expected, result)

        potentialCorpus = [[(0, 4.), (1, 2.), (2, 5.), (3, 8.)]]
        result = utils.is_corpus(potentialCorpus)
        expected = (True, potentialCorpus)
        self.assertEqual(expected, result)

        # several documents, one word
        potentialCorpus = [[(0, 4.)], [(1, 2.)]]
        result = utils.is_corpus(potentialCorpus)
        expected = (True, potentialCorpus)
        self.assertEqual(expected, result)

        potentialCorpus = [[(0, 4.)], [(1, 2.)], [(2, 5.)], [(3, 8.)]]
        result = utils.is_corpus(potentialCorpus)
        expected = (True, potentialCorpus)
        self.assertEqual(expected, result)

    def test_int_tuples(self):
        potentialCorpus = [[(0, 4)]]
        result = utils.is_corpus(potentialCorpus)
        expected = (True, potentialCorpus)
        self.assertEqual(expected, result)

    def test_invalid_formats(self):
        # test invalid formats
        # these are no corpus, because they do not consists of 2-tuples with
        # the form(int, float).
        potentials = list()
        potentials.append(["human"])
        potentials.append("human")
        potentials.append(["human", "star"])
        potentials.append([1, 2, 3, 4, 5, 5])
        potentials.append([[(0, 'string')]])
        for noCorpus in potentials:
            result = utils.is_corpus(noCorpus)
            expected = (False, noCorpus)
            self.assertEqual(expected, result)


class TestUtils(unittest.TestCase):
    def test_decode_entities(self):
        # create a string that fails to decode with unichr on narrow python builds
        body = u'It&#146;s the Year of the Horse. YES VIN DIESEL &#128588; &#128175;'
        expected = u'It\x92s the Year of the Horse. YES VIN DIESEL \U0001f64c \U0001f4af'
        self.assertEqual(utils.decode_htmlentities(body), expected)

    def test_open_file_existent_file(self):
        number_of_lines_in_file = 30
        with utils.open_file(datapath('testcorpus.mm')) as infile:
            self.assertEqual(sum(1 for _ in infile), number_of_lines_in_file)

    def test_open_file_non_existent_file(self):
        with self.assertRaises(Exception):
            with utils.open_file('non_existent_file.txt'):
                pass

    def test_open_file_existent_file_object(self):
        number_of_lines_in_file = 30
        file_obj = open(datapath('testcorpus.mm'))
        with utils.open_file(file_obj) as infile:
            self.assertEqual(sum(1 for _ in infile), number_of_lines_in_file)

    def test_open_file_non_existent_file_object(self):
        file_obj = None
        with self.assertRaises(Exception):
            with utils.open_file(file_obj):
                pass


class TestSampleDict(unittest.TestCase):
    def test_sample_dict(self):
        d = {1: 2, 2: 3, 3: 4, 4: 5}
        expected_dict = [(1, 2), (2, 3)]
        expected_dict_random = [(k, v) for k, v in iteritems(d)]
        sampled_dict = utils.sample_dict(d, 2, False)
        self.assertEqual(sampled_dict, expected_dict)
        sampled_dict_random = utils.sample_dict(d, 2)
        if sampled_dict_random in expected_dict_random:
            self.assertTrue(True)


class TestTrimVocabByFreq(unittest.TestCase):
    def test_trim_vocab(self):
        d = {"word1": 5, "word2": 1, "word3": 2}
        expected_dict = {"word1": 5, "word3": 2}
        utils.trim_vocab_by_freq(d, topk=2)
        self.assertEqual(d, expected_dict)

        d = {"word1": 5, "word2": 2, "word3": 2, "word4": 1}
        expected_dict = {"word1": 5, "word2": 2, "word3": 2}
        utils.trim_vocab_by_freq(d, topk=2)
        self.assertEqual(d, expected_dict)


class TestMergeDicts(unittest.TestCase):
    def test_merge_dicts(self):
        d1 = {"word1": 5, "word2": 1, "word3": 2}
        d2 = {"word1": 2, "word3": 3, "word4": 10}

        res_dict = utils.merge_counts(d1, d2)
        expected_dict = {"word1": 7, "word2": 1, "word3": 5, "word4": 10}
        self.assertEqual(res_dict, expected_dict)


class TestWindowing(unittest.TestCase):

    arr10_5 = np.array([
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8],
        [5, 6, 7, 8, 9]
    ])

    def _assert_arrays_equal(self, expected, actual):
        self.assertEqual(expected.shape, actual.shape)
        self.assertTrue((actual == expected).all())

    def test_strided_windows1(self):
        out = utils.strided_windows(range(5), 2)
        expected = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4]
        ])
        self._assert_arrays_equal(expected, out)

    def test_strided_windows2(self):
        input_arr = np.arange(10)
        out = utils.strided_windows(input_arr, 5)
        expected = self.arr10_5.copy()
        self._assert_arrays_equal(expected, out)
        out[0, 0] = 10
        self.assertEqual(10, input_arr[0], "should make view rather than copy")

    def test_strided_windows_window_size_exceeds_size(self):
        input_arr = np.array(['this', 'is', 'test'], dtype='object')
        out = utils.strided_windows(input_arr, 4)
        expected = np.ndarray((0, 0))
        self._assert_arrays_equal(expected, out)

    def test_strided_windows_window_size_equals_size(self):
        input_arr = np.array(['this', 'is', 'test'], dtype='object')
        out = utils.strided_windows(input_arr, 3)
        expected = np.array([input_arr.copy()])
        self._assert_arrays_equal(expected, out)

    def test_iter_windows_include_below_window_size(self):
        texts = [['this', 'is', 'a'], ['test', 'document']]
        out = utils.iter_windows(texts, 3, ignore_below_size=False)
        windows = [list(w) for w in out]
        self.assertEqual(texts, windows)

        out = utils.iter_windows(texts, 3)
        windows = [list(w) for w in out]
        self.assertEqual([texts[0]], windows)

    def test_iter_windows_list_texts(self):
        texts = [['this', 'is', 'a'], ['test', 'document']]
        windows = list(utils.iter_windows(texts, 2))
        list_windows = [list(iterable) for iterable in windows]
        expected = [['this', 'is'], ['is', 'a'], ['test', 'document']]
        self.assertListEqual(list_windows, expected)

    def test_iter_windows_uses_views(self):
        texts = [np.array(['this', 'is', 'a'], dtype='object'), ['test', 'document']]
        windows = list(utils.iter_windows(texts, 2))
        list_windows = [list(iterable) for iterable in windows]
        expected = [['this', 'is'], ['is', 'a'], ['test', 'document']]
        self.assertListEqual(list_windows, expected)
        windows[0][0] = 'modified'
        self.assertEqual('modified', texts[0][0])

    def test_iter_windows_with_copy(self):
        texts = [
            np.array(['this', 'is', 'a'], dtype='object'),
            np.array(['test', 'document'], dtype='object')
        ]
        windows = list(utils.iter_windows(texts, 2, copy=True))

        windows[0][0] = 'modified'
        self.assertEqual('this', texts[0][0])

        windows[2][0] = 'modified'
        self.assertEqual('test', texts[1][0])

    def test_flatten_nested(self):
        nested_list = [[[1, 2, 3], [4, 5]], 6]
        expected = [1, 2, 3, 4, 5, 6]
        self.assertEqual(utils.flatten(nested_list), expected)

    def test_flatten_not_nested(self):
        not_nested = [1, 2, 3, 4, 5, 6]
        expected = [1, 2, 3, 4, 5, 6]
        self.assertEqual(utils.flatten(not_nested), expected)


class TestSaveAsLineSentence(unittest.TestCase):
    def test_save_as_line_sentence_en(self):
        corpus_file = get_tmpfile('gensim_utils.tst')
        ref_sentences = [l.split() for l in utils.any2unicode('hello world\nhow are you').split('\n')]

        utils.save_as_line_sentence(ref_sentences, corpus_file)

        with utils.smart_open(corpus_file, encoding='utf8') as fin:
            sentences = [line.strip().split() for line in fin.read().strip().split('\n')]
            self.assertEqual(sentences, ref_sentences)

    def test_save_as_line_sentence_ru(self):
        corpus_file = get_tmpfile('gensim_utils.tst')
        ref_sentences = [l.split() for l in utils.any2unicode('привет мир\nкак ты поживаешь').split('\n')]
        utils.save_as_line_sentence(ref_sentences, corpus_file)

        with utils.smart_open(corpus_file, encoding='utf8') as fin:
            sentences = [line.strip().split() for line in fin.read().strip().split('\n')]
            self.assertEqual(sentences, ref_sentences)


def hash_main(alg):
    """Generate hash values for test from standard input."""
    import sys
    import six

    assert six.PY3, 'this only works under Py3'
    assert not DISABLE_CYTHON_TESTS, 'this only works if Cython extensions available'

    hashmap = {
        'py_broken': gensim.models.utils_any2vec._ft_hash_broken_py,
        'py_bytes': gensim.models.utils_any2vec._ft_hash_bytes_py,
        'cy_broken': gensim.models.utils_any2vec._ft_hash_broken_py,
        'cy_bytes': gensim.models.utils_any2vec._ft_hash_bytes_cy,
    }
    try:
        fun = hashmap[alg]
    except KeyError:
        raise KeyError('invalid alg: %r expected one of %r' % (alg, sorted(hashmap)))

    for line in sys.stdin:
        if 'bytes' in alg:
            words = line.encode('utf-8').rstrip().split(b' ')
        else:
            words = line.rstrip().split(' ')
        for word in words:
            print('u%r: %r,' % (word, fun(word)))


class HashTest(unittest.TestCase):
    def setUp(self):
        #
        # I obtained these expected values using:
        #
        # $ echo word1 ... wordN | python -c 'from gensim.test.test_utils import hash_main;hash_main("alg")'  # noqa: E501
        #
        # where alg is one of py_bytes, py_broken, cy_bytes, cy_broken.

        #
        self.expected = {
            u'команда': 1725507386,
            u'маленьких': 3011324125,
            u'друзей': 737001801,
            u'возит': 4225261911,
            u'грузы': 1301826944,
            u'всех': 706328732,
            u'быстрей': 1379730754,
            u'mysterious': 1903186891,
            u'asteroid': 1988297200,
            u'odyssey': 310195777,
            u'introduction': 2848265721,
            u'北海道': 4096045468,
            u'札幌': 3909947444,
            u'西区': 3653372632,
        }
        self.expected_broken = {
            u'команда': 962806708,
            u'маленьких': 3633597485,
            u'друзей': 214728041,
            u'возит': 3590926132,
            u'грузы': 3674544745,
            u'всех': 3931012458,
            u'быстрей': 822471432,
            u'mysterious': 1903186891,
            u'asteroid': 1988297200,
            u'odyssey': 310195777,
            u'introduction': 2848265721,
            u'北海道': 4017049120,
            u'札幌': 1706980764,
            u'西区': 1113327900,
        }

    def test_python(self):
        actual = {k: gensim.models.utils_any2vec._ft_hash_bytes_py(k.encode('utf-8')) for k in self.expected}
        self.assertEqual(self.expected, actual)

    @unittest.skipIf(DISABLE_CYTHON_TESTS, 'Cython functions are not properly compiled')
    def test_cython(self):
        actual = {k: gensim.models.utils_any2vec._ft_hash_bytes_cy(k.encode('utf-8')) for k in self.expected}
        self.assertEqual(self.expected, actual)

    def test_python_broken(self):
        actual = {k: gensim.models.utils_any2vec._ft_hash_broken_py(k) for k in self.expected}
        self.assertEqual(self.expected_broken, actual)

    @unittest.skipIf(DISABLE_CYTHON_TESTS, 'Cython functions are not properly compiled')
    def test_cython_broken(self):
        actual = {k: gensim.models.utils_any2vec._ft_hash_broken_cy(k) for k in self.expected}
        self.assertEqual(self.expected_broken, actual)


#
# Run with:
#
#   python -c 'import gensim.test.test_utils as t;t.ngram_main()' py_text 3 5
#
def ngram_main():
    """Generate ngrams for tests from standard input."""
    import sys
    import six

    alg = sys.argv[1]
    minn = int(sys.argv[2])
    maxn = int(sys.argv[3])

    assert six.PY3, 'this only works under Py3'
    assert not DISABLE_CYTHON_TESTS, 'this only works if Cython extensions available'
    assert minn <= maxn, 'expected sane command-line parameters'

    hashmap = {
        'py_text': gensim.models.utils_any2vec._compute_ngrams_py,
        'py_bytes': gensim.models.utils_any2vec._compute_ngrams_bytes_py,
        'cy_text': gensim.models.utils_any2vec._compute_ngrams_cy,
        'cy_bytes': gensim.models.utils_any2vec._compute_ngrams_bytes_cy,
    }
    try:
        fun = hashmap[alg]
    except KeyError:
        raise KeyError('invalid alg: %r expected one of %r' % (alg, sorted(hashmap)))

    for line in sys.stdin:
        word = line.rstrip('\n')
        ngrams = fun(word, minn, maxn)
        print("%r: %r," % (word, ngrams))


class NgramsTest(unittest.TestCase):
    def setUp(self):
        self.expected_text = {
            'test': ['<te', 'tes', 'est', 'st>', '<tes', 'test', 'est>', '<test', 'test>'],
            'at the': [
                '<at', 'at ', 't t', ' th', 'the', 'he>',
                '<at ', 'at t', 't th', ' the', 'the>', '<at t', 'at th', 't the', ' the>'
            ],
            'at\nthe': [
                '<at', 'at\n', 't\nt', '\nth', 'the', 'he>',
                '<at\n', 'at\nt', 't\nth', '\nthe', 'the>', '<at\nt', 'at\nth', 't\nthe', '\nthe>'
            ],
            'тест': ['<те', 'тес', 'ест', 'ст>', '<тес', 'тест', 'ест>', '<тест', 'тест>'],
            'テスト': ['<テス', 'テスト', 'スト>', '<テスト', 'テスト>', '<テスト>'],
            '試し': ['<試し', '試し>', '<試し>'],
        }
        self.expected_bytes = {
            'test': [b'<te', b'<tes', b'<test', b'tes', b'test', b'test>', b'est', b'est>', b'st>'],
            'at the': [
                b'<at', b'<at ', b'<at t', b'at ', b'at t', b'at th', b't t',
                b't th', b't the', b' th', b' the', b' the>', b'the', b'the>', b'he>'
            ],
            'тест': [
                b'<\xd1\x82\xd0\xb5', b'<\xd1\x82\xd0\xb5\xd1\x81', b'<\xd1\x82\xd0\xb5\xd1\x81\xd1\x82',
                b'\xd1\x82\xd0\xb5\xd1\x81', b'\xd1\x82\xd0\xb5\xd1\x81\xd1\x82', b'\xd1\x82\xd0\xb5\xd1\x81\xd1\x82>',
                b'\xd0\xb5\xd1\x81\xd1\x82', b'\xd0\xb5\xd1\x81\xd1\x82>', b'\xd1\x81\xd1\x82>'
            ],
            'テスト': [
                b'<\xe3\x83\x86\xe3\x82\xb9', b'<\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88',
                b'<\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88>', b'\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88',
                b'\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88>', b'\xe3\x82\xb9\xe3\x83\x88>'
            ],
            '試し': [b'<\xe8\xa9\xa6\xe3\x81\x97', b'<\xe8\xa9\xa6\xe3\x81\x97>', b'\xe8\xa9\xa6\xe3\x81\x97>'],
        }

        self.expected_text_wide_unicode = {
            '🚑🚒🚓🚕': [
                '<🚑🚒', '🚑🚒🚓', '🚒🚓🚕', '🚓🚕>',
                '<🚑🚒🚓', '🚑🚒🚓🚕', '🚒🚓🚕>', '<🚑🚒🚓🚕', '🚑🚒🚓🚕>'
             ],
        }
        self.expected_bytes_wide_unicode = {
            '🚑🚒🚓🚕': [
                b'<\xf0\x9f\x9a\x91\xf0\x9f\x9a\x92',
                b'<\xf0\x9f\x9a\x91\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93',
                b'<\xf0\x9f\x9a\x91\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93\xf0\x9f\x9a\x95',
                b'\xf0\x9f\x9a\x91\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93',
                b'\xf0\x9f\x9a\x91\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93\xf0\x9f\x9a\x95',
                b'\xf0\x9f\x9a\x91\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93\xf0\x9f\x9a\x95>',
                b'\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93\xf0\x9f\x9a\x95',
                b'\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93\xf0\x9f\x9a\x95>',
                b'\xf0\x9f\x9a\x93\xf0\x9f\x9a\x95>'
            ],
        }

    def test_text_py(self):
        for word in self.expected_text:
            expected = self.expected_text[word]
            actual = gensim.models.utils_any2vec._compute_ngrams_py(word, 3, 5)
            self.assertEqual(expected, actual)

    @unittest.skipIf(sys.maxunicode == 0xffff, "Python interpreter doesn't support UCS-4 (wide unicode)")
    def test_text_py_wide_unicode(self):
        for word in self.expected_text_wide_unicode:
            expected = self.expected_text_wide_unicode[word]
            actual = gensim.models.utils_any2vec._compute_ngrams_py(word, 3, 5)
            self.assertEqual(expected, actual)

    @unittest.skipIf(DISABLE_CYTHON_TESTS, 'Cython functions are not properly compiled')
    def test_text_cy(self):
        for word in self.expected_text:
            expected = self.expected_text[word]
            actual = gensim.models.utils_any2vec._compute_ngrams_cy(word, 3, 5)
            self.assertEqual(expected, actual)

    @unittest.skipIf(DISABLE_CYTHON_TESTS, 'Cython functions are not properly compiled')
    @unittest.skipIf(sys.maxunicode == 0xffff, "Python interpreter doesn't support UCS-4 (wide unicode)")
    def test_text_cy_wide_unicode(self):
        for word in self.expected_text_wide_unicode:
            expected = self.expected_text_wide_unicode[word]
            actual = gensim.models.utils_any2vec._compute_ngrams_cy(word, 3, 5)
            self.assertEqual(expected, actual)

    def test_bytes_py(self):
        for word in self.expected_bytes:
            expected = self.expected_bytes[word]
            actual = gensim.models.utils_any2vec._compute_ngrams_bytes_py(word, 3, 5)
            self.assertEqual(expected, actual)

            expected_text = self.expected_text[word]
            actual_text = [n.decode('utf-8') for n in actual]
            #
            # The text and byte implementations yield ngrams in different
            # order, so the test ignores ngram order.
            #
            self.assertEqual(sorted(expected_text), sorted(actual_text))

        for word in self.expected_bytes_wide_unicode:
            expected = self.expected_bytes_wide_unicode[word]
            actual = gensim.models.utils_any2vec._compute_ngrams_bytes_py(word, 3, 5)
            self.assertEqual(expected, actual)

            expected_text = self.expected_text_wide_unicode[word]
            actual_text = [n.decode('utf-8') for n in actual]

            self.assertEqual(sorted(expected_text), sorted(actual_text))

    @unittest.skipIf(DISABLE_CYTHON_TESTS, 'Cython functions are not properly compiled')
    def test_bytes_cy(self):
        for word in self.expected_bytes:
            expected = self.expected_bytes[word]
            actual = gensim.models.utils_any2vec._compute_ngrams_bytes_cy(word, 3, 5)
            self.assertEqual(expected, actual)

            expected_text = self.expected_text[word]
            actual_text = [n.decode('utf-8') for n in actual]
            self.assertEqual(sorted(expected_text), sorted(actual_text))

        for word in self.expected_bytes_wide_unicode:
            expected = self.expected_bytes_wide_unicode[word]
            actual = gensim.models.utils_any2vec._compute_ngrams_bytes_cy(word, 3, 5)
            self.assertEqual(expected, actual)

            expected_text = self.expected_text_wide_unicode[word]
            actual_text = [n.decode('utf-8') for n in actual]
            self.assertEqual(sorted(expected_text), sorted(actual_text))

    def test_fb(self):
        """Test against results from Facebook's implementation."""
        with smart_open.smart_open(datapath('fb-ngrams.txt'), 'r', encoding='utf-8') as fin:
            fb = dict(_read_fb(fin))

        for word, expected in fb.items():
            #
            # The model was trained with minn=3, maxn=6
            #
            actual = gensim.models.utils_any2vec._compute_ngrams_py(word, 3, 6)
            self.assertEqual(sorted(expected), sorted(actual))


def _read_fb(fin):
    """Read ngrams from output of the FB utility."""
    #
    # $ cat words.txt
    # test
    # at the
    # at\nthe
    # тест
    # テスト
    # 試し
    # 🚑🚒🚓🚕
    # $ while read w;
    # do
    #   echo "<start>";
    #   echo $w;
    #   ./fasttext print-ngrams gensim/test/test_data/crime-and-punishment.bin "$w";
    #   echo "<end>";
    # done < words.txt > gensim/test/test_data/fb-ngrams.txt
    #
    while fin:
        line = fin.readline().rstrip()
        if not line:
            break

        assert line == '<start>'
        word = fin.readline().rstrip()

        fin.readline()  # ignore this line, it contains an origin vector for the full term

        ngrams = []
        while True:
            line = fin.readline().rstrip()
            if line == '<end>':
                break

            columns = line.split(' ')
            term = ' '.join(columns[:-5])
            ngrams.append(term)

        yield word, ngrams


if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()
