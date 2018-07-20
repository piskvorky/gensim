#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2016 Olavur Mortensen <olavurmortensen@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for the author-topic model (AuthorTopicModel class). These tests
are based on the unit tests of LDA; the classes are quite similar, and the tests
needed are thus quite similar.
"""


import logging
import unittest
import numbers
from os import remove

import six
import numpy as np

from gensim.corpora import mmcorpus, Dictionary
from gensim.models import atmodel
from gensim import matutils
from gensim.test import basetmtests
from gensim.test.utils import (datapath,
    get_tmpfile, common_texts, common_dictionary as dictionary, common_corpus as corpus)
from gensim.matutils import jensen_shannon
# TODO:
# Test that computing the bound on new unseen documents works as expected (this is somewhat different
# in the author-topic model than in LDA).
# Perhaps test that the bound increases, in general (i.e. in several of the tests below where it makes
# sense. This is not tested in LDA either. Tests can also be made to check that automatic prior learning
# increases the bound.
# Test that models are compatiple across versions, as done in LdaModel.

# Assign some authors randomly to the documents above.
author2doc = {
    'john': [0, 1, 2, 3, 4, 5, 6],
    'jane': [2, 3, 4, 5, 6, 7, 8],
    'jack': [0, 2, 4, 6, 8],
    'jill': [1, 3, 5, 7]
}

doc2author = {
    0: ['john', 'jack'],
    1: ['john', 'jill'],
    2: ['john', 'jane', 'jack'],
    3: ['john', 'jane', 'jill'],
    4: ['john', 'jane', 'jack'],
    5: ['john', 'jane', 'jill'],
    6: ['john', 'jane', 'jack'],
    7: ['jane', 'jill'],
    8: ['jane', 'jack']
}

# More data with new and old authors (to test update method).
# Although the text is just a subset of the previous, the model
# just sees it as completely new data.
texts_new = common_texts[0:3]
author2doc_new = {'jill': [0], 'bob': [0, 1], 'sally': [1, 2]}
dictionary_new = Dictionary(texts_new)
corpus_new = [dictionary_new.doc2bow(text) for text in texts_new]


class TestAuthorTopicModel(unittest.TestCase, basetmtests.TestBaseTopicModel):
    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.class_ = atmodel.AuthorTopicModel
        self.model = self.class_(corpus, id2word=dictionary, author2doc=author2doc, num_topics=2, passes=100)

    def testTransform(self):
        passed = False
        # sometimes, training gets stuck at a local minimum
        # in that case try re-training the model from scratch, hoping for a
        # better random initialization
        for i in range(25):  # restart at most 5 times
            # create the transformation model
            model = self.class_(id2word=dictionary, num_topics=2, passes=100, random_state=0)
            model.update(corpus, author2doc)

            jill_topics = model.get_author_topics('jill')

            # NOTE: this test may easily fail if the author-topic model is altered in any way. The model's
            # output is sensitive to a lot of things, like the scheduling of the updates, or like the
            # author2id (because the random initialization changes when author2id changes). If it does
            # fail, simply be aware of whether we broke something, or if it just naturally changed the
            # output of the model slightly.
            vec = matutils.sparse2full(jill_topics, 2)  # convert to dense vector, for easier equality tests
            expected = [0.91, 0.08]
            # must contain the same values, up to re-ordering
            passed = np.allclose(sorted(vec), sorted(expected), atol=1e-1)
            if passed:
                break
            logging.warning(
                "Author-topic model failed to converge on attempt %i (got %s, expected %s)",
                i, sorted(vec), sorted(expected)
            )
        self.assertTrue(passed)

    def testBasic(self):
        # Check that training the model produces a positive topic vector for some author
        # Otherwise, many of the other tests are invalid.

        model = self.class_(corpus, author2doc=author2doc, id2word=dictionary, num_topics=2)

        jill_topics = model.get_author_topics('jill')
        jill_topics = matutils.sparse2full(jill_topics, model.num_topics)
        self.assertTrue(all(jill_topics > 0))

    def testEmptyDocument(self):
        _local_texts = common_texts + [['only_occurs_once_in_corpus_and_alone_in_doc']]
        _dictionary = Dictionary(_local_texts)
        _dictionary.filter_extremes(no_below=2)
        _corpus = [_dictionary.doc2bow(text) for text in _local_texts]
        _a2d = author2doc.copy()
        _a2d['joaquin'] = [len(_local_texts) - 1]
        try:
            _ = self.class_(_corpus, author2doc=_a2d, id2word=_dictionary, num_topics=2)
        except IndexError:
            raise IndexError("error occurs in 1.0.0 release tag")
        assert(_)

    def testAuthor2docMissing(self):
        # Check that the results are the same if author2doc is constructed automatically from doc2author.
        model = self.class_(
            corpus, author2doc=author2doc, doc2author=doc2author,
            id2word=dictionary, num_topics=2, random_state=0
        )
        model2 = self.class_(
            corpus, doc2author=doc2author, id2word=dictionary,
            num_topics=2, random_state=0
        )

        # Compare Jill's topics before in both models.
        jill_topics = model.get_author_topics('jill')
        jill_topics2 = model2.get_author_topics('jill')
        jill_topics = matutils.sparse2full(jill_topics, model.num_topics)
        jill_topics2 = matutils.sparse2full(jill_topics2, model.num_topics)
        self.assertTrue(np.allclose(jill_topics, jill_topics2))

    def testDoc2authorMissing(self):
        # Check that the results are the same if doc2author is constructed automatically from author2doc.
        model = self.class_(
            corpus, author2doc=author2doc, doc2author=doc2author,
            id2word=dictionary, num_topics=2, random_state=0
        )
        model2 = self.class_(
            corpus, author2doc=author2doc, id2word=dictionary,
            num_topics=2, random_state=0
        )

        # Compare Jill's topics before in both models.
        jill_topics = model.get_author_topics('jill')
        jill_topics2 = model2.get_author_topics('jill')
        jill_topics = matutils.sparse2full(jill_topics, model.num_topics)
        jill_topics2 = matutils.sparse2full(jill_topics2, model.num_topics)
        self.assertTrue(np.allclose(jill_topics, jill_topics2))

    def testUpdate(self):
        # Check that calling update after the model already has been trained works.
        model = self.class_(corpus, author2doc=author2doc, id2word=dictionary, num_topics=2)

        jill_topics = model.get_author_topics('jill')
        jill_topics = matutils.sparse2full(jill_topics, model.num_topics)

        model.update()
        jill_topics2 = model.get_author_topics('jill')
        jill_topics2 = matutils.sparse2full(jill_topics2, model.num_topics)

        # Did we learn something?
        self.assertFalse(all(np.equal(jill_topics, jill_topics2)))

    def testUpdateNewDataOldAuthor(self):
        # Check that calling update with new documents and/or authors after the model already has
        # been trained works.
        # Test an author that already existed in the old dataset.
        model = self.class_(corpus, author2doc=author2doc, id2word=dictionary, num_topics=2)

        jill_topics = model.get_author_topics('jill')
        jill_topics = matutils.sparse2full(jill_topics, model.num_topics)

        model.update(corpus_new, author2doc_new)
        jill_topics2 = model.get_author_topics('jill')
        jill_topics2 = matutils.sparse2full(jill_topics2, model.num_topics)

        # Did we learn more about Jill?
        self.assertFalse(all(np.equal(jill_topics, jill_topics2)))

    def testUpdateNewDataNewAuthor(self):
        # Check that calling update with new documents and/or authors after the model already has
        # been trained works.
        # Test a new author, that didn't exist in the old dataset.
        model = self.class_(corpus, author2doc=author2doc, id2word=dictionary, num_topics=2)

        model.update(corpus_new, author2doc_new)

        # Did we learn something about Sally?
        sally_topics = model.get_author_topics('sally')
        sally_topics = matutils.sparse2full(sally_topics, model.num_topics)
        self.assertTrue(all(sally_topics > 0))

    def testSerialized(self):
        # Test the model using serialized corpora. Basic tests, plus test of update functionality.

        model = self.class_(
            self.corpus, author2doc=author2doc, id2word=dictionary, num_topics=2,
            serialized=True, serialization_path=datapath('testcorpus_serialization.mm')
        )

        jill_topics = model.get_author_topics('jill')
        jill_topics = matutils.sparse2full(jill_topics, model.num_topics)
        self.assertTrue(all(jill_topics > 0))

        model.update()
        jill_topics2 = model.get_author_topics('jill')
        jill_topics2 = matutils.sparse2full(jill_topics2, model.num_topics)

        # Did we learn more about Jill?
        self.assertFalse(all(np.equal(jill_topics, jill_topics2)))

        model.update(corpus_new, author2doc_new)

        # Did we learn something about Sally?
        sally_topics = model.get_author_topics('sally')
        sally_topics = matutils.sparse2full(sally_topics, model.num_topics)
        self.assertTrue(all(sally_topics > 0))

        # Delete the MmCorpus used for serialization inside the author-topic model.
        remove(datapath('testcorpus_serialization.mm'))

    def testTransformSerialized(self):
        # Same as testTransform, using serialized corpora.
        passed = False
        # sometimes, training gets stuck at a local minimum
        # in that case try re-training the model from scratch, hoping for a
        # better random initialization
        for i in range(25):  # restart at most 5 times
            # create the transformation model
            model = self.class_(
                id2word=dictionary, num_topics=2, passes=100, random_state=0,
                serialized=True, serialization_path=datapath('testcorpus_serialization.mm')
            )
            model.update(self.corpus, author2doc)

            jill_topics = model.get_author_topics('jill')

            # NOTE: this test may easily fail if the author-topic model is altered in any way. The model's
            # output is sensitive to a lot of things, like the scheduling of the updates, or like the
            # author2id (because the random initialization changes when author2id changes). If it does
            # fail, simply be aware of whether we broke something, or if it just naturally changed the
            # output of the model slightly.
            vec = matutils.sparse2full(jill_topics, 2)  # convert to dense vector, for easier equality tests
            expected = [0.91, 0.08]
            # must contain the same values, up to re-ordering
            passed = np.allclose(sorted(vec), sorted(expected), atol=1e-1)

            # Delete the MmCorpus used for serialization inside the author-topic model.
            remove(datapath('testcorpus_serialization.mm'))
            if passed:
                break
            logging.warning(
                "Author-topic model failed to converge on attempt %i (got %s, expected %s)",
                i, sorted(vec), sorted(expected)
            )
        self.assertTrue(passed)

    def testAlphaAuto(self):
        model1 = self.class_(
            corpus, author2doc=author2doc, id2word=dictionary,
            alpha='symmetric', passes=10, num_topics=2
        )
        modelauto = self.class_(
            corpus, author2doc=author2doc, id2word=dictionary,
            alpha='auto', passes=10, num_topics=2
        )

        # did we learn something?
        self.assertFalse(all(np.equal(model1.alpha, modelauto.alpha)))

    def testAlpha(self):
        kwargs = dict(
            author2doc=author2doc,
            id2word=dictionary,
            num_topics=2,
            alpha=None
        )
        expected_shape = (2,)

        # should not raise anything
        self.class_(**kwargs)

        kwargs['alpha'] = 'symmetric'
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(all(model.alpha == np.array([0.5, 0.5])))

        kwargs['alpha'] = 'asymmetric'
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(np.allclose(model.alpha, [0.630602, 0.369398]))

        kwargs['alpha'] = 0.3
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(all(model.alpha == np.array([0.3, 0.3])))

        kwargs['alpha'] = 3
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(all(model.alpha == np.array([3, 3])))

        kwargs['alpha'] = [0.3, 0.3]
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(all(model.alpha == np.array([0.3, 0.3])))

        kwargs['alpha'] = np.array([0.3, 0.3])
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(all(model.alpha == np.array([0.3, 0.3])))

        # all should raise an exception for being wrong shape
        kwargs['alpha'] = [0.3, 0.3, 0.3]
        self.assertRaises(AssertionError, self.class_, **kwargs)

        kwargs['alpha'] = [[0.3], [0.3]]
        self.assertRaises(AssertionError, self.class_, **kwargs)

        kwargs['alpha'] = [0.3]
        self.assertRaises(AssertionError, self.class_, **kwargs)

        kwargs['alpha'] = "gensim is cool"
        self.assertRaises(ValueError, self.class_, **kwargs)

    def testEtaAuto(self):
        model1 = self.class_(
            corpus, author2doc=author2doc, id2word=dictionary,
            eta='symmetric', passes=10, num_topics=2
        )
        modelauto = self.class_(
            corpus, author2doc=author2doc, id2word=dictionary,
            eta='auto', passes=10, num_topics=2
        )

        # did we learn something?
        self.assertFalse(all(np.equal(model1.eta, modelauto.eta)))

    def testEta(self):
        kwargs = dict(
            author2doc=author2doc,
            id2word=dictionary,
            num_topics=2,
            eta=None
        )
        num_terms = len(dictionary)
        expected_shape = (num_terms,)

        # should not raise anything
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == np.array([0.5] * num_terms)))

        kwargs['eta'] = 'symmetric'
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == np.array([0.5] * num_terms)))

        kwargs['eta'] = 0.3
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == np.array([0.3] * num_terms)))

        kwargs['eta'] = 3
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == np.array([3] * num_terms)))

        kwargs['eta'] = [0.3] * num_terms
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == np.array([0.3] * num_terms)))

        kwargs['eta'] = np.array([0.3] * num_terms)
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == np.array([0.3] * num_terms)))

        # should be ok with num_topics x num_terms
        testeta = np.array([[0.5] * len(dictionary)] * 2)
        kwargs['eta'] = testeta
        self.class_(**kwargs)

        # all should raise an exception for being wrong shape
        kwargs['eta'] = testeta.reshape(tuple(reversed(testeta.shape)))
        self.assertRaises(AssertionError, self.class_, **kwargs)

        kwargs['eta'] = [0.3]
        self.assertRaises(AssertionError, self.class_, **kwargs)

        kwargs['eta'] = [0.3] * (num_terms + 1)
        self.assertRaises(AssertionError, self.class_, **kwargs)

        kwargs['eta'] = "gensim is cool"
        self.assertRaises(ValueError, self.class_, **kwargs)

        kwargs['eta'] = "asymmetric"
        self.assertRaises(ValueError, self.class_, **kwargs)

    def testTopTopics(self):
        top_topics = self.model.top_topics(corpus)

        for topic, score in top_topics:
            self.assertTrue(isinstance(topic, list))
            self.assertTrue(isinstance(score, float))

            for v, k in topic:
                self.assertTrue(isinstance(k, six.string_types))
                self.assertTrue(isinstance(v, float))

    def testGetTopicTerms(self):
        topic_terms = self.model.get_topic_terms(1)

        for k, v in topic_terms:
            self.assertTrue(isinstance(k, numbers.Integral))
            self.assertTrue(isinstance(v, float))

    def testGetAuthorTopics(self):

        model = self.class_(
            corpus, author2doc=author2doc, id2word=dictionary, num_topics=2,
            passes=100, random_state=np.random.seed(0)
        )

        author_topics = []
        for a in model.id2author.values():
            author_topics.append(model.get_author_topics(a))

        for topic in author_topics:
            self.assertTrue(isinstance(topic, list))
            for k, v in topic:
                self.assertTrue(isinstance(k, int))
                self.assertTrue(isinstance(v, float))

    def testTermTopics(self):

        model = self.class_(
            corpus, author2doc=author2doc, id2word=dictionary, num_topics=2,
            passes=100, random_state=np.random.seed(0)
        )

        # check with word_type
        result = model.get_term_topics(2)
        for topic_no, probability in result:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(isinstance(probability, float))

        # if user has entered word instead, check with word
        result = model.get_term_topics(str(model.id2word[2]))
        for topic_no, probability in result:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(isinstance(probability, float))

    def testNewAuthorTopics(self):

        model = self.class_(
            corpus, author2doc=author2doc, id2word=dictionary, num_topics=2,
            passes=100, random_state=np.random.seed(0)
        )
        author2doc_newauthor = {}
        author2doc_newauthor["test"] = [0, 1]
        model.update(corpus=corpus[0:2], author2doc=author2doc_newauthor)

        # temp save model state vars before get_new_author_topics is called
        state_gamma_len = len(model.state.gamma)
        author2doc_len = len(model.author2doc)
        author2id_len = len(model.author2id)
        id2author_len = len(model.id2author)
        doc2author_len = len(model.doc2author)

        new_author_topics = model.get_new_author_topics(corpus=corpus[0:2])

        # sanity check
        for k, v in new_author_topics:
            self.assertTrue(isinstance(k, int))
            self.assertTrue(isinstance(v, float))

        # make sure topics are similar enough
        similarity = 1 / (1 + jensen_shannon(model["test"], new_author_topics))
        self.assertTrue(similarity >= 0.9)

        # produce an error to test if rollback occurs
        with self.assertRaises(TypeError):
            model.get_new_author_topics(corpus=corpus[0])

        # assure rollback was successful and the model state is as before
        self.assertEqual(state_gamma_len, len(model.state.gamma))
        self.assertEqual(author2doc_len, len(model.author2doc))
        self.assertEqual(author2id_len, len(model.author2id))
        self.assertEqual(id2author_len, len(model.id2author))
        self.assertEqual(doc2author_len, len(model.doc2author))

    def testPasses(self):
        # long message includes the original error message with a custom one
        self.longMessage = True
        # construct what we expect when passes aren't involved
        test_rhots = []
        model = self.class_(id2word=dictionary, chunksize=1, num_topics=2)

        def final_rhot(model):
            return pow(model.offset + (1 * model.num_updates) / model.chunksize, -model.decay)

        # generate 5 updates to test rhot on
        for _ in range(5):
            model.update(corpus, author2doc)
            test_rhots.append(final_rhot(model))

        for passes in [1, 5, 10, 50, 100]:
            model = self.class_(id2word=dictionary, chunksize=1, num_topics=2, passes=passes)
            self.assertEqual(final_rhot(model), 1.0)
            # make sure the rhot matches the test after each update
            for test_rhot in test_rhots:
                model.update(corpus, author2doc)

                msg = "{}, {}, {}".format(passes, model.num_updates, model.state.numdocs)
                self.assertAlmostEqual(final_rhot(model), test_rhot, msg=msg)

            self.assertEqual(model.state.numdocs, len(corpus) * len(test_rhots))
            self.assertEqual(model.num_updates, len(corpus) * len(test_rhots))

    def testPersistence(self):
        fname = get_tmpfile('gensim_models_atmodel.tst')
        model = self.model
        model.save(fname)
        model2 = self.class_.load(fname)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(np.allclose(model.expElogbeta, model2.expElogbeta))
        self.assertTrue(np.allclose(model.state.gamma, model2.state.gamma))

    def testPersistenceIgnore(self):
        fname = get_tmpfile('gensim_models_atmodel_testPersistenceIgnore.tst')
        model = atmodel.AuthorTopicModel(corpus, author2doc=author2doc, num_topics=2)
        model.save(fname, ignore='id2word')
        model2 = atmodel.AuthorTopicModel.load(fname)
        self.assertTrue(model2.id2word is None)

        model.save(fname, ignore=['id2word'])
        model2 = atmodel.AuthorTopicModel.load(fname)
        self.assertTrue(model2.id2word is None)

    def testPersistenceCompressed(self):
        fname = get_tmpfile('gensim_models_atmodel.tst.gz')
        model = self.model
        model.save(fname)
        model2 = self.class_.load(fname, mmap=None)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(np.allclose(model.expElogbeta, model2.expElogbeta))

        # Compare Jill's topics before and after save/load.
        jill_topics = model.get_author_topics('jill')
        jill_topics2 = model2.get_author_topics('jill')
        jill_topics = matutils.sparse2full(jill_topics, model.num_topics)
        jill_topics2 = matutils.sparse2full(jill_topics2, model.num_topics)
        self.assertTrue(np.allclose(jill_topics, jill_topics2))

    def testLargeMmap(self):
        fname = get_tmpfile('gensim_models_atmodel.tst')
        model = self.model

        # simulate storing large arrays separately
        model.save(fname, sep_limit=0)

        # test loading the large model arrays with mmap
        model2 = self.class_.load(fname, mmap='r')
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(isinstance(model2.expElogbeta, np.memmap))
        self.assertTrue(np.allclose(model.expElogbeta, model2.expElogbeta))

        # Compare Jill's topics before and after save/load.
        jill_topics = model.get_author_topics('jill')
        jill_topics2 = model2.get_author_topics('jill')
        jill_topics = matutils.sparse2full(jill_topics, model.num_topics)
        jill_topics2 = matutils.sparse2full(jill_topics2, model.num_topics)
        self.assertTrue(np.allclose(jill_topics, jill_topics2))

    def testLargeMmapCompressed(self):
        fname = get_tmpfile('gensim_models_atmodel.tst.gz')
        model = self.model

        # simulate storing large arrays separately
        model.save(fname, sep_limit=0)

        # test loading the large model arrays with mmap
        self.assertRaises(IOError, self.class_.load, fname, mmap='r')

    def testDtypeBackwardCompatibility(self):
        atmodel_3_0_1_fname = datapath('atmodel_3_0_1_model')
        expected_topics = [(0, 0.068200842977296727), (1, 0.93179915702270333)]

        # save model to use in test
        # self.model.save(atmodel_3_0_1_fname)

        # load a model saved using a 3.0.1 version of Gensim
        model = self.class_.load(atmodel_3_0_1_fname)

        # and test it on a predefined document
        topics = model['jane']
        self.assertTrue(np.allclose(expected_topics, topics))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
