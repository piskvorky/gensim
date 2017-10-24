#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automated tests for the parsing module.
"""

import logging
import unittest
import numpy as np
from gensim.utils.text_utils import remove_stopwords, strip_punctuation2, strip_tags, strip_short, strip_numeric, \
    strip_non_alphanum, strip_multiple_whitespaces, split_alphanum, stem_text

# several documents
doc1 = """C'est un trou de verdure où chante une rivière,
Accrochant follement aux herbes des haillons
D'argent ; où le soleil, de la montagne fière,
Luit : c'est un petit val qui mousse de rayons."""

doc2 = """Un soldat jeune, bouche ouverte, tête nue,
Et la nuque baignant dans le frais cresson bleu,
Dort ; il est étendu dans l'herbe, sous la nue,
Pâle dans son lit vert où la lumière pleut."""

doc3 = """Les pieds dans les glaïeuls, il dort. Souriant comme
Sourirait un enfant malade, il fait un somme :
Nature, berce-le chaudement : il a froid."""

doc4 = """Les parfums ne font pas frissonner sa narine ;
Il dort dans le soleil, la main sur sa poitrine,
Tranquille. Il a deux trous rouges au côté droit."""

doc5 = """While it is quite useful to be able to search a
large collection of documents almost instantly for a joint
occurrence of a collection of exact words,
for many searching purposes, a little fuzziness would help. """


dataset = [strip_punctuation2(x.lower()) for x in [doc1, doc2, doc3, doc4]]
# doc1 and doc2 have class 0, doc3 and doc4 avec class 1
classes = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])


class TestPreprocessing(unittest.TestCase):

    def testStripNumeric(self):
        self.assertEqual(strip_numeric("salut les amis du 59"), "salut les amis du ")

    def testStripShort(self):
        self.assertEqual(strip_short("salut les amis du 59", 3), "salut les amis")

    def testStripTags(self):
        self.assertEqual(strip_tags("<i>Hello</i> <b>World</b>!"), "Hello World!")

    def testStripMultipleWhitespaces(self):
        self.assertEqual(strip_multiple_whitespaces("salut  les\r\nloulous!"), "salut les loulous!")

    def testStripNonAlphanum(self):
        self.assertEqual(strip_non_alphanum("toto nf-kappa titi"), "toto nf kappa titi")

    def testSplitAlphanum(self):
        self.assertEqual(split_alphanum("toto diet1 titi"), "toto diet 1 titi")
        self.assertEqual(split_alphanum("toto 1diet titi"), "toto 1 diet titi")

    def testStripStopwords(self):
        self.assertEqual(remove_stopwords("the world is square"), "world square")

    def testStemText(self):
        target = \
            "while it is quit us to be abl to search a larg " + \
            "collect of document almost instantli for a joint occurr " + \
            "of a collect of exact words, for mani search purposes, " + \
            "a littl fuzzi would help."
        self.assertEqual(stem_text(doc5), target)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
