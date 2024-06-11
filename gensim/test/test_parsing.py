#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automated tests for the parsing module.
"""

import logging
import unittest
from unittest import mock

import numpy as np

from gensim.parsing.preprocessing import (
    remove_short_tokens,
    remove_stopword_tokens,
    remove_stopwords,
    stem_text,
    split_alphanum,
    split_on_space,
    strip_multiple_whitespaces,
    strip_non_alphanum,
    strip_numeric,
    strip_punctuation,
    strip_short,
    strip_tags,
)

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


dataset = [strip_punctuation(x.lower()) for x in [doc1, doc2, doc3, doc4]]
# doc1 and doc2 have class 0, doc3 and doc4 avec class 1
classes = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])


class TestPreprocessing(unittest.TestCase):

    def test_strip_numeric(self):
        self.assertEqual(strip_numeric("salut les amis du 59"), "salut les amis du ")

    def test_strip_short(self):
        self.assertEqual(strip_short("salut les amis du 59", 3), "salut les amis")

    def test_strip_tags(self):
        self.assertEqual(strip_tags("<i>Hello</i> <b>World</b>!"), "Hello World!")

    def test_strip_multiple_whitespaces(self):
        self.assertEqual(strip_multiple_whitespaces("salut  les\r\nloulous!"), "salut les loulous!")

    def test_strip_non_alphanum(self):
        self.assertEqual(strip_non_alphanum("toto nf-kappa titi"), "toto nf kappa titi")

    def test_split_alphanum(self):
        self.assertEqual(split_alphanum("toto diet1 titi"), "toto diet 1 titi")
        self.assertEqual(split_alphanum("toto 1diet titi"), "toto 1 diet titi")

    def test_strip_stopwords(self):
        self.assertEqual(remove_stopwords("the world is square"), "world square")

        # confirm redifining the global `STOPWORDS` working
        with mock.patch('gensim.parsing.preprocessing.STOPWORDS', frozenset(["the"])):
            self.assertEqual(remove_stopwords("the world is square"), "world is square")

    def test_strip_stopword_tokens(self):
        self.assertEqual(remove_stopword_tokens(["the", "world", "is", "sphere"]), ["world", "sphere"])

        # confirm redifining the global `STOPWORDS` working
        with mock.patch('gensim.parsing.preprocessing.STOPWORDS', frozenset(["the"])):
            self.assertEqual(
                remove_stopword_tokens(["the", "world", "is", "sphere"]),
                ["world", "is", "sphere"]
            )

    def test_strip_short_tokens(self):
        self.assertEqual(remove_short_tokens(["salut", "les", "amis", "du", "59"], 3), ["salut", "les", "amis"])

    def test_split_on_space(self):
        self.assertEqual(split_on_space(" salut   les  amis du 59 "), ["salut", "les", "amis", "du", "59"])

    def test_stem_text(self):
        target = \
            "while it is quit us to be abl to search a larg " + \
            "collect of document almost instantli for a joint occurr " + \
            "of a collect of exact words, for mani search purposes, " + \
            "a littl fuzzi would help."
        self.assertEqual(stem_text(doc5), target)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
