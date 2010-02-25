#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# author Radim Rehurek, radimrehurek@seznam.cz

"""
This module contains various general utility functions.
"""


import logging
import re
import unicodedata
import cPickle


PAT_ALPHABETIC = re.compile('(((?![\d])\w)+)', re.UNICODE)


def deaccent(text):
    """
    Remove accentuation from the given string.
    
    Input text is either a unicode string or utf8 encoded bytestring. Return
    input string with accents removed, as unicode.
    
    >>> deaccent("Já to v libutf dělám, v příkladu pro ICU to také dělají")
    u'Ja to v libutf delam, v prikladu pro ICU to take delaji'
    """
    if not isinstance(text, unicode):
        text = unicode(text, 'utf8') # assume utf8 for byte strings, use default (strict) error handling 
    norm = unicodedata.normalize("NFD", text)
    result = u''.join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize("NFC", result)


def tokenize(text, lowercase = False, deacc = False, errors = "strict", toLower = False, lower = False):
    """
    Iteratively yield tokens as unicode strings, optionally also lowercasing them 
    and removing accent marks.
    
    Input text may be either unicode or utf8-encoded byte string.

    The tokens on output are maximal contiguous sequences of alphabetic 
    characters (no digits!).
    
    >>> list(tokenize('Nic nemůže letět rychlostí vyšší, než 300 tisíc kilometrů za sekundu!', deacc = True))
    [u'Nic', u'nemuze', u'letet', u'rychlosti', u'vyssi', u'nez', u'tisic', u'kilometru', u'za', u'sekundu']
    """
    lowercase = lowercase or toLower or lower
    if not isinstance(text, unicode):
        text = unicode(text, encoding = 'utf8', errors = errors)
    if lowercase:
        text = text.lower()
    if deacc:
        text = deaccent(text)
    for match in PAT_ALPHABETIC.finditer(text):
        yield match.group()


def toUtf8(text):
    if isinstance(text, unicode):
        return text.encode('utf8')
    return unicode(text, 'utf8').encode('utf8') # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8


class SaveLoad(object):
    """
    Objects which inherit from this class have save/load functions, which un/pickle 
    them to disk.
    
    This uses cPickle for serializing objects, so objects must not contains unpicklable 
    attributes, such as lambda functions etc.
    """
    @classmethod
    def load(cls, fname):
        """
        Load a previously saved object from file (also see save()).
        """
        logging.info("loading %s object from %s" % (cls.__name__, fname))
        return cPickle.load(open(fname))


    def save(self, fname):
        """
        Save the object to file via pickling (also see load()).
        """
        logging.info("saving %s object to %s" % (self.__class__.__name__, fname))
        f = open(fname, 'w')
        cPickle.dump(self, f, protocol = -1) # use the highest available protocol, for efficiency
        f.close()
#endclass SaveLoad


def identity(p):
    return p
