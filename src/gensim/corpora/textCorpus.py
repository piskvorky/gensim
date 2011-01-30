#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import with_statement

import logging

from gensim.corpora.dictionary import Dictionary
from gensim import matutils

logger = logging.getLogger('TextCorpus')
logger.setLevel(logging.INFO)


class TextCorpus(Dictionary):
    """
    Both Dictionary and TextCorpus are at the same level (corpus objects). So
    TextCorpus inheriting from Dictionary may look weird, but this saves us
    from having to wrap the Dictionary functions that we need.


    We can save the TextCorpus in the following formats:
        * entire TextCorpus as .cpickle: save()/load() [DONE]
        * TextCorpus as human readable text format:
          saveAsText()/loadFromText() [DONE]
        * TODO
    """

    def __init__(self, documents=None):
        """
        TODO
        """
        super(TextCorpus, self).__init__(documents)

        self.filters = None
        self.documentfolder = None

    def saveAsText(self, fname):
        """
        Store the corpus to disk, in a human-readable text format.

        Token to integer mapping, as a text file `fname_wordids.txt`.

        Store id->word mapping to a file, in format:
        `id[TAB]word_utf8[TAB]document frequency[NEWLINE]`.

        TODO: perhaps use csv module from the python std for this.
        """
        logger.info("saving dictionary mapping to %s" % fname)
        fname += '_wordids.txt'
        with open(fname, 'w') as fout:
            for token, tokenId in sorted(self.token2id.iteritems()):
                fout.write("%i\t%s\t%i\n" % (tokenId, token,
                    self.docFreq[tokenId]))

    #def saveAsMatrixMarket(self, fname):
        #"""
        #Document-term co-occurence frequency counts (bag-of-words), as
        #a Matrix Market file `fname_bow.mm`.
        #"""
        ##tmp = self._asdict()
        #print type(self)
        #matutils.MmWriter.writeCorpus(fname + '_bow.mm', self,
               #progressCnt=10000)

    #def _asdict(self):
        #result = {}
        #for key, word in self.id2word.items():
            #result[int(key)] = word # docFreq not used
        #return result

    @staticmethod
    def loadFromText(fname):
        """
        Load previously stored mapping between words and their ids.

        The result can be used as the `id2word` parameter for input to
        transformations.

        TODO: perhaps use csv module from the python std for this.
        """
        result = TextCorpus()
        with open(fname) as f:
            for lineNo, line in enumerate(f):
                try:
                    wordId, word, docFreq = line[:-1].split('\t')
                except Exception:
                    raise ValueError("invalid line in dictionary file %s: %s"
                            % (fname, line.strip()))
                wordId = int(wordId)
                result.token2id[word] = wordId
                result.docFreq[wordId] = int(docFreq)
        return result
