#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import logging

logger = logging.getLogger("gensim.models.phrases")

MAX_VOCAB_SIZE = 500000000 # Maximum 500M entries in the vocabulary, taken from word2phrase

class VocabWord(object):
    """A single vocabulary item, used internally for constructing binary trees (incl. both word leaves and inner nodes)."""
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "<" + ', '.join(vals) + ">"

class Phrases(object):
    """
    Class learns from senteces interables. It does so by collecting statistics on words and joining common
    adjacent words with the '_' character.
        
    It can be used to generate phrases on the fly.
        
    """

    vocab = {}  # mapping from a word (string) to a Vocab object

    def  __init__(self, sentences, min_count = 5, threshold = 100):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (unicode strings) that will be used for training.

        The `sentences` iterable can be simply a list, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.
        See :class:`BrownCorpus`, :class:`Text8Corpus` or :class:`LineSentence` in
        the :module:`word2vec` module for such examples.        

        `min_count` ignore all words with total frequency lower than this. By default (`min_count`=5).
        
        `threshold` represents a threshold for forming the phrases (higher means less phrases). By (`threshold`=100).
        """
        self.vocab = self._learn_vocab(sentences)
        self.train_words = len(self.vocab)
        self.min_count = min_count
        self.threshold = threshold
    
    @staticmethod
    def _learn_vocab(sentences):
        
        def add_or_increase(word, vocab):
            """ Add or increase word in a  vocabulary
            """
            if word in vocab:
                vocab[word].count += 1
            else:
                vocab[word] = VocabWord(count=1)
                
        min_reduce = 1
        sentence_no, vocab = -1, {}
        total_words = 0
        logging.info("collecting all words and their counts")
        for sentence_no, sentence in enumerate(sentences):
            if sentence_no % 10000 == 0:
                logger.info("PROGRESS: at sentence #%i, processed %i words and %i word types" %
                            (sentence_no, total_words, len(vocab)))
                     
            for bigram in zip(sentence, sentence[1:]):
                word = bigram[0]
                bigram_word = "%s_%s" % bigram
                total_words += 1
                add_or_increase(word, vocab)
                add_or_increase(bigram_word, vocab)                
                
            if len(sentence) > 0:    # add last word skipped by previous loop
                word = sentence[-1]
                add_or_increase(word, vocab)
                
            if len(vocab) > MAX_VOCAB_SIZE * 0.7:
                # reduce vocabulary
                to_delete = []         
                for w in vocab.iterkeys():
                    if vocab[w].count < min_reduce:
                        logging.debug("Reducing vocabulary eliminating word %s with %d" % (w, min_reduce))
                        to_delete.append(w) # this way we avoid getting a copy of the whole word list
                min_reduce += 1
                for w in to_delete:
                    del vocab[w]
            
 
        logger.info("collected %i word types from a corpus of %i words (unigram + bigrams) and %i sentences" %
                    (len(vocab), total_words, sentence_no + 1))
    
        return vocab

    def __getitem__(self, sentences):
        """
        Return a iterable of the original sentences with common words / phrases 
        joined by '_'
        Example::

          >>> sentences = Text8Corpus(path_to_corpus)
          >>> bigram = Phrases(sentences, min_count=5, threshold=100) 
          >>> for sentence in phrases[sentences]:
          >>>>    print(' '.join(s))
            he refuted nechaev other anarchists sometimes identified as pacifist anarchists advocated complete 
            nonviolence leo_tolstoy
        
        """
        for sentence_no, s in enumerate(sentences):
            if sentence_no % 10000 == 0:
                logger.info("PROGRESS: at sentence #%i" % (sentence_no))
                                     
            new_s = []
            last_bigram = False            
            for bigram in zip(s, s[1:]):
                bigram_word = "%s_%s" % bigram               
                
                if all(uni in self.vocab  for uni in bigram) and \
                    bigram_word in self.vocab and not last_bigram:                    
                    
                    pa = float(self.vocab[bigram[0]].count) 
                    pb = float(self.vocab[bigram[1]].count)
                    pab = float(self.vocab[bigram_word].count)
                    score = (pab - self.min_count) / pa / pb * self.train_words
                    #logger.debug("score for %s: %.2f" % (bigram_word, score))
                        
                    if score > self.threshold:
                        new_s.append(bigram_word)
                        last_bigram = True
                    else:
                        last_bigram = False
                    
                if not last_bigram:
                    
                    new_s.append(bigram[0])
                    last_bigram = False
                    
                
            if len(s) > 0:
                w = s[-1]
                if w in self.vocab and not last_bigram:
                    new_s.append(w)
            yield new_s


if __name__ == '__main__':
    import sys
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, format="%(asctime)s\t%(levelname)-8s\t%(filename)s:%(lineno)-4d\t%(message)s")
    from gensim.models.word2vec import Text8Corpus

    sentences = Text8Corpus("/Users/miguel/Downloads/text8")

    #test_doc = LineSentence('test/test_data/testcorpus.txt')
    bigram = Phrases(sentences, min_count=5, threshold=100)
    for s in bigram[sentences]:        
        print ' '.join(s)

  

