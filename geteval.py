#!/usr/bin/env python2.5
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Generate data for the Word Intrusion and Topic Intrusion tasks, in csv format \
suitable for the Mechanical Turk. For a description of what this means, see
Chang et al.: Reading Tea Leaves: How Humans Interpret Topic Models

For word intrusion:
./geteval_wordintrusion.py NUM_TASKS CONCEPT_FILE
e.g. ./geteval_wordintrusion.py 1000 /Users/kofola/workspace/dml/data/results/gensim_eng.lsa_concepts300 \
1> /Users/kofola/workspace/dml/data/results/gensim_eng.lsa_concepts300.wordintrusion

For topic instrusion:
./geteval_topicintrusion.py NUM_TASKS CONCEPT_FILE
e.g. ./geteval_topicintrusion.py 500 /Users/kofola/workspace/dml/data/results/gensim_eng.lsa_concepts300 \
1> /Users/kofola/workspace/dml/data/results/gensim_eng.lsa_concepts300.topicintrusion
"""

# The functions in this module expect that topics have been previously saved to 
# disk in a specific format, for example via docsim.saveTopics().

import logging
import sys
import os
import random

# number of top words from the same concept, for the word intrusion task.
# one extra word from a different concept will be added, for a set of WI_WORDS+1 
# randomly shuffled words.
WI_WORDS = 5


def loadConcepts(fname):
    """
    Load concepts (words) from a file on disk. Ignore the word weights, only store
    the words themselves.
    
    Return list of concepts, where each concept is a list of words. A concept's 
    id is implicitly its position in the list.
    """
    logging.info("loading concepts from %s" % fname)
    concepts = []
    for line in open(fname):
        concept = [part.split(':')[0] for part in line.strip().split('\t')]
        concepts.append(concept) # concept id is implicitly position within the list of concepts
    logging.info("loaded %i concepts" % len(concepts))
    return concepts


class WordIntrusion(object):
    def __init__(self, fname):
        self.concepts = loadConcepts(fname)
        
    def getAlienWord(self, conceptId):
        """
        For a given concept, choose an 'alien' word, which 
        a) is unlikely for the input concept
        b) is likely in some other concept (called alient concept).
        
        Return the 2-tuple (alien concept id, alien word).
        """
        allWords = self.concepts[conceptId]
        badWords = set(allWords[int(0.6 * len(allWords)) : ]) # use the bottom 40% of words for alien candidates
        
        candidates = []
        for alienId, concept in enumerate(self.concepts):
            if alienId == conceptId:
                continue
            topAlienWords = concept[ : 10] # use 10 most significant words as alien concept representatives 
            alienOk = badWords.intersection(topAlienWords)
            candidates.extend((alienId, alienWord) for alienWord in alienOk)
        assert candidates, "for concept %s, method %s, there are no candidates for alien words!" % (conceptId, method)
    
        return random.choice(candidates)

    def wordIntrusion(self, numWords):
        """
        Generate data for a single word intrusion task instance.
        """
        # randomly pick the target topic and its most significant words
        conceptId = random.randint(0, len(self.concepts) - 1)
        words = self.concepts[conceptId][ : numWords]
        random.shuffle(words) # shuffle the words in place
        
        # randomly pick another word, significant in another topic, and inject it into this topic
        alienConceptId, alienWord = self.getAlienWord(conceptId)
        alienPos = random.randint(0, numWords) # position of the alien word, for insertion
        words.insert(alienPos, alienWord)
        return conceptId, alienConceptId, words, alienPos

    def printProtocol(self, numInstances):
        """
        Print a specified number of instances for the word intrusion test.
        
        Each instance contains:
        1) id of the concept tested for intrusion
        2) five words from this concept
        3) id of concept from which the alien word is taken
        4) one alien word
        
        This information is represented by six words (shuffled), one position (of 
        the alien word within the six), and two concept ids, per instance. 
        
        Each instance is saved as one line in a csv file (which can be included
        in Mechanical Turk or similar software, to be evaluated by humans). The file
        therefore contains numInstances+1 lines; the extra first line is the csv
        descriptor of fields.
        """
        fields = ['w%i' % i for i in xrange(WI_WORDS + 1)] + ['apos', 'cid', 'aid']
        template = ','.join("%(" + field + ')s' for field in fields)
        headerLine = template % dict(zip(fields, fields)) # first line in csv describes the fields
        print headerLine
        for i in xrange(numInstances):
            cid, aid, words, apos = self.wordIntrusion(numWords = WI_WORDS)
            w0, w1, w2, w3, w4, w5 = words # FIXME this assumes WI_WORDS==5, make more flexible
            print template % locals()
#endclass WordIntrusion


def topicIntrusion(useTop = 3, fromBottom = 10):
    method = getRandomMethod()
    document = getRandomDocument()
    conceptScores = getConceptScores(method, document)
    conceptScores.sort(reverse = True) # best scores first
    concepts = conceptScores[: useTop]
    alienConcept = random.choice(conceptScores[-fromBottom : ])
    alienPos = random.randint(0, useTop)
    concepts.insert(alienPos, alienConcept)
    return method, document, alienPos, concepts



# ============= main entry point ================
if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG)
    logging.info("running %s" % " ".join(sys.argv))
    
    program = os.path.basename(sys.argv[0])
    
    # make sure we have enough cmd line parameters
    if len(sys.argv) < 2:
        print globals()["__doc__"]
        sys.exit(1)
    
    # parse cmd line
    numInstances = int(sys.argv[1])
    conceptFile = sys.argv[2]
    if 'word' in program:
        wi = WordIntrusion(conceptFile)
        wi.printProtocol(numInstances)
    elif 'topic' in program:
        ti = TopicIntrusion(conceptFile)
        ti.printProtocol(numInstances)
    else:
        print globals()["__doc__"]
        sys.exit(1)

    logging.info("finished running %s" % program)

