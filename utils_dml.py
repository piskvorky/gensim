import logging
import math
import numpy
import codecs
import cPickle
import shutil

class Callable:
    def __init__(self, anycallable):
        self.__call__ = anycallable

def vect2ngrams(vect, n):
    """return set of all n-gram tuples on the input vector."""
    _ngrams = set() # store intermediate results in set for faster lookup (hash)
    for i in range(n, len(vect) + 1):
        ngr = tuple(vect[i - n: i])
        _ngrams.add(ngr)
    return list(_ngrams) #convert set of tuples into list of tuples

def text2dict(tokens):
    """return dictionary mapping token -> tokenid"""
    uniqueTokens = sorted(list(set(tokens)))
    result = dict(zip(uniqueTokens, range(len(uniqueTokens))))
    return result

def vect2bow(_ids):
    """Convert sequence of ids into (id->id_frequency) dictionary."""
    ids = sorted(_ids)
    result = {}
    last = None
    cnt = 0
    for id in ids:
        if last == id:
            cnt += 1
        else:
            result[last] = cnt
            cnt = 1
            last = id
    result[last] = cnt
    del result[None]
#    uniqueids = set(ids) # get unique ids
#    if None in uniqueids:
#        uniqueids.remove(None) # remove out-of-dictionary id
#    result = dict(zip(uniqueids, map(ids.count, uniqueids)))
    return result

def vect2bow_sorted(_ids):
    """Convert sequence of ids into (id->id_frequency) list, sorted by 'id' increasingly."""
    if len(_ids) == 0:
        return []
    ids = sorted(_ids)
    result = []
    last = ids[0]
    cnt = 0
    total = 0
    for id in ids:
        if last == id:
            cnt += 1
        else:
            result.append((last, cnt))
            total += cnt
            cnt = 1
            last = id
    total += cnt
    result.append((last, cnt))
    return result, total

def text2vect(wrds, dictionary, newwords = {}):
    """Convert sequence of strings into list of integers, using dictionary. updates dictionary with new words."""
    result = []
    for word in wrds:
        if word not in dictionary:
            a = len(dictionary)
            dictionary[word] = a
            newwords[word] = a
        else:
            a = dictionary[word]
        result.append(a)
    return result

def readfile(fname, encoding = 'iso-8859-2'):
    """Read in file as a single huge string."""
    result = ''
    try:
        f = codecs.open(fname, 'r', encoding)
    except IOError, detail:
        logging.error('failed to open file : ' + str(detail))
    else:
        result = f.read()
        f.close()
    return result

def reverseMap(dictionary):
    """for a->b map, return b->a map; conflicting elements can be resolved randomly"""
    return dict([(v, k) for k, v in dictionary.iteritems()])

reverseDict = reverseMap

def loadPickle(fname):
    f = open(fname, 'r')
    try:
        self.index = cPickle.load(f)
    except Exception, mess:
        logging.critical("could not load from %s: %s" % (fname, mess))

def savePickle(obj, fname):
    f = open(fname, 'w')
    try:
        cPickle.dump(obj, f)
    except Exception, mess:
        logging.critical("could not save to %s: %s" % (fname, mess))

def backup(filename):
    try:
        shutil.copyfile(filename, filename + ".bup")
    except Exception, mess:
        logging.error("could not backup file %s: %s" % (fname, mess))
    
def segAsHtml(filename, fragments):
    import codecs
    fout = codecs.open(filename, 'w', 'utf8')
    fout.write('<HTML>\n<HEAD />\n<BODY>\n<PRE>\n<CODE>\n')
    colours = ['yellow', 'red']
    col = 0
    for fragment in fragments:
        fout.write('<FONT style="BACKGROUND-COLOR: %s">' % colours[col])
        fout.write(fragment)
        fout.write('</FONT>')
        col = 1 - col
    fout.write('\n</CODE>/n</PRE>\n</BODY>\n</HTML>')

    fout.close()

