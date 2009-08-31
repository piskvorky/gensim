#!/usr/bin/env python

import logging
import re

import common
import ArticleDB
import Article
import mscs

MIN_ARTICLES = 60

BAD_CHARS = range(0, 32) + [ord(ch) for ch in u'<>&']
BAD_MAP = dict(zip(BAD_CHARS, [ord(' ')] * len(BAD_CHARS)))

def getArts(fname, minFreq):
    db = ArticleDB.ArticleDB(common.dataFile(fname), mode = 'open')
    import ipyutils
    language = 'eng'
    ipyutils.loadDicts(prefix = 'gensim_' + language)
    arts = [Article.Article(rec) for rec in db.db if rec['language'] == language]
    for art in arts:
        art.msc = list(set(mscs.niceMSC(msc, prefix = 2)[0] for msc in art.msc))
    logging.info('loaded %i articles from %s' % (len(arts), fname))
    arts = [art for art in arts if len(art.msc)==1 and art.body and art.id_int in ipyutils.rdocids]
    logging.info('extracted %i articles with exactly one MSC and non-empty body' % (len(arts)))
    okmsc = mscs.getFreqMSCs(arts, minFreq, useMains = True)
    mscs.printStats(arts)
    for art in arts:
        art.body = art.body.decode('utf8')
#        art.body = art.body.decode('ascii', 'ignore')
        art.body = art.body.translate(BAD_MAP).encode('utf8')
        if art.title:
            art.title = art.title.decode('utf8')
    #        art.title = art.title.decode('ascii', 'ignore')
            art.title = art.title.translate(BAD_MAP).encode('utf8')
        art.msc = [mscs.niceMSC(art.msc[0], prefix = 2)[0]]
    arts = [art for art in arts if art.msc[0] in okmsc]
    allmsc, mainmsc = mscs.getMSCCnts(arts)
    for msc, mscarts in allmsc.iteritems():
        logging.info("class %s: %i articles" % (msc, len(mscarts)))
    logging.info("======================")
    logging.info("sum: %i articles" % sum(len(mscarts) for mscarts in allmsc.itervalues()))
        
    logging.debug('using %i articles from all %s msc classes that are covered by at least %i articles' % (len(arts), sorted(list(okmsc)), minFreq))
    return arts

PAT_MATH = re.compile('\$.*?\$', re.UNICODE)
PAT_1TOKEN = re.compile('\W', re.UNICODE)

def removeMath(s):
    return re.sub(PAT_MATH, "", s)

def acceptMath(s):
#    import sys
#    sys.path.extend(['..', '/home/radim/workspace/plagiarism/src'])
#    import document
#    df = document.DocumentFactory(lowercase = True, sourceType = 'string', keepTexts = False, keepPositions = False, contentType = 'alphanum', encoding = 'utf8')
#    doc = df.createDocument(s)
    maths = s.split('$')
    res = []
    for num, part in enumerate(maths):
        if num % 2 == 1:
            res.append(re.sub(PAT_1TOKEN, '_', part))
        else:
            res.append(part)
    result = '$'.join(res)
    return result

def noop(s):
    return s

def saveAsCorpus(arts, fname, fnc = lambda c: c):
    logging.info('saving corpus as %s' % fname)
    f = open(common.dataFile(fname), 'w')
    f.write('<?xml version="1.0" encoding="utf-8" ?>\n')
    f.write('<articles>\n')
    for art in arts:
        f.write('<article id="%s" lang="%s">\n' % (art.id_int, art.language))
        if art.msc:
            f.write('<category>\n')
#            assert len(art.msc) == 1
            f.write(art.msc[0])
            f.write('\n</category>\n')
        if art.title:
            f.write('<title>\n')
            f.write(fnc(art.title))
            f.write('\n</title>\n')            
        if art.body:
            f.write('<text>\n')
            f.write(fnc(art.body))
            f.write('\n</text>\n')
        if art.references:
            f.write('<references>\n')
            f.write(art.body)
            f.write('\n</references>\n')
        f.write('</article>\n')
    f.write('</articles>\n')
    f.close()

def loadMSCdict(fname):
    import cPickle
    mscs = cPickle.load(open(common.dataFile(fname), 'r'))
    return mscs

if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG)
#    arts = getArts('main_cmj.pdl')
#    mscs = loadMSCdict('mscs_primary.pkl')
#    saveAsCorpus(arts, 'cmj_eng.corpus')

#    arts = getArts('tex_casopis.pdl', 40)
#    mscs = loadMSCdict('mscs_primary.pkl')
#    saveAsCorpus(arts, 'tex.corpus')

#    arts = getArts('tex_casopis.pdl', 40)
#    mscs = loadMSCdict('mscs_primary.pkl')
#    saveAsCorpus(arts, 'tex_nomath.corpus', fnc = removeMath)

    arts = getArts('serial_msc.pdl', MIN_ARTICLES)
#    mscs = loadMSCdict('mscs_primary.pkl')
    saveAsCorpus(arts, 'serial_msc_eng.min%i.corpus' % MIN_ARTICLES, fnc = noop)
