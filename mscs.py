#!/usr/bin/env python

# needs python>=2.4 because of sets, sorted, etc.

import PyDbLite
import logging
import Article
import common
import os.path

def makeDefs():
    deffile = common.dataFile('mscdefs.txt')
    result = {}
    for mscdef in open(deffile):
        result[mscdef[ : 2]] = mscdef
    import cPickle
    f = open(common.dataFile('mscdefs.pkl'), 'w')
    cPickle.dump(result, f)
    f.close()
    return result

def niceMSC(_msc, prefix = 2):
    """get rid of weird characters, split into top classification (first two digits) and the rest"""
    msc = _msc.strip()
    isMain = msc.startswith('*')
    if isMain:
        msc = msc[1 : ]
    if len(msc) == 4:
        msc = '0' + msc
#    if not len(msc) == 5:
#        raise Exception("strange MSC format in %s" % _msc)
    top = (msc[ : prefix])
    rest = msc[prefix : ]
    return top, rest, isMain

def printStats(arts):
    """print some MSC stats for an article database (ArticleDB)"""
    mscs, mains = getMSCCnts(arts)
    uniq_mscs = set(mscs.keys())
    logging.info("#categories present in the db = %i" % len(uniq_mscs))
#    print uniq_mscs
    mscdefs = set(makeDefs().keys())
    if not uniq_mscs.issubset(mscdefs):
#        print 'db:', uniq_mscs
#        print 'defs:', mscdefs
        logging.warning("unrecognized MSC 2-digit code(s) present in the database: %s" % sorted(uniq_mscs - mscdefs))
#    uniq_mscs = uniq_mscs.union(mscdefs)
    logging.info('id\ttotal\tprimary')
    logging.info('==============================')
    for msc in sorted(list(uniq_mscs)):
        logging.info("%s\t%i\t%i" % (msc, len(mscs.setdefault(msc, [])), len(mains.setdefault(msc, []))))
    logging.info('==============================')
    len_mscs = [len(val) for val in mscs.itervalues()]
    len_mains = [len(val) for val in mains.itervalues()]
    logging.info('avg\t%i\t%i' % (sum(len_mscs) / len(mscs), sum(len_mains) / len(mscs)))
    logging.info('median\t%i\t%i' % (sorted(len_mscs)[len(mscs) / 2], sorted(len_mains)[len(mains) / 2]))
    lens = [len(art.msc) for art in arts if art.msc != None]
    logging.info('average MSC codes per article = %.2f' % (1.0 * sum(lens) / len(arts)))
    import cPickle
    cPickle.dump(mscs, open(common.dataFile('mscs_all.pkl'), 'w'), protocol = -1)
    cPickle.dump(mains, open(common.dataFile('mscs_primary.pkl'), 'w'), protocol = -1)    

def getMSCCnts(arts):
    mscs = {}
    mains = {}
    for art in arts:
        if art.msc != None:
            for msc in art.msc:
                top, rest, isMain = niceMSC(msc, prefix = 2)
                if len(art.msc) == 1:
                    isMain = True
                try:
                    if int(top) < 0 or int(top) > 99:
                        raise 'strange MSC %s' % (top)
                except Exception, e:
                    logging.error("%s; ignoring %s" % (e, art))
                else:
                    old = mscs.get(top, [])
                    old.append(art)
                    mscs[top] = old
                    if isMain:
                        old = mains.get(top, [])
                        old.append(art)
                        mains[top] = old
    return mscs, mains
    
def getFreqMSCs(arts, minFreq = 50, useMains = False):
    mscs, mains = getMSCCnts(arts)
    if useMains:
        mscs = mains
    uniqMSCs = set(mscs.keys())
    result = set([msc for msc in sorted(list(uniqMSCs)) if len(mscs[msc]) >= minFreq])
#    print [(msc, len(mscs[msc])) for msc in result]
    return result

if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG)
    import ArticleDB
#    print sorted(makeDefs().keys())

    db = ArticleDB.ArticleDB(common.dataFile('tex_casopis.pdl'), mode = 'open')
    arts = [Article.Article(rec) for rec in db.db if rec['msc']]
#    print len(db)
    logging.info('gathering MSC stats from %i articles' % len(arts))
    printStats(arts)
