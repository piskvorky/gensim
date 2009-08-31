#!/usr/bin/env python

import logging
import math
import os.path

import numpy

import docsim
import matutils
import common
import DocumentCollection
import mscs
import ipyutils
import utils_dml
import ArticleDB
import Article
import utils_iddb


ARTS_FILE = common.dbFile('mscs', 'serial') # persistent db (file) where all mscs info is stored


def rmseFile(matfile1, matfile2):
    data1 = matutils.loadMatrix(common.matrixFile(matfile1))
    data2 = matutils.loadMatrix(common.matrixFile(matfile2))
    return matutils.rmse(data1, data2)

def loadMsc2Id(lang):
    cats = [item.strip().split('\t') for item in open(common.dataFile('serial_mscids_%s.txt' % lang)).readlines()]
    cats = dict([(int(id), val) for id, val in cats]) # map int -> id
    rcats = dict([(v, k) for k, v in cats.iteritems()]) # map id -> int
    return cats, rcats

def saveMsc2Id(cats, lang):
    DocumentCollection.saveLex(cats, common.dataFile('serial_mscids_%s.txt' % lang))

def createMscsDb():
    """Create MSC database of all languages."""
    db = ArticleDB.ArticleDB(ARTS_FILE, mode = 'override', autocommit = False)
    baseDir = ''

    proc_total = 0
    logging.info("processing directory %s" % common.inputPath(baseDir))
    for root, dirs, files in os.walk(common.inputPath(baseDir)):
        root = os.path.normpath(root)
        if os.path.basename(root).startswith('#'):
            proc_total += 1
            try:
                meta = utils_iddb.parseMeta(os.path.join(root, 'meta.xml'))
                try:
                    meta['body'] = open(os.path.join(root, 'fulltext.txt')).read()
                except Exception, e:
                    meta['body'] = None
                meta['id_int'] = root[len(common.INPUT_PATH) + 1: ]
                meta['references'] = None # TODO add
                art = Article.Article(record = meta)
                db.insertArticle(art)
            except Exception, e:
                logging.warning('invalid entries in %s; ignoring article (%s)' % (root, e))
                continue
    db.commit()

    logging.info('%i directories processed' % proc_total)
    logging.info('%i articles in the database' % len(db))
    return db

def buildMscPureMatrix(lang):
    if os.path.exists(ARTS_FILE):
        logging.warning("SKIPPING creating MSC database from meta.xml files (using old %s); is this what you want?" % ARTS_FILE)
    else:
        logging.info("creating MSC database from meta.xml files")
        createMscsDb() # only run this when collection changes (test for file existence / delete file explicitly?)
    arts = [art for art in docsim.getArts(ARTS_FILE, acceptNoBody = True, acceptNoMsc = False) if art.language == lang or lang == 'any']
    
    cats = dict(enumerate(mscs.getFreqMSCs(arts, minFreq = 1, useMains = False)))
    rcats = utils_dml.reverseMap(cats)
    saveMsc2Id(cats, lang)
    
    logging.info("building MSC binary matrix")
    resultBin = numpy.zeros((len(cats), len(cats)), dtype = numpy.float32) # binary msc similarity (with fixed msc hierarchy = identity)
    for idi, cati in cats.iteritems():
        for idj, catj in cats.iteritems():
#            print idi, cati, idj, catj
            if idi == idj:
                resultBin[idi, idj] = 1.0
            else:
                resultBin[idi, idj] = 0.0
    matutils.saveMatrix(resultBin, common.matrixFile("mscs_bin_%s.mm" % lang), sparse = False)
    return resultBin
    
def buildMscOverlapMatrix(lang):
    logging.info("building MSC overlap matrix")

    arts = [art for art in docsim.getArts(ARTS_FILE, acceptNoBody = True, acceptNoMsc = False) if art.language == lang or lang == 'any']
    cats, rcats = loadMsc2Id(lang) # from buildPure
    
    logging.info("computing MSC matrices")
    
    overlap = numpy.zeros((len(cats), len(cats)), dtype = int) # binary msc similarity (with fixed msc hierarchy = identity)
    for art in arts:
        for msc1 in art.msc:
            for msc2 in art.msc:
                overlap[rcats[mscs.niceMSC(msc1)[0]], rcats[mscs.niceMSC(msc2)[0]]] += 1
                
    resultOverlap = numpy.zeros((len(cats), len(cats)), dtype = numpy.float32)
    for i in xrange(resultOverlap.shape[0]):
        max = numpy.max(overlap[i])
        for j in xrange(resultOverlap.shape[1]):
            resultOverlap[i, j] = math.log(1.0 + 100.0 * overlap[i, j] / max) / math.log(101)
    
    matutils.saveMatrix(resultOverlap, common.matrixFile("mscs_overlap_%s.mm" % lang), sparse = False)
    return resultOverlap

def buildMscCentroidMatrix(language):
    logging.info("building MSC centroid matrix from %s" % ARTS_FILE)
    arts = [art for art in docsim.getArts(ARTS_FILE, acceptNoBody = False, acceptNoMsc = False) if art.language == language or language == 'any']
    prefix = 'mscs_serial_%s_' % language
    matFile = common.matrixFile(prefix + 'TFIDF_T.mm')
    if os.path.exists(matFile):
         logging.warning('SKIPPING creating TFIDF matrix for %s (file %s present). Is this what you wanted?' % (language, matFile))
         tfidf = matutils.loadMatrix(matFile).tocsr()
    else:
         logging.info('creating TFIDF matrix for %s to %s' % (language, matFile))
         tfidf = docsim.buildTFIDFMatrices(arts, prefix = prefix, saveMatrices = False).tocsr()
    
    ipyutils.loadDicts(prefix = prefix)
    arts = [art for art in arts if art.id_int in ipyutils.rdocids] # remove articles that had empty body (according to their tfidf vector)
    if len(ipyutils.rdocids) != len(arts):
        logging.error("no. of TFIDF document = %i, but there are %i documents in the database (mismatch)" % (len(ipyutils.rdocids), len(arts)))
        raise Exception("different size of database/dictionary; version mismatch?")

    cats, rcats = loadMsc2Id(language) # from buildPure
#    print "mscs:", cats

    logging.info("loading tfidf collection matrix (for centroids)")
    tfidf = matutils.loadMatrix(common.matrixFile('gensim_' + language + 'TFIDF_T.mm')).tocsr()
    logging.debug("loaded %ix%i matrix" % tfidf.shape)

    logging.info("computing centroids")
    centroids = numpy.zeros((len(cats), tfidf.shape[1]), numpy.float)
#    print "centroids.shape =", centroids.shape
    num = numpy.zeros((len(cats),), numpy.int)
    artCnt = 0
    for art in arts:
        if not art.id_int in ipyutils.rdocids:
            logging.warning("article not found among docids: %s" % art)
            continue
        artCnt += 1
        artId = ipyutils.rdocids[art.id_int]
        tops = [mscs.niceMSC(msc)[0] for msc in art.msc]
        tops = set(tops) # only count each top-level once (comment out this line to count e.g. 30H55 and 30.13 twice for this article, as cat. 30)
        for top in tops:
            mscId = rcats[top]
            vec = tfidf[artId].toarray()
            vec.shape = (vec.size, )
#            print "vec.shape = ", vec.shape
            centroids[mscId] += vec
            num[mscId] += 1
        if artCnt < 10 or artCnt % 1000 == 0:
            logging.debug("sanity check - article %s has id %i and has mscs=%s, mscsIds=%s" % 
                          (art.id_int, artId, art.msc, [rcats[mscs.niceMSC(msc)[0]] for msc in art.msc]))
    if not artCnt == tfidf.shape[0]:
        raise Exception("not all articles used; database/matrix mismatch?")
    for i, vec in enumerate(centroids):
        logging.info("centroid for msc %s (id %i) is an average of %i vectors" % (cats[i], i, num[i]))
        if numpy.sum(numpy.abs(vec)) == 0:
            logging.warning("empty centroid for msc %s (msc int id %i)" % (cats[i], i))
    for mscId in cats.iterkeys():
        centroids[mscId] /= num[mscId]
    logging.info("used %i articles for %i vectors (articles may have more than one msc and so can be counted more than once)" % (artCnt, sum(num)))

    logging.info("computing MSC centroid matrix")
    resultCentroid = numpy.zeros((len(cats), len(cats)), dtype = numpy.float32)
    for idi, cati in cats.iteritems():
        for idj, catj in cats.iteritems():
#            print idi, cati, idj, catj
            sim = matutils.cossim(centroids[idi], centroids[idj])
            if numpy.isfinite(sim):
                resultCentroid[idi, idj] = sim
            else:
                resultCentroid[idi, idj] = 0.0

    matutils.saveMatrix(resultCentroid, common.matrixFile("mscs_centroid_%s.mm" % language), sparse = False)

def visualize(docdoc, arts):
    import matplotlib
    mscid = [(art.msc, art.id_int) for art in arts]
    mscid.sort()
#    sort arts by (primary?) msc, plot 2d gray scale matrix of docdoc, docdoc vs docmsc matrix
                           
def buildMSCMatrices(language):
    buildMscPureMatrix(language)
    buildMscOverlapMatrix(language)
    buildMscCentroidMatrix(language)
    logging.info("for %s language:", language)
    logging.info("rmse(resultBin, resultOverlap) = %f", rmseFile("mscs_bin.mm", "mscs_overlap.mm"))
    logging.info("rmse(resultBin, resultCentroid) = %f", rmseFile("mscs_bin.mm", "mscs_centroid_%s.mm" % language))
    logging.info("rmse(resultOverlap, resultCentroid) = %f", rmseFile("mscs_overlap.mm", "mscs_centroid_%s.mm" % language))

def buildDocDoc(arts, type, language):
    ipyutils.loadDicts(prefix = 'gensim_' + language)
    arts = [art for art in arts if art.id_int in ipyutils.rdocids]
    assert(len(arts) == len(ipyutils.rdocids))

    logging.info("loading msc<->id mapping")
    cats, rcats = loadMsc2Id(language)

    mscsFile = common.matrixFile("mscs_%s.mm" % type)
    matMsc = matutils.loadMatrix(mscsFile)
    mscDict = {}
    for art in arts:
        artId = ipyutils.rdocids[art.id_int]
        mscIds = [rcats[mscs.niceMSC(msc)[0]] for msc in art.msc]
        mscDict[artId] = mscIds
    
    logging.info("computing doc*doc similarity matrix based on %s" % mscsFile)
    docdoc = numpy.zeros((len(arts), len(arts)), numpy.float32)

    for i in xrange(len(arts)):
        if i % 100 == 0:
            logging.info("PROGRESS: %i/%i" % (i, len(arts)))
        art1Id = ipyutils.rdocids[arts[i].id_int]
        for j in xrange(i, len(arts)):
            art2Id = ipyutils.rdocids[arts[j].id_int]
            bestScore = 0.0
            for msc1Id in mscDict[art1Id]:
                for msc2Id in mscDict[art2Id]:
                    bestScore = max(bestScore, matMsc[msc1Id, msc2Id])
            docdoc[art1Id, art2Id] = docdoc[art2Id, art1Id] = bestScore

    matutils.saveMatrix(docdoc, common.matrixFile("docdoc_" + language + "_%s.mm" % type), sparse = False)
    return docdoc

if __name__ == "__main__":
    import sys
    import os.path
    logging.basicConfig(level = common.PRINT_LEVEL) # set to logging.INFO to print statistics as well
    logging.root.level = 10

    language = 'eng'
    buildMSCMatrices(language)
#    arts = docsim.getArts(common.dbFile('gensim', language), acceptNoBody = False, acceptNoMsc = False)
#    buildDocDoc(arts, "bin", language)
#    buildDocDoc(arts, "overlap", language)
#    buildDocDoc(arts, "centroid_%s" % language, language)
    
#    data1 = matutils.loadMatrix(common.matrixFile("docdoc_eng_bin.mm"))
#    import utils_plot
#    utils_plot.viewMatrix(data1)
#    ascas
#    print "rmse(bin, tfidf_cossim) = ", rmseFile("docdoc_eng_bin.mm", "gensim_engTFIDFsim.mm")
#    print "rmse(overlap, tfidf_cossim) = ", rmseFile("docdoc_eng_overlap.mm", "gensim_engTFIDFsim.mm")
#    print "rmse(centroid, tfidf_cossim) = ", rmseFile("docdoc_eng_centroid.mm", "gensim_engTFIDFsim.mm")
#    print "rmse(bin, lsi_cossim) = ", rmseFile("docdoc_eng_bin.mm", "gensim_engLSIsim.mm")
#    print "rmse(overlap, lsi_cossim) = ", rmseFile("docdoc_eng_overlap.mm", "gensim_engLSIsim.mm")
#    print "rmse(centroid, lsi_cossim) = ", rmseFile("docdoc_eng_centroid.mm", "gensim_engLSIsim.mm")
        
#build doc*doc matrix of msc
#    a_ij = max{msc_ij pro i,j vsechny kombinace msc obou clanku]}
#    
#korelace tfidf cossim matice s msc matici
