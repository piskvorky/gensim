# TODO pridat DocumentCollection.check(): test na prazdne dokumenty, konzistenci, chybove stavy atd

import logging
import os.path
import os
import math
import numpy
import scipy
import utils_dml
import bisect
import itertools
import codecs

def saveMapping(lex, fname):
    logging.info("saving lexicon to %s" % fname)
    if isinstance(lex, dict):
        items = sorted(lex.iteritems())
    else:
        items = enumerate(lex)
    f = codecs.open(fname, 'w', encoding = 'utf8')
    for item in items:
        f.write("%s\t%s\n" % item)
    f.close()

class DocumentCollection:

    def __init__(self):
        self.docs = []
        self.setIDFs(None)
        self.dictionary = None
    
    def saveLex(self, fpath, prefix):
        saveMapping(self.getDocIds(), os.path.join(fpath, prefix + '_docids.txt'))
        dic = self.getDictionary()
        rdic = utils_dml.reverseDict(dic)
        saveMapping(rdic, os.path.join(fpath, prefix + '_tokenids.txt'))
        saveMapping(dict([(rdic[k], v) for k, v in self.getDocFreqMap().iteritems()]), os.path.join(fpath, prefix + '_docfreq.txt'))
        saveMapping(dict([(rdic[k], v) for k, v in self.getFreqMap().iteritems()]), os.path.join(fpath, prefix + '_colfreq.txt'))

    def docPos(self, docid):
        """return self.docs position of document with docid; -1 if no such document exists"""
        for i in xrange(len(self.docs)):
            if self.docs[i].getId() == docid:
                return i
        return -1

    def docExists(self, docid):
        return self.docPos(docid) >= 0
    
    def addDocument(self, doc):
        docid = doc.getId()
        if (self.docExists(docid)):
            logging.error("cannot add document: id '%s' already exists!" % docid)
        self.docs.append(doc)

    def addDocuments(self, docs):
        ids = set(self.getDocIds())
        for doc in docs:
            docid = doc.getId()
            if not docid in ids: # TODO add dirty flags for content change and simply call addDocument() repeatedly
                self.docs.append(doc)
                ids.add(docid)
            else:
                logging.error("cannot add document: id '%s' already exists!" % docid)

    def removeDocument(self, docid):
        pos = self.docPos(docid)
        if (pos >= 0):
            del self.docs[pos]
        else:
            logging.error("cannot remove document: id '%s' does not exist!" % docid)

    def removeEmptyDocuments(self):
        empty = []
        for doc in self.getDocs():
            if len(doc.getTokenIds()) == 0:
               logging.warning("removing empty article at %s" % doc.getId())
               empty.append(doc.getId())
        for empt in empty:
            self.removeDocument(empt)

    def numDocs(self):
        return len(self.docs)

    def getDocMap(self):
        """return collection as a docid -> Document mapping"""
        return dict(zip(self.getDocIds(), self.getDocs()))
    
    def getDocs(self):
        """return collection as a list of Document references"""
        return self.docs

    def getDocIds(self):
        """return ids of documents in the collection"""
        return [doc.getId() for doc in self.getDocs()]

    def mergeTokenIds(self):
        result = []
        for doc in self.getDocs():
            result.extend(doc.getTokenIds())
        return result

    def getNumTokens(self):
        result = 0
        for doc in self.getDocs():
            result += len(doc.getTokenIds())
        return result

    def getUniqueTokens(self):
        result = set()
        for doc in self.getDocs():
            result.update(doc.getTokenIds())
        return result

    def getNumUniqueTokens(self):
        return len(self.getUniqueTokens())

    def clearTokens(self):
        for doc in self.getDocs():
            doc.setTokens(None)

    def clearTexts(self):
        for doc in self.getDocs():
            doc.setText(None)

    def clearPositions(self):
        for doc in self.getDocs():
            doc.setTokenPositions(None)

    def clearIds(self):
        for doc in self.getDocs():
            doc.setTokenIds(None)

    def filterIds(self, filterSet, rebuild = False):
        """from all collection documents, remove all token ids contained in filterSet
        optionally rebuilds the dictionary to get rid of gaps in id sequence"""
        ## FIXME: !!doesn't update tokens and tokenPositions!!
        filterSet = frozenset(filterSet)
        logging.debug("filtering out %i terms" % len(filterSet))
        for doc in self.getDocs():
            tokenIds = doc.getTokenIds()
            if tokenIds == None:
                logging.error('DocumentCollection.filterIds called without ids in %s' % doc.getId())
            keep = [i for i in xrange(len(tokenIds)) if tokenIds[i] not in filterSet]
            doc.setTokenIds([tokenIds[i] for i in keep])
            tokens = doc.getTokens()
            if tokens != None:
                doc.setTokens([tokens[i] for i in keep])
            pos = doc.getTokenPositions()
            if pos != None:
                doc.setTokenPositions([pos[i] for i in keep])
        if rebuild:
            self.rebuildDictionary()
    
    def getDictionary(self):
        """return map that assigns each distinct token a unique number; the unique number sequence starts at 1"""
        if self.dictionary == None:
            self.createDictionary()
        return self.dictionary

    def createDictionary(self):
        tokens = []
        for doc in self.getDocs():
            doctokens = doc.getTokens()
            if doctokens == None:
                logging.error("DocumentCollection.createDictionary called but keepTokens is False for %s" % doc.getId())
                return {}
            tokens.extend(doctokens)
        self.dictionary = utils_dml.text2dict(tokens)
        for doc in self.getDocs():
            doc.setTokenIds(utils_dml.text2vect(doc.getTokens(), self.dictionary))

    def rebuildDictionary(self):
        """remove entries which have no corresponding ids in the database, shrink resulting gaps"""
        logging.debug("rebuilding dictionary, shrinking gaps")
        uids = sorted(self.getUniqueTokens()) # get unique ids present in the database
        newids = range(len(uids)) # new ids
        revold = utils_dml.reverseMap(self.getDictionary())
        idmap = dict(zip(uids, newids))
        newdict = {}
        for uid in uids: # generate new dictionary
            newdict[revold[uid]] = idmap[uid]
        for doc in self.getDocs(): # update ids in documents to reflect the new dictionary
            doc.setTokenIds([idmap[i] for i in doc.getTokenIds()])
        self.dictionary = newdict

    def getBOWMatrix(self): # FIXME not sparse atm; scipy.sparse is crap
        """return collection as sparse term-by-document matrix (index order is array[term,document]=frequency)"""

        dictionary = self.getDictionary()
        docs = self.getDocs()
        numterms = max(dictionary.values()) + 1
        logging.info("constructing %ix%i BOW matrix" % (numterms, len(docs)))

        # build the matrix
        result = numpy.empty((numterms, len(docs)), dtype = int)

        for i in xrange(len(docs)):
            result[:, i] = docs[i].getFull(length = numterms)
##        result = numpy.column_stack([doc.getFull() for doc in self.getDocs()])
        
        # print some stats
        reverseDictionary = utils_dml.reverseMap(dictionary)
        docids = self.getDocIds()
        marginal = [numpy.sum(result, axis = ax) for ax in range(2)] # marginal sums along either axis
        empty = [numpy.compress(marginal[ax] == 0, range(len(marginal[ax]))) for ax in range(2)] # indices of empty columns/rows
        logging.info("%i empty BOW document vectors: " % len(empty[0]) + str(zip(empty[0], map(docids.__getitem__, empty[0]))))
        logging.info("%i empty BOW term vectors:" % len(empty[1]) + str(zip(empty[1], map(reverseDictionary.get, empty[1]))))
        zeroElems = len(result.nonzero()[0])
        logging.info("BOW sparsity: %i/%i = %.3f%%" % (zeroElems, result.size, 100.0 * zeroElems / result.size))
        return result

    def getIDFs(self, bow = None):
        if (self.idfs != None):
            return self.idfs
        if bow == None:
            bow = self.getBOWMatrix()
        docfreq = numpy.array([len(numpy.compress(row > 0, row)) for row in bow])
        result = numpy.log((1.0 * bow.shape[1]) / docfreq) / numpy.log(2)
        result[numpy.isinf(result)] = 0.0 # HAX replace INFs with 0.0
        return result

    def setIDFs(self, idfs):
        self.idfs = idfs

    def getTFIDFMatrix(self, sparse = False):
        """construct tf*idf collection matrix. """

        docs = self.getDocs()
        dictionary = self.getDictionary()
        if not dictionary:
            raise Exception("cannot build matrix from an empty collection; chances are something went wrong!")
        numterms = max(dictionary.values()) + 1

        logging.info("constructing %ix%i TFIDF matrix, sparse = %s" % (numterms, len(docs), sparse))

        if sparse:
            result = scipy.sparse.lil_matrix((numterms, len(docs)), dtype = numpy.float32)
            for i in xrange(len(docs)):
                if (i + 1) % 1000 == 0:
                    logging.info(" progress: at vector #%i/%i" % (i, len(docs)))
                vec, total = utils_dml.vect2bow_sorted(docs[i].getTokenIds())
                if total > 0:
                    itotal = 1.0 / total
                    for id, idfreq in vec:
                        result.rows[id].append(i) # vec must be sorted by id in order for this to work!
                        result.data[id].append(itotal * idfreq)
            nnz = numpy.array([len(result.data[j]) for j in xrange(result.shape[0])])
            idfs = numpy.log((1.0 * result.shape[1]) / nnz) / numpy.log(2)
            idfs[numpy.isinf(idfs)] = 0.0 # HAX replace INFs with 0.0 - is this ok?
            self.setIDFs(idfs)
            for iterm in xrange(result.shape[0]): # multiply by IDF
                if (iterm + 1) % 10000 == 0:
                    logging.info(" progress: idf#%i" % iterm)
                result.data[iterm] = [val * idfs[iterm] for val in result.data[iterm]]
            result = result.tocsr()
            logging.info("TFIDF sparsity: %i/%i = %.3f%%" % (result.getnnz(), result.shape[0] * result.shape[1], 100.0 * result.getnnz() / (result.shape[0] * result.shape[1])))

        else:
            result = numpy.empty((numterms, len(docs)), dtype = numpy.float32)
            for i in xrange(len(docs)):
                if (i + 1) % 10000 == 0:
                    logging.debug(" progress: vec#%i" % i)
                vec = docs[i].getFull(numterms)
                sm = 1.0 * numpy.sum(vec)
                if sm == 0:
                    result[:, i] = 0.0 # HACK: don't scale zero vectors (INF bad?)
                else:
                    result[:, i] = vec / sm

            idfs = self.getIDFs(result)
            self.setIDFs(idfs)

            for iterm in xrange(result.shape[0]): # multiply by IDF
                if (iterm + 1) % 10000 == 0:
                    logging.debug(" progress: idf#%i" % iterm)
                result[iterm, :] *= idfs[iterm]
            # print some stats
            reverseDictionary = utils_dml.reverseMap(self.getDictionary())
            docids = self.getDocIds()
            marginal = [numpy.sum(result, axis = ax) for ax in range(2)] # marginal sums along both axes
            empty = [numpy.compress(marginal[ax] == 0, range(len(marginal[ax]))) for ax in range(2)] # indices of empty columns/rows
            logging.debug("%i empty TFIDF document vectors: " % len(empty[0]) + str(zip(empty[0], map(docids.__getitem__, empty[0]))))
            logging.debug("%i empty TFIDF term vectors: " % len(empty[1])  + str(zip(empty[1], map(reverseDictionary.get, empty[1]))))
            zeroElems = len(result.nonzero()[0])
            nearZeroElems = len(result.compress(result.flat >= 1e-3))
            logging.info("TFIDF sparsity: %i/%i = %.3f%% (%.3f%% < 1e-3)" % (zeroElems, result.size, 100.0 * zeroElems / result.size, 100.0 * nearZeroElems / result.size))
        return result
            
    def getFreqMap(self):
        """return tokenid -> collection frequency mapping"""
        merged = self.mergeTokenIds()
        merged.sort()
        result = {}
        last = None
        cnt = 0
        for token in merged:
            if last == token:
                cnt += 1
            else:
                result[last] = cnt
                cnt = 1
                last = token
        result[last] = cnt
        del result[None]
#        unique = frozenset(merged)
#        result = dict(zip(unique, map(merged.count, unique)))
        return result

    def getDocFreqMap(self):
        """return tokenid -> document frequency mapping"""
        docs = self.getDocs()
        doctokenIds = []
        for doc in docs:
            doctokenIds.extend(set(doc.getTokenIds()))
        doctokenIds.sort()
        unique = frozenset(doctokenIds)
        freqs = map(lambda tokenid: bisect.bisect_right(doctokenIds, tokenid) - bisect.bisect_left(doctokenIds, tokenid), unique) # OPT suboptimal, make linear
        result = dict(zip(unique, freqs))
        return result

    def freqRange(self, freqmap, minFreq, maxFreq):
        """return token ids that have freqmap frequency of minFreq <= frequency < maxFreq"""
        result = [tokenid for tokenid, freq in freqmap.items() if minFreq <= freq < maxFreq]
        return result

    def filterDF1(self, dfm = None, rebuild = True):
        """filter out tokens that appear in only one document"""
        if (dfm == None): # TODO set 'dirty' flag for the collection & recompute self.maps automatically at call time + set dirty on addDocument()
            dfm = self.getDocFreqMap()
        rem = self.freqRange(dfm, 0, 2)
        self.filterIds(rem, rebuild = rebuild)

    def filterExtremes(self, extremes = 1, dfm = None, rebuild = True):
        """filter out token ids that appear in at most 'extremes' documents or at least '#docs-extremes' documents"""
        if (dfm == None):
            dfm = self.getDocFreqMap()
        rem = self.freqRange(dfm, 0, extremes + 1) # terms with DF <= extremes
        rem.extend(self.freqRange(dfm, self.numDocs() - extremes, self.numDocs() + 1)) # terms with DF >= (#docs - extremes)
        self.filterIds(rem, rebuild = rebuild)

    def fromDir(directory, documentFactory, walk = False, accept = None):
        """create document collection from files in directory (+all subdirs if walk == True)"""
        if documentFactory.sourceType != 'file':
            logging.warning('DocumentCollection.fromDir called with source type different from "file"!')

        if walk:
            filenames = []
            for root, dirs, files in os.walk(directory):
                filenames.extend([os.path.join(root, filename) for filename in files])
        else:        
            filenames = [os.path.join(directory, fname) for fname in os.listdir(directory) if os.path.isfile(os.path.join(directory, fname))]
        if accept != None:
            filenames = [filename for filename in filenames if accept(filename)]
        filenames.sort()
#        filenames = filenames[:10]
        logging.info('creating collection of %i documents from %s' % (len(filenames), directory))
        result = DocumentCollection()
        result.dictionary = {}
        result.addDocuments(documentFactory.createDocuments(filenames, None, dictionary = result.dictionary)) # store files as documents, with docid = filename
        ids = result.mergeTokenIds()
        logging.info('%i tokens (%i unique) in %i documents' % (len(ids), len(set(ids)), result.numDocs()))
        return result

    fromDir = utils_dml.Callable(fromDir)


if __name__ == '__main__':
    import document
    
    logging.basicConfig(level = logging.DEBUG)
    texts = {
        'c1' : 'Human machine interface for ABC computer application.',
        'c2' : 'A survey of user opinion of computer system response time.',
        'c3' : 'The EPS user interface management system.',
        'c4' : 'System and human system engineering testing of EPS.',
        'c5' : 'Relation of user perceived response time to error measurement.',
        'm1' : 'The generation of random, binary, ordered trees.',
        'm2' : 'The intersection graph of paths in trees.',
        'm3' : 'Graph minors IV: Widths of trees and well-quasi-ordering.',
        'm4' : 'Graph minors: A survey.'}
    df = document.DocumentFactory(lowercase = True)
    coll = DocumentCollection()
    for docid, text in texts.iteritems():
        coll.addDocument(df.createDocument(text, docid))
    print coll.getDictionary()
    print coll.getDocFreqMap()
    coll.removeDocument('test')
    coll.removeDocument('m1')
    coll.addDocument(df.createDocument(".", "empty_doc"))
    coll.addDocument(df.createDocument("minors graph eps trees system computer survey user human time interface response.", "full_doc"))
    if not coll.docExists('m1'):
        coll.addDocument(df.createDocument(texts['m1'], 'brekeke'))
        
    coll.createDictionary()
    
    for doc in coll.getDocs():
        print "dc1: %s (%i):" % (doc.getId(), len(doc.getTokenIds())), doc.getTokenIds()
    print coll.getDictionary()
    
    mat = coll.getBOWMatrix()
    dfm = coll.getDocFreqMap()
    stopList = ['a','and','of','the',':', 'totallyirrelevant'] # fixed stoplist
    stopIds = utils_dml.text2vect(stopList, coll.getDictionary())
    stopIds.extend(coll.freqRange(dfm, 0, 2)) # extend stopIds with ids that have 0 <= document frequency < 2
    print 'stoplist = ', map(utils_dml.reverseMap(coll.getDictionary()).__getitem__, stopIds)

    for doc in coll.getDocs():
        print "before filter: %s (%i):" % (doc.getId(), len(doc.getTokenIds())), zip(doc.getTokenIds(), doc.getTokens())
        
    coll.filterIds(stopIds) # remove unwanted tokens
    
    for doc in coll.getDocs():
        print "after filter, before rebuild: %s (%i):" % (doc.getId(), len(doc.getTokenIds())), zip(doc.getTokenIds(), doc.getTokens())
        
    coll.rebuildDictionary()
    for doc in coll.getDocs():
        print "after rebuild: %s (%i):" % (doc.getId(), len(doc.getTokenIds())), zip(doc.getTokenIds(), doc.getTokens())
        
    coll.filterExtremes(1)

    for doc in coll.getDocs():
        print "after extremes: %s (%i):" % (doc.getId(), len(doc.getTokenIds())), zip(doc.getTokenIds(), doc.getTokens())

    coll.createDictionary()
    
    for doc in coll.getDocs():
        print "after create: %s (%i):" % (doc.getId(), len(doc.getTokenIds())), zip(doc.getTokenIds(), doc.getTokens())

    print 'new dictionary = ', coll.getDictionary()
    mat = coll.getBOWMatrix()
    print mat.dtype, mat
    mat = coll.getTFIDFMatrix()
    print mat.dtype, mat

    basedir = '/home/radim/workspace/plagiarism/data/Karrigell/tst/original/'
##    basedir = g:\\school\\phd\\20news\\alt.atheism'
    coll2 = DocumentCollection.fromDir(
        basedir,
        document.DocumentFactory(encoding = 'utf8', sourceType = 'file', lowercase = True, keepTexts = False, keepTokens = False),
        walk = True)
##    coll2.addDocuments(df.createDocuments(texts.values(), 0, dictionary = coll2.dictionary)) # add texts, with docid = document sequence number
    coll2.clearTexts()
    coll2.clearTokens()
#    import matutils
#    matutils.saveMatrix(coll2.getTFIDFMatrix(sparse = True), '/home/radim/workspace/plagiarism/data/tfidf.mm', sparse = True)
##    coll2.filterExtremes(1)
    coll2.filterDF1()
#    coll2.rebuildDictionary()
    dic = sorted(coll2.getDictionary().items(), key = lambda (a, b): b)
    print dic[:20], dic[-20:]
    mats = coll2.getTFIDFMatrix(sparse = True)
    mat = coll2.getTFIDFMatrix(sparse = False)
    print 'full[0]:', mat[0, :20]
    print '%i, sparse[0] (%i):' % (mats.getnnz(), mats[0, :].getnnz()),
    print mats[0, :20]
##    dic = coll2.getDictionary()
    print mat.shape, mat.dtype, len(dic)
