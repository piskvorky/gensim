import logging
import os.path
import numpy
import scipy.sparse
import utils_dml
from DocumentCollection import DocumentCollection

class Document:
    
    def __init__(self, id=''):
        self.setId(id)
        self.text = None
        self.tokens = None
        self.tokenIds = None
        self.tokenPositions = None
        
    def getId(self):
         return self.id
    
    def setId(self, id):
        self.id = id

    def getText(self):
        return self.text

    def setText(self, text):
        self.text = text
    
    def getTokens(self):
        return self.tokens

    def setTokens(self, tokens):
        self.tokens = tokens
    
    def getTokenIds(self):
        return self.tokenIds

    def setTokenIds(self, tokenIds):
#        if len(tokenIds) == 0:
#            logging.warning("empty article at %s" % self.getId())
        self.tokenIds = tokenIds
    
    def getTokenPositions(self):
        return self.tokenPositions
    
    def setTokenPositions(self, pos):
        self.tokenPositions = pos

    def getSparse(self, length, idfs = None):
        """return document as sparse BOW column vector (scipy.sparse.lil_matrix[tokenid,0]=frequency)"""
        
        bow = utils_dml.vect2bow(self.getTokenIds())  # lil_matrix access to element is O(log), so build a hash first to get O(1)
        if idfs == None:
            result = scipy.sparse.lil_matrix(shape = (1, length), dtype = int)
            for tokenid, freq in bow.iteritems():
                result[0, tokenid] = freq
        else:
            result = scipy.sparse.lil_matrix(shape = (1, length), dtype = numpy.float32)
            for tokenid, freq in bow.iteritems():
                result[0, tokenid] = freq * idfs[tokenid]
        return result

    def getFull(self, length, idfs = None):
        """return document as BOW column vector. optionally scale each element by idfs"""
        
        bow = utils_dml.vect2bow(self.getTokenIds())
        if (idfs == None):
            result = numpy.zeros(length, dtype = int)
            for tokenid, freq in bow.iteritems():
                result[tokenid] = freq
        else:
            result = numpy.zeros(length, dtype = numpy.float32)
            for tokenid, freq in bow.iteritems():
                result[tokenid] = freq * idfs[tokenid]
        return result
        

class DocumentFactory:
    """create a document according to interface set at constructor. concrete document/tokenizer class is not specified in advance."""
    def __init__(self, encoding = 'utf8', sourceType = 'string', contentType = 'text', lowercase = True,
                 keepTexts = True, keepTokens = True, keepPositions = True):
        """
        sourceType - document created from string passed as parameter/from filename passed as parameter
        contentType - type of text (used for domain knowledge in tokenization etc)
        """
        from tokenizers import TokenizerFactory
        self.tf = TokenizerFactory()
        self.tokenizer = self.tf.createTokenizer(contentType, encoding, lowercase)
        self.sourceType = sourceType
        self.contentType = contentType
        self.keepTexts = keepTexts
        self.keepTokens = keepTokens
        self.keepPositions = keepPositions
        
    def createDocument(self, source, docid = None, dictionary = None):
        """
        source - either text as string or filename (if sourceType=='file')
        docid - document id or filename
        """
        if self.sourceType == 'file':
            if docid == None:
                docid = source
##                docid = os.path.basename(source)
            source = utils_dml.readfile(source)
##        logging.debug("creating document %s" % str(docid))
        result = Document(docid)
        if self.keepTexts:
            result.setText(source)
        if self.keepTokens or self.keepPositions or dictionary != None:
            if self.keepPositions:
                tokens, pos = self.tokenizer.tokenize(source, returnPositions = self.keepPositions)
            else:
                tokens = self.tokenizer.tokenize(source, returnPositions = self.keepPositions)
            if self.keepTokens:
                result.setTokens(tokens)
            if self.keepPositions:
                result.setTokenPositions(pos)
            if dictionary != None:
                newwords = {}
                result.setTokenIds(utils_dml.text2vect(tokens, dictionary, newwords))
##                print 'for %s added %i (new length = %i) ' % (docid, len(newwords), len(dictionary))
        return result

    def createDocuments(self, sources, start = None, dictionary = None):
        """
        if sourceType == 'text' then docid will be 'start + sequence index in sources'
        if sourceType == 'file' and start == None, docid will be the filename as returned by os.path.basename()
        """
        result = []
        doNames = self.sourceType == 'file' and start == None
        if start == None:
            start = 0

        cnt = 0
        for source in sources:
            if (cnt + 1) % 1000 == 0:
                logging.debug("progress: doc#%i" % cnt)
            cnt += 1
            if doNames:
                doc = self.createDocument(source, None, dictionary = dictionary)
            else:
                doc = self.createDocument(source, start, dictionary = dictionary)
            result.append(doc)
            start += 1

        return result

    def getTokenizer(self):
        return self.tokenizer

