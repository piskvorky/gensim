#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>


"""
Server for vector space "find similar" service, using gensim as back-end.

The server performs 3 main functions:

1. converts documents to semantic representation
2. indexes documents in the semantic representation, for faster retrieval
3. for a given query document, returns the most similar documents from the index

"""

from __future__ import with_statement

import os
import logging
import random
import threading

import numpy

import gensim
from sqlitedict import SqliteDict


logger = logging.getLogger('gensim_server')



MODEL_METHOD = 'lsi' # use LSI to represent documents
#MODEL_METHOD = 'tfidf'
LSI_TOPICS = 400
TOP_SIMS = 100 # when precomputing similarities, only consider this many "most similar" documents
SHARD_SIZE = 32768 # spill index shards to disk in SHARD_SIZE-ed chunks of documents



def simple_preprocess(doc):
    """
    Convert a document into a list of tokens.

    This lowercases, tokenizes, stems, normalizes etc. -- the output are final,
    utf8 encoded strings that won't be processed any further.
    """
    tokens = [token.encode('utf8') for token in gensim.utils.tokenize(doc, lower=True, errors='ignore')
            if 2 <= len(token) <= 15 and not token.startswith('_')]
    return tokens


def merge_sims(oldsims, newsims, clip=TOP_SIMS):
    """Merge two precomputed lists, truncating the result to `clip` most similar items."""
    if oldsims is None:
        result = newsims or []
    elif newsims is None:
        result = oldsims
    else:
        result = sorted(oldsims + newsims, key=lambda item:-item[1])
    return result[: clip]



class SimIndex(gensim.utils.SaveLoad):
    """
    An index of documents. Used internally by SimServer.

    It uses the Similarity class to persist all document vectors to disk (via mmap).
    """
    def __init__(self, fname, num_features, shardsize=SHARD_SIZE, topsims=TOP_SIMS):
        """
        Spill index shards to disk after every `shardsize` documents.
        In similarity queries, return only the `topsims` most similar documents.
        """
        self.fname = fname
        self.shardsize = int(shardsize)
        self.topsims = int(topsims)
        self.id2pos = {} # map document id (string) to index position (integer)
        self.pos2id = {} # reverse mapping for id2pos; redundant, for performance
        self.id2sims = SqliteDict(self.fname + '.id2sims') # precomputed top similar: document id -> [(doc_id, similarity)]
        self.qindex = gensim.similarities.Similarity(self.fname + '.idx', corpus=None,
            num_best=None, num_features=num_features, shardsize=shardsize)
        self.length = 0

    def save(self, fname):
        tmp, self.id2sims = self.id2sims, None
        super(SimIndex, self).save(fname)
        self.id2sims = tmp

    @staticmethod
    def load(fname):
        result = gensim.utils.SaveLoad.load(fname)
        result.id2sims = SqliteDict(result.fname + '.id2sims')
        return result


    def terminate(self):
        """Delete all files created by this index, invalidating it. Use with care."""
        import glob
        for fname in glob.glob(self.fname + '*'):
            try:
                os.remove(fname)
                logger.info("deleted %s" % fname)
            except IOError:
                logger.warning("failed to delete %s" % fname)
        for val in self.__dict__.keys():
            delattr(self, val)


    def index_documents(self, fresh_docs, model):
        """
        Update fresh index with new documents (potentially replacing old ones with
        the same id). `fresh_docs` is a dictionary-like object (=dict, sqlitedict, shelve etc)
        that maps document_id->document.
        """
        docids = fresh_docs.keys()
        vectors = (model.docs2vecs(fresh_docs[docid] for docid in docids))
        logger.info("adding %i documents to %s" % (len(docids), self))
        self.qindex.add_documents(vectors)
        self.qindex.save()
        self.update_ids(docids)


    def update_ids(self, docids):
        """Update id->pos mapping with new document ids."""
        logger.info("updating %i id mappings" % len(docids))
        for docid in docids:
            if docid is not None:
                if docid in self.id2pos:
                    logger.info("replacing existing document %r in %s" % (docid, self))
                    del self.pos2id[self.id2pos[docid]]
                self.id2pos[docid] = self.length
                try:
                    del self.id2sims[docid]
                except:
                    pass
            self.length += 1
        self.id2sims.sync()
        self.update_mappings()


    def update_mappings(self):
        """Synchronize id<->position mappings."""
        self.pos2id = dict((v, k) for k, v in self.id2pos.iteritems())
        assert len(self.pos2id) == len(self.id2pos), "duplicate ids or positions detected"


    def delete(self, docids):
        """Delete documents (specified by their ids) from the index."""
        logger.debug("deleting %i documents from %s" % (len(docids), self))
        deleted = 0
        for docid in docids:
            try:
                del self.id2pos[docid]
                deleted += 1
                del self.id2sims[docid]
            except:
                pass
        self.id2sims.sync()
        if deleted:
            logger.info("deleted %i documents from %s" % (deleted, self))
        self.update_mappings()


    def sims2scores(self, sims):
        """Convert raw similarity vector to a list of (docid, similarity) results."""
        result = []
        if isinstance(sims, numpy.ndarray):
            sims = abs(sims) # TODO or maybe clip? are opposite vectors "similar" or "dissimilar"?!
            for pos in numpy.argsort(sims)[::-1]:
                if pos in self.pos2id and sims[pos] > 1e-8: # ignore deleted/rewritten documents
                    # convert positions of resulting docs back to ids
                    result.append((self.pos2id[pos], sims[pos]))
                    if len(result) == self.topsims:
                        break
        else:
            for pos, score in sims:
                if pos in self.pos2id and abs(score) > 1e-8: # ignore deleted/rewritten documents
                    # convert positions of resulting docs back to ids
                    result.append((self.pos2id[pos], abs(score)))
                    if len(result) == self.topsims:
                        break
        return result


    def sims_by_id(self, docid):
        """Find the most similar documents to the (already indexed) document with `docid`."""
        result = self.id2sims.get(docid, None)
        if result is None:
            sims = self.qindex.similarity_by_id(self.id2pos[docid])
            result = self.sims2scores(sims)
        return result


    def sims_by_doc(self, doc, model):
        """
        Find the most similar documents to a fulltext document.

        The document is first processed (tokenized etc) and converted to a vector
        in the same way the training documents were, during `train()`.
        """
        vec = model.doc2vec(doc) # convert document (text) to vector
        sims = self.qindex[vec] # query the index
        return self.sims2scores(sims)


    def merge(self, other):
        """Merge documents from the other index. Update precomputed similarities
        in the process."""
        other.qindex.normalize, other.qindex.num_best = False, self.topsims
        # update precomputed "most similar" for old documents (in case some of
        # the new docs make it to the top-N for some of the old documents)
        logger.info("updating old precomputed values")
        pos, lenself = 0, len(self.qindex)
        for chunk in self.qindex.iter_chunks():
            for sims in other.qindex[chunk]:
                if pos in self.pos2id:
                    # ignore masked entries (deleted, overwritten documents)
                    docid = self.pos2id[pos]
                    sims = self.sims2scores(sims)
                    self.id2sims[docid] = merge_sims(self.id2sims[docid], sims)
                pos += 1
                if pos % 10000 == 0:
                    logger.info("PROGRESS: updated doc #%i/%i" % (pos, lenself))
        self.id2sims.sync()

        logger.info("merging fresh index into optimized one")
        pos, docids = 0, []
        for chunk in other.qindex.iter_chunks():
            for vec in chunk:
                if pos in other.pos2id: # don't copy deleted documents
                    self.qindex.add_documents([vec])
                    docids.append(other.pos2id[pos])
                pos += 1
        self.qindex.save()
        self.update_ids(docids)

        logger.info("precomputing most similar for the fresh index")
        pos, lenother = 0, len(other.qindex)
        norm, self.qindex.normalize = self.qindex.normalize, False
        topsims, self.qindex.num_best = self.qindex.num_best, self.topsims
        for chunk in other.qindex.iter_chunks():
            for sims in self.qindex[chunk]:
                if pos in other.pos2id:
                    # ignore masked entries (deleted, overwritten documents)
                    docid = other.pos2id[pos]
                    self.id2sims[docid] = self.sims2scores(sims)
                pos += 1
                if pos % 10000 == 0:
                    logger.info("PROGRESS: precomputed doc #%i/%i" % (pos, lenother))
        self.qindex.normalize, self.qindex.num_best = norm, topsims
        self.id2sims.sync()


    def __len__(self):
        return len(self.id2pos)

    def __contains__(self, docid):
        return docid in self.id2pos

    def keys(self):
        return self.id2pos.keys()

    def __str__(self):
        return "SimIndex(%i docs, %i real size)" % (len(self), self.length)
#endclass SimIndex



class SimModel(gensim.utils.SaveLoad):
    """
    A semantic model responsible for translating between plain text and (semantic)
    vectors.

    These vectors can then be indexed/queried for similarity, see the `SimIndex`
    class. Used internally by `SimServer`.
    """
    def __init__(self, fresh_docs, dictionary=None, method=MODEL_METHOD, preprocess=simple_preprocess):
        """
        Train a model, using `fresh_docs` as training corpus.

        If `dictionary` is not specified, it is computed from the documents.

        `method` is currently one of "tfidf"/"lsi"/"lda".

        `preprocess` is a function that takes a text and returns a sequence of
        preprocessed tokens. It is used to parse documents.
        """
        # FIXME TODO: use subclassing/injection for different methods, instead of param..
        self.preprocess = preprocess
        self.method = method
        docids = fresh_docs.keys()
        random.shuffle(docids)
        logger.info("creating model from %s documents" % len(docids))

        logger.info("preprocessing texts")
        preprocessed = SqliteDict(gensim.utils.randfname(prefix='gensim'))
        for docid in docids:
            preprocessed[docid] = self.preprocess(fresh_docs[docid]['text'])

        # create id->word (integer->string) mapping
        logger.info("creating dictionary from %s documents" % len(fresh_docs))
        if dictionary is None:
            self.dictionary = gensim.corpora.Dictionary(preprocessed[docid] for docid in preprocessed)
            if len(fresh_docs) >= 1000:
                self.dictionary.filter_extremes(no_below=5, no_above=0.2, keep_n=50000)
            else:
                logger.warning("training model on only %i documents; is this intentional?" % len(fresh_docs))
                self.dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=50000)
        else:
            self.dictionary = dictionary

        if method == 'lsi':
            logger.info("training TF-IDF model")
            corpus = (self.dictionary.doc2bow(preprocessed[docid]) for docid in preprocessed)
            self.tfidf = gensim.models.TfidfModel(corpus, id2word=self.dictionary)
            logger.info("training LSI model")
            corpus = (self.dictionary.doc2bow(preprocessed[docid]) for docid in preprocessed)
            tfidf_corpus = self.tfidf[corpus]
            self.lsi = gensim.models.LsiModel(tfidf_corpus, id2word=self.dictionary, num_topics=LSI_TOPICS)
            self.lsi.projection.u = self.lsi.projection.u.astype(numpy.float32) # use single precision to save mem
            self.num_features = len(self.lsi.projection.s)
        else:
            msg = "unknown semantic method %s" % method
            logger.error(msg)
            raise NotImplementedError(msg)
        preprocessed.terminate()


    def doc2vec(self, doc):
        """Convert a single SimilarityDocument to vector."""
        # TODO take method into account
        tokens = self.preprocess(doc['text'])
        bow = self.dictionary.doc2bow(tokens)
        return self.lsi[self.tfidf[bow]]


    def docs2vecs(self, docs):
        """Convert multiple SimilarityDocuments to vectors (batch version of doc2vec)."""
        bows = (self.dictionary.doc2bow(self.preprocess(doc['text'])) for doc in docs)
        return self.lsi[self.tfidf[bows]]


    def __str__(self):
        return "SimModel(method=%s, dict=%s)" % (self.method, self.dictionary)
#endclass SimModel



class SimServer(object):
    """
    Top-level functionality for similarity services. A similarity server takes
    care of
    1. creating semantic models
    2. indexing documents using these models
    3. finding the most similar documents in an index.

    An object of this class can be shared across network via Pyro, to answer remote
    client requests. It is thread safe. Using a server concurrently from multiple
    processes is safe only for reading (answering similarity queries). Modifying
    (training/indexing) concurrently is not safe.
    """
    def __init__(self, basename):
        """
        All data will be stored under directory `basename`. If there is a server
        there already, it will be loaded (resumed).

        The server object is stateless -- its state is defined entirely by its location.
        There is therefore no need to store the server object.
        """
        if not os.path.isdir(basename):
            raise ValueError("%r must be a writable directory" % basename)
        self.basename = basename
        try:
            self.fresh_index = SimIndex.load(self.location('index_fresh'))
        except IOError:
            self.fresh_index = None
        try:
            self.opt_index = SimIndex.load(self.location('index_opt'))
        except IOError:
            self.opt_index = None
        try:
            self.model = SimModel.load(self.location('model'))
        except IOError:
            self.model = None
        self.fresh_docs = SqliteDict() # defaults to a random location in temp
        self.lock_update = threading.Lock() # only one thread can modify the server at a time
        logger.info("loaded %s" % self)


    def location(self, name):
        return os.path.join(self.basename, name)


    def flush(self, delete_fresh=True):
        """
        Commit all changes, clear all caches. If `delete_fresh`, also clear the
        document upload buffer of `add_documents()`.
        """
        if self.fresh_index is not None:
            self.fresh_index.save(self.location('index_fresh'))
        if self.opt_index is not None:
            self.opt_index.save(self.location('index_opt'))
        if self.model is not None:
            self.model.save(self.location('model'))
        if delete_fresh:
            self.fresh_docs.terminate() # erase all buffered documents + file on disk
            self.fresh_docs = SqliteDict()
        self.fresh_docs.sync()


    @gensim.utils.synchronous('lock_update')
    def train(self, corpus=None, method='lsi'):
        """
        Create an indexing model. Will overwrite the model if it already exists.
        All indexes become invalid, because documents in them use a now-obsolete
        representation.

        The model is trained on documents previously entered via `add_documents`,
        or directly on `corpus`, if specified.
        """
        if corpus is not None:
            self.flush(delete_fresh=True)
            self.add_documents(corpus)
        self.model = SimModel(self.fresh_docs, method=method)
        self.flush(delete_fresh=True)


    @gensim.utils.synchronous('lock_update')
    def index(self, corpus=None):
        """
        Permanently index all documents previously added via `add_documents`, or
        directly documents from `corpus`, if specified.

        The indexing model must already exist (see `train`) before this function
        is called.
        """
        if not self.model:
            msg = 'must initialize the model for %s before indexing documents' % self.basename
            logger.error(msg)
            raise AttributeError(msg)

        if corpus is not None:
            self.flush(delete_fresh=True)
            self.add_documents(corpus)

        if not self.fresh_index:
            logger.info("starting a new fresh index for %s" % self)
            self.fresh_index = SimIndex(self.location('index_fresh'), self.model.num_features)
        self.fresh_index.index_documents(self.fresh_docs, self.model)
        if self.opt_index is not None:
            self.opt_index.delete(self.fresh_docs.keys())
        self.flush(delete_fresh=True)


    @gensim.utils.synchronous('lock_update')
    def optimize(self):
        """
        Precompute top similarities for all indexed documents. This speeds up
        `find_similar` queries by id (but not queries by fulltext).

        Internally, documents are moved from a fresh index (=no precomputed similarities)
        to an optimized index (precomputed similarities). Similarity queries always
        query both indexes, so this split is transparent to clients.

        If you add documents later via `index`, they go to the fresh index again.
        To precompute top similarities for these new documents too, simply call
        `optimize` again.

        """
        if self.fresh_index is None:
            logger.warning("optimize called but there are no new documents")
            return # nothing to do!

        if self.opt_index is None:
            logger.info("starting a new optimized index for %s" % self)
            self.opt_index = SimIndex(self.location('index_opt'), self.model.num_features)

        self.opt_index.merge(self.fresh_index)
        self.fresh_index = self.fresh_index.terminate() # delete old files
        self.flush(delete_fresh=False)


    @gensim.utils.synchronous('lock_update')
    def drop_index(self, keep_model=True):
        """Drop all indexed documents. If `keep_model` is False, also dropped the model."""
        modelstr = "" if keep_model else "and model "
        logger.info("deleting similarity index " + modelstr + "from %s" % self.basename)
        for index in [self.fresh_index, self.opt_index]:
            if index is not None:
                index.terminate()
        self.fresh_index, self.opt_index = None, None
        if not keep_model and self.model is not None:
            fname = self.location('model')
            try:
                os.remove(fname)
                logger.info("deleted %s" % fname)
            except IOError:
                logger.warning("failed to delete %s" % fname)
            self.model = None
        self.flush(delete_fresh=True)


    @gensim.utils.synchronous('lock_update')
    def buffer(self, documents):
        """
        Add a sequence of documents to be processed (indexed or trained on).

        Here, the documents are simply collected; real processing is done later,
        during the `self.index` or `self.train` calls.

        `buffer` can be called repeatedly; the result is the same as if it was
        called once, with a concatenation of all the partial document batches.
        The point is to save memory when sending large corpora over network: the
        entire `documents` must be serialized into RAM. See `utils.upload_chunked()`.

        A call to `flush()` clears this documents-to-be-processed buffer (`flush`
        is also implicitly called when you call `index()` and `train()`).
        """
        logger.info("adding %i documents to temporary buffer of %s" % (len(documents), self))
        for doc in documents:
            docid = doc['id']
#            logger.debug("buffering document %r" % docid)
            if docid in self.fresh_docs:
                logger.warning("asked to re-add id %r; rewriting old value" % docid)
            self.fresh_docs[docid] = doc
        self.fresh_docs.sync()

    add_documents = buffer # alias


    @gensim.utils.synchronous('lock_update')
    def drop_documents(self, docids):
        """Delete specified documents from the index."""
        logger.info("asked to drop %i documents" % len(docids))
        for index in [self.opt_index, self.fresh_index]:
            if index is not None:
                index.delete(docids)
        self.flush(delete_fresh=False)


    def find_similar(self, doc, min_score=0.0, max_results=100):
        """
        Find at most `max_results` most similar articles in the index,
        each having similarity score of at least `min_score`.

        `doc` is either a string (document id, previously indexed) or a
        dict containing a 'text' key. This text is processed to produce a vector,
        which is then used as a query.

        The similar documents are returned in decreasing similarity order, as
        (doc_id, doc_score) pairs.
        """
        logger.debug("received query call with %r" % doc)
        if self.lock_update.locked():
            msg = "cannot query while the server is being updated"
            logger.error(msg)
            raise RuntimeError(msg)
        sims_opt, sims_fresh = None, None
        if isinstance(doc, basestring):
            # query by direct document id
            docid = doc
            if self.opt_index is not None and docid in self.opt_index:
                sims_opt = self.opt_index.sims_by_id(docid)
            if self.fresh_index is not None and docid in self.fresh_index:
                sims_fresh = self.fresh_index.sims_by_id(docid)
            if sims_fresh is None and sims_opt is None:
                raise ValueError("document %r not in index" % docid)
        else:
            # query by an arbitrary text (=string) inside doc['text']
            if self.opt_index is not None:
                sims_opt = self.opt_index.sims_by_doc(doc, self.model)
            if self.fresh_index is not None:
                sims_fresh = self.fresh_index.sims_by_doc(doc, self.model)

        result = []
        for docid, score in merge_sims(sims_opt, sims_fresh):
            if score < min_score or 0 < max_results <= len(result):
                break
            result.append((docid, score))
        return result


    def __str__(self):
        return ("SimServer(loc=%r, fresh=%s, opt=%s, model=%s, buffer=%s)" %
                (self.basename, self.fresh_index, self.opt_index, self.model, self.fresh_docs))

    def __contains__(self, docid):
        """Is document with `docid` in the index?"""
        return any(index is not None and docid in index
                   for index in [self.opt_index, self.fresh_index])

    def status(self):
        return str(self)

    def keys(self):
        """Return ids of all indexed documents."""
        return self.fresh_index.keys() + self.opt_index.keys()

    def memdebug(self):
        from guppy import hpy
        return str(hpy().heap())
#endclass SimServer
