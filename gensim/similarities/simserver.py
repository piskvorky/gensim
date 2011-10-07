#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>


"""
"Find similar" service, using gensim (=vector spaces) for backend.

The server performs 3 main functions:

1. converts documents to semantic representation (TF-IDF, LSA, LDA...)
2. indexes documents in the vector representation, for faster retrieval
3. for a given query document, return ids of the most similar documents from the index

SessionServer objects are transactional, so that you can rollback/commit an entire
set of changes.

The server is ready for concurrent requests (thread-safe). Indexing is incremental
and you can query the SessionServer even while it's being updated, so that there
is virtually no down-time.

"""

from __future__ import with_statement

import os
import logging
import random
import threading
import shutil

import numpy

import gensim
from sqlitedict import SqliteDict # needs sqlitedict: run "sudo easy_install sqlitedict"


logger = logging.getLogger('gensim.similarities.simserver')



MODEL_METHOD = 'lsi' # use LSI to represent documents
#MODEL_METHOD = 'tfidf'
LSI_TOPICS = 400
TOP_SIMS = 100 # when precomputing similarities, only consider this many "most similar" documents
SHARD_SIZE = 65536 # spill index shards to disk in SHARD_SIZE-ed chunks of documents



def merge_sims(oldsims, newsims, clip=TOP_SIMS):
    """Merge two precomputed similarity lists, truncating the result to `clip` most similar items."""
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
        result.check_moved(fname)
        result.id2sims = SqliteDict(result.fname + '.id2sims')
        return result


    def check_moved(self, fname):
        # Add extra logic to loading: if the location on filesystem changed,
        # update locations of all shard files.
        # The other option was making shard locations relative to a directory name.
        # That way we wouldn't have to update their locations on load, but on the
        # other hand we'd have to pass a dirname to each call that needs their
        # absolute location... annoying.
        if self.fname != fname:
            logger.info("index seems to have moved from %s to %s; updating locations" %
                        (self.fname, fname))
            self.fname = fname
            output_prefix = fname + '.idx'
            for shard in self.qindex.shards:
                shard.fname = shard.fname.replace(self.qindex.output_prefix, output_prefix, 1)
            self.qindex.output_prefix = output_prefix


    def terminate(self):
        """Delete all files created by this index, invalidating it. Use with care."""
        import glob
        for fname in glob.glob(self.fname + '*'):
            try:
                os.remove(fname)
                logger.info("deleted %s" % fname)
            except Exception, e:
                logger.warning("failed to delete %s: %s" % (fname, e))
        for val in self.__dict__.keys():
            try:
                delattr(self, val)
            except:
                pass


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
                pos = self.id2pos.get(docid, None)
                if pos is not None:
                    logger.info("replacing existing document %r in %s" % (docid, self))
                    del self.pos2id[pos]
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


    def sims2scores(self, sims, eps=1e-7):
        """Convert raw similarity vector to a list of (docid, similarity) results."""
        result = []
        if isinstance(sims, numpy.ndarray):
            sims = abs(sims) # TODO or maybe clip? are opposite vectors "similar" or "dissimilar"?!
            for pos in numpy.argsort(sims)[::-1]:
                if pos in self.pos2id and sims[pos] > eps: # ignore deleted/rewritten documents
                    # convert positions of resulting docs back to ids
                    result.append((self.pos2id[pos], sims[pos]))
                    if len(result) == self.topsims:
                        break
        else:
            for pos, score in sims:
                if pos in self.pos2id and abs(score) > eps: # ignore deleted/rewritten documents
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
    def __init__(self, fresh_docs, dictionary=None, method=MODEL_METHOD):
        """
        Train a model, using `fresh_docs` as training corpus.

        If `dictionary` is not specified, it is computed from the documents.

        `method` is currently one of "tfidf"/"lsi"/"lda".
        """
        # FIXME TODO: use subclassing/injection for different methods, instead of param..
        self.method = method
        logger.info("collecting %i document ids" % len(fresh_docs))
        docids = fresh_docs.keys()
        logger.info("creating model from %s documents" % len(docids))
        preprocessed = lambda : (fresh_docs[docid]['tokens'] for docid in docids)

        # create id->word (integer->string) mapping
        logger.info("creating dictionary from %s documents" % len(docids))
        if dictionary is None:
            self.dictionary = gensim.corpora.Dictionary(preprocessed())
            if len(docids) >= 1000:
                self.dictionary.filter_extremes(no_below=5, no_above=0.2, keep_n=50000)
            else:
                logger.warning("training model on only %i documents; is this intentional?" % len(docids))
                self.dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=50000)
        else:
            self.dictionary = dictionary
        corpus = lambda: (self.dictionary.doc2bow(tokens) for tokens in preprocessed())
        if method == 'lsi':
            logger.info("training TF-IDF model")
            self.tfidf = gensim.models.TfidfModel(corpus(), id2word=self.dictionary)
            logger.info("training LSI model")
            tfidf_corpus = self.tfidf[corpus()]
            self.lsi = gensim.models.LsiModel(tfidf_corpus, id2word=self.dictionary, num_topics=LSI_TOPICS)
            self.lsi.projection.u = self.lsi.projection.u.astype(numpy.float32) # use single precision to save mem
            self.num_features = len(self.lsi.projection.s)
        else:
            msg = "unknown semantic method %s" % method
            logger.error(msg)
            raise NotImplementedError(msg)


    def doc2vec(self, doc):
        """Convert a single SimilarityDocument to vector."""
        # FIXME take self.method into account!
        bow = self.dictionary.doc2bow(doc['tokens'])
        return self.lsi[self.tfidf[bow]]


    def docs2vecs(self, docs):
        """Convert multiple SimilarityDocuments to vectors (batch version of doc2vec)."""
        bows = (self.dictionary.doc2bow(doc['tokens']) for doc in docs)
        return self.lsi[self.tfidf[bows]]


    def __str__(self):
        return "SimModel(method=%s, dict=%s)" % (self.method, self.dictionary)
#endclass SimModel



class SimServer(object):
    """
    Top-level functionality for similarity services. A similarity server takes
    care of::

    1. creating semantic models
    2. indexing documents using these models
    3. finding the most similar documents in an index.

    An object of this class can be shared across network via Pyro, to answer remote
    client requests. It is thread safe. Using a server concurrently from multiple
    processes is safe for reading = answering similarity queries. Modifying
    (training/indexing) is realized via locking = serialized internally.
    """
    def __init__(self, basename):
        """
        All data will be stored under directory `basename`. If there is a server
        there already, it will be loaded (resumed).

        The server object is stateless in RAM -- its state is defined entirely by its location.
        There is therefore no need to store the server object.
        """
        if not os.path.isdir(basename):
            raise ValueError("%r must be a writable directory" % basename)
        self.basename = basename
        self.lock_update = threading.RLock() # only one thread can modify the server at a time
        try:
            self.fresh_index = SimIndex.load(self.location('index_fresh'))
        except:
            self.fresh_index = None
        try:
            self.opt_index = SimIndex.load(self.location('index_opt'))
        except:
            self.opt_index = None
        try:
            self.model = SimModel.load(self.location('model'))
        except:
            self.model = None
        self.payload = SqliteDict(self.location('payload'), autocommit=True)
        # save the opened objects right back. this is not necessary and costs extra
        # time, but is cleaner when there are server location changes (see `check_moved`).
        self.flush(save_index=True, save_model=True, clear_buffer=True)
        logger.info("loaded %s" % self)


    def location(self, name):
        return os.path.join(self.basename, name)


    @gensim.utils.synchronous('lock_update')
    def flush(self, save_index=False, save_model=False, clear_buffer=False):
        """Commit all changes, clear all caches."""
        if save_index:
            if self.fresh_index is not None:
                self.fresh_index.save(self.location('index_fresh'))
            if self.opt_index is not None:
                self.opt_index.save(self.location('index_opt'))
        if save_model:
            if self.model is not None:
                self.model.save(self.location('model'))
        self.payload.commit()
        if clear_buffer:
            if hasattr(self, 'fresh_docs'):
                try:
                    self.fresh_docs.terminate() # erase all buffered documents + file on disk
                except:
                    pass
            self.fresh_docs = SqliteDict() # buffer defaults to a random location in temp
        self.fresh_docs.sync()


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
        logger.info("adding documents to temporary buffer of %s" % (self))
        for doc in documents:
            docid = doc['id']
#            logger.debug("buffering document %r" % docid)
            if docid in self.fresh_docs:
                logger.warning("asked to re-add id %r; rewriting old value" % docid)
            self.fresh_docs[docid] = doc
        self.fresh_docs.sync()


    @gensim.utils.synchronous('lock_update')
    def train(self, corpus=None, method='lsi', clear_buffer=True):
        """
        Create an indexing model. Will overwrite the model if it already exists.
        All indexes become invalid, because documents in them use a now-obsolete
        representation.

        The model is trained on documents previously entered via `buffer`,
        or directly on `corpus`, if specified.
        """
        if corpus is not None:
            # use the supplied corpus only (erase existing buffer, if any)
            self.flush(clear_buffer=True)
            self.buffer(corpus)
        if not self.fresh_docs:
            msg = "train called but no training corpus specified for %s" % self
            logger.error(msg)
            raise ValueError(msg)
        self.model = SimModel(self.fresh_docs, method=method)
        self.flush(save_model=True, clear_buffer=clear_buffer)


    @gensim.utils.synchronous('lock_update')
    def index(self, corpus=None, clear_buffer=True):
        """
        Permanently index all documents previously added via `buffer`, or
        directly index documents from `corpus`, if specified.

        The indexing model must already exist (see `train`) before this function
        is called.
        """
        if not self.model:
            msg = 'must initialize model for %s before indexing documents' % self.basename
            logger.error(msg)
            raise AttributeError(msg)

        if corpus is not None:
            # use the supplied corpus only (erase existing buffer, if any)
            self.flush(clear_buffer=True)
            self.buffer(corpus)

        if not self.fresh_docs:
            msg = "index called but no training corpus specified for %s" % self
            logger.error(msg)
            raise ValueError(msg)

        if not self.fresh_index:
            logger.info("starting a new fresh index for %s" % self)
            self.fresh_index = SimIndex(self.location('index_fresh'), self.model.num_features)
        self.fresh_index.index_documents(self.fresh_docs, self.model)
        if self.opt_index is not None:
            self.opt_index.delete(self.fresh_docs.keys())
        logger.info("storing document payloads")
        for docid in self.fresh_docs:
            payload = self.fresh_docs[docid].get('payload', None)
            if payload is None:
                # TODO HACK: exit on first doc without a payload (=assume all docs have payload, or none does)
                break
            self.payload[docid] = payload
        self.flush(save_index=True, clear_buffer=clear_buffer)


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
        self.flush(save_index=True)


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
            except Exception, e:
                logger.warning("failed to delete %s" % fname)
            self.model = None
        self.flush(save_index=True, save_model=True, clear_buffer=True)


    @gensim.utils.synchronous('lock_update')
    def delete(self, docids):
        """Delete specified documents from the index."""
        logger.info("asked to drop %i documents" % len(docids))
        for index in [self.opt_index, self.fresh_index]:
            if index is not None:
                index.delete(docids)
        self.flush(save_index=True)


    def is_locked(self):
        return self.lock_update._RLock__count > 0


    def find_similar(self, doc, min_score=0.0, max_results=100):
        """
        Find at most `max_results` most similar articles in the index,
        each having similarity score of at least `min_score`.

        `doc` is either a string (document id, previously indexed) or a
        dict containing a 'tokens' key. These tokens are processed to produce a
        vector, which is then used as a query.

        The similar documents are returned in decreasing similarity order, as
        (doc_id, doc_score) pairs.
        """
        logger.debug("received query call with %r" % doc)
        if self.is_locked():
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
            # query by an arbitrary text (=tokens) inside doc['tokens']
            if self.opt_index is not None:
                sims_opt = self.opt_index.sims_by_doc(doc, self.model)
            if self.fresh_index is not None:
                sims_fresh = self.fresh_index.sims_by_doc(doc, self.model)

        result = []
        for docid, score in merge_sims(sims_opt, sims_fresh):
            if score < min_score or 0 < max_results <= len(result):
                break
            result.append((docid, score, self.payload.get(docid, None)))
        return result


    def __str__(self):
        return ("SimServer(loc=%r, fresh=%s, opt=%s, model=%s, buffer=%s)" %
                (self.basename, self.fresh_index, self.opt_index, self.model, self.fresh_docs))


    def __len__(self):
        return sum(len(index) for index in [self.opt_index, self.fresh_index]
                   if index is not None)


    def __contains__(self, docid):
        """Is document with `docid` in the index?"""
        return any(index is not None and docid in index
                   for index in [self.opt_index, self.fresh_index])

    def status(self):
        return str(self)

    def keys(self):
        """Return ids of all indexed documents."""
        result = []
        if self.fresh_index is not None:
            result += self.fresh_index.keys()
        if self.opt_index is not None:
            result += self.opt_index.keys()
        return result

    def memdebug(self):
        from guppy import hpy
        return str(hpy().heap())
#endclass SimServer



class SessionServer(gensim.utils.SaveLoad):
    """
    Similarity server on top of :class:`SimServer` that implements sessions = transactions.

    A transaction is a set of server modifications (index/delete/train calls) that
    may be either commited or rolled back entirely.

    Sessions are realized by:

    1. cloning (=copying) a SimServer at the beginning of a session
    2. serving read-only queries from the original server (the clone may be modified during queries)
    3. modifications affect only the clone
    4. at commit, the clone and the original are switched
    5. at rollback, do nothing (clone is discarded, next transaction starts from the original again)
    """
    def __init__(self, basedir, autosession=True):
        self.basedir = basedir
        self.autosession = autosession
        self.lock_update = threading.RLock()
        self.locs = ['a', 'b'] # directories under which to store stable.session data
        try:
            stable = open(self.location('stable')).read().strip()
            self.istable = self.locs.index(stable)
        except:
            self.istable = 0
            logger.info("stable index pointer not found or invalid; starting from %s" %
                        self.loc_stable)
        try:
            os.makedirs(self.loc_stable)
        except:
            pass
        self.write_istable()
        self.stable = SimServer(self.loc_stable)
        self.session = None


    def location(self, name):
        return os.path.join(self.basedir, name)

    @property
    def loc_stable(self):
        return self.location(self.locs[self.istable])

    @property
    def loc_session(self):
        return self.location(self.locs[1 - self.istable])

    def __contains__(self, docid):
        return docid in self.stable

    def __str__(self):
        return "SessionServer(\n\tstable=%s\n\tsession=%s\n)" % (self.stable, self.session)

    def __len__(self):
        return len(self.stable)

    def keys(self):
        return self.stable.keys()

    @gensim.utils.synchronous('lock_update')
    def check_session(self):
        """
        Make sure a session is open.

        If it's not and autosession is turned on, create a new session automatically.
        If it's not and autosession is off, raise an exception.
        """
        if self.session is None:
            if self.autosession:
                self.open_session()
            else:
                msg = "must open a session before modifying %s" % self
                raise RuntimeError(msg)

    @gensim.utils.synchronous('lock_update')
    def open_session(self):
        """
        Open a new session to modify this server.

        You can either call this fnc directly, or turn on autosession which will
        open/commit sessions for you transparently.
        """
        if self.session is not None:
            msg = "session already open; commit it or rollback before opening another one in %s" % self
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info("opening a new session")
        logger.info("removing %s" % self.loc_session)
        try:
            shutil.rmtree(self.loc_session)
        except:
            logger.info("failed to delete %s" % self.loc_session)
        logger.info("cloning server from %s to %s" %
                    (self.loc_stable, self.loc_session))
        shutil.copytree(self.loc_stable, self.loc_session)
        self.session = SimServer(self.loc_session)
        self.lock_update.acquire() # no other thread can call any modification methods until commit/rollback

    @gensim.utils.synchronous('lock_update')
    def buffer(self, *args, **kwargs):
        """Buffer documents, in the current session"""
        self.check_session()
        result = self.session.buffer(*args, **kwargs)
        return result

    @gensim.utils.synchronous('lock_update')
    def index(self, *args, **kwargs):
        """Index documents, in the current session"""
        self.check_session()
        result = self.session.index(*args, **kwargs)
        if self.autosession:
            self.commit()
        return result

    @gensim.utils.synchronous('lock_update')
    def train(self, *args, **kwargs):
        """Update semantic model, in the current session."""
        self.check_session()
        result = self.session.train(*args, **kwargs)
        if self.autosession:
            self.commit()
        return result

    @gensim.utils.synchronous('lock_update')
    def drop_index(self, keep_model=True):
        """Drop all indexed documents from the session. Optionally, drop model too."""
        self.check_session()
        result = self.session.drop_index(keep_model)
        if self.autosession:
            self.commit()
        return result

    @gensim.utils.synchronous('lock_update')
    def delete(self, docids):
        """Delete documents from the current session."""
        self.check_session()
        result = self.session.delete(docids)
        if self.autosession:
            self.commit()
        return result

    @gensim.utils.synchronous('lock_update')
    def optimize(self):
        """Optimize index for faster by-document-id queries."""
        self.check_session()
        result = self.session.optimize()
        if self.autosession:
            self.commit()
        return result

    @gensim.utils.synchronous('lock_update')
    def write_istable(self):
        with open(self.location('stable'), 'w') as fout:
            fout.write(os.path.basename(self.loc_stable))

    @gensim.utils.synchronous('lock_update')
    def commit(self):
        """Commit changes made by the latest session."""
        if self.session is not None:
            logger.info("committing transaction in %s" % self)
            self.stable, self.session = self.session, None
            self.istable = 1 - self.istable
            self.write_istable()
            self.lock_update.release()
        else:
            logger.warning("commit called but there's no open session in %s" % self)

    @gensim.utils.synchronous('lock_update')
    def rollback(self):
        """Ignore all changes made in the latest session (terminate the session)."""
        if self.session is not None:
            logger.info("rolling back transaction in %s" % self)
            self.session = None
            self.lock_update.release()
        else:
            logger.warning("rollback called but there's no open session in %s" % self)

    @gensim.utils.synchronous('lock_update')
    def set_autosession(self, value=None):
        """
        Turn autosession (automatic committing after each modification call) on/off.
        If value is None, only query current value (don't change anything)
        """
        if value is not None:
            self.rollback()
            self.autosession = value
        return self.autosession

    def find_similar(self, *args, **kwargs):
        """
        Find similar articles.

        With autosession off, use the index state *before* current session started,
        so that changes made in the session will not be visible here. With autosession
        on, close the current session first (so that session changes *are* committed
        and visible).
        """
        if self.session is not None and self.autosession:
            # with autosession on, commit the pending transaction first
            self.commit()
        return self.stable.find_similar(*args, **kwargs)

    # add some functions for remote access (RPC via Pyro)
    def debug_model(self):
        return self.stable.model

    def status(self): # str() alias
        return str(self)
