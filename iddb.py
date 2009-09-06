#!/usr/bin/env python

# needs python>=2.4 because of sets, sorted, etc.

import logging
import os
import os.path

import ArticleDB
import Article
import common
import utils_iddb

def create_refsdb(baseDir, downloadMSC = False):
    """read references from baseDir files and merge them together into an ArticleDB database.
    also check for consistency and print some stats
    store the resulting database on disk as 'refs_baseDir.pdl'.
    """
    import cPickle
    reffiles = utils_iddb.getRefFiles(common.inputPath(baseDir))
    cPickle.dump(reffiles, open(common.dataFile('reffiles.pkl'), 'w'), protocol = -1) # store filesystem paths to references.xml files
    refdb = {}
    for reffile, refid in reffiles.iteritems():
        refs = utils_iddb.parseReferences(reffile, downloadMSC = downloadMSC)
        refdb[refid] = refs
        logging.info('id=%s: retrieved %i references' % (refid, len(refs)))
        if downloadMSC:
            cPickle.dump(refdb, open(common.dataFile('refs_partial.pkl'), 'w'), protocol = -1) # dump database immediately after each iteration
    f = open(common.dataFile('refs.pkl'), 'w')
    cPickle.dump(refdb, f, protocol = -1) # dump database immediately after each iteration        
    f.close()
    #print some statistics
    logging.info("%i MSC download attemps (%i ok, %i failed)" % (utils_iddb.attempts_all, utils_iddb.attempts_success, utils_iddb.attempts_failed))
    logging.info('reference database size: %i references in %i articles' % (sum([len(refs) for refs in refdb.itervalues()]), len(refdb)))

    db = ArticleDB.ArticleDB(common.dbFile('refs', baseDir), mode = 'override')
    insert_errors = 0
    for id, reflist in refdb.iteritems():
        for num, ref in enumerate(reflist):
            ref.id_int = id + ':' + str(num + 1) # references.xml reference counting starts with '1'
            if not db.insertArticle(ref):
                insert_errors += 1
        #print '.',
    db.commit()
    logging.info('resulting database has %i records (originally %i)' % (len(db), sum([len(refs) for refs in refdb.itervalues()])))
    logging.info('detected %i inconsistency collisions' % insert_errors)

def create_maindb(dbId, dbBaseDir):
    """Look for any subdirs (arbitrary level under dbBaseDir) that start with '#'.
    From these subdirs, try to load ./fulltext.txt and ./meta.xml files, create an article and insert it into database. 
    Store the database persistently as PyDbLite (pickle) file 'gensim_dbId.pdl'.
    """
    db = ArticleDB.ArticleDB(common.dbFile('gensim', dbId), mode = 'override', autocommit = False)

    proc_total = 0
    logging.info("processing database %s, directory %s" % (dbId, dbBaseDir))
    for root, dirs, files in os.walk(dbBaseDir):
        root = os.path.normpath(root)
        if os.path.basename(root).startswith('#'):
            proc_total += 1
            try:
                meta = utils_iddb.parseMeta(os.path.join(root, 'meta.xml'))
                #meta = {'msc' : []}
                #meta['id_int'] = Article.idFromDir(root)
                meta['id_int'] = os.path.join(dbId, root[len(dbBaseDir) + 1: ])
                meta['body'] = unicode(open(os.path.join(root, 'fulltext.txt'), 'r').read(), 'utf8', 'ignore').encode('utf8')
                meta['references'] = None # TODO add
                art = Article.Article(record = meta)
                db.insertArticle(art)
            except Exception, e:
                logging.warning('invalid entries in %s; ignoring article (%s)' % (root, e))
                continue
    db.commit()

    logging.info('%i directories processed' % proc_total)
    logging.info('%i articles in the %s database' % (len(db), dbId))

def create_merged(baseDir):
    """merge main and reference databases, store them into merged_baseDir.pdl file"""
    db_main = ArticleDB.ArticleDB(common.dbFile('main', baseDir), mode = 'open')
    db_refs = ArticleDB.ArticleDB(common.dbFile('refs', baseDir), mode = 'open')
    db_merged = ArticleDB.ArticleDB(common.dbFile('merged', baseDir), mode = 'override')
    db_merged.mergeWith(db_main) # db_main comes first, so that potentially overlapping attributes stay from db_main, not db_refs (such as id_int and body)
    db_merged.mergeWith(db_refs)
    db_merged.commit()

    logging.info('%i total articles in the database (originally %i in reference database + %i in main database = %i)' % (len(db_merged), len(db_refs), len(db_main), len(db_refs) + len(db_main)))

def merge(inputs, dbout, acceptLanguage = 'any'):
    logging.info('merging %i databases, accepting "%s" language' % (len(inputs), acceptLanguage))

    db_merged = ArticleDB.ArticleDB(common.dbFile(dbout, acceptLanguage), mode = 'override')
    lang_failed = 0
    for dbId, dbBaseDir in inputs.iteritems():
        db_part = ArticleDB.ArticleDB(common.dbFile('gensim', dbId), mode = 'open')
        logging.info("processing %i articles from %s" % (len(db_part.db), dbId))
        inserted = 0
        for rec in db_part.db:
            if acceptLanguage == 'any' or rec['language'] == acceptLanguage:
                db_merged.insertArticle(Article.Article(rec))
                inserted += 1
            else:
                lang_failed += 1
        logging.info("accepted %i articles of %s language from %s" % (inserted, acceptLanguage, dbId))
    db_merged.commit()
    
    logging.info('%i total articles in the merged database; %i rejected due to different language' % (len(db_merged), lang_failed))


if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG)
    baseDir = common.inputPath('numdam')
#    create_refsdb(baseDir, downloadMSC = False)
    create_maindb(baseDir)
#    create_merged(baseDir)
