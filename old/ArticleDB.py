import logging
from PyDbLite import Base
import Article

INDEX_ON = ['id_int', 'idZBL', 'idMR'] # some code depends on id_int, idMR and idZBL; do not remove these (only possibly add others)

class ArticleDB:
    """class for persistent storage of articles.
    what is stored from each Article object is defined in Article.TO_SAVE
    """
    def __init__(self, dbfile, mode = 'open', autocommit = False):
        self.db = Base(dbfile)
        self.db.create(*Article.TO_SAVE, **{'mode': mode})
        self.db.create_index(*INDEX_ON)
        self.autocommit = autocommit

    def insertArticle(self, art):
        """insert article into database, with id consistency check"""
        present = []
        if art.id_int != None:
            present.extend(self.db._id_int[art.id_int])
#        if art.idZBL != None:
#            present.extend(self.db._idZBL[art.idZBL])
#        if art.idMR != None:        
#            present.extend(self.db._idMR[art.idMR])
        ids = list(set([rec['__id__'] for rec in present])) # unique ids
        present = [self.db[id] for id in ids] # remove duplicate identical entries (coming from matches on more than one id on the same article)
        new = art
        for old in present: # FIXME HACK turns off consistency checking
            try:
                new.mergeWith(Article.Article(record = old)) # article already present in database -- check if ids are consistent, update it with new info from art
            except Exception, e:
#                logging.error('inconsistent database contents (%i overlapping records); leaving database unchanged' % (len(present)))
                #logging.info('inconsistency between \n%s\n%s' % (new, Article.Article(old)))
                logging.warning('inconsistency between %s and %s' % (new, Article.Article(old)))
#                return False
        if len(present) == 0:
#            logging.debug('inserting a new article')
            pass
        else:
#            logging.debug('replacing %i old (consistent) record(s) for %s' % (len(present), new))
            pass
        self.db.delete(present)
        id = self.db.insert(**new.__dict__)

        if self.autocommit:
            self.commit()
        return True
    
    def mergeWith(self, db):
        """merge records with another ArticleDB"""
        for rec in db.db:
            self.insertArticle(Article.Article(rec))
        return self

    def __len__(self):
        return len(self.db)

    def commit(self):
        self.db.commit()
