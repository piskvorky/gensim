import logging

TO_SAVE = ['status', 'idZBL', 'number', 'references', 'category', 'access', 'idMR',
           'title', 'review', 'note', 'dml_error', 'msc', 'body', 'range_pages', 
           'idUlrych', 'id_int', 'lang_summary', 'language', 'author', 'range']

MANDATORY = []#'msc'] # mandatory fields for when initializing Article(record)

class Article:
    """article class; all info on a single article comes here"""

    def __init__(self, record = None):
        # set some default fields that are always present (and set them to None = error state)
        self.id_int = None
        self.body = None
        self.references = None
        for key in TO_SAVE:
            self.__dict__[key] = None
        # update fields with 'record' dictionary (such as obtained from meta.xml)
        if record != None:
            self.__dict__.update(record)
            if self.idMR:
                self.idMR = self.idMR.rjust(7, '0') # make MR id always have length 7 -- pad with 0's if necessary
            allOk = True
            for m in MANDATORY:
                if not self.__dict__[m]:
                    allOk = False
            if not allOk:
                logging.warning('some mandatory fields missing in %s' % self)
            
    def mergeWith(self, art2):
        """merge information about the same article from another source (another partially filled Article object)
        a check for id consistency is done.
        in case both Article objects have the same attribute filled, the original is retained (ie., the new attribute value is discarded)"""
        if not consistentArticle(self, art2):
            raise Exception("failed to merge inconsistent articles")
        for key, val in art2.__dict__.iteritems():
            if not self.__dict__.get(key, None):
                self.__dict__[key] = val
        return self

    def __str__(self):
        return '[ source: %s, id_int: %s, idZBL: %s, idMR: %s, msc: %s, title: %s, has_body: %s, references: %s ]' % (self.source, self.id_int, self.idZBL, self.idMR, self.msc, self.title, self.body != None, self.references)

def consistentIds(id1, id2):
    if id1 == None or id2 == None:
        return True
    return id1 == id2

def consistentArticle(art1, art2):
    #consistentIds(art1.id_int, art2.id_int) and  \ # FIXME don't check internal id consistency for now
    result = \
        consistentIds(art1.idZBL, art2.idZBL) and \
        consistentIds(art1.idMR, art2.idMR)
    return result

def areValidMSCs(mscs):
    """check to see whether a list of MSC codes contains ALL valid codes
    a valid code has either length 5, or starts with an * and has length 6"""
    lens = set([len(msc) for msc in mscs])
    return lens.issubset(set([5, 6]))

def mergeArticles(art1, art2):
    return Article().mergeWith(art1).mergeWith(art2)

def idFromDir(dir):
    import os.path
    head, tail = os.path.split(dir)
    result = os.path.split(head)[1] + '-' + tail[1:]
    return result


if __name__ == "__main__":
    _records = [
               ['1-1994:0', None, None, '12:0X', 'title1', 'nejaky text', None],
               ['1-1995:0', None, None, '19:05', 'title2', 'dlouhy clanek', ['1-1994:1']],
               ['1-1995:1', None, 155, '02:12', 'title3', 'e na druhou', ['1-1994:0']],
               ['1-1995:0', '0:12456', None, '19:05', 'title4', None, None],
               ['1-1994:1', '0:12455', None, 'unknown', 'title5', None, ['1-1994:0']]
               ]
    names = ['id_int', 'idZBL', 'idMR', 'msc', 'title', 'body', 'references']
    records = [dict(zip(names, rec)) for rec in _records]
    print records
    a1 = Article(record = records[1])
    print 'rec[1]', a1
    a1.mergeWith(Article(record = records[3]))
    print 'merged', a1
    print 'static merge', mergeArticles(Article(records[1]), Article(records[3]))