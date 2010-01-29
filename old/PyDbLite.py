"""PyDbLite.py

In-memory database management, with selection by list comprehension 
or generator expression

Fields are untyped : they can store anything that can be pickled.
Selected records are returned as dictionaries. Each record is 
identified by a unique id and has a version number incremented
at every record update, to detect concurrent access

Syntax :
    from PyDbLite import Base
    db = Base('dummy')
    # create new base with field names
    db.create('name','age','size')
    # existing base
    db.open()
    # insert new record
    db.insert(name='homer',age=23,size=1.84)
    # records are dictionaries with a unique integer key __id__
    # selection by list comprehension
    res = [ r for r in db if 30 > r['age'] >= 18 and r['size'] < 2 ]
    # or generator expression
    for r in (r for r in db if r['name'] in ('homer','marge') ):
    # delete a record or a list of records
    db.delete(one_record)
    db.delete(list_of_records)
    # delete a record by its id
    del db[rec_id]
    # direct access by id
    record = db[rec_id] # the record such that record['__id__'] == rec_id
    # create an index on a field
    db.create_index('age')
    # access by index
    records = db._age[23] # returns the list of records with age == 23
    # update
    db.update(record,age=24)
    # add and drop fields
    db.add_field('new_field')
    db.drop_field('name')
    # save changes on disk
    db.commit()
"""

import os
import cPickle
import bisect

# compatibility with Python 2.3
try:
    set([])
except NameError:
    from sets import Set as set
    
class Index:
    """Class used for indexing a base on a field
    The instance of Index is an attribute the Base instance"""

    def __init__(self,db,field):
        self.db = db # database object (instance of Base)
        self.field = field # field name

    def __iter__(self):
        return iter(self.db.indices[self.field])

    def keys(self):
        return self.db.indices[self.field].keys()

    def __getitem__(self,key):
        """Lookup by key : return the list of records where
        field value is equal to this key, or an empty list"""
        ids = self.db.indices[self.field].get(key,[])
        return [ self.db.records[_id] for _id in ids ]

class Base:

    def __init__(self,basename):
        self.name = basename

    def create(self,*fields,**kw):
        """Create a new base with specified field names
        A keyword argument mode can be specified ; it is used if a file
        with the base name already exists
        - if mode = 'open' : open the existing base, ignore the fields
        - if mode = 'override' : erase the existing base and create a
        new one with the specified fields"""
        self.mode = mode = kw.get("mode",None)
        if os.path.exists(self.name):
            if not os.path.isfile(self.name):
                raise IOError,"%s exists and is not a file" %self.name
            elif mode is None:
                raise IOError,"Base %s already exists" %self.name
            elif mode == "open":
                return self.open()
            elif mode == "override":
                os.remove(self.name)
        self.fields = list(fields)
        self.records = {}
        self.next_id = 0
        self.indices = {}
        self.commit()
        return self

    def create_index(self,*fields):
        """Create an index on the specified field names
        
        An index on a field is a mapping between the values taken by the field
        and the sorted list of the ids of the records whose field is equal to 
        this value
        
        For each indexed field, an attribute of self is created, an instance 
        of the class Index (see above). Its name it the field name, with the
        prefix _ to avoid name conflicts
        """
        reset = False
        for f in fields:
            if not f in self.fields:
                raise NameError,"%s is not a field name" %f
            # initialize the indices
            if self.mode == "open" and f in self.indices:
                continue
            reset = True
            self.indices[f] = {}
            for _id,record in self.records.iteritems():
                # use bisect to quickly insert the id in the list
                bisect.insort(self.indices[f].setdefault(record[f],[]),
                    _id)
            # create a new attribute of self, used to find the records
            # by this index
            setattr(self,'_'+f,Index(self,f))
        if reset:
            self.commit()

    def open(self):
        """Open an existing database and load its content into memory"""
        _in = open(self.name) # don't specify binary mode !
        self.fields = cPickle.load(_in)
        self.next_id = cPickle.load(_in)
        self.records = cPickle.load(_in)
        self.indices = cPickle.load(_in)
        for f in self.indices.keys():
            setattr(self,'_'+f,Index(self,f))
        _in.close()
        self.mode = "open"
        return self

    def commit(self):
        """Write the database to a file"""
        out = open(self.name,'wb')
        cPickle.dump(self.fields,out)
        cPickle.dump(self.next_id,out)
        cPickle.dump(self.records,out)
        cPickle.dump(self.indices,out)
        out.close()

    def insert(self,*args,**kw):
        """Insert a record in the database
        Parameters can be positional or keyword arguments. If positional
        they must be in the same order as in the create() method
        If some of the fields are missing the value is set to None
        Returns the record identifier
        """
        if args:
            kw = dict([(f,arg) for f,arg in zip(self.fields,args)])
        # initialize all fields to None
        record = dict([(f,None) for f in self.fields])
        # set keys and values
        for (k,v) in kw.iteritems():
            record[k]=v
        # add the key __id__ : record identifier
        record['__id__'] = self.next_id
        # add the key __version__ : version number
        record['__version__'] = 0
        # create an entry in the dictionary self.records, indexed by __id__
        self.records[self.next_id] = record
        # update index
        for ix in self.indices.keys():
            bisect.insort(self.indices[ix].setdefault(record[ix],[]),
                self.next_id)
        # increment the next __id__ to attribute
        self.next_id += 1
        return record['__id__']

    def delete(self,removed):
        """Remove a single record, or the records in an iterable
        Before starting deletion, test if all records are in the base
        and don't have twice the same __id__
        Return the number of deleted items
        """
        if isinstance(removed,dict):
            # remove a single record
            removed = [removed]
        else:
            # convert iterable into a list (to be able to sort it)
            removed = [ r for r in removed ]
        if not removed:
            return 0
        _ids = [ r['__id__'] for r in removed ]
        _ids.sort()
        keys = set(self.records.keys())
        # check if the records are in the base
        if not set(_ids).issubset(keys):
            missing = list(set(_ids).difference(keys))
            raise IndexError,'Delete aborted. Records with these ids' \
                ' not found in the base : %s' %str(missing)
        # raise exception if duplicate ids
        for i in range(len(_ids)-1):
            if _ids[i] == _ids[i+1]:
                raise IndexError,"Delete aborted. Duplicate id : %s" %_ids[i]
        deleted = len(removed)
        while removed:
            r = removed.pop()
            _id = r['__id__']
            # remove id from indices
            for indx in self.indices.keys():
                pos = bisect.bisect(self.indices[indx][r[indx]],_id)-1
                del self.indices[indx][r[indx]][pos]
                if not self.indices[indx][r[indx]]:
                    del self.indices[indx][r[indx]]
            # remove record from self.records
            del self.records[_id]
        return deleted

    def update(self,record,**kw):
        """Update the record with new keys and values and update indices"""
        # update indices
        _id = record["__id__"]
        for indx in self.indices.keys():
            if indx in kw.keys():
                if record[indx] == kw[indx]:
                    continue
                # remove id for the old value
                old_pos = bisect.bisect(self.indices[indx][record[indx]],_id)-1
                del self.indices[indx][record[indx]][old_pos]
                if not self.indices[indx][record[indx]]:
                    del self.indices[indx][record[indx]]
                # insert new value
                bisect.insort(self.indices[indx].setdefault(kw[indx],[]),_id)
        # update record values
        record.update(kw)
        # increment version number
        record["__version__"] += 1

    def add_field(self,field,default=None):
        if field in self.fields + ["__id__","__version__"]:
            raise ValueError,"Field %s already defined" %field
        for r in self:
            r[field] = default
        self.fields.append(field)
        self.commit()
    
    def drop_field(self,field):
        if field in ["__id__","__version__"]:
            raise ValueError,"Can't delete field %s" %field
        self.fields.remove(field)
        for r in self:
            del r[field]
        if field in self.indices:
            del self.indices[field]
        self.commit()
    
    def __getitem__(self,record_id):
        """Direct access by record id"""
        return self.records[record_id]
    
    def __len__(self):
        return len(self.records)

    def __delitem__(self,record_id):
        """Delete by record id"""
        self.delete(self[record_id])
        
    def __iter__(self):
        """Iteration on the records"""
        return self.records.itervalues()

if __name__ == '__main__':
    # test on a 1000 record base
    import random
    import datetime
    names = ['pierre','claire','simon','camille','jean',
                 'florence','marie-anne']
    db = Base('PyDbLite_test')
    db.create('name','age','size','birth',mode="override")
    for i in range(1000):
        db.insert(name=unicode(random.choice(names)),
             age=random.randint(7,47),size=random.uniform(1.10,1.95),
             birth=datetime.date(1990,10,10))
    db.create_index('age')
    db.commit()

    print 'Record #20 :',db[20]
    print '\nRecords with age=30 :'
    for rec in db._age[30]:
        print '%-10s | %2s | %s' %(rec['name'],rec['age'],round(rec['size'],2))
    db.insert(name=unicode(random.choice(names))) # missing fields
    print '\nNumber of records with 30 <= age < 33 :',
    print sum([1 for r in db if 33 > r['age'] >= 30])
    
    print db.delete([])

    d = db.delete([r for r in db if 32> r['age'] >= 30 and r['name']==u'pierre'])
    print "\nDeleting %s records with name == 'pierre' and 30 <= age < 32" %d
    print '\nAfter deleting records '
    for rec in db._age[30]:
        print '%-10s | %2s | %s' %(rec['name'],rec['age'],round(rec['size'],2))
    print '\n',sum([1 for r in db]),'records in the database'
    print '\nMake pierre uppercase for age > 27'
    for record in ([r for r in db if r['name']=='pierre' and r['age'] >27]) :
        db.update(record,name=u"Pierre")
    print len([r for r in db if r['name']==u'Pierre']),'Pierre'
    print len([r for r in db if r['name']==u'pierre']),'pierre'
    print len([r for r in db if r['name'] in [u'pierre',u'Pierre']]),'p/Pierre'
    print 'is unicode :',isinstance(db[20]['name'],unicode)
    db.commit()
    db.open()
    print '\nSame operation after commit + open'
    print len([r for r in db if r['name']==u'Pierre']),'Pierre'
    print len([r for r in db if r['name']==u'pierre']),'pierre'
    print len([r for r in db if r['name'] in [u'pierre',u'Pierre']]),'p/Pierre'
    print 'is unicode :',isinstance(db[20]['name'],unicode)
    
    print "\nDeleting record #20"
    del db[20]
    if not 20 in db:
        print "record 20 removed"

    print db[21]
    db.drop_field('name')
    print db[21]
    db.add_field('adate',datetime.date.today())
    print db[21]
    
    k = db._age.keys()[0]
    print "key",k
    print k in db._age
    db.delete(db._age[k])
    print db._age[k]
    print k in db._age
