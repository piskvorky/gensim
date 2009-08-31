#!/usr/bin/env python

# needs python>=2.4 because of sets, sorted, etc.

import logging
import os
import os.path
import findmsc
import Article
import common
import re

attempts_all = 0
attempts_success = 0
attempts_failed = 0

def getRefFiles(rootdir):
    result = {}
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file == 'references.xml':
                id = Article.idFromDir(root)
                result[os.path.join(root, file)] = id
    return result

def parseReferences(reffile, downloadMSC = False):
    """parse references file (references.xml) and return references as a list of Article objects"""
    global attempts_all, attempts_success, attempts_failed
    reflist = [] # the result
    f = open(reffile, 'r')
    for line in f:
        line = line.strip()
        if line.startswith('<title>'):
            art = Article.Article()
            art.title = line.replace("<title>", "").replace("</title>", "")
            line = f.next().strip()
            while not line.startswith("</reference>"):
                if line.startswith("<link "):
                    type = line[line.find('source="') + len('source="') : ]
                    type = type[: type.find('"')]
                    id = line[line.find('id="') + len('id="') : ]
                    id = id[: id.find('"')]
                    if type == "zbl":
                        art.idZBL = id
                    elif type == "mref":
                        art.idMR = id
                    else:
#                        logging.debug("unknown source in %s: %s" % (reffile, type))
                        pass
                    line = f.next().strip()
                    if type in ['zbl', 'mref'] and downloadMSC:
                        url = line[1 : line.rfind("</link>")]
                        msc = findmsc.findmsc(url)
                        attempts_all += 1                    
                        if not msc:
                            attempts_failed += 1
                            logging.warning("could not retrieve any MSC from url: %s" % url)
                        else:
                            attempts_success += 1
                            if art.msc == None:
                                art.msc = []
                            art.msc.extend(msc)
                            art.msc = list(set(art.msc))
                line = f.next().strip()
            reflist.append(art) # add the article into result
    f.close()
    return reflist

def _parseMeta(xmlfile):
    """parse out some interesting fields from meta.xml"""
    result = [None, None, None, []] # [id_zb, id_mr, title, [MSCs]]
    xml = open(xmlfile, 'r')
    for line in xml:
        line = line.strip()
        if line.startswith('<title lang="eng">'):
            result[2] = line[len('<title lang="eng">') : line.rfind('</title>')]
        elif line.startswith('<idMR>MR'):
            result[1] = line[len('<idMR>MR') : line.rfind('</idMR>')]
        elif line.startswith('<idZBL>'):
            result[0] = line[len('<idZBL>') : line.rfind('</idZBL>')]
        elif line.startswith('<msc>'):
            result[3].append(line[len('<msc>') : line.rfind('</msc>')])
    for i in xrange(len(result)):
        if not result[i]:
            result[i] = None # if any of the element content is "", remove the attribute (set to None)
    if result[3]:
        result[3][0] = '*' + result[3][0]# prepend '*' to the first MSC code
    xml.close()
    return tuple(result)

PAT_TAG = re.compile('<(.*?)>(.*)</.*?>')

def parseMeta(xmlfile):
    """parse out all fields from meta.xml, return as dictionary"""
    result = {'msc': []}
    xml = open(xmlfile, 'r')
    for line in xml:
        if line.find('<article>') >= 0:
            break
    for line in xml:
        if line.find('</article>') >= 0:
            break
        p = re.search(PAT_TAG, line) # HAX assumes one element = one line
        if p:
            name, cont = p.groups()
            name = name.split()[0]
            name, cont = name.strip(), cont.strip()
            if name == 'msc':
                old = result[name]
                if len(cont) != 5:
                    logging.warning('invalid MSC=%s in %s' % (cont, xmlfile))
                old.append(cont)
                result[name] = old
                continue
            if name == 'idMR':
                cont = cont[2:] # leave out MR from MR123456
            if name and cont:
                result[name] = cont
    xml.close()
    return result

def tstref():
    reffile = "/home/radim/workspace/plagiarism/data/dmlcz/cmj/30-1980-1/#2/references.xml"
    reffile = '/home/radim/workspace/plagiarism/data/dmlcz/cmj/16-1966-2/#2/references.xml'
    refs = parseReferences(reffile)
    print '\n=========-BEGIN DB-==========='
    for r in refs:
        print r
    print '========---END DB---=======\n'


if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG)
    tstref()
