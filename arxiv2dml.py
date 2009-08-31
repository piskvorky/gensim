#!/usr/bin/env python

import logging
import sys
import os
import os.path
import codecs
import re

import common

PAT_LANG = re.compile('langue=\"(.*?)\"')
PAT_IDMR = re.compile('mrid=\"(.*?)\"')
PAT_MSCS = re.compile('msc=\"(.*?)\"')
PAT_IDZBL = re.compile('zblid=\"(.*?)\"')
PAT_IDART = re.compile('idart=\"(.*?)\"')


LANG_MAP = { # map language id from numdam notation to dml notation
            'en' : 'eng', 
            'fr' : 'fre', 
            'de' : 'ger',
            'it' : 'ita',
            'unknown' : 'unknown' 
            }


META_XML = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<article>

\t<status>in progress</status>
\t<sr></sr>
\t<date></date>
\t<idMR>MR%(idmr)s</idMR>
\t<access>true</access>
\t<review></review>
\t<idZBL>%(idzbl)s</idZBL>
\t<category>math</category>
\t<author></author>
\t<range_pages></range_pages>
\t<language>%(lang)s</language>
\t<idJFM></idJFM>
\t<range></range>
\t<title lang="fre"></title>
\t<title lang="eng"></title>
\t<number></number>
\t<idUlrych></idUlrych>
\t%(mscsxml)s
\t<lang_summary></lang_summary>

</article>"""

MSC_XML = """<msc>%s</msc>"""

def saveArxivDml(contents, pathname):
    global saveUnder
    head, outfile = os.path.split(pathname)
    dirName = os.path.split(head)[1]
    finaldir = os.path.join(saveUnder, '#' + dirName)

    contents['mscsxml'] = '\n\t'.join(MSC_XML % msc for msc in contents['mscs'])
    metaxml = META_XML % contents
    
    logging.debug("saving article files to %s" % finaldir)
    
    # create directory structure if necessary
    if not os.path.isdir(finaldir):
        os.makedirs(finaldir)
    else:
        pass
    
    # write fulltext.txt
    f = codecs.open(os.path.join(finaldir, "fulltext.txt"), 'w', encoding = "utf8")
    f.write(contents['body'])
    f.close()
    
    #write meta.xml
    f = codecs.open(os.path.join(finaldir, "meta.xml"), 'w', encoding = "utf8")
    f.write(metaxml)
    f.close()

def processArxivXhtml(fname):
    logging.debug("processing %s" % fname)
    lines = codecs.open(fname, 'r', encoding="utf8").read().replace('\n', ' ')
    bodyPat = re.compile('<body>(.*)</body>', re.MULTILINE | re.UNICODE)
    body = re.search(bodyPat, lines)
    if not body:
        logging.error('no body found in %s' % fname)
        return False
    body = body.group(1)
    tagPat = re.compile('<.*?>', re.MULTILINE | re.UNICODE)
    body = re.sub(tagPat, ' ', body)
    lang = 'eng'
    idmr = ''
    idzbl = ''
    mscs = []
    saveArxivDml(locals(), os.path.realpath(fname))
    return True

def processDir(baseDir):
    proc_total = proc_ok = 0
    logging.info("processing directory %s" % common.inputPath(baseDir))
    for root, dirs, files in os.walk(baseDir):
        root = os.path.normpath(root)
        for f in filter(lambda f: f.endswith('.xhtml'), files):
            proc_total += 1
            fname = os.path.join(root, f)
            if processArxivXhtml(fname):
                proc_ok += 1

    logging.info('%i xhtml files processed; %i articles output' % (proc_total, proc_ok))
    
        
if __name__ == '__main__':
    import sys
    import os.path
    logging.basicConfig(level = common.PRINT_LEVEL) # set to logging.INFO to print statistics as well
    
    prog = os.path.basename(sys.argv[0])

    if len(sys.argv) < 3:
        print "USAGE: %s NAME INPUTPATH\n\tparse (tidy'ed) numdam xml files and output fulltext.txt and meta.xml under arxiv/NAME/" % prog
        sys.exit(1)

    logging.root.level = 10
    saveUnder = os.path.join(common.INPUT_PATH, 'arxiv', sys.argv[1])
    processDir(sys.argv[2])
