#!/usr/bin/env python

import logging
import sys
import os
import os.path
import codecs
import re

import common

PAT_LANG = re.compile('langue=\"(.*?)\"', re.MULTILINE)
PAT_IDMR = re.compile('mrid=\"(.*?)\"', re.MULTILINE)
PAT_MSCS = re.compile('msc=\"(.*?)\"', re.MULTILINE)
PAT_IDZBL = re.compile('zblid=\"(.*?)\"', re.MULTILINE)
PAT_IDART = re.compile('idart=\"(.*?)\"', re.MULTILINE)
PAT_TITLE = re.compile('<title>(.*?)</title>', re.MULTILINE + re.DOTALL)

LANG_MAP = { # map language id from numdam notation to dml notation
            'en' : 'eng', 
            'fr' : 'fre', 
            'de' : 'ger',
            'it' : 'ita',
            'ru' : 'rus',
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
\t<title lang="%(lang)s">%(title)s</title>
\t<number></number>
\t<idUlrych></idUlrych>
\t%(mscsxml)s
\t<lang_summary></lang_summary>

</article>"""

MSC_XML = """<msc>%s</msc>"""

def saveNumdamDML(contents, pathname):
    head, outfile = os.path.split(pathname)
    dirName = outfile.split('_')[0]
    finaldir = os.path.join(common.INPUT_PATHS['numdam'], dirName, '#' + os.path.splitext(outfile)[0])
    
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
    contents['mscsxml'] = '\n\t'.join(MSC_XML % msc for msc in contents['mscs'])
    metaxml = META_XML % contents
    f.write(metaxml)
    f.close()

def processNumdamXml(fname):
    logging.debug("processing %s" % fname)
    lines = codecs.open(fname, 'r', encoding="iso-8859-1").readlines()
    headLen = 0
    while headLen < len(lines) and lines[headLen].find('<body>') < 0:
        headLen += 1
    if headLen >= len(lines):
        raise Exception("bad file (no <body>) in %s" % fname)
    header = ' '.join(lines[ : headLen + 1]) # header are all the lines preceding the first line containing <body>, plus the <body> line itself
    lang = re.findall(PAT_LANG, header)
    if not lang:
        lang = 'unknown'
        logging.warning("missing language in %s" % fname)
#        if fname.find("MSMF") >= 0 or fname.find("AFST") >= 0 or fname.find("BSMF") >= 0 or fname.find("JEDP") >= 0 \
#        or fname.find("AMBP") >= 0 or fname.find("AIF") >= 0:
#            lang = "fr"
#            raise Exception("unspecified language (perhaps french?)")
#        elif fname.find("ASENS") >= 0 or fname.find("PMIHES") >= 0 or fname.find("AIHPC") >= 0 or \
#        fname.find("AIHPB") >= 0:
#            lang = "en"
#            raise Exception("unspecified language (perhaps english?)")
#        else:
#            raise Exception("missing language attribute")
    else:
        lang = lang[0]
    try:
        lang = LANG_MAP[lang]
    except Exception, e:
        logging.warning('unknown language "%s" in %s' % (lang, fname))
    idmr = re.findall(PAT_IDMR, header)[0]
    idart = re.findall(PAT_IDART, header)[0]
    idzbl = re.findall(PAT_IDZBL, header)[0]
    mscs = re.findall(PAT_MSCS, header)[0].split(' ')
    title = re.findall(PAT_TITLE, header, re.MULTILINE)[0]
    if not set([len(msc) for msc in mscs]) == set([5]):
        logging.warning("incorrect msc='%s' in article %s" % (re.findall(PAT_MSCS, header)[0], fname))
    body = ''.join(lines[headLen + 1 : -2])
    saveNumdamDML(locals(), os.path.realpath(fname))

def processDir(baseDir):
    proc_total = proc_ok = 0
    logging.info("processing directory %s" % baseDir)
    for root, dirs, files in os.walk(baseDir):
        root = os.path.normpath(root)
        for f in filter(lambda f: f.endswith('.xml'), files):
            proc_total += 1
            fname = os.path.join(root, f)
            processNumdamXml(fname)
#            try:
#                processNumdamXml(fname)
#            except Exception, e:
#                logging.warning('error processing file: %s; ignoring %s' % (e, fname))
#                continue
            proc_ok +=1

    logging.info('%i xml files processed; %i articles output' % (proc_total, proc_ok))
    
        
if __name__ == '__main__':
    import sys
    import os.path
    logging.basicConfig(level = logging.INFO) # set to logging.INFO to print statistics as well
    
    prog = os.path.basename(sys.argv[0])

    if len(sys.argv) < 2:
        print "USAGE: %s INPUTPATH\n\tparse (tidy'ed) numdam xml files and output fulltext.txt and meta.xml under directory %s" % (prog, common.INPUT_PATHS['numdam'])
        sys.exit(1)

    processDir(sys.argv[1])
