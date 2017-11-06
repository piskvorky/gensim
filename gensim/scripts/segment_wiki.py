#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Jayant Jain <jayant@rare-technologies.com>
# Copyright (C) 2016 RaRe Technologies

"""
Construct a corpus from a Wikipedia (or other MediaWiki-based) database dump (typical filename
is <LANG>wiki-<YYYYMMDD>-pages-articles.xml.bz2 or <LANG>wiki-latest-pages-articles.xml.bz2),
extract titles, section names, section content and save to json-line format,
that contains 3 fields ::

    'title' (str) - title of article,
    'section_titles' (list) - list of titles of sections,
    'section_texts' (list) - list of content from sections.

English Wikipedia dump available
`here <https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2>`_. Approximate time
for processing is 2.5 hours (i7-6700HQ, SSD).

Examples
--------

Convert wiki to json-lines format:
`python -m gensim.scripts.segment_wiki -f enwiki-latest-pages-articles.xml.bz2 | gzip > enwiki-latest.json.gz`

Read json-lines dump

>>> # iterate over the plain text file we just created
>>> for line in smart_open('enwiki-latest.json.gz'):
>>>    # decode JSON into a Python object
>>>    article = json.loads(line)
>>>
>>>    # each article has a "title", "section_titles" and "section_texts" fields
>>>    print("Article title: %s" % article['title'])
>>>    for section_title, section_text in zip(article['section_titles'], article['section_texts']):
>>>        print("Section title: %s" % section_title)
>>>        print("Section text: %s" % section_text)

"""

import argparse
import json
import logging
import multiprocessing
import re
import sys
from xml.etree import cElementTree

from gensim.corpora.wikicorpus import IGNORED_NAMESPACES, WikiCorpus, filter_wiki, get_namespace, utils
from smart_open import smart_open


logger = logging.getLogger(__name__)


def segment_all_articles(file_path):
    """Extract article titles and sections from a MediaWiki bz2 database dump.

    Parameters
    ----------
    file_path : str
        Path to MediaWiki dump, typical filename is <LANG>wiki-<YYYYMMDD>-pages-articles.xml.bz2
        or <LANG>wiki-latest-pages-articles.xml.bz2.

    Yields
    ------
    (str, list of (str, str))
        Structure contains (title, [(section_heading, section_content), ...]).

    """
    with smart_open(file_path, 'rb') as xml_fileobj:
        wiki_sections_corpus = _WikiSectionsCorpus(xml_fileobj)
        wiki_sections_corpus.metadata = True
        wiki_sections_text = wiki_sections_corpus.get_texts_with_sections()
        for article_title, article_sections in wiki_sections_text:
            yield article_title, article_sections


def segment_and_write_all_articles(file_path, output_file):
    """Write article title and sections to output_file,
    output_file is json-line file with 3 fields::

        'title' - title of article,
        'section_titles' - list of titles of sections,
        'section_texts' - list of content from sections.

    Parameters
    ----------
    file_path : str
        Path to MediaWiki dump, typical filename is <LANG>wiki-<YYYYMMDD>-pages-articles.xml.bz2
        or <LANG>wiki-latest-pages-articles.xml.bz2.

    output_file : str
        Path to output file in json-lines format.

    """
    if output_file is None:
        outfile = sys.stdout
    else:
        outfile = smart_open(output_file, 'wb')

    try:
        for idx, (article_title, article_sections) in enumerate(segment_all_articles(file_path)):
            output_data = {"title": article_title, "section_titles": [], "section_texts": []}
            for section_heading, section_content in article_sections:
                output_data["section_titles"].append(section_heading)
                output_data["section_texts"].append(section_content)
            if (idx + 1) % 100000 == 0:
                logger.info("Processed #%d articles", idx + 1)
            outfile.write(json.dumps(output_data) + "\n")
    finally:
        outfile.close()


def extract_page_xmls(f):
    """Extract pages from a MediaWiki database dump.

    Parameters
    ----------
    f : file
        File descriptor of MediaWiki dump.

    Yields
    ------
    str
        XML strings for page tags.

    """
    elems = (elem for _, elem in cElementTree.iterparse(f, events=("end",)))

    elem = next(elems)
    namespace = get_namespace(elem.tag)
    ns_mapping = {"ns": namespace}
    page_tag = "{%(ns)s}page" % ns_mapping

    for elem in elems:
        if elem.tag == page_tag:
            yield cElementTree.tostring(elem)
            # Prune the element tree, as per
            # http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
            # except that we don't need to prune backlinks from the parent
            # because we don't use LXML.
            # We do this only for <page>s, since we need to inspect the
            # ./revision/text element. The pages comprise the bulk of the
            # file, so in practice we prune away enough.
            elem.clear()


def segment(page_xml):
    """Parse the content inside a page tag

    Parameters
    ----------
    page_xml : str
        Content from page tag.

    Returns
    -------
    (str, list of (str, str))
        Structure contains (title, [(section_heading, section_content)]).

    """
    elem = cElementTree.fromstring(page_xml)
    filter_namespaces = ('0',)
    namespace = get_namespace(elem.tag)
    ns_mapping = {"ns": namespace}
    text_path = "./{%(ns)s}revision/{%(ns)s}text" % ns_mapping
    title_path = "./{%(ns)s}title" % ns_mapping
    ns_path = "./{%(ns)s}ns" % ns_mapping
    lead_section_heading = "Introduction"
    top_level_heading_regex = r"\n==[^=].*[^=]==\n"
    top_level_heading_regex_capture = r"\n==([^=].*[^=])==\n"

    title = elem.find(title_path).text
    text = elem.find(text_path).text
    ns = elem.find(ns_path).text
    if ns not in filter_namespaces:
        text = None

    if text is not None:
        section_contents = re.split(top_level_heading_regex, text)
        section_headings = [lead_section_heading] + re.findall(top_level_heading_regex_capture, text)
        assert len(section_contents) == len(section_headings)
    else:
        section_contents = []
        section_headings = []

    section_contents = [filter_wiki(section_content) for section_content in section_contents]
    sections = list(zip(section_headings, section_contents))
    return title, sections


class _WikiSectionsCorpus(WikiCorpus):
    """Treat a wikipedia articles dump (<LANG>wiki-<YYYYMMDD>-pages-articles.xml.bz2
    or <LANG>wiki-latest-pages-articles.xml.bz2) as a (read-only) corpus.

    The documents are extracted on-the-fly, so that the whole (massive) dump can stay compressed on disk.

    """
    def __init__(self, fileobj, processes=None, lemmatize=utils.has_pattern(), filter_namespaces=('0',)):
        """
        Parameters
        ----------
        fileobj : file
            File descriptor of MediaWiki dump.
        processes : int
            Number of processes, max(1, multiprocessing.cpu_count() - 1) if None.
        lemmatize : bool
            If `pattern` package is installed, use fancier shallow parsing to get token lemmas.
            Otherwise, use simple regexp tokenization.
        filter_namespaces : tuple of int
            Enumeration of namespaces that will be ignored.

        """
        self.fileobj = fileobj
        self.filter_namespaces = filter_namespaces
        self.metadata = False
        if processes is None:
            processes = max(1, multiprocessing.cpu_count() - 1)
        self.processes = processes
        self.lemmatize = lemmatize

    def get_texts_with_sections(self):
        """Iterate over the dump, returning titles and text versions of all sections of articles.

        Notes
        -----
        Only articles of sufficient length are returned (short articles & redirects
        etc are ignored).

        Note that this iterates over the **texts**; if you want vectors, just use
        the standard corpus interface instead of this function::

        >>> for vec in wiki_corpus:
        >>>     print(vec)

        Yields
        ------
        (str, list of (str, str))
            Structure contains (title, [(section_heading, section_content), ...]).

        """
        articles = 0
        page_xmls = extract_page_xmls(self.fileobj)
        pool = multiprocessing.Pool(self.processes)
        # process the corpus in smaller chunks of docs, because multiprocessing.Pool
        # is dumb and would load the entire input into RAM at once...
        for group in utils.chunkize(page_xmls, chunksize=10 * self.processes, maxsize=1):
            for article_title, sections in pool.imap(segment, group):  # chunksize=10):
                # article redirects are pruned here
                if any(article_title.startswith(ignore + ':') for ignore in IGNORED_NAMESPACES) \
                        or len(sections) == 0 \
                        or sections[0][1].lstrip().startswith("#REDIRECT"):
                    continue
                articles += 1
                yield (article_title, sections)
        pool.terminate()
        self.length = articles  # cache corpus length


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(processName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s", " ".join(sys.argv))

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=globals()['__doc__'])
    parser.add_argument('-f', '--file', help='Path to MediaWiki database dump', required=True)
    parser.add_argument('-o', '--output', help='Path to output file (stdout if not specified)')
    args = parser.parse_args()
    segment_and_write_all_articles(args.file, args.output)

    logger.info("finished running %s", sys.argv[0])
