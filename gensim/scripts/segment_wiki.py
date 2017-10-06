#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Jayant Jain <jayant@rare-technologies.com>
# Copyright (C) 2016 RaRe Technologies

"""
Construct a corpus from a Wikipedia (or other MediaWiki-based) database dump and extract sections of pages from it

If you have the `pattern` package installed, this module will use a fancy
lemmatization to get a lemma of each token (instead of plain alphabetic
tokenizer). The package is available at https://github.com/clips/pattern .

"""

import argparse
import json
import logging
import multiprocessing
import re
import sys
from xml.etree import cElementTree

from gensim.corpora.wikicorpus import ARTICLE_MIN_WORDS, IGNORED_NAMESPACES, WikiCorpus, \
    filter_wiki, get_namespace, tokenize, utils
from smart_open import smart_open


logger = logging.getLogger(__name__)


def segment_all_articles(file_path):
    """
    Extract article titles and sections from a MediaWiki bz2 database dump.

    Return an iterable over (str, list) which generates
    (title, [(section_heading, section_content)]) 2-tuples.

    """
    with smart_open(file_path, 'rb') as xml_fileobj:
        wiki_sections_corpus = WikiSectionsCorpus(xml_fileobj)
        wiki_sections_corpus.metadata = True
        wiki_sections_text = wiki_sections_corpus.get_texts_with_sections()
        for article_title, article_sections in wiki_sections_text:
            yield article_title, article_sections


def segment_and_print_all_articles(file_path, output_file):
    """
    Prints article title and sections to stdout, tab-separated
    article_title<tab>section_heading<tab>section_content<tab>section_heading<tab>section_content

    """
    with smart_open(output_file, 'wb') as outfile:
        for idx, (article_title, article_sections) in enumerate(segment_all_articles(file_path)):
            printed_components = [json.dumps(article_title)]
            for section_heading, section_content in article_sections:
                printed_components.append(json.dumps(section_heading))
                printed_components.append(json.dumps(section_content))
            if (idx + 1) % 100000 == 0:
                logger.info("Processed #%d articles", idx + 1)
            outfile.write(u"\t".join(printed_components).encode('utf-8') + "\n")


def extract_page_xmls(f):
    """
    Extract pages from a MediaWiki database dump = open file-like object `f`.

    Return an iterable which generates xml strings for page tags.

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
    """
    Parse the content inside a page tag, returning its content as a list of tokens
    (utf8-encoded strings).

    Returns a 2-tuple (str, list) -
    (title, [(section_heading, section_content)])

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
        assert(len(section_contents) == len(section_headings))
    else:
        section_contents = []
        section_headings = []

    section_contents = [filter_wiki(section_content) for section_content in section_contents]
    sections = list(zip(section_headings, section_contents))
    return title, sections


class WikiSectionsCorpus(WikiCorpus):
    """
    Treat a wikipedia articles dump (\*articles.xml.bz2) as a (read-only) corpus.

    The documents are extracted on-the-fly, so that the whole (massive) dump
    can stay compressed on disk.

    >>> wiki = WikiCorpus('enwiki-20100622-pages-articles.xml.bz2') # create word->word_id mapping, takes almost 8h
    >>> MmCorpus.serialize('wiki_en_vocab200k.mm', wiki) # another 8h, creates a file in MatrixMarket format plus file with id->word

    """
    def __init__(self, fileobj, processes=None, lemmatize=utils.has_pattern(), filter_namespaces=('0',)):
        """
        Initialize the corpus. Unless a dictionary is provided, this scans the
        corpus once, to determine its vocabulary.

        If `pattern` package is installed, use fancier shallow parsing to get
        token lemmas. Otherwise, use simple regexp tokenization. You can override
        this automatic logic by forcing the `lemmatize` parameter explicitly.

        """
        self.fileobj = fileobj
        self.filter_namespaces = filter_namespaces
        self.metadata = False
        if processes is None:
            processes = max(1, multiprocessing.cpu_count() - 1)
        self.processes = processes
        self.lemmatize = lemmatize

    def get_texts_with_sections(self):
        """
        Iterate over the dump, returning titles and text versions of all sections of articles as a list
        of 2-tuples [(article_title, [(section_heading, section_content)]].

        Only articles of sufficient length are returned (short articles & redirects
        etc are ignored).

        Note that this iterates over the **texts**; if you want vectors, just use
        the standard corpus interface instead of this function::

        >>> for vec in wiki_corpus:
        >>>     print(vec)
        """
        articles = 0
        page_xmls = extract_page_xmls(self.fileobj)
        pool = multiprocessing.Pool(self.processes)
        # process the corpus in smaller chunks of docs, because multiprocessing.Pool
        # is dumb and would load the entire input into RAM at once...
        for group in utils.chunkize(page_xmls, chunksize=10 * self.processes, maxsize=1):
            for article_title, sections in pool.imap(segment, group):  # chunksize=10):
                # article redirects and short stubs are pruned here
                num_total_tokens = 0
                for section_title, section_content in sections:
                    if self.lemmatize:
                        num_total_tokens += len(utils.lemmatize(section_content))
                    else:
                        num_total_tokens += len(tokenize(section_content))
                if num_total_tokens < ARTICLE_MIN_WORDS or any(article_title.startswith(ignore + ':') for ignore in IGNORED_NAMESPACES):
                    continue
                articles += 1
                yield (article_title, sections)
        pool.terminate()
        self.length = articles  # cache corpus length


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(processName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s", " ".join(sys.argv))

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=globals()['__doc__'])
    parser.add_argument('-f', '--file', help='path to mediawiki database dump', required=True)
    parser.add_argument('-o', '--output', help='path to output file', required=True)
    args = parser.parse_args()
    segment_and_print_all_articles(args.file, args.output)

    logger.info("finished running %s", sys.argv[0])
