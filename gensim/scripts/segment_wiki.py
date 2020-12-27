#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Jayant Jain <jayant@rare-technologies.com>
# Copyright (C) 2016 RaRe Technologies

"""This script using for extracting plain text out of a raw Wikipedia dump. Input is an xml.bz2 file provided
by MediaWiki that looks like <LANG>wiki-<YYYYMMDD>-pages-articles.xml.bz2 or <LANG>wiki-latest-pages-articles.xml.bz2
(e.g. 14 GB of https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2).

It streams through all the XML articles using multiple cores (#cores - 1, by default),
decompressing on the fly and extracting plain text from the articles and their sections.

For each extracted article, it prints its title, section names and plain text section contents, in json-line format.

How to use
----------
#. Process Wikipedia dump with this script ::

    python -m gensim.scripts.segment_wiki -i -f enwiki-latest-pages-articles.xml.bz2 -o enwiki-latest.json.gz

#. Read output in simple way:

.. sourcecode:: pycon

    >>> from gensim import utils
    >>> import json
    >>>
    >>> # iterate over the plain text data we just created
    >>> with utils.open('enwiki-latest.json.gz', 'rb') as f:
    >>>     for line in f:
    >>>         # decode each JSON line into a Python dictionary object
    >>>         article = json.loads(line)
    >>>
    >>>         # each article has a "title", a mapping of interlinks and a list of "section_titles" and
    >>>         # "section_texts".
    >>>         print("Article title: %s" % article['title'])
    >>>         print("Interlinks: %s" + article['interlinks'])
    >>>         for section_title, section_text in zip(article['section_titles'], article['section_texts']):
    >>>             print("Section title: %s" % section_title)
    >>>             print("Section text: %s" % section_text)


Notes
-----
Processing the entire English Wikipedia dump takes 1.7 hours (about 3 million articles per hour,
or 10 MB of XML per second) on an 8 core Intel i7-7700 @3.60GHz.


Command line arguments
----------------------

.. program-output:: python -m gensim.scripts.segment_wiki --help
   :ellipsis: 0, -10

"""

import argparse
import json
import logging
import multiprocessing
import re
import sys
from xml.etree import ElementTree
from functools import partial

from gensim.corpora.wikicorpus import IGNORED_NAMESPACES, WikiCorpus, filter_wiki, find_interlinks, get_namespace, utils
import gensim.utils

logger = logging.getLogger(__name__)


def segment_all_articles(file_path, min_article_character=200, workers=None, include_interlinks=False):
    """Extract article titles and sections from a MediaWiki bz2 database dump.

    Parameters
    ----------
    file_path : str
        Path to MediaWiki dump, typical filename is <LANG>wiki-<YYYYMMDD>-pages-articles.xml.bz2
        or <LANG>wiki-latest-pages-articles.xml.bz2.

    min_article_character : int, optional
        Minimal number of character for article (except titles and leading gaps).

    workers: int or None
        Number of parallel workers, max(1, multiprocessing.cpu_count() - 1) if None.

    include_interlinks: bool
        Whether or not interlinks should be included in the output

    Yields
    ------
    (str, list of (str, str), (Optionally) list of (str, str))
        Structure contains (title, [(section_heading, section_content), ...],
        (Optionally) [(interlink_article, interlink_text), ...]).

    """
    with gensim.utils.open(file_path, 'rb') as xml_fileobj:
        wiki_sections_corpus = _WikiSectionsCorpus(
            xml_fileobj, min_article_character=min_article_character, processes=workers,
            include_interlinks=include_interlinks)
        wiki_sections_corpus.metadata = True
        wiki_sections_text = wiki_sections_corpus.get_texts_with_sections()

        for article in wiki_sections_text:
            yield article


def segment_and_write_all_articles(file_path, output_file, min_article_character=200, workers=None,
                                   include_interlinks=False):
    """Write article title and sections to `output_file` (or stdout, if output_file is None).

    The output format is one article per line, in json-line format with 4 fields::

        'title' - title of article,
        'section_titles' - list of titles of sections,
        'section_texts' - list of content from sections,
        (Optional) 'section_interlinks' - list of interlinks in the article.

    Parameters
    ----------
    file_path : str
        Path to MediaWiki dump, typical filename is <LANG>wiki-<YYYYMMDD>-pages-articles.xml.bz2
        or <LANG>wiki-latest-pages-articles.xml.bz2.

    output_file : str or None
        Path to output file in json-lines format, or None for printing to stdout.

    min_article_character : int, optional
        Minimal number of character for article (except titles and leading gaps).

    workers: int or None
        Number of parallel workers, max(1, multiprocessing.cpu_count() - 1) if None.

    include_interlinks: bool
        Whether or not interlinks should be included in the output
    """
    if output_file is None:
        outfile = getattr(sys.stdout, 'buffer', sys.stdout)  # we want write bytes, so for py3 we used 'buffer'
    else:
        outfile = gensim.utils.open(output_file, 'wb')

    try:
        article_stream = segment_all_articles(file_path, min_article_character, workers=workers,
                                              include_interlinks=include_interlinks)
        for idx, article in enumerate(article_stream):
            article_title, article_sections = article[0], article[1]
            if include_interlinks:
                interlinks = article[2]

            output_data = {
                "title": article_title,
                "section_titles": [],
                "section_texts": [],
            }
            if include_interlinks:
                output_data["interlinks"] = interlinks

            for section_heading, section_content in article_sections:
                output_data["section_titles"].append(section_heading)
                output_data["section_texts"].append(section_content)

            if (idx + 1) % 100000 == 0:
                logger.info("processed #%d articles (at %r now)", idx + 1, article_title)
            outfile.write((json.dumps(output_data) + "\n").encode('utf-8'))

    finally:
        if output_file is not None:
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
    elems = (elem for _, elem in ElementTree.iterparse(f, events=("end",)))

    elem = next(elems)
    namespace = get_namespace(elem.tag)
    ns_mapping = {"ns": namespace}
    page_tag = "{%(ns)s}page" % ns_mapping

    for elem in elems:
        if elem.tag == page_tag:
            yield ElementTree.tostring(elem)
            # Prune the element tree, as per
            # http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
            # except that we don't need to prune backlinks from the parent
            # because we don't use LXML.
            # We do this only for <page>s, since we need to inspect the
            # ./revision/text element. The pages comprise the bulk of the
            # file, so in practice we prune away enough.
            elem.clear()


def segment(page_xml, include_interlinks=False):
    """Parse the content inside a page tag

    Parameters
    ----------
    page_xml : str
        Content from page tag.

    include_interlinks : bool
        Whether or not interlinks should be parsed.

    Returns
    -------
    (str, list of (str, str), (Optionally) list of (str, str))
        Structure contains (title, [(section_heading, section_content), ...],
        (Optionally) [(interlink_article, interlink_text), ...]).

    """
    elem = ElementTree.fromstring(page_xml)
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
        if include_interlinks:
            interlinks = find_interlinks(text)
        section_contents = re.split(top_level_heading_regex, text)
        section_headings = [lead_section_heading] + re.findall(top_level_heading_regex_capture, text)
        section_headings = [heading.strip() for heading in section_headings]
        assert len(section_contents) == len(section_headings)
    else:
        interlinks = []
        section_contents = []
        section_headings = []

    section_contents = [filter_wiki(section_content) for section_content in section_contents]
    sections = list(zip(section_headings, section_contents))

    if include_interlinks:
        return title, sections, interlinks
    else:
        return title, sections


class _WikiSectionsCorpus(WikiCorpus):
    """Treat a wikipedia articles dump (<LANG>wiki-<YYYYMMDD>-pages-articles.xml.bz2
    or <LANG>wiki-latest-pages-articles.xml.bz2) as a (read-only) corpus.

    The documents are extracted on-the-fly, so that the whole (massive) dump can stay compressed on disk.

    """

    def __init__(self, fileobj, min_article_character=200, processes=None,
                 lemmatize=None, filter_namespaces=('0',), include_interlinks=False):
        """
        Parameters
        ----------
        fileobj : file
            File descriptor of MediaWiki dump.
        min_article_character : int, optional
            Minimal number of character for article (except titles and leading gaps).
        processes : int, optional
            Number of processes, max(1, multiprocessing.cpu_count() - 1) if None.
        filter_namespaces : tuple of int, optional
            Enumeration of namespaces that will be ignored.
        include_interlinks: bool
            Whether or not interlinks should be included in the output

        """
        if lemmatize is not None:
            raise NotImplementedError(
                'The lemmatize parameter is no longer supported since Gensim 4.0.0. '
                'If you need to lemmatize, use e.g. https://github.com/clips/pattern '
                'to preprocess your corpus before submitting it to Gensim.'
            )

        self.fileobj = fileobj
        self.filter_namespaces = filter_namespaces
        self.metadata = False
        if processes is None:
            processes = max(1, multiprocessing.cpu_count() - 1)
        self.processes = processes
        self.min_article_character = min_article_character
        self.include_interlinks = include_interlinks

    def get_texts_with_sections(self):
        """Iterate over the dump, returning titles and text versions of all sections of articles.

        Notes
        -----
        Only articles of sufficient length are returned (short articles & redirects
        etc are ignored).

        Note that this iterates over the **texts**; if you want vectors, just use
        the standard corpus interface instead of this function:

        .. sourcecode:: pycon

            >>> for vec in wiki_corpus:
            >>>     print(vec)

        Yields
        ------
        (str, list of (str, str), list of (str, str))
            Structure contains (title, [(section_heading, section_content), ...],
            (Optionally)[(interlink_article, interlink_text), ...]).

        """
        skipped_namespace, skipped_length, skipped_redirect = 0, 0, 0
        total_articles, total_sections = 0, 0
        page_xmls = extract_page_xmls(self.fileobj)
        pool = multiprocessing.Pool(self.processes)
        # process the corpus in smaller chunks of docs, because multiprocessing.Pool
        # is dumb and would load the entire input into RAM at once...
        for group in utils.chunkize(page_xmls, chunksize=10 * self.processes, maxsize=1):
            for article in pool.imap(partial(segment, include_interlinks=self.include_interlinks),
                                     group):  # chunksize=10): partial(merge_names, b='Sons')
                article_title, sections = article[0], article[1]

                # article redirects are pruned here
                if any(article_title.startswith(ignore + ':') for ignore in IGNORED_NAMESPACES):  # filter non-articles
                    skipped_namespace += 1
                    continue
                if not sections or sections[0][1].lstrip().lower().startswith("#redirect"):  # filter redirect
                    skipped_redirect += 1
                    continue
                if sum(len(body.strip()) for (_, body) in sections) < self.min_article_character:
                    # filter stubs (incomplete, very short articles)
                    skipped_length += 1
                    continue
                total_articles += 1
                total_sections += len(sections)

                if self.include_interlinks:
                    interlinks = article[2]
                    yield (article_title, sections, interlinks)
                else:
                    yield (article_title, sections)

        logger.info(
            "finished processing %i articles with %i sections (skipped %i redirects, %i stubs, %i ignored namespaces)",
            total_articles, total_sections, skipped_redirect, skipped_length, skipped_namespace)
        pool.terminate()
        self.length = total_articles  # cache corpus length


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(module)s - %(levelname)s - %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=__doc__[:-136])
    default_workers = max(1, multiprocessing.cpu_count() - 1)
    parser.add_argument('-f', '--file', help='Path to MediaWiki database dump (read-only).', required=True)
    parser.add_argument(
        '-o', '--output',
        help='Path to output file (stdout if not specified). If ends in .gz or .bz2, '
             'the output file will be automatically compressed (recommended!).')
    parser.add_argument(
        '-w', '--workers',
        help='Number of parallel workers for multi-core systems. Default: %(default)s.',
        type=int,
        default=default_workers
    )
    parser.add_argument(
        '-m', '--min-article-character',
        help="Ignore articles with fewer characters than this (article stubs). Default: %(default)s.",
        type=int,
        default=200
    )
    parser.add_argument(
        '-i', '--include-interlinks',
        help='Include a mapping for interlinks to other articles in the dump. The mappings format is: '
             '"interlinks": [("article_title_1", "interlink_text_1"), ("article_title_2", "interlink_text_2"), ...]',
        action='store_true'
    )
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    segment_and_write_all_articles(
        args.file, args.output,
        min_article_character=args.min_article_character,
        workers=args.workers,
        include_interlinks=args.include_interlinks
    )

    logger.info("finished running %s", sys.argv[0])
