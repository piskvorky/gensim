#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2012 Lars Buitinck <larsmans@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
Construct a corpus from a Wikipedia (or other MediaWiki-based) database dump.

Notes
-----
If you have the `pattern` package installed, this module will use a fancy
lemmatization to get a lemma of each token (instead of plain alphabetic
tokenizer). The package is available at [1]_ .

See scripts/process_wiki.py for a canned (example) script based on this
module.

References
----------
.. [1] https://github.com/clips/pattern

"""


import bz2
import logging
import multiprocessing
import re
import signal
from xml.etree.cElementTree import \
    iterparse  # LXML isn't faster, so let's go with the built-in solution

from gensim import utils
# cannot import whole gensim.corpora, because that imports wikicorpus...
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.textcorpus import TextCorpus

logger = logging.getLogger(__name__)

ARTICLE_MIN_WORDS = 50
"""Ignore shorter articles (after full preprocessing)"""

# default thresholds for lengths of individual tokens
TOKEN_MIN_LEN = 2
TOKEN_MAX_LEN = 15


RE_P0 = re.compile(r'<!--.*?-->', re.DOTALL | re.UNICODE)
"""Comments"""
RE_P1 = re.compile(r'<ref([> ].*?)(</ref>|/>)', re.DOTALL | re.UNICODE)
"""Footnotes"""
RE_P2 = re.compile(r'(\n\[\[[a-z][a-z][\w-]*:[^:\]]+\]\])+$', re.UNICODE)
"""Links to languages"""
RE_P3 = re.compile(r'{{([^}{]*)}}', re.DOTALL | re.UNICODE)
"""Template"""
RE_P4 = re.compile(r'{{([^}]*)}}', re.DOTALL | re.UNICODE)
"""Template"""
RE_P5 = re.compile(r'\[(\w+):\/\/(.*?)(( (.*?))|())\]', re.UNICODE)
"""Remove URL, keep description"""
RE_P6 = re.compile(r'\[([^][]*)\|([^][]*)\]', re.DOTALL | re.UNICODE)
"""Simplify links, keep description"""
RE_P7 = re.compile(r'\n\[\[[iI]mage(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)
"""Keep description of images"""
RE_P8 = re.compile(r'\n\[\[[fF]ile(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)
"""Keep description of files"""
RE_P9 = re.compile(r'<nowiki([> ].*?)(</nowiki>|/>)', re.DOTALL | re.UNICODE)
"""External links"""
RE_P10 = re.compile(r'<math([> ].*?)(</math>|/>)', re.DOTALL | re.UNICODE)
"""Math content"""
RE_P11 = re.compile(r'<(.*?)>', re.DOTALL | re.UNICODE)
"""All other tags"""
RE_P12 = re.compile(r'\n(({\|)|(\|-)|(\|}))(.*?)(?=\n)', re.UNICODE)
"""Table formatting"""
RE_P13 = re.compile(r'\n(\||\!)(.*?\|)*([^|]*?)', re.UNICODE)
"""Table cell formatting"""
RE_P14 = re.compile(r'\[\[Category:[^][]*\]\]', re.UNICODE)
"""Categories"""
RE_P15 = re.compile(r'\[\[([fF]ile:|[iI]mage)[^]]*(\]\])', re.UNICODE)
"""Remove File and Image templates"""

IGNORED_NAMESPACES = [
    'Wikipedia', 'Category', 'File', 'Portal', 'Template',
    'MediaWiki', 'User', 'Help', 'Book', 'Draft', 'WikiProject',
    'Special', 'Talk'
]
"""MediaWiki namespaces [2]_ that ought to be ignored.

References
----------
.. [2] https://www.mediawiki.org/wiki/Manual:Namespace

"""


def filter_wiki(raw):
    """Filter out wiki markup from `raw`, leaving only text.

    Parameters
    ----------
    raw : str
        Unicode or utf-8 encoded string.

    Returns
    -------
    str

    """
    # parsing of the wiki markup is not perfect, but sufficient for our purposes
    # contributions to improving this code are welcome :)
    text = utils.to_unicode(raw, 'utf8', errors='ignore')
    text = utils.decode_htmlentities(text)  # '&amp;nbsp;' --> '\xa0'
    return remove_markup(text)


def remove_markup(text):
    """Filter out wiki markup from `text`, leaving only text.

    Parameters
    ----------
    text : str
        String containing markup

    Returns
    -------
    str

    """
    text = re.sub(RE_P2, '', text)  # remove the last list (=languages)
    # the wiki markup is recursive (markup inside markup etc)
    # instead of writing a recursive grammar, here we deal with that by removing
    # markup in a loop, starting with inner-most expressions and working outwards,
    # for as long as something changes.
    text = remove_template(text)
    text = remove_file(text)
    iters = 0
    while True:
        old, iters = text, iters + 1
        text = re.sub(RE_P0, '', text)  # remove comments
        text = re.sub(RE_P1, '', text)  # remove footnotes
        text = re.sub(RE_P9, '', text)  # remove outside links
        text = re.sub(RE_P10, '', text)  # remove math content
        text = re.sub(RE_P11, '', text)  # remove all remaining tags
        text = re.sub(RE_P14, '', text)  # remove categories
        text = re.sub(RE_P5, '\\3', text)  # remove urls, keep description
        text = re.sub(RE_P6, '\\2', text)  # simplify links, keep description only
        # remove table markup
        text = text.replace('||', '\n|')  # each table cell on a separate line
        text = re.sub(RE_P12, '\n', text)  # remove formatting lines
        text = re.sub(RE_P13, '\n\\3', text)  # leave only cell content
        # remove empty mark-up
        text = text.replace('[]', '')
        # stop if nothing changed between two iterations or after a fixed number of iterations
        if old == text or iters > 2:
            break

    # the following is needed to make the tokenizer see '[[socialist]]s' as a single word 'socialists'
    # TODO is this really desirable?
    text = text.replace('[', '').replace(']', '')  # promote all remaining markup to plain text
    return text


def remove_template(s):
    """Remove template wikimedia markup.

    Return a copy of `s` with all the wikimedia markup template removed. See
    [4]_ for wikimedia templates
    details.

    Parameters
    ----------
    s : str
        String containing markup template

    Returns
    -------
    str

    Notes
    -----
    Since template can be nested, it is difficult remove them using
    regular expresssions.

    References
    ----------
    .. [4] http://meta.wikimedia.org/wiki/Help:Template

    """

    # Find the start and end position of each template by finding the opening
    # '{{' and closing '}}'
    n_open, n_close = 0, 0
    starts, ends = [], []
    in_template = False
    prev_c = None
    for i, c in enumerate(iter(s)):
        if not in_template:
            if c == '{' and c == prev_c:
                starts.append(i - 1)
                in_template = True
                n_open = 1
        if in_template:
            if c == '{':
                n_open += 1
            elif c == '}':
                n_close += 1
            if n_open == n_close:
                ends.append(i)
                in_template = False
                n_open, n_close = 0, 0
        prev_c = c

    # Remove all the templates
    return ''.join([s[end + 1:start] for start, end in zip(starts + [None], [-1] + ends)])


def remove_file(s):
    """Remove the 'File:' and 'Image:' markup, keeping the file caption.

    Return a copy of `s` with all the 'File:' and 'Image:' markup replaced by
    their corresponding captions. [3]_

    Parameters
    ----------
    s : str
        String containing 'File:' and 'Image:' markup

    Returns
    -------
    str

    References
    ----------
    .. [3] http://www.mediawiki.org/wiki/Help:Images

    """
    # The regex RE_P15 match a File: or Image: markup
    for match in re.finditer(RE_P15, s):
        m = match.group(0)
        caption = m[:-2].split('|')[-1]
        s = s.replace(m, caption, 1)
    return s


def tokenize(content, token_min_len=TOKEN_MIN_LEN, token_max_len=TOKEN_MAX_LEN, lower=True):
    """Tokenize a piece of text from wikipedia.

    Set `token_min_len`, `token_max_len` as character length (not bytes!)
    thresholds for individual tokens.

    Parameters
    ----------
    content : str
        String without markup (see `filter_wiki()`)
    token_min_len : int
    token_max_len : int
    lower : bool
         Whether to lowercase content

    Returns
    -------
    list of str

    """
    # TODO maybe ignore tokens with non-latin characters? (no chinese, arabic, russian etc.)
    return [
        utils.to_unicode(token) for token in utils.tokenize(content, lower=lower, errors='ignore')
        if token_min_len <= len(token) <= token_max_len and not token.startswith('_')
    ]


def get_namespace(tag):
    """Returns the namespace of tag.

    Parameters
    ----------
    tag : str

    Returns
    -------
    str

    """
    m = re.match("^{(.*?)}", tag)
    namespace = m.group(1) if m else ""
    if not namespace.startswith("http://www.mediawiki.org/xml/export-"):
        raise ValueError("%s not recognized as MediaWiki dump namespace" % namespace)
    return namespace


_get_namespace = get_namespace


def extract_pages(f, filter_namespaces=False):
    """Extract pages from a MediaWiki database dump = open file-like object `f`.

    Parameters
    ----------
    f : File
        File-like object

    filter_namespaces : list of str or bool
         Namespaces to consider

    Yields
    ------
    tuple of (str or None, str, str)

    """
    elems = (elem for _, elem in iterparse(f, events=("end",)))

    # We can't rely on the namespace for database dumps, since it's changed
    # it every time a small modification to the format is made. So, determine
    # those from the first element we find, which will be part of the metadata,
    # and construct element paths.
    elem = next(elems)
    namespace = get_namespace(elem.tag)
    ns_mapping = {"ns": namespace}
    page_tag = "{%(ns)s}page" % ns_mapping
    text_path = "./{%(ns)s}revision/{%(ns)s}text" % ns_mapping
    title_path = "./{%(ns)s}title" % ns_mapping
    ns_path = "./{%(ns)s}ns" % ns_mapping
    pageid_path = "./{%(ns)s}id" % ns_mapping

    for elem in elems:
        if elem.tag == page_tag:
            title = elem.find(title_path).text
            text = elem.find(text_path).text

            if filter_namespaces:
                ns = elem.find(ns_path).text
                if ns not in filter_namespaces:
                    text = None

            pageid = elem.find(pageid_path).text
            yield title, text or "", pageid     # empty page will yield None

            # Prune the element tree, as per
            # http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
            # except that we don't need to prune backlinks from the parent
            # because we don't use LXML.
            # We do this only for <page>s, since we need to inspect the
            # ./revision/text element. The pages comprise the bulk of the
            # file, so in practice we prune away enough.
            elem.clear()


_extract_pages = extract_pages  # for backward compatibility


def process_article(args, tokenizer_func=tokenize, token_min_len=TOKEN_MIN_LEN,
                    token_max_len=TOKEN_MAX_LEN, lower=True):
    """Parse a wikipedia article, returning its content as a list of tokens
    (utf8-encoded strings).

    Set `tokenizer_func` (defaults to `tokenize`) parameter for languages like japanese or thai to perform better
    tokenization. The `tokenizer_func` needs to take 4 parameters: (text, token_min_len, token_max_len, lower).

    Parameters
    ----------
    args :

    tokenizer_func :
         (Default value = tokenize)
    token_min_len : int
    token_max_len : int
    lower : bool
         Whether to lowercase result

    Returns
    -------
    tuple(list of str, str, str)

    """
    text, lemmatize, title, pageid = args
    text = filter_wiki(text)
    if lemmatize:
        result = utils.lemmatize(text)
    else:
        result = tokenizer_func(text, token_min_len, token_max_len, lower)
    return result, title, pageid


def init_to_ignore_interrupt():
    """Enables interruption ignoring.

    Notes
    -----
    Should only be used when master is prepared to handle termination of
    child processes.

    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _process_article(args):
    """Same as `process_article`, but with args in list format.

    Parameters
    ----------
    args : list of (function, int, int, bool)

    Returns
    -------
    tuple(list of str, str, str)

    Notes
    -----
    Should not be called explicitly. Use `process_article` instead.

    """

    tokenizer_func, token_min_len, token_max_len, lower = args[-1]
    args = args[:-1]

    return process_article(
        args, tokenizer_func=tokenizer_func, token_min_len=token_min_len,
        token_max_len=token_max_len, lower=lower
    )


class WikiCorpus(TextCorpus):
    """
    Treat a wikipedia articles dump as a (read-only) corpus.

    Supported dump formats:

    *<LANG>wiki-<YYYYMMDD>-pages-articles.xml.bz2*

    *<LANG>wiki-latest-pages-articles.xml.bz2*

    The documents are extracted on-the-fly, so that the whole (massive) dump
    can stay compressed on disk.

    Notes
    -----
    "Multistream" archives are *not* supported in Python 2 due to
    `limitations in the core bz2 library
    <https://docs.python.org/2/library/bz2.html#de-compression-of-files>`_.

    Examples
    --------
    >>> wiki = WikiCorpus('enwiki-20100622-pages-articles.xml.bz2') # create word->word_id mapping, takes almost 8h
    >>> MmCorpus.serialize('wiki_en_vocab200k.mm', wiki) # another 8h, creates a file in MatrixMarket format and mapping

    """
    def __init__(self, fname, processes=None, lemmatize=utils.has_pattern(), dictionary=None,
                 filter_namespaces=('0',), tokenizer_func=tokenize, article_min_tokens=ARTICLE_MIN_WORDS,
                 token_min_len=TOKEN_MIN_LEN, token_max_len=TOKEN_MAX_LEN, lower=True):
        """Initialize the corpus.

        Unless a dictionary is provided, this scans the corpus once,
        to determine its vocabulary.

        Parameters
        ----------
        fname : str
            Filename
        processes : int or None
            Number of processes to run, defaults to *number of cpu - 1*
        lemmatize : bool
            Whether to use lemmatization instead of simple regexp
            tokenization. Defaults to `True` if *pattern* package installed
            and to `False` otherwise
        dictionary : `corpora.Dictionary` or None
        filter_namespaces : tuple of str
            Namespaces to consider
        tokenizer_func : function(text, token_min_len, token_max_len, lower)
            Returns list of tokens. Set this parameter for languages like
            japanese or thai to perform better tokenization
        article_min_tokens : int
            Minimum tokens in article. Article ignored if number of tokens is
            less
        token_min_len : int
        token_max_len : int
        lower : bool
            Whether to lowercase texts

        Attributes
        ----------
        metadata : bool
            Whether to write articles titles to serialized corpus.

        """
        self.fname = fname
        self.filter_namespaces = filter_namespaces
        self.metadata = False
        if processes is None:
            processes = max(1, multiprocessing.cpu_count() - 1)
        self.processes = processes
        self.lemmatize = lemmatize
        self.tokenizer_func = tokenizer_func
        self.article_min_tokens = article_min_tokens
        self.token_min_len = token_min_len
        self.token_max_len = token_max_len
        self.lower = lower

        if dictionary is None:
            self.dictionary = Dictionary(self.get_texts())
        else:
            self.dictionary = dictionary

    def get_texts(self):
        """Iterate over the dump, yielding list of tokens for each article.

        Yields
        ------
        (list of str) or tuple(list of str, tuple(str, str)))

        Notes
        -----
        Only articles of sufficient length are returned (short articles,
        redirects, etc. are ignored). This is control by
        `article_min_tokens` on the class instance.

        Examples
        --------
        Note that this iterates over the **texts**; if you want vectors,
        just use the standard corpus interface instead of this function:

        >>> for vec in wiki_corpus:
        >>>     print(vec)

        """

        articles, articles_all = 0, 0
        positions, positions_all = 0, 0

        tokenization_params = (self.tokenizer_func, self.token_min_len, self.token_max_len, self.lower)
        texts = \
            ((text, self.lemmatize, title, pageid, tokenization_params)
             for title, text, pageid
             in extract_pages(bz2.BZ2File(self.fname), self.filter_namespaces))
        pool = multiprocessing.Pool(self.processes, init_to_ignore_interrupt)

        try:
            # process the corpus in smaller chunks of docs, because multiprocessing.Pool
            # is dumb and would load the entire input into RAM at once...
            for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1):
                for tokens, title, pageid in pool.imap(_process_article, group):
                    articles_all += 1
                    positions_all += len(tokens)
                    # article redirects and short stubs are pruned here
                    if len(tokens) < self.article_min_tokens or \
                            any(title.startswith(ignore + ':') for ignore in IGNORED_NAMESPACES):
                        continue
                    articles += 1
                    positions += len(tokens)
                    if self.metadata:
                        yield (tokens, (pageid, title))
                    else:
                        yield tokens
        except KeyboardInterrupt:
            logger.warn(
                "user terminated iteration over Wikipedia corpus after %i documents with %i positions "
                "(total %i articles, %i positions before pruning articles shorter than %i words)",
                articles, positions, articles_all, positions_all, ARTICLE_MIN_WORDS
            )
        else:
            logger.info(
                "finished iterating over Wikipedia corpus of %i documents with %i positions "
                "(total %i articles, %i positions before pruning articles shorter than %i words)",
                articles, positions, articles_all, positions_all, ARTICLE_MIN_WORDS
            )
            self.length = articles  # cache corpus length
        finally:
            pool.terminate()
