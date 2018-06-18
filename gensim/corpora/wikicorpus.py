#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2012 Lars Buitinck <larsmans@gmail.com>
# Copyright (C) 2018 Emmanouil Stergiadis <em.stergiadis@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""Construct a corpus from a Wikipedia (or other MediaWiki-based) database dump.

Notes
-----
If you have the `pattern` package installed, this module will use a fancy lemmatization to get a lemma
of each token (instead of plain alphabetic tokenizer). The package is available at [1]_ .

See :mod:`~gensim.scripts.make_wiki` for a canned (example) script based on this module.

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

from six import raise_from


logger = logging.getLogger(__name__)

ARTICLE_MIN_WORDS = 50
"""Ignore shorter articles (after full preprocessing)."""

# default thresholds for lengths of individual tokens
TOKEN_MIN_LEN = 2
TOKEN_MAX_LEN = 15

RE_P0 = re.compile(r'<!--.*?-->', re.DOTALL | re.UNICODE)
"""Comments."""
RE_P1 = re.compile(r'<ref([> ].*?)(</ref>|/>)', re.DOTALL | re.UNICODE)
"""Footnotes."""
RE_P2 = re.compile(r'(\n\[\[[a-z][a-z][\w-]*:[^:\]]+\]\])+$', re.UNICODE)
"""Links to languages."""
RE_P3 = re.compile(r'{{([^}{]*)}}', re.DOTALL | re.UNICODE)
"""Template."""
RE_P4 = re.compile(r'{{([^}]*)}}', re.DOTALL | re.UNICODE)
"""Template."""
RE_P5 = re.compile(r'\[(\w+):\/\/(.*?)(( (.*?))|())\]', re.UNICODE)
"""Remove URL, keep description."""
RE_P6 = re.compile(r'\[([^][]*)\|([^][]*)\]', re.DOTALL | re.UNICODE)
"""Simplify links, keep description."""
RE_P7 = re.compile(r'\n\[\[[iI]mage(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)
"""Keep description of images."""
RE_P8 = re.compile(r'\n\[\[[fF]ile(.*?)(\|.*?)*\|(.*?)\]\]', re.UNICODE)
"""Keep description of files."""
RE_P9 = re.compile(r'<nowiki([> ].*?)(</nowiki>|/>)', re.DOTALL | re.UNICODE)
"""External links."""
RE_P10 = re.compile(r'<math([> ].*?)(</math>|/>)', re.DOTALL | re.UNICODE)
"""Math content."""
RE_P11 = re.compile(r'<(.*?)>', re.DOTALL | re.UNICODE)
"""All other tags."""
RE_P12 = re.compile(r'(({\|)|(\|-(?!\d))|(\|}))(.*?)(?=\n)', re.UNICODE)
"""Table formatting."""
RE_P13 = re.compile(r'(?<=(\n[ ])|(\n\n)|([ ]{2})|(.\n)|(.\t))(\||\!)([^[\]\n]*?\|)*', re.UNICODE)
"""Table cell formatting."""
RE_P14 = re.compile(r'\[\[Category:[^][]*\]\]', re.UNICODE)
"""Categories."""
RE_P15 = re.compile(r'\[\[([fF]ile:|[iI]mage)[^]]*(\]\])', re.UNICODE)
"""Remove File and Image templates."""
RE_P16 = re.compile(r'\[{2}(.*?)\]{2}', re.UNICODE)
"""Capture interlinks text and article linked"""
RE_P17 = re.compile(
    r'(\n.{0,4}((bgcolor)|(\d{0,1}[ ]?colspan)|(rowspan)|(style=)|(class=)|(align=)|(scope=))(.*))|'
    '(^.{0,2}((bgcolor)|(\d{0,1}[ ]?colspan)|(rowspan)|(style=)|(class=)|(align=))(.*))',
    re.UNICODE
)
"""Table markup"""
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


def find_interlinks(raw):
    """Find all interlinks to other articles in the dump.

    Parameters
    ----------
    raw : str
        Unicode or utf-8 encoded string.

    Returns
    -------
    dict
        Mapping from the linked article to the actual text found.
    """
    filtered = filter_wiki(raw, promote_remaining=False, simplify_links=False)
    interlinks_raw = re.findall(RE_P16, filtered)

    interlinks = {}
    for parts in [i.split('|') for i in interlinks_raw]:
        actual_title = parts[0]
        try:
            interlink_text = parts[1]
            interlinks[actual_title] = interlink_text
        except IndexError:
            interlinks[actual_title] = actual_title

    legit_interlinks = {i: j for i, j in interlinks.items() if '[' not in i and ']' not in i}
    return legit_interlinks


def filter_wiki(raw, promote_remaining=True, simplify_links=True):
    """Filter out wiki markup from `raw`, leaving only text.

    Parameters
    ----------
    raw : str
        Unicode or utf-8 encoded string.
    promote_remaining : bool
        Whether uncaught markup should be promoted to plain text.
    simplify_links : bool
        Whether links should be simplified keeping only their description text.

    Returns
    -------
    str
        `raw` without markup.
    """
    # parsing of the wiki markup is not perfect, but sufficient for our purposes
    # contributions to improving this code are welcome :)
    text = utils.to_unicode(raw, 'utf8', errors='ignore')
    text = utils.decode_htmlentities(text)  # '&amp;nbsp;' --> '\xa0'
    return remove_markup(text, promote_remaining, simplify_links)


def remove_markup(text, promote_remaining=True, simplify_links=True):
    """Filter out wiki markup from `text`, leaving only text.

    Parameters
    ----------
    text : str
        String containing markup.
    promote_remaining : bool
        Whether uncaught markup should be promoted to plain text.
    simplify_links : bool
        Whether links should be simplified keeping only their description text.

    Returns
    -------
    str
        `text` without markup.

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

        if simplify_links:
            text = re.sub(RE_P6, '\\2', text)  # simplify links, keep description only
        # remove table markup
        text = text.replace("!!", "\n|")  # each table head cell on a separate line
        text = text.replace("|-||", "\n|")  # for cases where a cell is filled with '-'
        text = re.sub(RE_P12, '\n', text)  # remove formatting lines
        text = text.replace('|||', '|\n|')  # each table cell on a separate line(where |{{a|b}}||cell-content)
        text = text.replace('||', '\n|')  # each table cell on a separate line
        text = re.sub(RE_P13, '\n', text)  # leave only cell content
        text = re.sub(RE_P17, '\n', text)  # remove formatting lines

        # remove empty mark-up
        text = text.replace('[]', '')
        # stop if nothing changed between two iterations or after a fixed number of iterations
        if old == text or iters > 2:
            break

    if promote_remaining:
        text = text.replace('[', '').replace(']', '')  # promote all remaining markup to plain text

    return text


def remove_template(s):
    """Remove template wikimedia markup.

    Parameters
    ----------
    s : str
        String containing markup template.

    Returns
    -------
    str
        Сopy of `s` with all the wikimedia markup template removed. See [4]_ for wikimedia templates details.

    Notes
    -----
    Since template can be nested, it is difficult remove them using regular expressions.

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

    Parameters
    ----------
    s : str
        String containing 'File:' and 'Image:' markup.

    Returns
    -------
    str
        Сopy of `s` with all the 'File:' and 'Image:' markup replaced by their corresponding captions. [3]_

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

    Set `token_min_len`, `token_max_len` as character length (not bytes!) thresholds for individual tokens.

    Parameters
    ----------
    content : str
        String without markup (see :func:`~gensim.corpora.wikicorpus.filter_wiki`).
    token_min_len : int
        Minimal token length.
    token_max_len : int
        Maximal token length.
    lower : bool
         If True - convert `content` to lower case.

    Returns
    -------
    list of str
        List of tokens from `content`.

    """
    # TODO maybe ignore tokens with non-latin characters? (no chinese, arabic, russian etc.)
    return [
        utils.to_unicode(token) for token in utils.tokenize(content, lower=lower, errors='ignore')
        if token_min_len <= len(token) <= token_max_len and not token.startswith('_')
    ]


def get_namespace(tag):
    """Get the namespace of tag.

    Parameters
    ----------
    tag : str
        Namespace or tag.

    Returns
    -------
    str
        Matched namespace or tag.

    """
    m = re.match("^{(.*?)}", tag)
    namespace = m.group(1) if m else ""
    if not namespace.startswith("http://www.mediawiki.org/xml/export-"):
        raise ValueError("%s not recognized as MediaWiki dump namespace" % namespace)
    return namespace


_get_namespace = get_namespace


def extract_pages(f, filter_namespaces=False, filter_articles=None):
    """Extract pages from a MediaWiki database dump.

    Parameters
    ----------
    f : file
        File-like object.
    filter_namespaces : list of str or bool
         Namespaces that will be extracted.

    Yields
    ------
    tuple of (str or None, str, str)
        Title, text and page id.

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

            if filter_articles is not None:
                if filter_articles(elem, namespace=namespace, title=title,
                                   text=text, page_tag=page_tag,
                                   text_path=text_path, title_path=title_path,
                                   ns_path=ns_path, pageid_path=pageid_path) is None:
                    text = None

            pageid = elem.find(pageid_path).text
            yield title, text or "", pageid  # empty page will yield None

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
    """Parse a wikipedia article, extract all tokens.

    Notes
    -----
    Set `tokenizer_func` (defaults is :func:`~gensim.corpora.wikicorpus.tokenize`) parameter for languages
    like japanese or thai to perform better tokenization.
    The `tokenizer_func` needs to take 4 parameters: (text: str, token_min_len: int, token_max_len: int, lower: bool).

    Parameters
    ----------
    args : (str, bool, str, int)
        Article text, lemmatize flag (if True, :func:`~gensim.utils.lemmatize` will be used), article title,
        page identificator.
    tokenizer_func : function
        Function for tokenization (defaults is :func:`~gensim.corpora.wikicorpus.tokenize`).
        Needs to have interface:
        tokenizer_func(text: str, token_min_len: int, token_max_len: int, lower: bool) -> list of str.
    token_min_len : int
        Minimal token length.
    token_max_len : int
        Maximal token length.
    lower : bool
         If True - convert article text to lower case.

    Returns
    -------
    (list of str, str, int)
        List of tokens from article, title and page id.

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

    Warnings
    --------
    Should only be used when master is prepared to handle termination of
    child processes.

    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _process_article(args):
    """Same as :func:`~gensim.corpora.wikicorpus.process_article`, but with args in list format.

    Parameters
    ----------
    args : [(str, bool, str, int), (function, int, int, bool)]
        First element - same as `args` from :func:`~gensim.corpora.wikicorpus.process_article`,
        second element is tokenizer function, token minimal length, token maximal length, lowercase flag.

    Returns
    -------
    (list of str, str, int)
        List of tokens from article, title and page id.

    Warnings
    --------
    Should not be called explicitly. Use :func:`~gensim.corpora.wikicorpus.process_article` instead.

    """
    tokenizer_func, token_min_len, token_max_len, lower = args[-1]
    args = args[:-1]

    return process_article(
        args, tokenizer_func=tokenizer_func, token_min_len=token_min_len,
        token_max_len=token_max_len, lower=lower
    )


class WikiCorpus(TextCorpus):
    """Treat a wikipedia articles dump as a **read-only** corpus.

    Supported dump formats:

    * <LANG>wiki-<YYYYMMDD>-pages-articles.xml.bz2
    * <LANG>wiki-latest-pages-articles.xml.bz2

    The documents are extracted on-the-fly, so that the whole (massive) dump can stay compressed on disk.

    Notes
    -----
    Dumps for English wikipedia can be founded `here <https://dumps.wikimedia.org/enwiki/>`_.

    Attributes
    ----------
    metadata : bool
        Whether to write articles titles to serialized corpus.

    Warnings
    --------
    "Multistream" archives are *not* supported in Python 2 due to `limitations in the core bz2 library
    <https://docs.python.org/2/library/bz2.html#de-compression-of-files>`_.

    Examples
    --------
    >>> from gensim.corpora import WikiCorpus, MmCorpus
    >>>
    >>> wiki = WikiCorpus('enwiki-20100622-pages-articles.xml.bz2') # create word->word_id mapping, takes almost 8h
    >>> MmCorpus.serialize('wiki_en_vocab200k.mm', wiki) # another 8h, creates a file in MatrixMarket format and mapping

    """

    def __init__(self, fname, processes=None, lemmatize=utils.has_pattern(), dictionary=None,
                 filter_namespaces=('0',), tokenizer_func=tokenize, article_min_tokens=ARTICLE_MIN_WORDS,
                 token_min_len=TOKEN_MIN_LEN, token_max_len=TOKEN_MAX_LEN, lower=True, filter_articles=None):
        """Initialize the corpus.

        Unless a dictionary is provided, this scans the corpus once,
        to determine its vocabulary.

        Parameters
        ----------
        fname : str
            Path to file with wikipedia dump.
        processes : int, optional
            Number of processes to run, defaults to **number of cpu - 1**.
        lemmatize : bool
            Whether to use lemmatization instead of simple regexp tokenization.
            Defaults to `True` if *pattern* package installed.
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            Dictionary, if not provided,  this scans the corpus once, to determine its vocabulary
            (this needs **really long time**).
        filter_namespaces : tuple of str
            Namespaces to consider.
        tokenizer_func : function, optional
            Function that will be used for tokenization. By default, use :func:`~gensim.corpora.wikicorpus.tokenize`.
            Need to support interface:
            tokenizer_func(text: str, token_min_len: int, token_max_len: int, lower: bool) -> list of str.
        article_min_tokens : int, optional
            Minimum tokens in article. Article will be ignored if number of tokens is less.
        token_min_len : int, optional
            Minimal token length.
        token_max_len : int, optional
            Maximal token length.
        lower : bool, optional
             If True - convert all text to lower case.
        filter_articles: callable or None, optional
            If set, each XML article element will be passed to this callable before being processed. Only articles
            where the callable returns an XML element are processed, returning None allows filtering out
            some articles based on customised rules.

        """
        self.fname = fname
        self.filter_namespaces = filter_namespaces
        self.filter_articles = filter_articles
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
        self.dictionary = dictionary or Dictionary(self.get_texts())

    def get_texts(self):
        """Iterate over the dump, yielding list of tokens for each article.

        Notes
        -----
        This iterates over the **texts**. If you want vectors, just use the standard corpus interface
        instead of this method:

        >>> for vec in wiki_corpus:
        >>>     print(vec)

        Yields
        ------
        list of str
            If `metadata` is False, yield only list of token extracted from the article.
        (list of str, (int, str))
            List of tokens (extracted from the article), page id and article title otherwise.

        """

        articles, articles_all = 0, 0
        positions, positions_all = 0, 0

        tokenization_params = (self.tokenizer_func, self.token_min_len, self.token_max_len, self.lower)
        texts = \
            ((text, self.lemmatize, title, pageid, tokenization_params)
             for title, text, pageid
             in extract_pages(bz2.BZ2File(self.fname), self.filter_namespaces, self.filter_articles))
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
        except PicklingError as exc:
            raise_from(PicklingError('Can not send filtering function {} to multiprocessing, '\
                'make sure the function can be pickled.'.format(self.filter_articles)), exc)
        else:
            logger.info(
                "finished iterating over Wikipedia corpus of %i documents with %i positions "
                "(total %i articles, %i positions before pruning articles shorter than %i words)",
                articles, positions, articles_all, positions_all, ARTICLE_MIN_WORDS
            )
            self.length = articles  # cache corpus length
        finally:
            pool.terminate()
