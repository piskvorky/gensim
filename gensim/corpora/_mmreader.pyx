from __future__ import with_statement

from gensim import utils

from six import string_types
from six.moves import xrange
import logging

cimport cython
from libc.stdio cimport sscanf


logger = logging.getLogger(__name__)


cdef class MmReader(object):
    """
    Wrap a term-document matrix on disk (in matrix-market format), and present it
    as an object which supports iteration over the rows (~documents).

    Note that the file is read into memory one document at a time, not the whole
    matrix at once (unlike scipy.io.mmread). This allows us to process corpora
    which are larger than the available RAM.
    """
    cdef public input
    cdef public bint transposed
    cdef public int num_docs, num_terms, num_nnz

    def __init__(self, input, transposed=True):
        """
        Initialize the matrix reader.

        The `input` refers to a file on local filesystem, which is expected to
        be in the sparse (coordinate) Matrix Market format. Documents are assumed
        to be rows of the matrix (and document features are columns).

        `input` is either a string (file path) or a file-like object that supports
        `seek()` (e.g. gzip.GzipFile, bz2.BZ2File). File-like objects are not closed automatically.
        """
        logger.info("initializing cython corpus reader from %s", input)
        self.input, self.transposed = input, transposed
        with utils.file_or_filename(self.input) as lines:
            try:
                header = utils.to_unicode(next(lines)).strip()
                if not header.lower().startswith('%%matrixmarket matrix coordinate real general'):
                    raise ValueError(
                        "File %s not in Matrix Market format with coordinate real general; instead found: \n%s" %
                        (self.input, header)
                    )
            except StopIteration:
                pass

            self.num_docs = self.num_terms = self.num_nnz = 0
            for lineno, line in enumerate(lines):
                line = utils.to_unicode(line)
                if not line.startswith('%'):
                    self.num_docs, self.num_terms, self.num_nnz = (int(x) for x in line.split())
                    if not self.transposed:
                        self.num_docs, self.num_terms = self.num_terms, self.num_docs
                    break

        logger.info(
            "accepted corpus with %i documents, %i features, %i non-zero entries",
            self.num_docs, self.num_terms, self.num_nnz
        )

    def __len__(self):
        return self.num_docs

    def __str__(self):
        return ("MmCorpus(%i documents, %i features, %i non-zero entries)" %
                (self.num_docs, self.num_terms, self.num_nnz))

    def skip_headers(self, input_file):
        """
        Skip file headers that appear before the first document.
        """
        for line in input_file:
            if line.startswith(b'%'):
                continue
            break

    def __iter__(self):
        """
        Iteratively yield vectors from the underlying file, in the format (row_no, vector),
        where vector is a list of (col_no, value) 2-tuples.

        Note that the total number of vectors returned is always equal to the
        number of rows specified in the header; empty documents are inserted and
        yielded where appropriate, even if they are not explicitly stored in the
        Matrix Market file.
        """
        cdef int docid, termid, previd
        cdef double val = 0

        with utils.file_or_filename(self.input) as lines:
            self.skip_headers(lines)

            previd = -1
            for line in lines:
                #docid, termid, val = utils.to_unicode(line).split()  # needed for python3
                if (sscanf(line, "%d %d %lf", &docid, &termid, &val) != 3):
                    raise ValueError("unable to parse line: {}".format(line))

                if not self.transposed:
                    termid, docid = docid, termid

                # -1 because matrix market indexes are 1-based => convert to 0-based
                docid -= 1
                termid -= 1

                assert previd <= docid, "matrix columns must come in ascending order"
                if docid != previd:
                    # change of document: return the document read so far (its id is prevId)
                    if previd >= 0:
                        yield previd, document  # noqa:F821

                    # return implicit (empty) documents between previous id and new id
                    # too, to keep consistent document numbering and corpus length
                    for previd in xrange(previd + 1, docid):
                        yield previd, []

                    # from now on start adding fields to a new document, with a new id
                    previd = docid
                    document = []

                document.append((termid, val,))  # add another field to the current document

        # handle the last document, as a special case
        if previd >= 0:
            yield previd, document

        # return empty documents between the last explicit document and the number
        # of documents as specified in the header
        for previd in xrange(previd + 1, self.num_docs):
            yield previd, []

    def docbyoffset(self, offset):
        """Return document at file offset `offset` (in bytes)"""
        # empty documents are not stored explicitly in MM format, so the index marks
        # them with a special offset, -1.
        cdef int docid, termid, previd
        cdef double val

        if offset == -1:
            return []
        if isinstance(self.input, string_types):
            fin, close_fin = utils.smart_open(self.input), True
        else:
            fin, close_fin = self.input, False

        fin.seek(offset)  # works for gzip/bz2 input, too
        previd, document = -1, []
        for line in fin:
            if (sscanf(line, "%d %d %lf", &docid, &termid, &val) != 3):
                raise ValueError("unable to parse line: {}".format(line))

            if not self.transposed:
                termid, docid = docid, termid

            # -1 because matrix market indexes are 1-based => convert to 0-based
            docid -= 1
            termid -= 1

            assert previd <= docid, "matrix columns must come in ascending order"
            if docid != previd:
                if previd >= 0:
                    break
                previd = docid

            document.append((termid, val,))  # add another field to the current document

        if close_fin:
            fin.close()
        return document
