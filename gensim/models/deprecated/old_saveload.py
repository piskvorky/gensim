#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
"""Class containing the old SaveLoad class with modeified `unpickle` function is support loading models saved using
an older gensim version.
"""
from __future__ import with_statement

import logging

try:
    from html.entities import name2codepoint as n2cp
except ImportError:
    from htmlentitydefs import name2codepoint as n2cp
try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle

import re
import sys

import numpy as np
import scipy.sparse

from six import iteritems

from smart_open import smart_open

if sys.version_info[0] >= 3:
    unicode = str

logger = logging.getLogger(__name__)


PAT_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE)
RE_HTML_ENTITY = re.compile(r'&(#?)([xX]?)(\w{1,8});', re.UNICODE)


class SaveLoad(object):
    """Class which inherit from this class have save/load functions, which un/pickle them to disk.

    Warnings
    --------
    This uses pickle for de/serializing, so objects must not contain unpicklable attributes,
    such as lambda functions etc.

    """
    @classmethod
    def load(cls, fname, mmap=None):
        """Load a previously saved object (using :meth:`~gensim.utils.SaveLoad.save`) from file.

        Parameters
        ----------
        fname : str
            Path to file that contains needed object.
        mmap : str, optional
            Memory-map option.  If the object was saved with large arrays stored separately, you can load these arrays
            via mmap (shared memory) using `mmap='r'.
            If the file being loaded is compressed (either '.gz' or '.bz2'), then `mmap=None` **must be** set.

        See Also
        --------
        :meth:`~gensim.utils.SaveLoad.save`

        Returns
        -------
        object
            Object loaded from `fname`.

        Raises
        ------
        IOError
            When methods are called on instance (should be called from class).

        """
        logger.info("loading %s object from %s", cls.__name__, fname)

        compress, subname = SaveLoad._adapt_by_suffix(fname)

        obj = unpickle(fname)
        obj._load_specials(fname, mmap, compress, subname)
        logger.info("loaded %s", fname)
        return obj

    def _load_specials(self, fname, mmap, compress, subname):
        """Loads any attributes that were stored specially, and gives the same opportunity
        to recursively included :class:`~gensim.utils.SaveLoad` instances.

        Parameters
        ----------
        fname : str
            Path to file that contains needed object.
        mmap : str
            Memory-map option.
        compress : bool
            Set to True if file is compressed.
        subname : str
            ...


        """
        def mmap_error(obj, filename):
            return IOError(
                'Cannot mmap compressed object %s in file %s. ' % (obj, filename) +
                'Use `load(fname, mmap=None)` or uncompress files manually.'
            )

        for attrib in getattr(self, '__recursive_saveloads', []):
            cfname = '.'.join((fname, attrib))
            logger.info("loading %s recursively from %s.* with mmap=%s", attrib, cfname, mmap)
            getattr(self, attrib)._load_specials(cfname, mmap, compress, subname)

        for attrib in getattr(self, '__numpys', []):
            logger.info("loading %s from %s with mmap=%s", attrib, subname(fname, attrib), mmap)

            if compress:
                if mmap:
                    raise mmap_error(attrib, subname(fname, attrib))

                val = np.load(subname(fname, attrib))['val']
            else:
                val = np.load(subname(fname, attrib), mmap_mode=mmap)

            setattr(self, attrib, val)

        for attrib in getattr(self, '__scipys', []):
            logger.info("loading %s from %s with mmap=%s", attrib, subname(fname, attrib), mmap)
            sparse = unpickle(subname(fname, attrib))
            if compress:
                if mmap:
                    raise mmap_error(attrib, subname(fname, attrib))

                with np.load(subname(fname, attrib, 'sparse')) as f:
                    sparse.data = f['data']
                    sparse.indptr = f['indptr']
                    sparse.indices = f['indices']
            else:
                sparse.data = np.load(subname(fname, attrib, 'data'), mmap_mode=mmap)
                sparse.indptr = np.load(subname(fname, attrib, 'indptr'), mmap_mode=mmap)
                sparse.indices = np.load(subname(fname, attrib, 'indices'), mmap_mode=mmap)

            setattr(self, attrib, sparse)

        for attrib in getattr(self, '__ignoreds', []):
            logger.info("setting ignored attribute %s to None", attrib)
            setattr(self, attrib, None)

    @staticmethod
    def _adapt_by_suffix(fname):
        """Give appropriate compress setting and filename formula.

        Parameters
        ----------
        fname : str
            Input filename.

        Returns
        -------
        (bool, function)
            First argument will be True if `fname` compressed.

        """
        compress, suffix = (True, 'npz') if fname.endswith('.gz') or fname.endswith('.bz2') else (False, 'npy')
        return compress, lambda *args: '.'.join(args + (suffix,))

    def _smart_save(self, fname, separately=None, sep_limit=10 * 1024**2, ignore=frozenset(), pickle_protocol=2):
        """Save the object to file.

        Parameters
        ----------
        fname : str
            Path to file.
        separately : list, optional
            Iterable of attributes than need to store distinctly.
        sep_limit : int, optional
            Limit for separation.
        ignore : frozenset, optional
            Attributes that shouldn't be store.
        pickle_protocol : int, optional
            Protocol number for pickle.

        Notes
        -----
        If `separately` is None, automatically detect large
        numpy/scipy.sparse arrays in the object being stored, and store
        them into separate files. This avoids pickle memory errors and
        allows mmap'ing large arrays back on load efficiently.

        You can also set `separately` manually, in which case it must be
        a list of attribute names to be stored in separate files. The
        automatic check is not performed in this case.

        See Also
        --------
        :meth:`~gensim.utils.SaveLoad.load`

        """
        logger.info("saving %s object under %s, separately %s", self.__class__.__name__, fname, separately)

        compress, subname = SaveLoad._adapt_by_suffix(fname)

        restores = self._save_specials(fname, separately, sep_limit, ignore, pickle_protocol,
                                       compress, subname)
        try:
            pickle(self, fname, protocol=pickle_protocol)
        finally:
            # restore attribs handled specially
            for obj, asides in restores:
                for attrib, val in iteritems(asides):
                    setattr(obj, attrib, val)
        logger.info("saved %s", fname)

    def _save_specials(self, fname, separately, sep_limit, ignore, pickle_protocol, compress, subname):
        """Save aside any attributes that need to be handled separately, including
        by recursion any attributes that are themselves :class:`~gensim.utils.SaveLoad` instances.

        Parameters
        ----------
        fname : str
            Output filename.
        separately : list or None
            Iterable of attributes than need to store distinctly
        sep_limit : int
            Limit for separation.
        ignore : iterable of str
            Attributes that shouldn't be store.
        pickle_protocol : int
            Protocol number for pickle.
        compress : bool
            If True - compress output with :func:`numpy.savez_compressed`.
        subname : function
            Produced by :meth:`~gensim.utils.SaveLoad._adapt_by_suffix`

        Returns
        -------
        list of (obj, {attrib: value, ...})
            Settings that the caller should use to restore each object's attributes that were set aside
            during the default :func:`~gensim.utils.pickle`.

        """
        asides = {}
        sparse_matrices = (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)
        if separately is None:
            separately = []
            for attrib, val in iteritems(self.__dict__):
                if isinstance(val, np.ndarray) and val.size >= sep_limit:
                    separately.append(attrib)
                elif isinstance(val, sparse_matrices) and val.nnz >= sep_limit:
                    separately.append(attrib)

        # whatever's in `separately` or `ignore` at this point won't get pickled
        for attrib in separately + list(ignore):
            if hasattr(self, attrib):
                asides[attrib] = getattr(self, attrib)
                delattr(self, attrib)

        recursive_saveloads = []
        restores = []
        for attrib, val in iteritems(self.__dict__):
            if hasattr(val, '_save_specials'):  # better than 'isinstance(val, SaveLoad)' if IPython reloading
                recursive_saveloads.append(attrib)
                cfname = '.'.join((fname, attrib))
                restores.extend(val._save_specials(cfname, None, sep_limit, ignore, pickle_protocol, compress, subname))

        try:
            numpys, scipys, ignoreds = [], [], []
            for attrib, val in iteritems(asides):
                if isinstance(val, np.ndarray) and attrib not in ignore:
                    numpys.append(attrib)
                    logger.info("storing np array '%s' to %s", attrib, subname(fname, attrib))

                    if compress:
                        np.savez_compressed(subname(fname, attrib), val=np.ascontiguousarray(val))
                    else:
                        np.save(subname(fname, attrib), np.ascontiguousarray(val))

                elif isinstance(val, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)) and attrib not in ignore:
                    scipys.append(attrib)
                    logger.info("storing scipy.sparse array '%s' under %s", attrib, subname(fname, attrib))

                    if compress:
                        np.savez_compressed(
                            subname(fname, attrib, 'sparse'),
                            data=val.data,
                            indptr=val.indptr,
                            indices=val.indices
                        )
                    else:
                        np.save(subname(fname, attrib, 'data'), val.data)
                        np.save(subname(fname, attrib, 'indptr'), val.indptr)
                        np.save(subname(fname, attrib, 'indices'), val.indices)

                    data, indptr, indices = val.data, val.indptr, val.indices
                    val.data, val.indptr, val.indices = None, None, None

                    try:
                        # store array-less object
                        pickle(val, subname(fname, attrib), protocol=pickle_protocol)
                    finally:
                        val.data, val.indptr, val.indices = data, indptr, indices
                else:
                    logger.info("not storing attribute %s", attrib)
                    ignoreds.append(attrib)

            self.__dict__['__numpys'] = numpys
            self.__dict__['__scipys'] = scipys
            self.__dict__['__ignoreds'] = ignoreds
            self.__dict__['__recursive_saveloads'] = recursive_saveloads
        except Exception:
            # restore the attributes if exception-interrupted
            for attrib, val in iteritems(asides):
                setattr(self, attrib, val)
            raise
        return restores + [(self, asides)]

    def save(self, fname_or_handle, separately=None, sep_limit=10 * 1024**2, ignore=frozenset(), pickle_protocol=2):
        """Save the object to file.

        Parameters
        ----------
        fname_or_handle : str or file-like
            Path to output file or already opened file-like object. If the object is a file handle,
            no special array handling will be performed, all attributes will be saved to the same file.
        separately : list of str or None, optional
            If None -  automatically detect large numpy/scipy.sparse arrays in the object being stored, and store
            them into separate files. This avoids pickle memory errors and allows mmap'ing large arrays
            back on load efficiently.
            If list of str - this attributes will be stored in separate files, the automatic check
            is not performed in this case.
        sep_limit : int
            Limit for automatic separation.
        ignore : frozenset of str
            Attributes that shouldn't be serialize/store.
        pickle_protocol : int
            Protocol number for pickle.

        See Also
        --------
        :meth:`~gensim.utils.SaveLoad.load`

        """
        try:
            _pickle.dump(self, fname_or_handle, protocol=pickle_protocol)
            logger.info("saved %s object", self.__class__.__name__)
        except TypeError:  # `fname_or_handle` does not have write attribute
            self._smart_save(fname_or_handle, separately, sep_limit, ignore, pickle_protocol=pickle_protocol)


def unpickle(fname):
    """Load object from `fname`.

    Parameters
    ----------
    fname : str
        Path to pickle file.

    Returns
    -------
    object
        Python object loaded from `fname`.

    """
    with smart_open(fname, 'rb') as f:
        # Because of loading from S3 load can't be used (missing readline in smart_open)
        file_bytes = f.read()
        file_bytes = file_bytes.replace(b'gensim.models.word2vec', b'gensim.models.deprecated.word2vec')
        file_bytes = file_bytes.replace(b'gensim.models.keyedvectors', b'gensim.models.deprecated.keyedvectors')
        file_bytes = file_bytes.replace(b'gensim.models.doc2vec', b'gensim.models.deprecated.doc2vec')
        file_bytes = file_bytes.replace(b'gensim.models.fasttext', b'gensim.models.deprecated.fasttext')
        file_bytes = file_bytes.replace(
            b'gensim.models.wrappers.fasttext', b'gensim.models.deprecated.fasttext_wrapper')
        if sys.version_info > (3, 0):
            return _pickle.loads(file_bytes, encoding='latin1')
        else:
            return _pickle.loads(file_bytes)


def pickle(obj, fname, protocol=2):
    """Pickle object `obj` to file `fname`.

    Parameters
    ----------
    obj : object
        Any python object.
    fname : str
        Path to pickle file.
    protocol : int, optional
        Pickle protocol number, default is 2 to support compatible across python 2.x and 3.x.

    """
    with smart_open(fname, 'wb') as fout:  # 'b' for binary, needed on Windows
        _pickle.dump(obj, fout, protocol=protocol)
