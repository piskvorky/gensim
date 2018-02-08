#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).


"""
Utilities for streaming from several file-like data storages: S3 / HDFS / standard
filesystem / compressed files..., using a single, Pythonic API.

The streaming makes heavy use of generators and pipes, to avoid loading
full file contents into memory, allowing work with arbitrarily large files.

The main methods are:

* `smart_open()`, which opens the given file for reading/writing
* `s3_iter_bucket()`, which goes over all keys in an S3 bucket in parallel

"""

import codecs
import logging
import os
import subprocess
import sys
import requests
import io
import warnings

from boto.compat import BytesIO, urlsplit, six
import boto.s3.connection
import boto.s3.key
from ssl import SSLError
import sys


IS_PY2 = (sys.version_info[0] == 2)

if IS_PY2:
    import cStringIO as StringIO

if sys.version_info[0] == 2:
    import httplib

elif sys.version_info[0] == 3:
    import io as StringIO
    import http.client as httplib


logger = logging.getLogger(__name__)

# Multiprocessing is unavailable in App Engine (and possibly other sandboxes).
# The only method currently relying on it is s3_iter_bucket, which is instructed
# whether to use it by the MULTIPROCESSING flag.
MULTIPROCESSING = False
try:
    import multiprocessing.pool
    MULTIPROCESSING = True
except ImportError:
    logger.warning("multiprocessing could not be imported and won't be used")
    from itertools import imap

if IS_PY2:
    from bz2file import BZ2File
else:
    from bz2 import BZ2File

import gzip
import smart_open.s3 as smart_open_s3


WEBHDFS_MIN_PART_SIZE = 50 * 1024**2  # minimum part size for HDFS multipart uploads

SYSTEM_ENCODING = sys.getdefaultencoding()

_ISSUE_146_FSTR = (
    "You have explicitly specified encoding=%(encoding)s, but smart_open does "
    "not currently support decoding text via the %(scheme)s scheme. "
    "Re-open the file without specifying an encoding to suppress this warning."
)

DEFAULT_ERRORS = 'strict'


def smart_open(uri, mode="rb", **kw):
    """
    Open the given S3 / HDFS / filesystem file pointed to by `uri` for reading or writing.

    The only supported modes for now are 'rb' (read, default) and 'wb' (replace & write).

    The reads/writes are memory efficient (streamed) and therefore suitable for
    arbitrarily large files.

    The `uri` can be either:

    1. a URI for the local filesystem (compressed ``.gz`` or ``.bz2`` files handled automatically):
       `./lines.txt`, `/home/joe/lines.txt.gz`, `file:///home/joe/lines.txt.bz2`
    2. a URI for HDFS: `hdfs:///some/path/lines.txt`
    3. a URI for Amazon's S3 (can also supply credentials inside the URI):
       `s3://my_bucket/lines.txt`, `s3://my_aws_key_id:key_secret@my_bucket/lines.txt`
    4. an instance of the boto.s3.key.Key class.

    Examples::

      >>> # stream lines from http; you can use context managers too:
      >>> with smart_open.smart_open('http://www.google.com') as fin:
      ...     for line in fin:
      ...         print line

      >>> # stream lines from S3; you can use context managers too:
      >>> with smart_open.smart_open('s3://mybucket/mykey.txt') as fin:
      ...     for line in fin:
      ...         print line

      >>> # you can also use a boto.s3.key.Key instance directly:
      >>> key = boto.connect_s3().get_bucket("my_bucket").get_key("my_key")
      >>> with smart_open.smart_open(key) as fin:
      ...     for line in fin:
      ...         print line

      >>> # stream line-by-line from an HDFS file
      >>> for line in smart_open.smart_open('hdfs:///user/hadoop/my_file.txt'):
      ...    print line

      >>> # stream content *into* S3:
      >>> with smart_open.smart_open('s3://mybucket/mykey.txt', 'wb') as fout:
      ...     for line in ['first line', 'second line', 'third line']:
      ...          fout.write(line + '\n')

      >>> # stream from/to (compressed) local files:
      >>> for line in smart_open.smart_open('/home/radim/my_file.txt'):
      ...    print line
      >>> for line in smart_open.smart_open('/home/radim/my_file.txt.gz'):
      ...    print line
      >>> with smart_open.smart_open('/home/radim/my_file.txt.gz', 'wb') as fout:
      ...    fout.write("hello world!\n")
      >>> with smart_open.smart_open('/home/radim/another.txt.bz2', 'wb') as fout:
      ...    fout.write("good bye!\n")
      >>> # stream from/to (compressed) local files with Expand ~ and ~user constructions:
      >>> for line in smart_open.smart_open('~/my_file.txt'):
      ...    print line
      >>> for line in smart_open.smart_open('my_file.txt'):
      ...    print line

    """
    logger.debug('%r', locals())

    #
    # This is a work-around for the problem described in Issue #144.
    # If the user has explicitly specified an encoding, then assume they want
    # us to open the destination in text mode, instead of the default binary.
    #
    # If we change the default mode to be text, and match the normal behavior
    # of Py2 and 3, then the above assumption will be unnecessary.
    #
    if kw.get('encoding') is not None and 'b' in mode:
        mode = mode.replace('b', '')

    # validate mode parameter
    if not isinstance(mode, six.string_types):
        raise TypeError('mode should be a string')

    if isinstance(uri, six.string_types):
        # this method just routes the request to classes handling the specific storage
        # schemes, depending on the URI protocol in `uri`
        parsed_uri = ParseUri(uri)

        if parsed_uri.scheme in ("file", ):
            # local files -- both read & write supported
            # compression, if any, is determined by the filename extension (.gz, .bz2)
            encoding = kw.pop('encoding', None)
            errors = kw.pop('errors', DEFAULT_ERRORS)
            return file_smart_open(parsed_uri.uri_path, mode, encoding=encoding, errors=errors)
        elif parsed_uri.scheme in ("s3", "s3n", 's3u'):
            return s3_open_uri(parsed_uri, mode, **kw)
        elif parsed_uri.scheme in ("hdfs", ):
            encoding = kw.pop('encoding', None)
            if encoding is not None:
                warnings.warn(_ISSUE_146_FSTR % {'encoding': encoding, 'scheme': parsed_uri.scheme})
            if mode in ('r', 'rb'):
                return HdfsOpenRead(parsed_uri, **kw)
            if mode in ('w', 'wb'):
                return HdfsOpenWrite(parsed_uri, **kw)
            else:
                raise NotImplementedError("file mode %s not supported for %r scheme", mode, parsed_uri.scheme)
        elif parsed_uri.scheme in ("webhdfs", ):
            encoding = kw.pop('encoding', None)
            if encoding is not None:
                warnings.warn(_ISSUE_146_FSTR % {'encoding': encoding, 'scheme': parsed_uri.scheme})
            if mode in ('r', 'rb'):
                return WebHdfsOpenRead(parsed_uri, **kw)
            elif mode in ('w', 'wb'):
                return WebHdfsOpenWrite(parsed_uri, **kw)
            else:
                raise NotImplementedError("file mode %s not supported for %r scheme", mode, parsed_uri.scheme)
        elif parsed_uri.scheme.startswith('http'):
            encoding = kw.pop('encoding', None)
            if encoding is not None:
                warnings.warn(_ISSUE_146_FSTR % {'encoding': encoding, 'scheme': parsed_uri.scheme})
            if mode in ('r', 'rb'):
                return HttpOpenRead(parsed_uri, **kw)
            else:
                raise NotImplementedError("file mode %s not supported for %r scheme", mode, parsed_uri.scheme)
        else:
            raise NotImplementedError("scheme %r is not supported", parsed_uri.scheme)
    elif isinstance(uri, boto.s3.key.Key):
        return s3_open_key(uri, mode, **kw)
    elif hasattr(uri, 'read'):
        # simply pass-through if already a file-like
        return uri
    else:
        raise TypeError('don\'t know how to handle uri %s' % repr(uri))


def s3_open_uri(parsed_uri, mode, **kwargs):
    logger.debug('%r', locals())
    if parsed_uri.access_id is not None:
        kwargs['aws_access_key_id'] = parsed_uri.access_id
    if parsed_uri.access_secret is not None:
        kwargs['aws_secret_access_key'] = parsed_uri.access_secret

    # Get an S3 host. It is required for sigv4 operations.
    host = kwargs.pop('host', None)
    if host is not None:
        kwargs['endpoint_url'] = 'http://' + host

    #
    # TODO: this is the wrong place to handle ignore_extension.
    # It should happen at the highest level in the smart_open function, because
    # it influences other file systems as well, not just S3.
    #
    if kwargs.pop("ignore_extension", False):
        codec = None
    else:
        codec = _detect_codec(parsed_uri.key_id)

    #
    # Codecs work on a byte-level, so the underlying S3 object should
    # always be reading bytes.
    #
    if mode in (smart_open_s3.READ, smart_open_s3.READ_BINARY):
        s3_mode = smart_open_s3.READ_BINARY
    elif mode in (smart_open_s3.WRITE, smart_open_s3.WRITE_BINARY):
        s3_mode = smart_open_s3.WRITE_BINARY
    else:
        raise NotImplementedError('mode %r not implemented for S3' % mode)

    #
    # TODO: I'm not sure how to handle this with boto3.  Any ideas?
    #
    # https://github.com/boto/boto3/issues/334
    #
    # _setup_unsecured_mode()

    encoding = kwargs.get('encoding')
    errors = kwargs.get('errors', DEFAULT_ERRORS)
    fobj = smart_open_s3.open(parsed_uri.bucket_id, parsed_uri.key_id, s3_mode, **kwargs)
    decompressed_fobj = _CODECS[codec](fobj, mode)
    decoded_fobj = encoding_wrapper(decompressed_fobj, mode, encoding=encoding, errors=errors)
    return decoded_fobj


def _setup_unsecured_mode(parsed_uri, kwargs):
    port = kwargs.pop('port', parsed_uri.port)
    if port != 443:
        kwargs['port'] = port

    if not kwargs.pop('is_secure', parsed_uri.scheme != 's3u'):
        kwargs['is_secure'] = False
        # If the security model docker is overridden, honor the host directly.
        kwargs['calling_format'] = boto.s3.connection.OrdinaryCallingFormat()


def s3_open_key(key, mode, **kwargs):
    logger.debug('%r', locals())
    #
    # TODO: handle boto3 keys as well
    #
    host = kwargs.pop('host', None)
    if host is not None:
        kwargs['endpoint_url'] = 'http://' + host

    if kwargs.pop("ignore_extension", False):
        codec = None
    else:
        codec = _detect_codec(key.name)

    #
    # Codecs work on a byte-level, so the underlying S3 object should
    # always be reading bytes.
    #
    if mode in (smart_open_s3.READ, smart_open_s3.READ_BINARY):
        s3_mode = smart_open_s3.READ_BINARY
    elif mode in (smart_open_s3.WRITE, smart_open_s3.WRITE_BINARY):
        s3_mode = smart_open_s3.WRITE_BINARY
    else:
        raise NotImplementedError('mode %r not implemented for S3' % mode)

    logging.debug('codec: %r mode: %r s3_mode: %r', codec, mode, s3_mode)
    encoding = kwargs.get('encoding')
    errors = kwargs.get('errors', DEFAULT_ERRORS)
    fobj = smart_open_s3.open(key.bucket.name, key.name, s3_mode, **kwargs)
    decompressed_fobj = _CODECS[codec](fobj, mode)
    decoded_fobj = encoding_wrapper(decompressed_fobj, mode, encoding=encoding, errors=errors)
    return decoded_fobj


def _detect_codec(filename):
    if filename.endswith(".gz"):
        return 'gzip'
    return None


def _wrap_gzip(fileobj, mode):
    return gzip.GzipFile(fileobj=fileobj, mode=mode)


def _wrap_none(fileobj, mode):
    return fileobj


_CODECS = {
    None: _wrap_none,
    'gzip': _wrap_gzip,
    #
    # TODO: add support for other codecs here.
    #
}


class ParseUri(object):
    """
    Parse the given URI.

    Supported URI schemes are "file", "s3", "s3n", "s3u" and "hdfs".

      * s3 and s3n are treated the same way.
      * s3u is s3 but without SSL.

    Valid URI examples::

      * s3://my_bucket/my_key
      * s3://my_key:my_secret@my_bucket/my_key
      * s3://my_key:my_secret@my_server:my_port@my_bucket/my_key
      * hdfs:///path/file
      * hdfs://path/file
      * webhdfs://host:port/path/file
      * ./local/path/file
      * ~/local/path/file
      * local/path/file
      * ./local/path/file.gz
      * file:///home/user/file
      * file:///home/user/file.bz2

    """
    def __init__(self, uri, default_scheme="file"):
        """
        Assume `default_scheme` if no scheme given in `uri`.

        """
        if os.name == 'nt':
            # urlsplit doesn't work on Windows -- it parses the drive as the scheme...
            if '://' not in uri:
                # no protocol given => assume a local file
                uri = 'file://' + uri
        parsed_uri = urlsplit(uri, allow_fragments=False)
        self.scheme = parsed_uri.scheme if parsed_uri.scheme else default_scheme

        if self.scheme == "hdfs":
            self.uri_path = parsed_uri.netloc + parsed_uri.path
            self.uri_path = "/" + self.uri_path.lstrip("/")

            if not self.uri_path:
                raise RuntimeError("invalid HDFS URI: %s" % uri)
        elif self.scheme == "webhdfs":
            self.uri_path = parsed_uri.netloc + "/webhdfs/v1" + parsed_uri.path
            if parsed_uri.query:
                self.uri_path += "?" + parsed_uri.query

            if not self.uri_path:
                raise RuntimeError("invalid WebHDFS URI: %s" % uri)
        elif self.scheme in ("s3", "s3n", "s3u"):
            self.bucket_id = (parsed_uri.netloc + parsed_uri.path).split('@')
            self.key_id = None
            self.port = 443
            self.host = boto.config.get('s3', 'host', 's3.amazonaws.com')
            self.ordinary_calling_format = False
            if len(self.bucket_id) == 1:
                # URI without credentials: s3://bucket/object
                self.bucket_id, self.key_id = self.bucket_id[0].split('/', 1)
                # "None" credentials are interpreted as "look for credentials in other locations" by boto
                self.access_id, self.access_secret = None, None
            elif len(self.bucket_id) == 2 and len(self.bucket_id[0].split(':')) == 2:
                # URI in full format: s3://key:secret@bucket/object
                # access key id: [A-Z0-9]{20}
                # secret access key: [A-Za-z0-9/+=]{40}
                acc, self.bucket_id = self.bucket_id
                self.access_id, self.access_secret = acc.split(':')
                self.bucket_id, self.key_id = self.bucket_id.split('/', 1)
            elif len(self.bucket_id) == 3 and len(self.bucket_id[0].split(':')) == 2:
                # or URI in extended format: s3://key:secret@server[:port]@bucket/object
                acc,  server, self.bucket_id = self.bucket_id
                self.access_id, self.access_secret = acc.split(':')
                self.bucket_id, self.key_id = self.bucket_id.split('/', 1)
                server = server.split(':')
                self.ordinary_calling_format = True
                self.host = server[0]
                if len(server) == 2:
                    self.port = int(server[1])
            else:
                # more than 2 '@' means invalid uri
                # Bucket names must be at least 3 and no more than 63 characters long.
                # Bucket names must be a series of one or more labels.
                # Adjacent labels are separated by a single period (.).
                # Bucket names can contain lowercase letters, numbers, and hyphens.
                # Each label must start and end with a lowercase letter or a number.
                raise RuntimeError("invalid S3 URI: %s" % uri)
        elif self.scheme == 'file':
            self.uri_path = parsed_uri.netloc + parsed_uri.path

            # '~/tmp' may be expanded to '/Users/username/tmp'
            self.uri_path = os.path.expanduser(self.uri_path)

            if not self.uri_path:
                raise RuntimeError("invalid file URI: %s" % uri)
        elif self.scheme.startswith('http'):
            self.uri_path = uri
        else:
            raise NotImplementedError("unknown URI scheme %r in %r" % (self.scheme, uri))


class HdfsOpenRead(object):
    """
    Implement streamed reader from HDFS, as an iterable & context manager.

    """
    def __init__(self, parsed_uri):
        if parsed_uri.scheme != "hdfs":
            raise TypeError("can only process HDFS files")
        self.parsed_uri = parsed_uri

    def __iter__(self):
        hdfs = subprocess.Popen(["hdfs", "dfs", '-text', self.parsed_uri.uri_path], stdout=subprocess.PIPE)
        return hdfs.stdout

    def read(self, size=None):
        raise NotImplementedError("read() not implemented yet")

    def seek(self, offset, whence=None):
        raise NotImplementedError("seek() not implemented yet")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass


class HdfsOpenWrite(object):
    """
    Implement streamed writer from HDFS, as an iterable & context manager.

    """
    def __init__(self, parsed_uri):
        if parsed_uri.scheme != "hdfs":
            raise TypeError("can only process HDFS files")
        self.parsed_uri = parsed_uri
        self.out_pipe = subprocess.Popen(
            ["hdfs", "dfs", "-put", "-f", "-", self.parsed_uri.uri_path], stdin=subprocess.PIPE
        )

    def write(self, b):
        self.out_pipe.stdin.write(b)

    def seek(self, offset, whence=None):
        raise NotImplementedError("seek() not implemented yet")

    def __enter__(self):
        return self

    def close(self):
        self.out_pipe.stdin.close()

    def __exit__(self, type, value, traceback):
        self.close()


class WebHdfsOpenRead(object):
    """
    Implement streamed reader from WebHDFS, as an iterable & context manager.
    NOTE: it does not support kerberos authentication yet

    """
    def __init__(self, parsed_uri):
        if parsed_uri.scheme != "webhdfs":
            raise TypeError("can only process WebHDFS files")
        self.parsed_uri = parsed_uri
        self.offset = 0

    def __iter__(self):
        payload = {"op": "OPEN"}
        response = requests.get("http://" + self.parsed_uri.uri_path, params=payload, stream=True)
        return response.iter_lines()

    def read(self, size=None):
        """
        Read the specific number of bytes from the file

        Note read() and line iteration (`for line in self: ...`) each have their
        own file position, so they are independent. Doing a `read` will not affect
        the line iteration, and vice versa.
        """
        if not size or size < 0:
            payload = {"op": "OPEN", "offset": self.offset}
            self.offset = 0
        else:
            payload = {"op": "OPEN", "offset": self.offset, "length": size}
            self.offset += size
        response = requests.get("http://" + self.parsed_uri.uri_path, params=payload, stream=True)
        return response.content

    def seek(self, offset, whence=0):
        """
        Seek to the specified position.

        Only seeking to the beginning (offset=0) supported for now.

        """
        if whence == 0 and offset == 0:
            self.offset = 0
        elif whence == 0:
            self.offset = offset
        else:
            raise NotImplementedError("operations with whence not implemented yet")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass


def make_closing(base, **attrs):
    """
    Add support for `with Base(attrs) as fout:` to the base class if it's missing.
    The base class' `close()` method will be called on context exit, to always close the file properly.

    This is needed for gzip.GzipFile, bz2.BZ2File etc in older Pythons (<=2.6), which otherwise
    raise "AttributeError: GzipFile instance has no attribute '__exit__'".
    """
    if not hasattr(base, '__enter__'):
        attrs['__enter__'] = lambda self: self

    if not hasattr(base, '__exit__'):
        attrs['__exit__'] = lambda self, type, value, traceback: self.close()

    return type('Closing' + base.__name__, (base, object), attrs)


class ClosingBZ2File(make_closing(BZ2File)):
    """
    Implements wrapper for BZ2File that closes file object receieved as argument

    """
    def __init__(self, inner_stream, *args, **kwargs):
        self.inner_stream = inner_stream
        super(ClosingBZ2File, self).__init__(inner_stream, *args, **kwargs)

    def close(self):
        super(ClosingBZ2File, self).close()
        if not self.inner_stream.closed:
            self.inner_stream.close()

class ClosingGzipFile(make_closing(gzip.GzipFile)):
    """
    Implement wrapper for GzipFile that closes file object receieved from arguments

    """
    def close(self):
        fileobj = self.fileobj
        super(ClosingGzipFile, self).close()
        if not fileobj.closed:
            fileobj.close()


def compression_wrapper(file_obj, filename, mode):
    """
    This function will wrap the file_obj with an appropriate
    [de]compression mechanism based on the extension of the filename.

    file_obj must either be a filehandle object, or a class which behaves
        like one.

    If the filename extension isn't recognized, will simply return the original
    file_obj.
    """
    _, ext = os.path.splitext(filename)
    if ext == '.bz2':
        return ClosingBZ2File(file_obj, mode)
    elif ext == '.gz':
        return ClosingGzipFile(fileobj=file_obj, mode=mode)
    else:
        return file_obj


def encoding_wrapper(fileobj, mode, encoding=None, errors=DEFAULT_ERRORS):
    """Decode bytes into text, if necessary.

    If mode specifies binary access, does nothing, unless the encoding is
    specified.  A non-null encoding implies text mode.

    :arg fileobj: must quack like a filehandle object.
    :arg str mode: is the mode which was originally requested by the user.
    :arg str encoding: The text encoding to use.  If mode is binary, overrides mode.
    :arg str errors: The method to use when handling encoding/decoding errors.
    :returns: a file object
    """
    logger.debug('encoding_wrapper: %r', locals())

    #
    # If the mode is binary, but the user specified an encoding, assume they
    # want text.  If we don't make this assumption, ignore the encoding and
    # return bytes, smart_open behavior will diverge from the built-in open:
    #
    #   open(filename, encoding='utf-8') returns a text stream in Py3
    #   smart_open(filename, encoding='utf-8') would return a byte stream
    #       without our assumption, because the default mode is rb.
    #
    if 'b' in mode and encoding is None:
        return fileobj

    if encoding is None:
        encoding = SYSTEM_ENCODING

    if mode[0] == 'r':
        decoder = codecs.getreader(encoding)
    else:
        decoder = codecs.getwriter(encoding)
    return decoder(fileobj, errors=errors)


def file_smart_open(fname, mode='rb', encoding=None, errors=DEFAULT_ERRORS):
    """
    Stream from/to local filesystem, transparently (de)compressing gzip and bz2
    files if necessary.

    :arg str fname: The path to the file to open.
    :arg str mode: The mode in which to open the file.
    :arg str encoding: The text encoding to use.
    :arg str errors: The method to use when handling encoding/decoding errors.
    :returns: A file object
    """
    #
    # This is how we get from the filename to the end result.
    # Decompression is optional, but it always accepts bytes and returns bytes.
    # Decoding is also optional, accepts bytes and returns text.
    # The diagram below is for reading, for writing, the flow is from right to
    # left, but the code is identical.
    #
    #           open as binary         decompress?          decode?
    # filename ---------------> bytes -------------> bytes ---------> text
    #                          raw_fobj        decompressed_fobj   decoded_fobj
    #
    try:  # TODO need to fix this place (for cases with r+ and so on)
        raw_mode = {'r': 'rb', 'w': 'wb', 'a': 'ab'}[mode]
    except KeyError:
        raw_mode = mode
    raw_fobj = open(fname, raw_mode)
    decompressed_fobj = compression_wrapper(raw_fobj, fname, raw_mode)
    decoded_fobj = encoding_wrapper(decompressed_fobj, mode, encoding=encoding, errors=errors)
    return decoded_fobj


class HttpReadStream(object):
    """
    Implement streamed reader from a web site, as an iterable & context manager.
    Supports Kerberos and Basic HTTP authentication.

    As long as you don't mix different access patterns (readline vs readlines vs
    read(n) vs read() vs iteration) this will load efficiently in memory.

    """
    def __init__(self, url, mode='r', kerberos=False, user=None, password=None):
        """
        If Kerberos is True, will attempt to use the local Kerberos credentials.
        Otherwise, will try to use "basic" HTTP authentication via username/password.

        If none of those are set, will connect unauthenticated.
        """
        if kerberos:
            import requests_kerberos
            auth = requests_kerberos.HTTPKerberosAuth()
        elif user is not None and password is not None:
            auth = (user, password)
        else:
            auth = None

        self.response = requests.get(url, auth=auth, stream=True)

        if not self.response.ok:
            self.response.raise_for_status()

        self.mode = mode
        self._read_buffer = None
        self._read_iter = None
        self._readline_iter = None

    def __iter__(self):
        return self.response.iter_lines()

    def binary_content(self):
        """Return the content of the request as bytes."""
        return self.response.content

    def readline(self):
        """
        Mimics the readline call to a filehandle object.
        """
        if self._readline_iter is None:
            self._readline_iter = self.response.iter_lines()

        try:
            return next(self._readline_iter)
        except StopIteration:
            # When readline runs out of data, it just returns an empty string
            return ''

    def readlines(self):
        """
        Mimics the readlines call to a filehandle object.
        """
        return list(self.response.iter_lines())

    def seek(self):
        raise NotImplementedError('seek() is not implemented')

    def read(self, size=None):
        """
        Mimics the read call to a filehandle object.
        """
        if size is None:
            return self.response.content
        else:
            if self._read_iter is None:
                self._read_iter = self.response.iter_content(size)
                self._read_buffer = next(self._read_iter)

            while len(self._read_buffer) < size:
                try:
                    self._read_buffer += next(self._read_iter)
                except StopIteration:
                    # Oops, ran out of data early.
                    retval = self._read_buffer
                    self._read_buffer = ''
                    if len(retval) == 0:
                        # When read runs out of data, it just returns empty
                        return ''
                    else:
                        return retval

            # If we got here, it means we have enough data in the buffer
            # to return to the caller.
            retval = self._read_buffer[:size]
            self._read_buffer = self._read_buffer[size:]
            return retval

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        self.response.close()


def HttpOpenRead(parsed_uri, mode='r', **kwargs):
    if parsed_uri.scheme not in ('http', 'https'):
        raise TypeError("can only process http/https urls")
    if mode not in ('r', 'rb'):
        raise NotImplementedError('Streaming write to http not supported')

    url = parsed_uri.uri_path

    response = HttpReadStream(url, **kwargs)

    fname = urlsplit(url, allow_fragments=False).path.split('/')[-1]

    if fname.endswith('.gz'):
        #  Gzip needs a seek-able filehandle, so we need to buffer it.
        buffer = make_closing(io.BytesIO)(response.binary_content())
        return compression_wrapper(buffer, fname, mode)
    else:
        return compression_wrapper(response, fname, mode)


class WebHdfsOpenWrite(object):
    """
    Context manager for writing into webhdfs files

    """
    def __init__(self, parsed_uri, min_part_size=WEBHDFS_MIN_PART_SIZE):
        if parsed_uri.scheme != "webhdfs":
            raise TypeError("can only process WebHDFS files")
        self.parsed_uri = parsed_uri
        self.closed = False
        self.min_part_size = min_part_size
        # creating empty file first
        payload = {"op": "CREATE", "overwrite": True}
        init_response = requests.put("http://" + self.parsed_uri.uri_path, params=payload, allow_redirects=False)
        if not init_response.status_code == httplib.TEMPORARY_REDIRECT:
            raise WebHdfsException(str(init_response.status_code) + "\n" + init_response.content)
        uri = init_response.headers['location']
        response = requests.put(uri, data="", headers={'content-type': 'application/octet-stream'})
        if not response.status_code == httplib.CREATED:
            raise WebHdfsException(str(response.status_code) + "\n" + response.content)
        self.lines = []
        self.parts = 0
        self.chunk_bytes = 0
        self.total_size = 0

    def upload(self, data):
        payload = {"op": "APPEND"}
        init_response = requests.post("http://" + self.parsed_uri.uri_path, params=payload, allow_redirects=False)
        if not init_response.status_code == httplib.TEMPORARY_REDIRECT:
            raise WebHdfsException(str(init_response.status_code) + "\n" + init_response.content)
        uri = init_response.headers['location']
        response = requests.post(uri, data=data, headers={'content-type': 'application/octet-stream'})
        if not response.status_code == httplib.OK:
            raise WebHdfsException(str(response.status_code) + "\n" + response.content)

    def write(self, b):
        """
        Write the given bytes (binary string) into the WebHDFS file from constructor.

        """
        if self.closed:
            raise ValueError("I/O operation on closed file")
        if isinstance(b, six.text_type):
            # not part of API: also accept unicode => encode it as utf8
            b = b.encode('utf8')

        if not isinstance(b, six.binary_type):
            raise TypeError("input must be a binary string")

        self.lines.append(b)
        self.chunk_bytes += len(b)
        self.total_size += len(b)

        if self.chunk_bytes >= self.min_part_size:
            buff = b"".join(self.lines)
            logger.info(
                "uploading part #%i, %i bytes (total %.3fGB)",
                self.parts, len(buff), self.total_size / 1024.0 ** 3
            )
            self.upload(buff)
            logger.debug("upload of part #%i finished", self.parts)
            self.parts += 1
            self.lines, self.chunk_bytes = [], 0

    def seek(self, offset, whence=None):
        raise NotImplementedError("seek() not implemented yet")

    def close(self):
        buff = b"".join(self.lines)
        if buff:
            logger.info(
                "uploading last part #%i, %i bytes (total %.3fGB)",
                self.parts, len(buff), self.total_size / 1024.0 ** 3
            )
            self.upload(buff)
            logger.debug("upload of last part #%i finished", self.parts)
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


def s3_iter_bucket_process_key_with_kwargs(kwargs):
    return s3_iter_bucket_process_key(**kwargs)


def s3_iter_bucket_process_key(key, retries=3):
    """
    Conceptually part of `s3_iter_bucket`, but must remain top-level method because
    of pickling visibility.

    """
    # Sometimes, https://github.com/boto/boto/issues/2409 can happen because of network issues on either side.
    # Retry up to 3 times to ensure its not a transient issue.
    for x in range(0, retries + 1):
        try:
            return key, key.get_contents_as_string()
        except SSLError:
            # Actually fail on last pass through the loop
            if x == retries:
                raise
            # Otherwise, try again, as this might be a transient timeout
            pass


def s3_iter_bucket(bucket, prefix='', accept_key=lambda key: True, key_limit=None, workers=16, retries=3):
    """
    Iterate and download all S3 files under `bucket/prefix`, yielding out
    `(key, key content)` 2-tuples (generator).

    `accept_key` is a function that accepts a key name (unicode string) and
    returns True/False, signalling whether the given key should be downloaded out or
    not (default: accept all keys).

    If `key_limit` is given, stop after yielding out that many results.

    The keys are processed in parallel, using `workers` processes (default: 16),
    to speed up downloads greatly. If multiprocessing is not available, thus
    MULTIPROCESSING is False, this parameter will be ignored.

    Example::

      >>> mybucket = boto.connect_s3().get_bucket('mybucket')

      >>> # get all JSON files under "mybucket/foo/"
      >>> for key, content in s3_iter_bucket(mybucket, prefix='foo/', accept_key=lambda key: key.endswith('.json')):
      ...     print key, len(content)

      >>> # limit to 10k files, using 32 parallel workers (default is 16)
      >>> for key, content in s3_iter_bucket(mybucket, key_limit=10000, workers=32):
      ...     print key, len(content)

    """
    total_size, key_no = 0, -1
    keys = ({'key': key, 'retries': retries} for key in bucket.list(prefix=prefix) if accept_key(key.name))

    if MULTIPROCESSING:
        logger.info("iterating over keys from %s with %i workers", bucket, workers)
        pool = multiprocessing.pool.Pool(processes=workers)
        iterator = pool.imap_unordered(s3_iter_bucket_process_key_with_kwargs, keys)
    else:
        logger.info("iterating over keys from %s without multiprocessing", bucket)
        iterator = imap(s3_iter_bucket_process_key_with_kwargs, keys)

    for key_no, (key, content) in enumerate(iterator):
        if key_no % 1000 == 0:
            logger.info(
                "yielding key #%i: %s, size %i (total %.1fMB)",
                key_no, key, len(content), total_size / 1024.0 ** 2
            )

        yield key, content
        key.close()
        total_size += len(content)

        if key_limit is not None and key_no + 1 >= key_limit:
            # we were asked to output only a limited number of keys => we're done
            break

    if MULTIPROCESSING:
        pool.terminate()

    logger.info("processed %i keys, total size %i" % (key_no + 1, total_size))


class WebHdfsException(Exception):
    def __init__(self, msg=str()):
        self.msg = msg
        super(WebHdfsException, self).__init__(self.msg)
