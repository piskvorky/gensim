#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Radim Rehurek <me@radimrehurek.com>
#
# This code is distributed under the terms and conditions
# from the MIT License (MIT).

import unittest
import logging
import tempfile
import os
import sys
import hashlib

import boto
import mock
from moto import mock_s3
import responses
from ssl import SSLError
import gzip

import smart_open
from smart_open import smart_open_lib

logger = logging.getLogger(__name__)

PY2 = sys.version_info[0] == 2
CURR_DIR = os.path.abspath(os.path.dirname(__file__))


class ParseUriTest(unittest.TestCase):
    """
    Test ParseUri class.

    """
    def test_scheme(self):
        """Do URIs schemes parse correctly?"""
        # supported schemes
        for scheme in ("s3", "s3n", "hdfs", "file", "http", "https"):
            parsed_uri = smart_open.ParseUri(scheme + "://mybucket/mykey")
            self.assertEqual(parsed_uri.scheme, scheme)

        # unsupported scheme => NotImplementedError
        self.assertRaises(NotImplementedError, smart_open.ParseUri, "foobar://mybucket/mykey")

        # unknown scheme => default_scheme
        parsed_uri = smart_open.ParseUri("blah blah")
        self.assertEqual(parsed_uri.scheme, "file")

    def test_s3_uri(self):
        """Do S3 URIs parse correctly?"""
        # correct uri without credentials
        parsed_uri = smart_open.ParseUri("s3://mybucket/mykey")
        self.assertEqual(parsed_uri.scheme, "s3")
        self.assertEqual(parsed_uri.bucket_id, "mybucket")
        self.assertEqual(parsed_uri.key_id, "mykey")
        self.assertEqual(parsed_uri.access_id, None)
        self.assertEqual(parsed_uri.access_secret, None)

        # correct uri, key contains slash
        parsed_uri = smart_open.ParseUri("s3://mybucket/mydir/mykey")
        self.assertEqual(parsed_uri.scheme, "s3")
        self.assertEqual(parsed_uri.bucket_id, "mybucket")
        self.assertEqual(parsed_uri.key_id, "mydir/mykey")
        self.assertEqual(parsed_uri.access_id, None)
        self.assertEqual(parsed_uri.access_secret, None)

        # correct uri with credentials
        parsed_uri = smart_open.ParseUri("s3://ACCESSID456:acces/sse_cr-et@mybucket/mykey")
        self.assertEqual(parsed_uri.scheme, "s3")
        self.assertEqual(parsed_uri.bucket_id, "mybucket")
        self.assertEqual(parsed_uri.key_id, "mykey")
        self.assertEqual(parsed_uri.access_id, "ACCESSID456")
        self.assertEqual(parsed_uri.access_secret, "acces/sse_cr-et")

        # correct uri, contains credentials
        parsed_uri = smart_open.ParseUri("s3://accessid:access/secret@mybucket/mykey")
        self.assertEqual(parsed_uri.scheme, "s3")
        self.assertEqual(parsed_uri.bucket_id, "mybucket")
        self.assertEqual(parsed_uri.key_id, "mykey")
        self.assertEqual(parsed_uri.access_id, "accessid")
        self.assertEqual(parsed_uri.access_secret, "access/secret")

        # incorrect uri - only two '@' in uri are allowed
        self.assertRaises(RuntimeError, smart_open.ParseUri, "s3://access_id@access_secret@mybucket@port/mykey")

    def test_webhdfs_uri(self):
        """Do webhdfs URIs parse correctly"""
        # valid uri, no query
        parsed_uri = smart_open.ParseUri("webhdfs://host:port/path/file")
        self.assertEqual(parsed_uri.scheme, "webhdfs")
        self.assertEqual(parsed_uri.uri_path, "host:port/webhdfs/v1/path/file")

        # valid uri, with query
        parsed_uri = smart_open.ParseUri("webhdfs://host:port/path/file?query_part_1&query_part2")
        self.assertEqual(parsed_uri.scheme, "webhdfs")
        self.assertEqual(parsed_uri.uri_path, "host:port/webhdfs/v1/path/file?query_part_1&query_part2")


class SmartOpenHttpTest(unittest.TestCase):
    """
    Test reading from HTTP connections in various ways.

    """
    @responses.activate
    def test_http_read(self):
        """Does http read method work correctly"""
        responses.add(responses.GET, "http://127.0.0.1/index.html", body='line1\nline2')
        smart_open_object = smart_open.HttpOpenRead(smart_open.ParseUri("http://127.0.0.1/index.html"))
        self.assertEqual(smart_open_object.read().decode("utf-8"), "line1\nline2")

    @responses.activate
    def test_https_readline(self):
        """Does https readline method work correctly"""
        responses.add(responses.GET, "https://127.0.0.1/index.html", body='line1\nline2')
        smart_open_object = smart_open.HttpOpenRead(smart_open.ParseUri("https://127.0.0.1/index.html"))
        self.assertEqual(smart_open_object.readline().decode("utf-8"), "line1")

    @responses.activate
    def test_http_pass(self):
        """Does http authentication work correctly"""
        responses.add(responses.GET, "http://127.0.0.1/index.html", body='line1\nline2')
        _ = smart_open.HttpOpenRead(smart_open.ParseUri("http://127.0.0.1/index.html"), user='me', password='pass')
        self.assertEqual(len(responses.calls), 1)
        actual_request = responses.calls[0].request
        self.assertTrue('Authorization' in actual_request.headers)
        self.assertTrue(actual_request.headers['Authorization'].startswith('Basic '))

    @responses.activate
    def test_http_gz(self):
        """Can open gzip via http?"""
        fpath = os.path.join(CURR_DIR, 'test_data/crlf_at_1k_boundary.warc.gz')
        with open(fpath, 'rb') as infile:
            data = infile.read()

        responses.add(responses.GET, "http://127.0.0.1/data.gz",
                      body=data)
        smart_open_object = smart_open.HttpOpenRead(
            smart_open.ParseUri("http://127.0.0.1/data.gz?some_param=some_val"))

        m = hashlib.md5(smart_open_object.read())
        # decompress the gzip and get the same md5 hash
        self.assertEqual(m.hexdigest(), '18473e60f8c7c98d29d65bf805736a0d')

    @responses.activate
    def test_http_bz2(self):
        """Can open bz2 via http?"""
        test_string = b'Hello World Compressed.'
        with tempfile.NamedTemporaryFile('wb', suffix='.bz2', delete=False) as infile:
            test_file = infile.name

        with smart_open.smart_open(test_file, 'wb') as outfile:
            outfile.write(test_string)

        with open(test_file, 'rb') as infile:
            compressed_data = infile.read()

        if os.path.isfile(test_file):
            os.unlink(test_file)

        responses.add(responses.GET, "http://127.0.0.1/data.bz2", body=compressed_data)
        smart_open_object = smart_open.HttpOpenRead(
            smart_open.ParseUri("http://127.0.0.1/data.bz2")
        )

        # decompress the gzip and get the same md5 hash
        self.assertEqual(smart_open_object.read(), test_string)


class SmartOpenReadTest(unittest.TestCase):
    """
    Test reading from files under various schemes.

    """

    def test_open_with_keywords(self):
        """This test captures Issue #142."""
        fpath = os.path.join(CURR_DIR, 'test_data/cp852.tsv.txt')
        with open(fpath, 'rb') as fin:
            expected = fin.read().decode('cp852')
        with smart_open.smart_open(fpath, encoding='cp852') as fin:
            actual = fin.read()
        self.assertEqual(expected, actual)

    def test_open_with_keywords_explicit_r(self):
        fpath = os.path.join(CURR_DIR, 'test_data/cp852.tsv.txt')
        with open(fpath, 'rb') as fin:
            expected = fin.read().decode('cp852')
        with smart_open.smart_open(fpath, mode='r', encoding='cp852') as fin:
            actual = fin.read()
        self.assertEqual(expected, actual)

    @mock_s3
    def test_read_never_returns_none(self):
        """read should never return None."""
        conn = boto.connect_s3()
        conn.create_bucket("mybucket")
        test_string = u"ветер по морю гуляет..."
        with smart_open.smart_open("s3://mybucket/mykey", "wb") as fout:
            fout.write(test_string.encode('utf8'))

        r = smart_open.smart_open("s3://mybucket/mykey", "rb")
        self.assertEqual(r.read(), test_string.encode("utf-8"))
        self.assertEqual(r.read(), b"")
        self.assertEqual(r.read(), b"")

    @mock_s3
    def test_readline(self):
        """Does readline() return the correct file content?"""
        conn = boto.connect_s3()
        conn.create_bucket("mybucket")
        test_string = u"hello žluťoučký world!\nhow are you?".encode('utf8')
        with smart_open.smart_open("s3://mybucket/mykey", "wb") as fout:
            fout.write(test_string)

        reader = smart_open.smart_open("s3://mybucket/mykey", "rb")
        self.assertEqual(reader.readline(), u"hello žluťoučký world!\n".encode("utf-8"))

    @mock_s3
    def test_readline_iter(self):
        """Does __iter__ return the correct file content?"""
        conn = boto.connect_s3()
        conn.create_bucket("mybucket")
        lines = [u"всем привет!\n", u"что нового?"]
        with smart_open.smart_open("s3://mybucket/mykey", "wb") as fout:
            fout.write("".join(lines).encode("utf-8"))

        reader = smart_open.smart_open("s3://mybucket/mykey", "rb")

        actual_lines = [l.decode("utf-8") for l in reader]
        self.assertEqual(2, len(actual_lines))
        self.assertEqual(lines[0], actual_lines[0])
        self.assertEqual(lines[1], actual_lines[1])

    @mock_s3
    def test_readline_eof(self):
        """Does readline() return empty string on EOF?"""
        conn = boto.connect_s3()
        conn.create_bucket("mybucket")
        with smart_open.smart_open("s3://mybucket/mykey", "wb"):
            pass

        reader = smart_open.smart_open("s3://mybucket/mykey", "rb")

        self.assertEqual(reader.readline(), b"")
        self.assertEqual(reader.readline(), b"")
        self.assertEqual(reader.readline(), b"")

    @mock_s3
    def test_s3_iter_lines(self):
        """Does s3_iter_lines give correct content?"""
        # create fake bucket and fake key
        conn = boto.connect_s3()
        conn.create_bucket("mybucket")
        test_string = u"hello žluťoučký world!\nhow are you?".encode('utf8')
        with smart_open.smart_open("s3://mybucket/mykey", "wb") as fin:
            fin.write(test_string)

        # call s3_iter_lines and check output
        reader = smart_open.smart_open("s3://mybucket/mykey", "rb")
        output = list(reader)
        self.assertEqual(b''.join(output), test_string)

    @mock_s3
    def test_s3_iter_lines_without_key(self):
        """Does s3_iter_lines fail on invalid input?"""
        # cannot use context manager for assertRaise in py2.6
        try:
            for i in smart_open.s3_open_key(None):
                pass
        except TypeError:
            pass
        else:
            self.fail("s3_iter_lines expected to fail on non-`boto.key.Key` inputs")

        try:
            for i in smart_open.s3_open_key("test"):
                pass
        except TypeError:
            pass
        else:
            self.fail("s3_iter_lines extected to fail on non-`boto.key.Key` inputs")

    # TODO: add more complex test for file://
    @mock.patch('smart_open.smart_open_lib.file_smart_open')
    def test_file(self, mock_smart_open):
        """Is file:// line iterator called correctly?"""
        prefix = "file://"
        full_path = '/tmp/test.txt'
        read_mode = "rb"
        smart_open_object = smart_open.smart_open(prefix+full_path, read_mode)
        smart_open_object.__iter__()
        # called with the correct path?
        mock_smart_open.assert_called_with(full_path, read_mode, encoding=None, errors='strict')

        full_path = '/tmp/test#hash##more.txt'
        read_mode = "rb"
        smart_open_object = smart_open.smart_open(prefix+full_path, read_mode)
        smart_open_object.__iter__()
        # called with the correct path?
        mock_smart_open.assert_called_with(full_path, read_mode, encoding=None, errors='strict')

        full_path = 'aa#aa'
        read_mode = "rb"
        smart_open_object = smart_open.smart_open(full_path, read_mode)
        smart_open_object.__iter__()
        # called with the correct path?
        mock_smart_open.assert_called_with(full_path, read_mode, encoding=None, errors='strict')

        short_path = "~/tmp/test.txt"
        full_path = os.path.expanduser(short_path)

        smart_open_object = smart_open.smart_open(prefix+short_path, read_mode, errors='strict')
        smart_open_object.__iter__()
        # called with the correct expanded path?
        mock_smart_open.assert_called_with(full_path, read_mode, encoding=None, errors='strict')

    # couldn't find any project for mocking up HDFS data
    # TODO: we want to test also a content of the files, not just fnc call params
    @mock.patch('smart_open.smart_open_lib.subprocess')
    def test_hdfs(self, mock_subprocess):
        """Is HDFS line iterator called correctly?"""
        mock_subprocess.PIPE.return_value = "test"
        smart_open_object = smart_open.HdfsOpenRead(smart_open.ParseUri("hdfs:///tmp/test.txt"))
        smart_open_object.__iter__()
        # called with the correct params?
        mock_subprocess.Popen.assert_called_with(["hdfs", "dfs", "-text", "/tmp/test.txt"], stdout=mock_subprocess.PIPE)

        # second possibility of schema
        smart_open_object = smart_open.HdfsOpenRead(smart_open.ParseUri("hdfs://tmp/test.txt"))
        smart_open_object.__iter__()
        mock_subprocess.Popen.assert_called_with(["hdfs", "dfs", "-text", "/tmp/test.txt"], stdout=mock_subprocess.PIPE)

    @mock.patch('smart_open.smart_open_lib.subprocess')
    def test_hdfs_encoding(self, mock_subprocess):
        """Is HDFS line iterator called correctly?"""
        mock_subprocess.PIPE.return_value = "test"
        with mock.patch('warnings.warn') as warn:
            smart_open.smart_open("hdfs:///tmp/test.txt", encoding='utf-8')
            expected = smart_open.smart_open_lib._ISSUE_146_FSTR % {
                'encoding': 'utf-8', 'scheme': 'hdfs'
            }
            warn.assert_called_with(expected)

    @responses.activate
    def test_webhdfs(self):
        """Is webhdfs line iterator called correctly"""
        responses.add(responses.GET, "http://127.0.0.1:8440/webhdfs/v1/path/file", body='line1\nline2')
        smart_open_object = smart_open.WebHdfsOpenRead(smart_open.ParseUri("webhdfs://127.0.0.1:8440/path/file"))
        iterator = iter(smart_open_object)
        self.assertEqual(next(iterator).decode("utf-8"), "line1")
        self.assertEqual(next(iterator).decode("utf-8"), "line2")

    @mock.patch('smart_open.smart_open_lib.subprocess')
    def test_webhdfs_encoding(self, mock_subprocess):
        """Is HDFS line iterator called correctly?"""
        url = "webhdfs://127.0.0.1:8440/path/file"
        mock_subprocess.PIPE.return_value = "test"
        with mock.patch('warnings.warn') as warn:
            smart_open.smart_open(url, encoding='utf-8')
            expected = smart_open.smart_open_lib._ISSUE_146_FSTR % {
                'encoding': 'utf-8', 'scheme': 'webhdfs'
            }
            warn.assert_called_with(expected)

    @responses.activate
    def test_webhdfs_read(self):
        """Does webhdfs read method work correctly"""
        responses.add(responses.GET, "http://127.0.0.1:8440/webhdfs/v1/path/file", body='line1\nline2')
        smart_open_object = smart_open.WebHdfsOpenRead(smart_open.ParseUri("webhdfs://127.0.0.1:8440/path/file"))
        self.assertEqual(smart_open_object.read().decode("utf-8"), "line1\nline2")

    @mock_s3
    def test_s3_iter_moto(self):
        """Are S3 files iterated over correctly?"""
        # a list of strings to test with
        expected = [b"*" * 5 * 1024**2] + [b'0123456789'] * 1024 + [b"test"]

        # create fake bucket and fake key
        conn = boto.connect_s3()
        conn.create_bucket("mybucket")

        with smart_open.smart_open("s3://mybucket/mykey", "wb", s3_min_part_size=5 * 1024**2) as fout:
            # write a single huge line (=full multipart upload)
            fout.write(expected[0] + b'\n')

            # write lots of small lines
            for lineno, line in enumerate(expected[1:-1]):
                fout.write(line + b'\n')

            # ...and write the last line too, no newline at the end
            fout.write(expected[-1])

        # connect to fake s3 and read from the fake key we filled above
        smart_open_object = smart_open.smart_open("s3://mybucket/mykey")
        output = [line.rstrip(b'\n') for line in smart_open_object]
        self.assertEqual(output, expected)

        # same thing but using a context manager
        with smart_open.smart_open("s3://mybucket/mykey") as smart_open_object:
            output = [line.rstrip(b'\n') for line in smart_open_object]
            self.assertEqual(output, expected)

    @mock_s3
    def test_s3_read_moto(self):
        """Are S3 files read correctly?"""
        conn = boto.connect_s3()
        conn.create_bucket("mybucket")

        # write some bogus key so we can check it below
        content = u"hello wořld\nhow are you?".encode('utf8')
        with smart_open.smart_open("s3://mybucket/mykey", "wb") as fout:
            fout.write(content)

        smart_open_object = smart_open.smart_open("s3://mybucket/mykey")
        self.assertEqual(content[:6], smart_open_object.read(6))
        self.assertEqual(content[6:14], smart_open_object.read(8))  # ř is 2 bytes

        self.assertEqual(content[14:], smart_open_object.read())  # read the rest

    @unittest.skip('seek functionality for S3 currently disabled because of Issue #152')
    @mock_s3
    def test_s3_seek_moto(self):
        """Does seeking in S3 files work correctly?"""
        conn = boto.connect_s3()
        conn.create_bucket("mybucket")

        # write some bogus key so we can check it below
        content = u"hello wořld\nhow are you?".encode('utf8')
        with smart_open.smart_open("s3://mybucket/mykey", "wb") as fout:
            fout.write(content)

        smart_open_object = smart_open.smart_open("s3://mybucket/mykey")
        self.assertEqual(content[:6], smart_open_object.read(6))
        self.assertEqual(content[6:14], smart_open_object.read(8))  # ř is 2 bytes

        smart_open_object.seek(0)
        self.assertEqual(content, smart_open_object.read())  # no size given => read whole file

        smart_open_object.seek(0)
        self.assertEqual(content, smart_open_object.read(-1))  # same thing


@mock.patch('boto3.Session')
class SmartOpenS3KwargsTest(unittest.TestCase):
    def test_no_kwargs(self, mock_session):
        smart_open.smart_open('s3://mybucket/mykey')
        mock_session.assert_called_with(profile_name=None)
        mock_session.return_value.resource.assert_called_with('s3')

    def test_credentials(self, mock_session):
        smart_open.smart_open('s3://access_id:access_secret@mybucket/mykey')
        mock_session.assert_called_with(profile_name=None)
        mock_session.return_value.resource.assert_called_with(
            's3', aws_access_key_id='access_id', aws_secret_access_key='access_secret'
        )

    def test_profile(self, mock_session):
        smart_open.smart_open('s3://mybucket/mykey', profile_name='my_credentials')
        mock_session.assert_called_with(profile_name='my_credentials')
        mock_session.return_value.resource.assert_called_with('s3')

    def test_host(self, mock_session):
        smart_open.smart_open("s3://access_id:access_secret@mybucket/mykey", host='aa.domain.com')
        mock_session.return_value.resource.assert_called_with(
            's3', aws_access_key_id='access_id', aws_secret_access_key='access_secret',
            endpoint_url='http://aa.domain.com'
        )


class SmartOpenTest(unittest.TestCase):
    """
    Test reading and writing from/into files.

    """
    @mock.patch('smart_open.smart_open_lib.boto')
    @mock.patch('smart_open.smart_open_lib.file_smart_open')
    def test_file_mode_mock(self, mock_file, mock_boto):
        """Are file:// open modes passed correctly?"""
        # incorrect file mode
        self.assertRaises(
            NotImplementedError, smart_open.smart_open, "s3://bucket/key", "x"
        )

        # correct read modes
        smart_open.smart_open("blah", "r")
        mock_file.assert_called_with("blah", "r", encoding=None, errors='strict')

        smart_open.smart_open("blah", "rb")
        mock_file.assert_called_with("blah", "rb", encoding=None, errors='strict')

        short_path = "~/blah"
        full_path = os.path.expanduser(short_path)
        smart_open.smart_open(short_path, "rb")
        mock_file.assert_called_with(full_path, "rb", encoding=None, errors='strict')

        # correct write modes, incorrect scheme
        self.assertRaises(NotImplementedError, smart_open.smart_open, "hdfs:///blah.txt", "wb+")
        self.assertRaises(NotImplementedError, smart_open.smart_open, "http:///blah.txt", "w")
        self.assertRaises(NotImplementedError, smart_open.smart_open, "s3://bucket/key", "wb+")

        # correct write mode, correct file:// URI
        smart_open.smart_open("blah", "w")
        mock_file.assert_called_with("blah", "w", encoding=None, errors='strict')

        smart_open.smart_open("file:///some/file.txt", "wb")
        mock_file.assert_called_with("/some/file.txt", "wb", encoding=None, errors='strict')

        smart_open.smart_open("file:///some/file.txt", "wb+")
        mock_file.assert_called_with("/some/file.txt", "wb+", encoding=None, errors='strict')

        smart_open.smart_open("file:///some/file.txt", "w+")
        mock_file.assert_called_with("/some/file.txt", "w+", encoding=None, errors='strict')

    @mock.patch('boto3.Session')
    def test_s3_mode_mock(self, mock_session):
        """Are s3:// open modes passed correctly?"""

        # correct write mode, correct s3 URI
        smart_open.smart_open("s3://mybucket/mykey", "w", host='s3.amazonaws.com')
        mock_session.return_value.resource.assert_called_with(
            's3', endpoint_url='http://s3.amazonaws.com'
        )

    @mock.patch('smart_open.smart_open_lib.subprocess')
    def test_hdfs(self, mock_subprocess):
        """Is HDFS write called correctly"""
        smart_open_object = smart_open.HdfsOpenWrite(smart_open.ParseUri("hdfs:///tmp/test.txt"))
        smart_open_object.write("test")
        # called with the correct params?
        mock_subprocess.Popen.assert_called_with(
            ["hdfs", "dfs", "-put", "-f", "-", "/tmp/test.txt"], stdin=mock_subprocess.PIPE
        )

        # second possibility of schema
        smart_open_object = smart_open.HdfsOpenWrite(smart_open.ParseUri("hdfs://tmp/test.txt"))
        smart_open_object.write("test")
        mock_subprocess.Popen.assert_called_with(
            ["hdfs", "dfs", "-put", "-f", "-", "/tmp/test.txt"], stdin=mock_subprocess.PIPE
        )

    @unittest.skip('Not sure how to implement unsecured mode with boto3')
    @mock.patch('smart_open.smart_open_lib.boto')
    @mock.patch('smart_open.smart_open_lib.S3OpenWrite')
    def test_s3_unsecured_mode_mock(self, mock_write, mock_boto):
        """Are s3u:// open modes passed correctly?"""
        # Configure the mock boto.config.get to return default host
        smart_open.smart_open_lib.boto.config.get.return_value = 'example.com'

        # correct write mode, correct s3u URI
        smart_open.smart_open("s3u://user:secret@example.com@mybucket/mykey", "w")
        mock_boto.connect_s3.assert_called_with(
            aws_access_key_id='user', aws_secret_access_key='secret',
            profile_name=None, host='example.com', is_secure=False,
            calling_format=mock_boto.s3.connection.OrdinaryCallingFormat())
        mock_boto.connect_s3().lookup.return_value = True
        mock_boto.connect_s3().get_bucket.assert_called_with("mybucket")
        self.assertTrue(mock_write.called)

    @unittest.skip('Not sure how to implement unsecured mode with boto3')
    @mock.patch('smart_open.smart_open_lib.boto')
    @mock.patch('smart_open.smart_open_lib.S3OpenWrite')
    def test_s3_unsecured_mode_with_port_mock(self, mock_write, mock_boto):
        """Are s3u:// open modes passed correctly?"""
        # Configure the mock boto.config.get to return default host
        smart_open.smart_open_lib.boto.config.get.return_value = 'example.com'

        # correct write mode, correct s3u URI
        smart_open.smart_open("s3u://user:secret@example.com:2048@mybucket/mykey", "w")
        mock_boto.connect_s3.assert_called_with(
            aws_access_key_id='user', aws_secret_access_key='secret',
            profile_name=None, host='example.com', is_secure=False, port=2048,
            calling_format=mock_boto.s3.connection.OrdinaryCallingFormat())
        mock_boto.connect_s3().lookup.return_value = True
        mock_boto.connect_s3().get_bucket.assert_called_with("mybucket")
        self.assertTrue(mock_write.called)

    @mock_s3
    def test_s3_modes_moto(self):
        """Do s3:// open modes work correctly?"""
        # fake bucket and key
        conn = boto.connect_s3()
        conn.create_bucket("mybucket")
        test_string = b"second test"

        # correct write mode, correct s3 URI
        with smart_open.smart_open("s3://mybucket/newkey", "wb") as fout:
            logger.debug('fout: %r', fout)
            fout.write(test_string)

        logger.debug("write successfully completed")

        output = list(smart_open.smart_open("s3://mybucket/newkey", "rb"))

        self.assertEqual(output, [test_string])

    @mock_s3
    def test_write_bad_encoding_strict(self):
        """Should abort on encoding error."""
        text = u'欲しい気持ちが成長しすぎて'

        with self.assertRaises(UnicodeEncodeError):
            with tempfile.NamedTemporaryFile('wb', delete=True) as infile:
                with smart_open.smart_open(infile.name, 'w', encoding='koi8-r',
                                           errors='strict') as fout:
                    fout.write(text)

    @mock_s3
    def test_write_bad_encoding_replace(self):
        """Should replace characters that failed to encode."""
        text = u'欲しい気持ちが成長しすぎて'
        expected = u'?' * len(text)

        with tempfile.NamedTemporaryFile('wb', delete=True) as infile:
            with smart_open.smart_open(infile.name, 'w', encoding='koi8-r',
                                       errors='replace') as fout:
                fout.write(text)
            with smart_open.smart_open(infile.name, 'r', encoding='koi8-r') as fin:
                actual = fin.read()

        self.assertEqual(expected, actual)


class WebHdfsWriteTest(unittest.TestCase):
    """
    Test writing into webhdfs files.

    """

    @responses.activate
    def test_initialize_write(self):
        def request_callback(_):
            resp_body = ""
            headers = {'location': 'http://127.0.0.1:8440/file'}
            return 307, headers, resp_body

        responses.add_callback(responses.PUT, "http://127.0.0.1:8440/webhdfs/v1/path/file", callback=request_callback)
        responses.add(responses.PUT, "http://127.0.0.1:8440/file", status=201)
        smart_open.WebHdfsOpenWrite(smart_open.ParseUri("webhdfs://127.0.0.1:8440/path/file"))

        assert len(responses.calls) == 2
        path, params = responses.calls[0].request.url.split("?")
        assert path == "http://127.0.0.1:8440/webhdfs/v1/path/file"
        assert params == "overwrite=True&op=CREATE" or params == "op=CREATE&overwrite=True"
        assert responses.calls[1].request.url == "http://127.0.0.1:8440/file"

    @responses.activate
    def test_write(self):
        def request_callback(_):
            resp_body = ""
            headers = {'location': 'http://127.0.0.1:8440/file'}
            return 307, headers, resp_body

        responses.add_callback(responses.PUT, "http://127.0.0.1:8440/webhdfs/v1/path/file", callback=request_callback)
        responses.add(responses.PUT, "http://127.0.0.1:8440/file", status=201)
        smart_open_object = smart_open.WebHdfsOpenWrite(smart_open.ParseUri("webhdfs://127.0.0.1:8440/path/file"))

        def write_callback(request):
            assert request.body == u"žluťoučký koníček".encode('utf8')
            headers = {}
            return 200, headers, ""

        test_string = u"žluťoučký koníček".encode('utf8')
        responses.add_callback(responses.POST, "http://127.0.0.1:8440/webhdfs/v1/path/file", callback=request_callback)
        responses.add_callback(responses.POST, "http://127.0.0.1:8440/file", callback=write_callback)
        smart_open_object.write(test_string)
        smart_open_object.close()

        assert len(responses.calls) == 4
        assert responses.calls[2].request.url == "http://127.0.0.1:8440/webhdfs/v1/path/file?op=APPEND"
        assert responses.calls[3].request.url == "http://127.0.0.1:8440/file"


class S3IterBucketTest(unittest.TestCase):
    """
    Test parallel iteration of given bucket.

    """
    def test_s3_iter_bucket_process_key_mock(self):
        """Is s3_iter_bucket_process_key called correctly?"""
        attrs = {"name": "fileA", "get_contents_as_string.return_value": b"contentA"}
        mykey = mock.Mock(spec=["name", "get_contents_as_string"])
        mykey.configure_mock(**attrs)

        key, content = smart_open.s3_iter_bucket_process_key(mykey)
        self.assertEqual(key, mykey)
        self.assertEqual(content, b"contentA")

    def test_s3_iter_bucket_process_key_with_SSLError_mock(self):
        attrs = {"name": "fileA", "get_contents_as_string.return_value": b"contentA"}
        mykey = mock.Mock(spec=["name", "get_contents_as_string"])
        mykey.configure_mock(**attrs)

        # when get_contents_as_string always returns SSLError
        mykey.get_contents_as_string.side_effect = SSLError
        self.assertRaises(SSLError, smart_open.s3_iter_bucket_process_key, mykey)

        # when get_contents_as_string only returns SSLError once, can still recover
        mykey.get_contents_as_string.side_effect = [SSLError, b"contentA"]
        key, content = smart_open.s3_iter_bucket_process_key(mykey)
        self.assertEqual(key, mykey)
        self.assertEqual(content, b"contentA")

        # when get_contents_as_string fails up to three times, can still recover
        mykey.get_contents_as_string.side_effect = [SSLError, SSLError, SSLError, b"contentA"]
        key, content = smart_open.s3_iter_bucket_process_key(mykey)
        self.assertEqual(key, mykey)
        self.assertEqual(content, b"contentA")

        # but not more than three times ....
        mykey.get_contents_as_string.side_effect = [SSLError, SSLError, SSLError, SSLError, b"contentA"]
        self.assertRaises(SSLError, smart_open.s3_iter_bucket_process_key, mykey)

        # unless you specify more retries ....
        mykey.get_contents_as_string.side_effect = [SSLError, SSLError, SSLError, SSLError, b"contentA"]
        key, content = smart_open.s3_iter_bucket_process_key(mykey, retries=4)
        self.assertEqual(key, mykey)
        self.assertEqual(content, b"contentA")

        # some other exception always fails, and never retries
        mykey.get_contents_as_string.side_effect = [Exception, b"contentA"]
        self.assertRaises(Exception, smart_open.s3_iter_bucket_process_key, mykey)

    @mock_s3
    def test_s3_iter_bucket_process_key_moto(self):
        """Does s3_iter_bucket_process_key work correctly?"""
        conn = boto.connect_s3()
        conn.create_bucket("mybucket")
        mybucket = conn.get_bucket("mybucket")

        mykey = boto.s3.key.Key(mybucket)
        mykey.key = "mykey"
        mykey.set_contents_from_string("contentA")

        key, content = smart_open.s3_iter_bucket_process_key(mykey)
        self.assertEqual(key, mykey)
        self.assertEqual(content, b"contentA")

    @mock.patch('smart_open.multiprocessing.pool')
    def test_s3_iter_bucket_mock(self, mock_pool):
        """Is s3_iter_bucket called correctly?"""
        attrs = {"name": "fileA", "get_contents_as_string.return_value": "contentA"}
        mykey = mock.Mock(spec=["name", "get_contents_as_string"])
        mykey.configure_mock(**attrs)

        attrs = {"list.return_value": [mykey]}
        mybucket = mock.Mock(spec=["list"])
        mybucket.configure_mock(**attrs)

        for key, content in smart_open.s3_iter_bucket(mybucket):
            mock_pool.Pool.assert_called_with(processes=16)
            mock_pool.Pool().imap_unordered.assert_called_with()

        mock_pool.Pool.assert_called_with(processes=16)
        self.assertTrue(mock_pool.Pool().imap_unordered.called)

    @mock_s3
    def test_s3_iter_bucket_moto(self):
        """Does s3_iter_bucket work correctly?"""
        conn = boto.connect_s3()
        conn.create_bucket("mybucket")
        mybucket = conn.get_bucket("mybucket")

        # first, create some keys in the bucket
        expected = {}
        for key_no in range(200):
            key_name = "mykey%s" % key_no
            with smart_open.smart_open("s3://mybucket/%s" % key_name, 'wb') as fout:
                content = '\n'.join("line%i%i" % (key_no, line_no) for line_no in range(10)).encode('utf8')
                fout.write(content)
                expected[key_name] = content

        # read all keys + their content back, in parallel, using s3_iter_bucket
        result = {}
        for k, c in smart_open.s3_iter_bucket(mybucket):
            result[k.name] = c
        self.assertEqual(expected, result)

        # read some of the keys back, in parallel, using s3_iter_bucket
        result = {}
        for k, c in smart_open.s3_iter_bucket(mybucket, accept_key=lambda fname: fname.endswith('4')):
            result[k.name] = c
        self.assertEqual(result, dict((k, c) for k, c in expected.items() if k.endswith('4')))

        # read some of the keys back, in parallel, using s3_iter_bucket
        result = dict(smart_open.s3_iter_bucket(mybucket, key_limit=10))
        self.assertEqual(len(result), min(len(expected), 10))

        for workers in [1, 4, 8, 16, 64]:
            result = {}
            for k, c in smart_open.s3_iter_bucket(mybucket):
                result[k.name] = c
            self.assertEqual(result, expected)

    @mock.patch('smart_open.multiprocessing.pool.Pool.imap_unordered', map)
    def test_s3_iter_bucket_with_SSLError_moto(self):
        attrs = {"name": "fileA", "get_contents_as_string.return_value": b"contentA"}
        mykey = mock.Mock(spec=["name", "get_contents_as_string"])
        mykey.configure_mock(**attrs)

        attrs = {"list.return_value": [mykey]}
        mybucket = mock.Mock(spec=["list"])
        mybucket.configure_mock(**attrs)

        # when get_contents_as_string always returns SSLError
        mykey.get_contents_as_string.side_effect = SSLError
        self.assertRaises(SSLError, lambda x: next(smart_open.s3_iter_bucket(x)), mybucket)

        # when get_contents_as_string only returns SSLError once, can still recover
        mykey.get_contents_as_string.side_effect = [SSLError, b"contentA"]
        key, content = next(smart_open.s3_iter_bucket(mybucket))
        self.assertEqual(key, mykey)
        self.assertEqual(content, b"contentA")

        # when get_contents_as_string fails up to three times, can still recover
        mykey.get_contents_as_string.side_effect = [SSLError, SSLError, SSLError, b"contentA"]
        key, content = next(smart_open.s3_iter_bucket(mybucket))
        self.assertEqual(key, mykey)
        self.assertEqual(content, b"contentA")

        # but not more than three times ....
        mykey.get_contents_as_string.side_effect = [SSLError, SSLError, SSLError, SSLError, b"contentA"]
        self.assertRaises(SSLError, lambda x: next(smart_open.s3_iter_bucket(x)), mybucket)

        # unless you specify more retries ....
        mykey.get_contents_as_string.side_effect = [SSLError, SSLError, SSLError, SSLError, b"contentA"]
        key, content = next(smart_open.s3_iter_bucket(mybucket, retries=4))
        self.assertEqual(key, mykey)
        self.assertEqual(content, b"contentA")

        # some other exception always fails, and never retries
        mykey.get_contents_as_string.side_effect = [Exception, b"contentA"]
        self.assertRaises(Exception, lambda x: next(smart_open.s3_iter_bucket(x)), mybucket)


class CompressionFormatTest(unittest.TestCase):
    """
    Test that compression
    """

    TEXT = 'Hello'

    def write_read_assertion(self, test_file):
        with smart_open.smart_open(test_file, 'wb') as fout:  # 'b' for binary, needed on Windows
            fout.write(self.TEXT.encode('utf8'))

        with smart_open.smart_open(test_file, 'rb') as fin:
            self.assertEqual(fin.read().decode('utf8'), self.TEXT)

        if os.path.isfile(test_file):
            os.unlink(test_file)

    def test_open_gz(self):
        """Can open gzip?"""
        fpath = os.path.join(CURR_DIR, 'test_data/crlf_at_1k_boundary.warc.gz')
        with smart_open.smart_open(fpath) as infile:
            data = infile.read()
        m = hashlib.md5(data)
        assert m.hexdigest() == '18473e60f8c7c98d29d65bf805736a0d', \
            'Failed to read gzip'

    def test_write_read_gz(self):
        """Can write and read gzip?"""
        with tempfile.NamedTemporaryFile('wb', suffix='.gz', delete=False) as infile:
            test_file_name = infile.name
        self.write_read_assertion(test_file_name)

    def test_write_read_bz2(self):
        """Can write and read bz2?"""
        with tempfile.NamedTemporaryFile('wb', suffix='.bz2', delete=False) as infile:
            test_file_name = infile.name
        self.write_read_assertion(test_file_name)


class MultistreamsBZ2Test(unittest.TestCase):
    """
    Test that multistream bzip2 compressed files can be read.

    """

    # note: these tests are derived from the Python 3.x tip bz2 tests.

    TEXT_LINES = [
        b'root:x:0:0:root:/root:/bin/bash\n',
        b'bin:x:1:1:bin:/bin:\n',
        b'daemon:x:2:2:daemon:/sbin:\n',
        b'adm:x:3:4:adm:/var/adm:\n',
        b'lp:x:4:7:lp:/var/spool/lpd:\n',
        b'sync:x:5:0:sync:/sbin:/bin/sync\n',
        b'shutdown:x:6:0:shutdown:/sbin:/sbin/shutdown\n',
        b'halt:x:7:0:halt:/sbin:/sbin/halt\n',
        b'mail:x:8:12:mail:/var/spool/mail:\n',
        b'news:x:9:13:news:/var/spool/news:\n',
        b'uucp:x:10:14:uucp:/var/spool/uucp:\n',
        b'operator:x:11:0:operator:/root:\n',
        b'games:x:12:100:games:/usr/games:\n',
        b'gopher:x:13:30:gopher:/usr/lib/gopher-data:\n',
        b'ftp:x:14:50:FTP User:/var/ftp:/bin/bash\n',
        b'nobody:x:65534:65534:Nobody:/home:\n',
        b'postfix:x:100:101:postfix:/var/spool/postfix:\n',
        b'niemeyer:x:500:500::/home/niemeyer:/bin/bash\n',
        b'postgres:x:101:102:PostgreSQL Server:/var/lib/pgsql:/bin/bash\n',
        b'mysql:x:102:103:MySQL server:/var/lib/mysql:/bin/bash\n',
        b'www:x:103:104::/var/www:/bin/false\n',
    ]

    TEXT = b''.join(TEXT_LINES)

    DATA = \
        b'BZh91AY&SY.\xc8N\x18\x00\x01>_\x80\x00\x10@\x02\xff\xf0\x01\x07n\x00?\xe7\xff\xe00\x01\x99\xaa\x00' \
        b'\xc0\x03F\x86\x8c#&\x83F\x9a\x03\x06\xa6\xd0\xa6\x93M\x0fQ\xa7\xa8\x06\x804hh\x12$\x11\xa4i4\xf14S' \
        b'\xd2<Q\xb5\x0fH\xd3\xd4\xdd\xd5\x87\xbb\xf8\x94\r\x8f\xafI\x12\xe1\xc9\xf8/E\x00pu\x89\x12]\xc9' \
        b'\xbbDL\nQ\x0e\t1\x12\xdf\xa0\xc0\x97\xac2O9\x89\x13\x94\x0e\x1c7\x0ed\x95I\x0c\xaaJ\xa4\x18L\x10' \
        b'\x05#\x9c\xaf\xba\xbc/\x97\x8a#C\xc8\xe1\x8cW\xf9\xe2\xd0\xd6M\xa7\x8bXa<e\x84t\xcbL\xb3\xa7\xd9' \
        b'\xcd\xd1\xcb\x84.\xaf\xb3\xab\xab\xad`n}\xa0lh\tE,\x8eZ\x15\x17VH>\x88\xe5\xcd9gd6\x0b\n\xe9\x9b' \
        b'\xd5\x8a\x99\xf7\x08.K\x8ev\xfb\xf7xw\xbb\xdf\xa1\x92\xf1\xdd|/";\xa2\xba\x9f\xd5\xb1#A\xb6\xf6' \
        b'\xb3o\xc9\xc5y\\\xebO\xe7\x85\x9a\xbc\xb6f8\x952\xd5\xd7"%\x89>V,\xf7\xa6z\xe2\x9f\xa3\xdf\x11' \
        b'\x11"\xd6E)I\xa9\x13^\xca\xf3r\xd0\x03U\x922\xf26\xec\xb6\xed\x8b\xc3U\x13\x9d\xc5\x170\xa4\xfa^' \
        b'\x92\xacDF\x8a\x97\xd6\x19\xfe\xdd\xb8\xbd\x1a\x9a\x19\xa3\x80ankR\x8b\xe5\xd83]\xa9\xc6\x08' \
        b'\x82f\xf6\xb9"6l$\xb8j@\xc0\x8a\xb0l1..\xbak\x83ls\x15\xbc\xf4\xc1\x13\xbe\xf8E\xb8\x9d\r\xa8\x9dk' \
        b'\x84\xd3n\xfa\xacQ\x07\xb1%y\xaav\xb4\x08\xe0z\x1b\x16\xf5\x04\xe9\xcc\xb9\x08z\x1en7.G\xfc]\xc9\x14' \
        b'\xe1B@\xbb!8`'

    def create_temp_bz2(self, streams=1):
        f = tempfile.NamedTemporaryFile('wb', suffix='.bz2', delete=False)
        f.write(self.DATA * streams)
        f.close()
        return f.name

    def cleanup_temp_bz2(self, test_file):
        if os.path.isfile(test_file):
            os.unlink(test_file)

    def test_can_read_multistream_bz2(self):
        if PY2:
            # this is a backport from Python 3
            from bz2file import BZ2File
        else:
            from bz2 import BZ2File

        test_file = self.create_temp_bz2(streams=5)
        with BZ2File(test_file) as bz2f:
            self.assertEqual(bz2f.read(), self.TEXT * 5)
        self.cleanup_temp_bz2(test_file)

    def test_python2_stdlib_bz2_cannot_read_multistream(self):
        # Multistream bzip is included in Python 3
        if not PY2:
            return
        import bz2

        test_file = self.create_temp_bz2(streams=5)
        bz2f = bz2.BZ2File(test_file)
        self.assertNotEqual(bz2f.read(), self.TEXT * 5)
        bz2f.close()
        self.cleanup_temp_bz2(test_file)

    def test_file_smart_open_can_read_multistream_bz2(self):
        test_file = self.create_temp_bz2(streams=5)
        with smart_open_lib.file_smart_open(test_file) as bz2f:
            self.assertEqual(bz2f.read(), self.TEXT * 5)
        self.cleanup_temp_bz2(test_file)


class S3OpenTest(unittest.TestCase):

    @mock_s3
    def test_r(self):
        """Reading a UTF string should work."""
        conn = boto.connect_s3()
        conn.create_bucket("bucket")
        bucket = conn.get_bucket("bucket")
        key = boto.s3.key.Key(bucket)
        key.key = "key"

        text = u"физкульт-привет!"
        key.set_contents_from_string(text.encode("utf-8"))

        with smart_open.s3_open_key(key, "rb") as fin:
            self.assertEqual(fin.read(), text.encode('utf-8'))

        with smart_open.s3_open_key(key, "r", encoding='utf-8') as fin:
            self.assertEqual(fin.read(), text)

        parsed_uri = smart_open.ParseUri("s3://bucket/key")
        with smart_open.s3_open_uri(parsed_uri, "r", encoding='utf-8') as fin:
            self.assertEqual(fin.read(), text)

    def test_bad_mode(self):
        """Bad mode should raise and exception."""
        uri = smart_open.ParseUri("s3://bucket/key")
        self.assertRaises(NotImplementedError, smart_open.s3_open_uri, uri, "x")

    @mock_s3
    def test_rw_encoding(self):
        """Should read and write text, respecting encodings, etc."""
        conn = boto.connect_s3()
        conn.create_bucket("bucket")

        uri = smart_open.ParseUri("s3://bucket/key")
        text = u"расцветали яблони и груши"

        with smart_open.s3_open_uri(uri, "w", encoding="koi8-r") as fout:
            fout.write(text)

        with smart_open.s3_open_uri(uri, "r", encoding="koi8-r") as fin:
            self.assertEqual(text, fin.read())

        with smart_open.s3_open_uri(uri, "rb") as fin:
            self.assertEqual(text.encode("koi8-r"), fin.read())

        with smart_open.s3_open_uri(uri, "r", encoding="euc-jp") as fin:
            self.assertRaises(UnicodeDecodeError, fin.read)

        with smart_open.s3_open_uri(uri, "r", encoding="euc-jp",
                                    errors="replace") as fin:
            fin.read()

    @mock_s3
    def test_rw_gzip(self):
        """Should read/write gzip files, implicitly and explicitly."""
        conn = boto.connect_s3()
        conn.create_bucket("bucket")
        uri = smart_open.ParseUri("s3://bucket/key.gz")

        text = u"не слышны в саду даже шорохи"
        with smart_open.s3_open_uri(uri, "wb") as fout:
            fout.write(text.encode("utf-8"))

        #
        # Check that what we've created is a gzip.
        #
        with smart_open.s3_open_uri(uri, "rb", ignore_extension=True) as fin:
            gz = gzip.GzipFile(fileobj=fin)
            self.assertEqual(gz.read().decode("utf-8"), text)

        #
        # We should be able to read it back as well.
        #
        with smart_open.s3_open_uri(uri, "rb") as fin:
            self.assertEqual(fin.read().decode("utf-8"), text)

    @mock_s3
    def test_gzip_write_mode(self):
        """Should always open in binary mode when writing through a codec."""
        conn = boto.connect_s3()
        conn.create_bucket("bucket")
        uri = smart_open.ParseUri("s3://bucket/key.gz")

        with mock.patch('smart_open.smart_open_s3.open') as mock_open:
            smart_open.s3_open_uri(uri, "wb")
            mock_open.assert_called_with('bucket', 'key.gz', 'wb')

    @mock_s3
    def test_gzip_read_mode(self):
        """Should always open in binary mode when reading through a codec."""
        conn = boto.connect_s3()
        conn.create_bucket("bucket")
        uri = smart_open.ParseUri("s3://bucket/key.gz")

        text = u"если-б я был султан и имел трёх жён, то тройной красотой был бы окружён"
        with smart_open.s3_open_uri(uri, "wb") as fout:
            fout.write(text.encode("utf-8"))

        with mock.patch('smart_open.smart_open_s3.open') as mock_open:
            smart_open.s3_open_uri(uri, "r")
            mock_open.assert_called_with('bucket', 'key.gz', 'rb')

    @mock_s3
    def test_read_encoding(self):
        """Should open the file with the correct encoding, explicit text read."""
        conn = boto.connect_s3()
        conn.create_bucket('bucket')
        key = "s3://bucket/key.txt"
        text = u'это знала ева, это знал адам, колеса любви едут прямо по нам'
        with smart_open.smart_open(key, 'wb') as fout:
            fout.write(text.encode('koi8-r'))
        with smart_open.smart_open(key, 'r', encoding='koi8-r') as fin:
            actual = fin.read()
        self.assertEqual(text, actual)

    @mock_s3
    def test_read_encoding_implicit_text(self):
        """Should open the file with the correct encoding, implicit text read."""
        conn = boto.connect_s3()
        conn.create_bucket('bucket')
        key = "s3://bucket/key.txt"
        text = u'это знала ева, это знал адам, колеса любви едут прямо по нам'
        with smart_open.smart_open(key, 'wb') as fout:
            fout.write(text.encode('koi8-r'))
        with smart_open.smart_open(key, encoding='koi8-r') as fin:
            actual = fin.read()
        self.assertEqual(text, actual)

    @mock_s3
    def test_write_encoding(self):
        """Should open the file for writing with the correct encoding."""
        conn = boto.connect_s3()
        conn.create_bucket('bucket')
        key = "s3://bucket/key.txt"
        text = u'какая боль, какая боль, аргентина - ямайка, 5-0'

        with smart_open.smart_open(key, 'w', encoding='koi8-r') as fout:
            fout.write(text)
        with smart_open.smart_open(key, encoding='koi8-r') as fin:
            actual = fin.read()
        self.assertEqual(text, actual)

    @mock_s3
    def test_write_bad_encoding_strict(self):
        """Should open the file for writing with the correct encoding."""
        conn = boto.connect_s3()
        conn.create_bucket('bucket')
        key = "s3://bucket/key.txt"
        text = u'欲しい気持ちが成長しすぎて'

        with self.assertRaises(UnicodeEncodeError):
            with smart_open.smart_open(key, 'w', encoding='koi8-r', errors='strict') as fout:
                fout.write(text)

    @mock_s3
    def test_write_bad_encoding_replace(self):
        """Should open the file for writing with the correct encoding."""
        conn = boto.connect_s3()
        conn.create_bucket('bucket')
        key = "s3://bucket/key.txt"
        text = u'欲しい気持ちが成長しすぎて'
        expected = u'?' * len(text)

        with smart_open.smart_open(key, 'w', encoding='koi8-r', errors='replace') as fout:
            fout.write(text)
        with smart_open.smart_open(key, encoding='koi8-r') as fin:
            actual = fin.read()
        self.assertEqual(expected, actual)

    @mock_s3
    def test_write_text_gzip(self):
        """Should open the file for writing with the correct encoding."""
        conn = boto.connect_s3()
        conn.create_bucket('bucket')
        key = "s3://bucket/key.txt.gz"
        text = u'какая боль, какая боль, аргентина - ямайка, 5-0'

        with smart_open.smart_open(key, 'w', encoding='utf-8') as fout:
            fout.write(text)
        with smart_open.smart_open(key, 'r', encoding='utf-8') as fin:
            actual = fin.read()
        self.assertEqual(text, actual)

    @mock_s3
    def test_write_text_gzip_key(self):
        """Should open the boto S3 key for writing with the correct encoding."""
        conn = boto.connect_s3()
        mybucket = conn.create_bucket('bucket')
        mykey = boto.s3.key.Key(mybucket, 'key.txt.gz')
        text = u'какая боль, какая боль, аргентина - ямайка, 5-0'

        with smart_open.smart_open(mykey, 'w', encoding='utf-8') as fout:
            fout.write(text)
        with smart_open.smart_open(mykey, 'r', encoding='utf-8') as fin:
            actual = fin.read()
        self.assertEqual(text, actual)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
