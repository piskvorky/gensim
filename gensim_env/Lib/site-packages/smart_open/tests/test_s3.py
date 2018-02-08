# -*- coding: utf-8 -*-
import logging
import gzip
import io
import os
import uuid
import unittest

import boto3
import moto

import smart_open
import smart_open.s3


BUCKET_NAME = 'test-smartopen-{}'.format(uuid.uuid4().hex)  # generate random bucket (avoid race-condition in CI)
KEY_NAME = 'test-key'
WRITE_KEY_NAME = 'test-write-key'

logger = logging.getLogger(__name__)


def maybe_mock_s3(func):
    if os.environ.get('SO_DISABLE_MOCKS') == "1":
        return func
    else:
        return moto.mock_s3(func)


def cleanup_bucket(s3, delete_bucket=False):
    for bucket in s3.buckets.all():
        if bucket.name == BUCKET_NAME:
            for key in bucket.objects.all():
                key.delete()

            if delete_bucket:
                bucket.delete()
                return False
            return True
    return False


def create_bucket_and_key(bucket_name=BUCKET_NAME, key_name=KEY_NAME, contents=None):
    # fake (or not) connection, bucket and key
    logger.debug('%r', locals())
    s3 = boto3.resource('s3')
    bucket_exist = cleanup_bucket(s3)

    if not bucket_exist:
        mybucket = s3.create_bucket(Bucket=bucket_name)

    mybucket = s3.Bucket(bucket_name)
    mykey = s3.Object(bucket_name, key_name)
    if contents is not None:
        mykey.put(Body=contents)
    return mybucket, mykey


@maybe_mock_s3
class SeekableBufferedInputBaseTest(unittest.TestCase):
    def setUp(self):
        # lower the multipart upload size, to speed up these tests
        self.old_min_part_size = smart_open.s3.DEFAULT_MIN_PART_SIZE
        smart_open.s3.DEFAULT_MIN_PART_SIZE = 5 * 1024**2

    def tearDown(self):
        smart_open.s3.DEFAULT_MIN_PART_SIZE = self.old_min_part_size
        s3 = boto3.resource('s3')
        cleanup_bucket(s3, delete_bucket=True)

    def test_iter(self):
        """Are S3 files iterated over correctly?"""
        # a list of strings to test with
        expected = u"hello wořld\nhow are you?".encode('utf8')
        create_bucket_and_key(contents=expected)

        # connect to fake s3 and read from the fake key we filled above
        fin = smart_open.s3.SeekableBufferedInputBase(BUCKET_NAME, KEY_NAME)
        output = [line.rstrip(b'\n') for line in fin]
        self.assertEqual(output, expected.split(b'\n'))

    def test_iter_context_manager(self):
        # same thing but using a context manager
        expected = u"hello wořld\nhow are you?".encode('utf8')
        create_bucket_and_key(contents=expected)
        with smart_open.s3.SeekableBufferedInputBase(BUCKET_NAME, KEY_NAME) as fin:
            output = [line.rstrip(b'\n') for line in fin]
            self.assertEqual(output, expected.split(b'\n'))

    def test_read(self):
        """Are S3 files read correctly?"""
        content = u"hello wořld\nhow are you?".encode('utf8')
        create_bucket_and_key(contents=content)
        logger.debug('content: %r len: %r', content, len(content))

        fin = smart_open.s3.SeekableBufferedInputBase(BUCKET_NAME, KEY_NAME)
        self.assertEqual(content[:6], fin.read(6))
        self.assertEqual(content[6:14], fin.read(8))  # ř is 2 bytes
        self.assertEqual(content[14:], fin.read())  # read the rest

    def test_seek_beginning(self):
        """Does seeking to the beginning of S3 files work correctly?"""
        content = u"hello wořld\nhow are you?".encode('utf8')
        create_bucket_and_key(contents=content)

        fin = smart_open.s3.SeekableBufferedInputBase(BUCKET_NAME, KEY_NAME)
        self.assertEqual(content[:6], fin.read(6))
        self.assertEqual(content[6:14], fin.read(8))  # ř is 2 bytes

        fin.seek(0)
        self.assertEqual(content, fin.read())  # no size given => read whole file

        fin.seek(0)
        self.assertEqual(content, fin.read(-1))  # same thing

    def test_seek_start(self):
        """Does seeking from the start of S3 files work correctly?"""
        content = u"hello wořld\nhow are you?".encode('utf8')
        create_bucket_and_key(contents=content)

        fin = smart_open.s3.SeekableBufferedInputBase(BUCKET_NAME, KEY_NAME)
        seek = fin.seek(6)
        self.assertEqual(seek, 6)
        self.assertEqual(fin.tell(), 6)
        self.assertEqual(fin.read(6), u'wořld'.encode('utf-8'))

    def test_seek_current(self):
        """Does seeking from the middle of S3 files work correctly?"""
        content = u"hello wořld\nhow are you?".encode('utf8')
        create_bucket_and_key(contents=content)

        fin = smart_open.s3.SeekableBufferedInputBase(BUCKET_NAME, KEY_NAME)
        self.assertEqual(fin.read(5), b'hello')
        seek = fin.seek(1, whence=smart_open.s3.CURRENT)
        self.assertEqual(seek, 6)
        self.assertEqual(fin.read(6), u'wořld'.encode('utf-8'))

    def test_seek_end(self):
        """Does seeking from the end of S3 files work correctly?"""
        content = u"hello wořld\nhow are you?".encode('utf8')
        create_bucket_and_key(contents=content)

        fin = smart_open.s3.SeekableBufferedInputBase(BUCKET_NAME, KEY_NAME)
        seek = fin.seek(-4, whence=smart_open.s3.END)
        self.assertEqual(seek, len(content) - 4)
        self.assertEqual(fin.read(), b'you?')

    def test_detect_eof(self):
        content = u"hello wořld\nhow are you?".encode('utf8')
        create_bucket_and_key(contents=content)

        fin = smart_open.s3.SeekableBufferedInputBase(BUCKET_NAME, KEY_NAME)
        fin.read()
        eof = fin.tell()
        self.assertEqual(eof, len(content))
        fin.seek(0, whence=smart_open.s3.END)
        self.assertEqual(eof, fin.tell())

    def test_read_gzip(self):
        expected = u'раcцветали яблони и груши, поплыли туманы над рекой...'.encode('utf-8')
        buf = io.BytesIO()
        buf.close = lambda: None  # keep buffer open so that we can .getvalue()
        with gzip.GzipFile(fileobj=buf, mode='w') as zipfile:
            zipfile.write(expected)
        create_bucket_and_key(contents=buf.getvalue())

        #
        # Make sure we're reading things correctly.
        #
        with smart_open.s3.SeekableBufferedInputBase(BUCKET_NAME, KEY_NAME) as fin:
            self.assertEqual(fin.read(), buf.getvalue())

        #
        # Make sure the buffer we wrote is legitimate gzip.
        #
        sanity_buf = io.BytesIO(buf.getvalue())
        with gzip.GzipFile(fileobj=sanity_buf) as zipfile:
            self.assertEqual(zipfile.read(), expected)

        logger.debug('starting actual test')
        with smart_open.s3.SeekableBufferedInputBase(BUCKET_NAME, KEY_NAME) as fin:
            with gzip.GzipFile(fileobj=fin) as zipfile:
                actual = zipfile.read()

        self.assertEqual(expected, actual)

    def test_readline(self):
        content = b'englishman\nin\nnew\nyork\n'
        create_bucket_and_key(contents=content)

        with smart_open.s3.BufferedInputBase(BUCKET_NAME, KEY_NAME) as fin:
            actual = list(fin)

        expected = [b'englishman\n', b'in\n', b'new\n', b'york\n']
        self.assertEqual(expected, actual)

    def test_readline_tiny_buffer(self):
        content = b'englishman\nin\nnew\nyork\n'
        create_bucket_and_key(contents=content)

        with smart_open.s3.BufferedInputBase(BUCKET_NAME, KEY_NAME, buffer_size=8) as fin:
            actual = list(fin)

        expected = [b'englishman\n', b'in\n', b'new\n', b'york\n']
        self.assertEqual(expected, actual)


@maybe_mock_s3
class BufferedOutputBaseTest(unittest.TestCase):
    """
    Test writing into s3 files.

    """
    def tearDown(self):
        s3 = boto3.resource('s3')
        cleanup_bucket(s3, delete_bucket=True)

    def test_write_01(self):
        """Does writing into s3 work correctly?"""
        create_bucket_and_key()
        test_string = u"žluťoučký koníček".encode('utf8')

        # write into key
        with smart_open.s3.BufferedOutputBase(BUCKET_NAME, WRITE_KEY_NAME) as fout:
            fout.write(test_string)

        # read key and test content
        output = list(smart_open.smart_open("s3://{}/{}".format(BUCKET_NAME, WRITE_KEY_NAME), "rb"))

        self.assertEqual(output, [test_string])

    def test_write_01a(self):
        """Does s3 write fail on incorrect input?"""
        create_bucket_and_key()

        try:
            with smart_open.s3.BufferedOutputBase(BUCKET_NAME, WRITE_KEY_NAME) as fin:
                fin.write(None)
        except TypeError:
            pass
        else:
            self.fail()

    def test_write_02(self):
        """Does s3 write unicode-utf8 conversion work?"""
        create_bucket_and_key()

        smart_open_write = smart_open.s3.BufferedOutputBase(BUCKET_NAME, WRITE_KEY_NAME)
        smart_open_write.tell()
        logger.info("smart_open_write: %r", smart_open_write)
        with smart_open_write as fout:
            fout.write(u"testžížáč".encode("utf-8"))
            self.assertEqual(fout.tell(), 14)

    def test_write_03(self):
        """Does s3 multipart chunking work correctly?"""
        create_bucket_and_key()

        # write
        smart_open_write = smart_open.s3.BufferedOutputBase(
            BUCKET_NAME, WRITE_KEY_NAME, min_part_size=10
        )
        with smart_open_write as fout:
            fout.write(b"test")
            self.assertEqual(fout._buf.tell(), 4)

            fout.write(b"test\n")
            self.assertEqual(fout._buf.tell(), 9)
            self.assertEqual(fout._total_parts, 0)

            fout.write(b"test")
            self.assertEqual(fout._buf.tell(), 0)
            self.assertEqual(fout._total_parts, 1)

        # read back the same key and check its content
        output = list(smart_open.smart_open("s3://{}/{}".format(BUCKET_NAME, WRITE_KEY_NAME)))
        self.assertEqual(output, [b"testtest\n", b"test"])

    def test_write_04(self):
        """Does writing no data cause key with an empty value to be created?"""
        _ = create_bucket_and_key()

        smart_open_write = smart_open.s3.BufferedOutputBase(BUCKET_NAME, WRITE_KEY_NAME)
        with smart_open_write as fout:  # noqa
            pass

        # read back the same key and check its content
        output = list(smart_open.smart_open("s3://{}/{}".format(BUCKET_NAME, WRITE_KEY_NAME)))

        self.assertEqual(output, [])

    def test_gzip(self):
        create_bucket_and_key()

        expected = u'а не спеть ли мне песню... о любви'.encode('utf-8')
        with smart_open.s3.BufferedOutputBase(BUCKET_NAME, WRITE_KEY_NAME) as fout:
            with gzip.GzipFile(fileobj=fout, mode='w') as zipfile:
                zipfile.write(expected)

        with smart_open.s3.SeekableBufferedInputBase(BUCKET_NAME, WRITE_KEY_NAME) as fin:
            with gzip.GzipFile(fileobj=fin) as zipfile:
                actual = zipfile.read()

        self.assertEqual(expected, actual)

    def test_text_iterator(self):
        expected = u"выйду ночью в поле с конём".split(u' ')
        create_bucket_and_key(contents="\n".join(expected).encode('utf-8'))
        with smart_open.s3.open(BUCKET_NAME, KEY_NAME, 'r') as fin:
            actual = [line.rstrip() for line in fin]
        self.assertEqual(expected, actual)

    def test_binary_iterator(self):
        expected = u"выйду ночью в поле с конём".encode('utf-8').split(b' ')
        create_bucket_and_key(contents=b"\n".join(expected))
        with smart_open.s3.open(BUCKET_NAME, KEY_NAME, 'rb') as fin:
            actual = [line.rstrip() for line in fin]
        self.assertEqual(expected, actual)

    def test_nonexisting_bucket(self):
        expected = u"выйду ночью в поле с конём".encode('utf-8')
        with self.assertRaises(ValueError):
            with smart_open.s3.open('thisbucketdoesntexist', 'mykey', 'wb') as fout:
                fout.write(expected)


class ClampTest(unittest.TestCase):
    def test(self):
        self.assertEqual(smart_open.s3._clamp(5, 0, 10), 5)
        self.assertEqual(smart_open.s3._clamp(11, 0, 10), 10)
        self.assertEqual(smart_open.s3._clamp(-1, 0, 10), 0)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    unittest.main()
