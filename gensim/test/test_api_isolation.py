"""Check downloader test isolation without downloading any data."""
import os
import tempfile
import unittest
from unittest import mock

import gensim.downloader as api
from gensim.test import test_api


class TestApiIsolation(unittest.TestCase):
    def test_cache_isolation_and_cleanup(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                cache = os.path.join(directory, 'existing-cache')
                os.mkdir(cache)
                sentinel = os.path.join(cache, 'existing-model')
                with open(sentinel, 'w') as fout:
                    fout.write('preserve this cache')
                paths = []

                class OfflineTestApi(test_api.TestApi):
                    __unittest_skip__ = False

                    def runTest(self):
                        paths.append((api.BASE_DIR, api._PARENT_DIR))
                        self.assertEqual(api.base_dir, api.BASE_DIR)
                        self.assertEqual(api._PARENT_DIR, os.path.dirname(api.BASE_DIR))
                        if fail:
                            api._create_base_dir()
                            raise RuntimeError('exercise cleanup after failure')
                        self.test_base_dir_creation()

                with mock.patch.multiple(api, BASE_DIR=cache, base_dir=cache, _PARENT_DIR=directory):
                    result = unittest.TestResult()
                    OfflineTestApi().run(result)
                    self.assertEqual(len(result.errors), int(fail))
                    self.assertEqual(result.failures, [])
                    self.assertEqual((api.BASE_DIR, api.base_dir, api._PARENT_DIR), (cache, cache, directory))
                    with open(sentinel) as fin:
                        self.assertEqual(fin.read(), 'preserve this cache')
                    self.assertEqual(len(paths), 1)
                    self.assertFalse(os.path.exists(paths[0][1]))
