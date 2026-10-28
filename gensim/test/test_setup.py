#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html

"""
Automated tests for the build configuration in setup.py.
"""

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

SETUP_PATH = Path(__file__).resolve().parents[2] / 'setup.py'


def load_setup_module():
    spec = importlib.util.spec_from_file_location('gensim_setup', SETUP_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch('setuptools.setup'):
        spec.loader.exec_module(module)
    return module


class TestNeedCython(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.setup = load_setup_module()

    def test_cython_available(self):
        with mock.patch.object(self.setup, 'find_spec', return_value=object()):
            self.assertTrue(self.setup.need_cython())

    def test_cython_available_with_prebuilt_sources(self):
        with mock.patch.object(self.setup, 'find_spec', return_value=object()), \
                mock.patch('os.path.isfile', return_value=True):
            self.assertTrue(self.setup.need_cython())

    def test_cython_missing_with_prebuilt_sources(self):
        with mock.patch.object(self.setup, 'find_spec', return_value=None), \
                mock.patch('os.path.isfile', return_value=True):
            self.assertFalse(self.setup.need_cython())

    def test_cython_missing_without_prebuilt_sources(self):
        with mock.patch.object(self.setup, 'find_spec', return_value=None), \
                mock.patch('os.path.isfile', return_value=False):
            self.assertTrue(self.setup.need_cython())


if __name__ == '__main__':
    unittest.main()
