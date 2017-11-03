#!/usr/bin/env python
# encoding: utf-8


"""
Common utils for tests
"""
import tempfile
import os

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder


def datapath(fname):
    return os.path.join(module_path, 'test_data', fname)


def get_tmpfile(suffix):
    return os.path.join(tempfile.gettempdir(), suffix)
