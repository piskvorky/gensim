"""Path protocol support for native model serialization."""

import io
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from gensim import utils
from gensim.models import Word2Vec


class CustomPath(os.PathLike):
    def __init__(self, path):
        self.path = path

    def __fspath__(self):
        return str(self.path)


class BytesPath(CustomPath):
    def __fspath__(self):
        return os.fsencode(self.path)


@pytest.mark.parametrize('suffix', ['', '.gz', '.bz2'])
@pytest.mark.parametrize('path_type', [Path, CustomPath, BytesPath])
def test_saveload_pathlike_round_trip(tmp_path, suffix, path_type):
    model = Word2Vec(
        [['cat', 'purrs'], ['dog', 'barks'], ['cat', 'dog']],
        vector_size=4, min_count=1, workers=1, epochs=1, sample=0,
    )
    path = path_type(tmp_path / ('词向量.model' + suffix))
    model.save(path, sep_limit=0)
    restored = Word2Vec.load(path)
    assert restored.wv.index_to_key == model.wv.index_to_key
    np.testing.assert_array_equal(restored.wv.vectors, model.wv.vectors)
    np.testing.assert_array_equal(restored.syn1neg, model.syn1neg)
    trained_words, _ = restored.train([['cat', 'dog']], total_examples=1, epochs=1)
    assert trained_words == 2


def test_saveload_pathlike_load_existing_string_path(tmp_path):
    model = utils.SaveLoad()
    model.array = np.arange(12).reshape(3, 4)
    path = tmp_path / 'vectors.model'
    model.save(str(path), separately=['array'])
    restored = utils.SaveLoad.load(path, mmap='r')
    assert isinstance(restored.array, np.memmap)
    np.testing.assert_array_equal(restored.array, model.array)


def test_saveload_file_handle_is_unchanged():
    model = utils.SaveLoad()
    model.array = np.arange(4)
    handle = io.BytesIO()
    model.save(handle)
    assert not handle.closed
    handle.seek(0)
    restored = pickle.load(handle)
    np.testing.assert_array_equal(restored.array, model.array)
