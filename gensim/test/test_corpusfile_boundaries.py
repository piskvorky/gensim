"""Native corpus-file batches and worker ranges must conserve training examples."""
import pickle

import numpy as np
import pytest

from gensim.models import FastText, Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec_corpusfile import CythonLineSentence


CASES = [
    b'alpha beta\ngamma delta\n',
    b'alpha beta\ngamma delta',
    b'\n \t\nalpha beta\n\ngamma delta\n \t',
    'café 橙子\n東京 mañana\r\n\r\n'.encode('utf8'),
    b'alpha beta\r\ngamma delta\r\n',
    b'alpha beta',
    b'alpha beta ' * 3000 + b'\n' + b'gamma delta ' * 3000 + b'\ntail context\n',
    b'alpha beta ' * 5000 + b'\nsuffix marker\n',
    b'alpha beta ' * 11500 + b'last\n',
]


def chunks(data, limit=10000):
    """Independent byte/line reference matching LineSentence's chunk semantics."""
    return [
        words[start:start + limit]
        for line in data.split(b'\n')
        for words in [line.split()]
        for start in range(0, len(words), limit)
    ]


def consume(reader):
    result = []
    while not reader.is_eof():
        batch = reader.next_batch()
        assert sum(map(len, batch)) <= reader.max_words_in_batch
        assert all(batch)
        result.extend(batch)
    return result


@pytest.mark.parametrize('contents', CASES[:6] + [b'', b'\n \t\n', b'a b c d e f g h i j k l'])
@pytest.mark.parametrize('limit', [1, 5, 7])
def test_reader_conserves_order_and_reset(tmp_path, contents, limit):
    path = tmp_path / 'corpus.txt'
    path.write_bytes(contents)
    expected = chunks(contents, limit)
    reader = CythonLineSentence(str(path), max_sentence_length=limit, end_offset=len(contents))
    assert consume(reader) == expected
    reader.reset()
    reader.next_batch()
    reader.reset()
    assert consume(reader) == expected
    assert list(reader) == expected


@pytest.mark.parametrize('contents', CASES[:6] + [b'', b'\n \t\n'])
@pytest.mark.parametrize('workers', [1, 2, 3, 7, 17])
def test_disjoint_byte_partitions(tmp_path, contents, workers):
    path = tmp_path / 'corpus.txt'
    path.write_bytes(contents)
    starts, ends = Word2Vec._get_corpusfile_offsets(str(path), workers)
    boundaries = [0] + [i + 1 for i, byte in enumerate(contents) if byte == 10] + [len(contents)]
    expected = [min(pos for pos in boundaries if pos >= len(contents) * worker // workers)
                for worker in range(workers)]
    assert starts == expected
    assert ends == starts[1:] + [len(contents)]
    actual = []
    for start, end in zip(starts, ends):
        reader = CythonLineSentence(str(path), offset=start, end_offset=end, max_sentence_length=5)
        expected_partition = chunks(contents[start:end], 5)
        reader.next_batch()
        assert consume(pickle.loads(pickle.dumps(reader))) == expected_partition
        reader.reset()
        actual.extend(consume(reader))
    assert actual == chunks(contents, 5)


class RecordWorkers:
    """Observe the existing native return contract without replacing any training."""
    def _do_train_epoch(self, *args, **kwargs):
        result = super()._do_train_epoch(*args, **kwargs)
        self.observed.append((args[1], args[5], args[2], kwargs.get('end_offsets'), result))
        return result


class RecordingWord2Vec(RecordWorkers, Word2Vec):
    pass


class RecordingFastText(RecordWorkers, FastText):
    pass


def new_model(model_class, sg, workers):
    return model_class(
        vector_size=5, window=2, min_count=1, sample=0, workers=workers,
        hs=1, negative=0, sg=sg, shrink_windows=False, seed=17,
        alpha=0.001, min_alpha=0.001,
        **({'bucket': 128} if issubclass(model_class, FastText) else {}),
    )


@pytest.mark.parametrize('contents', CASES)
@pytest.mark.parametrize('workers', [1, 2, 3, 7])
@pytest.mark.parametrize('sg', [0, 1])
@pytest.mark.parametrize('model_class', [RecordingWord2Vec, RecordingFastText])
def test_public_training_exact_worker_counts(tmp_path, contents, workers, sg, model_class):
    path = tmp_path / 'corpus.txt'
    path.write_bytes(contents)
    model = new_model(model_class, sg, workers)
    model.observed = []
    model.build_vocab(corpus_file=str(path))
    actual = model.train(
        corpus_file=str(path), total_examples=model.corpus_count,
        total_words=model.corpus_total_words, epochs=2,
    )
    expected_words = sum(map(len, chunks(contents)))
    assert actual == (2 * expected_words, 2 * expected_words)
    assert len(model.observed) == 2 * workers
    for thread, epoch, start, ends, (examples, effective, raw) in model.observed:
        expected = chunks(contents[start:ends[thread]])
        assert (examples, effective, raw) == (len(expected), sum(map(len, expected)), sum(map(len, expected)))
    assert sum(row[4][0] for row in model.observed) == 2 * len(chunks(contents))


@pytest.mark.parametrize('contents', CASES)
@pytest.mark.parametrize('sg', [0, 1])
@pytest.mark.parametrize('model_class', [Word2Vec, FastText])
def test_single_worker_weights_match_iterable(tmp_path, contents, sg, model_class):
    path = tmp_path / 'corpus.txt'
    path.write_bytes(contents)
    models = []
    for file_mode in [True, False]:
        model = new_model(model_class, sg, 1)
        model.build_vocab(corpus_file=str(path))
        data = {'corpus_file': str(path)} if file_mode else {'corpus_iterable': LineSentence(str(path))}
        model.train(**data, total_examples=model.corpus_count, total_words=model.corpus_total_words, epochs=2)
        models.append(model)
    actual, expected = models
    np.testing.assert_array_equal(actual.wv.vectors, expected.wv.vectors)
    np.testing.assert_array_equal(actual.syn1, expected.syn1)
    if model_class is FastText:
        np.testing.assert_array_equal(actual.wv.vectors_vocab, expected.wv.vectors_vocab)
        np.testing.assert_array_equal(actual.wv.vectors_ngrams, expected.wv.vectors_ngrams)


@pytest.mark.parametrize('buffered', [False, True])
def test_reader_rejects_nonprogressing_batch_budget(tmp_path, buffered):
    path = tmp_path / 'corpus.txt'
    path.write_bytes(b'one two three four\nfive six seven eight')
    reader = CythonLineSentence(str(path), max_sentence_length=5)
    if buffered:
        reader.next_batch()
    reader.max_words_in_batch = 3
    with pytest.raises(ValueError, match='too small'):
        reader.next_batch()
    reader = CythonLineSentence(str(path), max_sentence_length=5)
    reader.max_words_in_batch = 0
    with pytest.raises(ValueError, match='positive'):
        reader.next_batch()


@pytest.mark.parametrize('model_class', [Word2Vec, FastText])
@pytest.mark.parametrize('workers', [1, 3])
@pytest.mark.parametrize('rewrite_event', ['on_train_begin', 'on_epoch_begin'])
def test_callback_rewrite_refreshes_boundaries(tmp_path, model_class, workers, rewrite_event):
    from gensim.models.callbacks import CallbackAny2Vec

    path = tmp_path / 'corpus.txt'
    path.write_text('a a\nb b\n')

    class RewriteCorpus(CallbackAny2Vec):
        epoch = 0

        def rewrite(self):
            text = 'longtoken longtoken\nb b\n' if self.epoch % 2 == 0 else 'a a\nb b\n'
            path.write_text(text)
            self.epoch += 1

        def on_train_begin(self, model):
            if rewrite_event == 'on_train_begin':
                self.rewrite()

        def on_epoch_begin(self, model):
            if rewrite_event == 'on_epoch_begin':
                self.rewrite()

    model = new_model(model_class, 1, workers)
    model.build_vocab([['a', 'b', 'longtoken']])
    assert model.train(
        corpus_file=str(path), total_words=4, total_examples=2, epochs=2, callbacks=[RewriteCorpus()],
    ) == (8, 8)
