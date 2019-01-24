import gensim.downloader as api
import itertools as it
from gensim.models.utils_any2vec import compute_ngrams_bytes, ft_hash_bytes

words = tuple(it.chain.from_iterable(api.load("text8")))
assert len(words) == 17005207  # long enough

words = words[:100000]


def benchmark(words=words, ngram_func=compute_ngrams_bytes, hash_func=ft_hash_bytes):
    for w in words:
        arr = [hash_func(ngram) % 10000 for ngram in ngram_func(w, 3, 6)]
