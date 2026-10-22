"""Check the initialization of dynamic topic models against standalone LDA."""

import numpy as np
import pytest

from gensim.models import LdaModel, LdaSeqModel
from gensim.test.utils import common_corpus, common_dictionary


@pytest.mark.parametrize('chunksize, repeats', [(2, 1), (4, 1), (2000, 1), (None, 15)])
def test_ldaseq_initialization_uses_chunksize(chunksize, repeats):
    corpus = common_corpus * repeats
    expected_chunksize = 100 if chunksize is None else chunksize
    initial = LdaModel(
        corpus, id2word=common_dictionary, num_topics=2,
        chunksize=expected_chunksize, passes=2, alpha=0.01,
        random_state=42, dtype=np.float64,
    )
    kwargs = {} if chunksize is None else {'chunksize': chunksize}
    model = LdaSeqModel(
        corpus, id2word=common_dictionary, num_topics=2,
        time_slice=[len(corpus) // 2, len(corpus) - len(corpus) // 2],
        passes=2, random_state=42, lda_inference_max_iter=5,
        em_min_iter=1, em_max_iter=0, **kwargs,
    )

    np.testing.assert_allclose(model.sstats, initial.state.sstats.T)
    assert np.isfinite(model.gammas).all()
    for doc_index in range(len(corpus)):
        topic_probs = model.doc_topics(doc_index)
        assert np.isfinite(topic_probs).all()
        np.testing.assert_allclose(topic_probs.sum(), 1.0)
