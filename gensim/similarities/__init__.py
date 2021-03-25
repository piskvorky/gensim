"""
This package contains implementations of pairwise similarity queries.
"""

# bring classes directly into package namespace, to save some typing
import warnings
try:
    import Levenshtein  # noqa:F401
except ImportError:
    msg = (
        "The gensim.similarities.levenshtein submodule is disabled, because the optional "
        "Levenshtein package <https://pypi.org/project/python-Levenshtein/> is unavailable. "
        "Install Levenhstein (e.g. `pip install python-Levenshtein`) to suppress this warning."
    )
    warnings.warn(msg)
    LevenshteinSimilarityIndex = None
else:
    from .levenshtein import LevenshteinSimilarityIndex  # noqa:F401
from .docsim import (  # noqa:F401
    Similarity,
    MatrixSimilarity,
    SparseMatrixSimilarity,
    SoftCosineSimilarity,
    WmdSimilarity)
from .termsim import (  # noqa:F401
    TermSimilarityIndex,
    UniformTermSimilarityIndex,
    WordEmbeddingSimilarityIndex,
    SparseTermSimilarityMatrix)
