"""
This package contains implementations of pairwise similarity queries.
"""

# bring classes directly into package namespace, to save some typing

from .docsim import (  # noqa:F401
    Similarity,
    MatrixSimilarity,
    SparseMatrixSimilarity,
    SoftCosineSimilarity,
    WmdSimilarity)
from .termsim import (  # noqa:F401
    TermSimilarityIndex,
    UniformTermSimilarityIndex,
    SparseTermSimilarityMatrix)
from .levenshtein import LevenshteinSimilarityIndex  # noqa:F401
