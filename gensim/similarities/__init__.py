"""
This package contains implementations of pairwise similarity queries.
"""

from .docsim import (
    MatrixSimilarity,
    Similarity,  # noqa:F401
    SoftCosineSimilarity,
    SparseMatrixSimilarity,
    WmdSimilarity,
)

# bring classes directly into package namespace, to save some typing
from .levenshtein import LevenshteinSimilarityIndex  # noqa:F401
from .termsim import (
    SparseTermSimilarityMatrix,  # noqa:F401
    TermSimilarityIndex,
    UniformTermSimilarityIndex,
    WordEmbeddingSimilarityIndex,
)
