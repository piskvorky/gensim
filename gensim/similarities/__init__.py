"""
This package contains implementations of pairwise similarity queries.
"""

# bring classes directly into package namespace, to save some typing
from .docsim import Similarity, MatrixSimilarity, SparseMatrixSimilarity, SoftCosineSimilarity, WmdSimilarity  # noqa:F401
from .termsim import TermSimilarityIndex, UniformTermSimilarityIndex, SparseTermSimilarityMatrix  # noqa:F401
from .levenshtein import LevenshteinSimilarityIndex  # noqa:F401

from . import levenshtein  # noqa:F401
