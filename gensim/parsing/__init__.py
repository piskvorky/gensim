"""
This package contains functions to preprocess raw text
"""

# bring model classes directly into package namespace, to save some typing
from .porter import PorterStemmer  # noqa:F401
from .preprocessing import (remove_stopwords, strip_punctuation, strip_punctuation2,  # noqa:F401
                            strip_tags, strip_short, strip_numeric,
                            strip_non_alphanum, strip_multiple_whitespaces,
                            split_alphanum, stem_text, preprocess_string,
                            preprocess_documents, read_file, read_files)
