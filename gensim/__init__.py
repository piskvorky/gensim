"""
This package contains interfaces and functionality to compute pair-wise document
similarities within a corpus of documents.
"""

import logging

try:
    __version__ = __import__('pkg_resources').get_distribution('gensim').version
except:
    __version__ = '?'

class NullHandler(logging.Handler):
    """For python versions <= 2.6; same as `logging.NullHandler` in 2.7."""
    def emit(self, record):
        pass
logger = logging.getLogger('gensim')
if len(logger.handlers) == 0:	# To ensure reload() doesn't add another one
    logger.addHandler(NullHandler())

# try to initialize pyximport before importing models otherwise cython code in word2vec and doc2doc does not compile
try:
    import pyximport
    def get_includes():
        """
            get folder paths that contain .h files
        """
        import os.path
        # voidptr.h under models
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        return [
            model_path,
        ]
    
    from numpy import get_include as np_get_include
    include_dirs = [np_get_include()] + get_includes()
    pyximport.install(setup_args={"include_dirs":include_dirs}, reload_support=True)
except ImportError as err:
    logger.warning("Fail to initialize pyximport: {0}".format(err))

from gensim import parsing, matutils, interfaces, models, corpora, similarities, summarization
