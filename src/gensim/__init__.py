"""
This package contains interfaces and functionality to compute pair-wise document 
similarities within a corpus of documents. 
"""

# for IPython tab-completion
import utils, matutils, interfaces, corpora, models, similarities
import logging

# TODO: if version_info < 2.7?
class NullHandler(logging.Handler):
    """For python versions <= 2.6; same as `logging.NullHandler` in 2.7."""
    def emit(self, record):
        pass
 
logger = logging.getLogger('gensim')
logger.setLevel(logging.INFO)
if len(logger.handlers) == 0:	# To ensure reload() doesn't add another one
    logger.addHandler(NullHandler())
