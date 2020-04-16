"""This package contains interfaces and functionality to compute pair-wise document similarities within a corpus
of documents.
"""

import logging

__version__ = '3.8.1'

logger = logging.getLogger('gensim')
if len(logger.handlers) == 0:  # To ensure reload() doesn't add another one
    logger.addHandler(logging.NullHandler())

#
# Older versions of smart_open lack a NullHandler, meaning they can
# inadvertedly call logging.basicConfig upon import.  This has been fixed in
# newer versions.  The below NullHandler is necessary because gensim is pinned
# to an older smart_open version for Python 2.7 users.
#
logging.getLogger('smart_open').addHandler(logging.NullHandler())

#
# Perform the imports after logging handlers are set up, to prevent
#
#   - https://github.com/RaRe-Technologies/smart_open/issues/474
#   - https://github.com/RaRe-Technologies/smart_open/issues/475
#
from gensim import parsing, corpora, matutils, interfaces, models, similarities, summarization, utils  # noqa:F401
