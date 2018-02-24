"""
URLs and filepaths for the vis javascript libraries
"""

import os
from . import __path__
import warnings

__all__ = ["D3_URL", "D3_LOCAL", "LDAVIS_LOCAL", "LDAVIS_CSS_LOCAL"]

D3_URL = "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"

LOCAL_JS_DIR = os.path.join(__path__[0], "js")
D3_LOCAL = os.path.join(LOCAL_JS_DIR, "d3.v3.min.js")

LDAVIS_LOCAL = os.path.join(LOCAL_JS_DIR, "ldavis.js")

LDAVIS_CSS_LOCAL = os.path.join(LOCAL_JS_DIR, "ldavis.css")