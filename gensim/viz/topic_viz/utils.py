"""
Utility routines for the pyLDAvis package
"""

import json
import os
import re
import shutil
import warnings
from functools import wraps
import numpy as np
from . import urls

# Make sure that DeprecationWarning gets printed
warnings.simplefilter("always", DeprecationWarning)


def html_id_ok(objid, html5=False):
    """Check whether objid is valid as an HTML id attribute.

    If html5 == True, then use the more liberal html5 rules.
    """
    if html5:
        return not re.search('\s', objid)
    else:
        return bool(re.match("^[a-zA-Z][a-zA-Z0-9\-\.\:\_]*$", objid))


def get_id(obj, suffix="", prefix="el", warn_on_invalid=True):
    """Get a unique id for the object"""
    if not suffix:
        suffix = ""
    if not prefix:
        prefix = ""

    objid = prefix + str(os.getpid()) + str(id(obj)) + suffix

    if warn_on_invalid and not html_id_ok(objid):
        warnings.warn('"{0}" is not a valid html ID. This may cause problems')

    return objid


def write_ipynb_local_js(location=None, d3_src=None, ldavis_src=None, ldavis_css=None):
    """
    Write the pyLDAvis and d3 javascript libraries to the given file location.

    This utility is used by the IPython notebook tools to enable easy use
    of pyLDAvis with no web connection.

    Parameters
    ----------
    location : string (optioal)
        the directory in which the d3 and pyLDAvis javascript libraries will be
        written. If not specified, the IPython nbextensions directory will be
        used. If IPython doesn't support nbextensions (< 2.0),
        the current working directory will be used.
    d3_src : string (optional)
        the source location of the d3 library. If not specified, the standard
        path in pyLDAvis.urls.D3_LOCAL will be used.
    ldavis_src : string (optional)
        the source location of the pyLDAvis library. If not specified, the
        standard path in pyLDAvis.urls.LDAVIS_LOCAL will be used.

    Returns
    -------
    d3_url, ldavis_url : string
        The URLs to be used for loading these js files.
    """
    if location is None:
        try:
            from IPython.html import install_nbextension
        except ImportError:
            location = os.getcwd()
            nbextension = False
        else:
            nbextension = True
    else:
        nbextension = False

    if d3_src is None:
        d3_src = urls.D3_LOCAL
    if ldavis_src is None:
        ldavis_src = urls.LDAVIS_LOCAL
    if ldavis_css is None:
        ldavis_css = urls.LDAVIS_CSS_LOCAL

    d3js = os.path.basename(d3_src)
    ldavisjs = os.path.basename(ldavis_src)
    ldaviscss = os.path.basename(ldavis_css)

    if not os.path.exists(d3_src):
        raise ValueError("d3 src not found at '{0}'".format(d3_src))
    if not os.path.exists(ldavis_src):
        raise ValueError("pyLDAvis src not found at '{0}'".format(ldavis_src))
    if not os.path.exists(ldavis_css):
        raise ValueError("pyLDAvis src not found at '{0}'".format(ldavis_css))

    if nbextension:
        # IPython 2.0+.
        # This will not work if a url prefix is added
        prefix = '/nbextensions/'

        try:
            [install_nbextension(ext) for ext in [d3_src, ldavis_src, ldavis_css]]
        except IOError:
            # files may be read only. We'll try deleting them and re-installing
            from IPython.utils.path import get_ipython_dir
            nbext = os.path.join(get_ipython_dir(), "nbextensions")

            for src in [d3_src, ldavis_src]:
                dest = os.path.join(nbext, os.path.basename(src))
                if os.path.exists(dest):
                    os.remove(dest)
            [install_nbextension(ext) for ext in [d3_src, ldavis_src, ldavis_css]]

    else:
        # IPython < 2.0 or explicit path.
        # This won't work if users have changed the kernel directory.
        prefix = '/files/'

        d3_dest = os.path.join(location, d3js)
        ldavis_dest = os.path.join(location, ldavisjs)
        ldavis_css_dest = os.path.join(location, ldaviscss)

        for src, dest in [(d3_src, d3_dest), (ldavis_src, ldavis_dest), (ldavis_css, ldavis_css_dest)]:
            try:
                shutil.copyfile(src, dest)
            except IOError:
                # file may be read only. We'll try deleting it first
                if os.path.exists(dest):
                    os.remove(dest)
                shutil.copyfile(src, dest)


    return prefix + d3js, prefix + ldavisjs, prefix + ldaviscss

class NumPyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float64) or isinstance(obj, np.float32):
            return float(obj)
        if np.iscomplexobj(obj):
            return np.real(obj)
        return json.JSONEncoder.default(self, obj)
