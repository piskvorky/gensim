# -*- coding: utf-8 -*-
"""
Topic Models (e.g. LDA) visualization using D3
=============================================

Functions: General Use
----------------------
:func:`prepare`
    transform and prepare a LDA model's data for visualization

:func:`prepared_data_to_html`
    convert prepared data to an html string

:func:`show`
    launch a web server to view the visualization

:func:`save_html`
    save a visualization to a standalone html file

:func:`save_json`
    save the visualization JSON data of to a file


Functions: IPython Notebook
---------------------------
:func:`display`
    display a figure in an IPython notebook

:func:`enable_notebook`
    enable automatic D3 display of prepared model data in the IPython notebook.

:func:`disable_notebook`
    disable automatic D3 display of prepared model data in the IPython notebook.
"""

__all__ = ["prepare", "js_PCoA",
           "PreparedData", "prepared_data_to_html",
           "display", "show", "save_html", "save_json",
           "enable_notebook", "disable_notebook"]

from .display import *
from .prepare import prepare, js_PCoA, PreparedData
