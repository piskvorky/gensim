Gensim Sphinx Gallery README
============================

This README.rst file describes the mechanisms behind our documentation examples.
The intended audience is Gensim developers and documentation authors trying to understand how this stuff works.

Overview
--------

We use `Sphinx Gallery <https://sphinx-gallery.github.io/index.html>`__.
The top-level ``README.txt`` describes the gallery.
Each subdirectory is a gallery of examples, each with its own ``README.txt`` file.
Each example is a Python script.

Sphinx Gallery reads these scripts and renders them as HTML and Jupyter
notebooks.  If the script is unchanged from the previous run, then Sphinx skips
rendering and running it.  This saves considerable time, as running all the
examples can take several hours.

Subdirectories
--------------

There are three important subdirectories for the gallery:

1. ``docs/src/gallery`` contains Python scripts
2. ``docs/src/auto_examples`` contains Jupyter notebooks and RST rendered from the Python scripts
3. ``docs/auto_examples`` contains HTML rendered from the Python scripts

We keep all 1) and 2) under version control in our git repository.
The rendering takes a fair bit of time (one or two hours to run everything) so it's worth keeping the result.
On the contrary, it doesn't take a lot of time to generate 3) from 2), so we don't keep 3) under version control.

.. Note::
    I'm not sure if there's some way we can merge 2) and 3) - that may make more
    sense than keeping them separate.

File naming
-----------

Each example file is a Python script.
We prefix each script with a ``run_`` prefix: this tells Gallery that it should run the file in order to render HTML and Jupyter notebooks.

If we remove that prefix from a file, then Sphinx will skip running it and just render it.
This is helpful when the example fails to run for some reason, and we want to temporarily skip running it.
It's best to avoid this unless absolutely necessary, because running the script ensures the example is still correct and valid.

Configuration
-------------

The Gallery relies on the general Sphinx configuration script in ``docs/src/conf.py``.
Within that script, there is a ``sphinx_gallery_conf`` dictionary that contains all the config options.
If you go tweaking those options, see the `Sphinx Gallery Documentation <https://sphinx-gallery.github.io/configuration.html>`__.

The order in which subgalleries and examples appear is an important part of the configuration.
For the subgalleries, we list them explicitly using the ``subsection_order`` parameter.
We do a similar thing for examples, except we need to write a little bit of code to make that happen.
See the ``sort_key`` function in ``conf.py`` for details.
