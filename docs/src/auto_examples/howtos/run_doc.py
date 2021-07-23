r"""
How to Author Gensim Documentation
==================================

How to author documentation for Gensim.
"""

###############################################################################
# Background
# ----------
#
# Gensim is a large project with a wide range of functionality.
# Unfortunately, not all of this functionality is documented **well**, and some of it is not documented at all.
# Without good documentation, users are unable to unlock Gensim's full potential.
# Therefore, authoring new documentation and improving existing documentation is of great value to the Gensim project.
#
# If you implement new functionality in Gensim, please include **helpful** documentation.
# By "helpful", we mean that your documentation answers questions that Gensim users may have.
# For example:
#
# - What is this new functionality?
# - **Why** is it important?
# - **How** is it relevant to Gensim?
# - **What** can I do with it? What are some real-world applications?
# - **How** do I use it to achieve those things?
# - ... and others (if you can think of them, please add them here)
#
# Before you author documentation, I suggest reading
# `"What nobody tells you about documentation" <https://www.divio.com/blog/documentation/>`__
# or watching its `accompanying video <https://www.youtube.com/watch?v=t4vKPhjcMZg>`__
# (or even both, if you're really keen).
#
# The summary of the above presentation is: there are four distinct kinds of documentation, and you really need them all:
#
# 1. Tutorials
# 2. Howto guides
# 3. Explanations
# 4. References
#
# Each kind has its own intended audience, purpose, and writing style.
# When you make a PR with new functionality, please consider authoring each kind of documentation.
# At the very least, you will (indirectly) author reference documentation through module, class and function docstrings.
#
# Mechanisms
# ----------
#
# We keep our documentation as individual Python scripts.
# These scripts live under :file:`docs/src/gallery` in one of several subdirectories:
#
# - core: core tutorials.  We try to keep this part small, avoid putting stuff here.
# - tutorials: tutorials.
# - howtos: howto guides.
#
# Pick a subdirectory and save your script under it.
# Prefix the name of the script with ``run_``: this way, the the documentation builder will run your script each time it builds our docs.
#
# The contents of the script are straightforward.
# At the very top, you need a docstring describing what your script does.

r"""
Title
=====

Brief description.
"""

###############################################################################
# The title is what will show up in the gallery.
# Keep this short and descriptive.
#
# The description will appear as a tooltip in the gallery.
# When people mouse-over the title, they will see the description.
# Keep this short too.
#

###############################################################################
# The rest of the script is Python, formatted in a special way so that Sphinx Gallery can parse it.
# The most important properties of this format are:
#
# - Sphinx Gallery will split your script into blocks
# - A block can be Python source or RST-formatted comments
# - To indicate that a block is in RST, prefix it with a line of 80 hash (#) characters.
# - All other blocks will be interpreted as Python source
#
# Read `this link <https://sphinx-gallery.github.io/syntax.html>`__ for more details.
# If you need further examples, check out other ``gensim`` tutorials and guides.
# All of them (including this one!) have a download link at the bottom of the page, which exposes the Python source they were generated from.
#
# You should be able to run your script directly from the command line::
#
#   python myscript.py
#
# and it should run to completion without error, occasionally printing stuff to standard output.
#

###############################################################################
# Authoring Workflow
# ------------------
#
# There are several ways to author documentation.
# The simplest and most straightforward is to author your ``script.py`` from scratch.
# You'll have the following cycle:
#
# 1. Make changes
# 2. Run ``python script.py``
# 3. Check standard output, standard error and return code
# 4. If everything works well, stop.
# 5. Otherwise, go back to step 1).
#
# If the above is not your cup of tea, you can also author your documentation as a Jupyter notebook.
# This is a more flexible approach that enables you to tweak parts of the documentation and re-run them as necessary.
#
# Once you're happy with the notebook, convert it to a script.py.
# There's a helpful `script <https://github.com/RaRe-Technologies/gensim/blob/develop/docs/src/tools/to_python.py>`__ that will do it for you.
# To use it::
#
#     python to_python.py < notebook.ipynb > script.py
#
# You may have to touch up the resulting ``script.py``.
# More specifically:
#
# - Update the title
# - Update the description
# - Fix any issues that the markdown-to-RST converter could not deal with
#
# Once your script.py works, put it in a suitable subdirectory.
# Please don't include your original Jupyter notebook in the repository - we won't be using it.

###############################################################################
# Correctness
# -----------
#
# Incorrect documentation can be worse than no documentation at all.
# Take the following steps to ensure correctness:
#
# - Run Python's doctest module on your docstrings
# - Run your documentation scripts from scratch, removing any temporary files/results
#
# Using data in your documentation
# --------------------------------
#
# Some parts of the documentation require real-world data to be useful.
# For example, you may need more than just a toy example to demonstrate the benefits of one model over another.
# This subsection provides some tips for including data in your documentation.
#
# If possible, use data available via Gensim's
# `downloader API <https://radimrehurek.com/gensim/gensim_numfocus/auto_examples/010_tutorials/run_downloader_api.html>`__.
# This will reduce the risk of your documentation becoming obsolete because required data is no longer available.
#
# Use the smallest possible dataset: avoid making people unnecessarily load large datasets and models.
# This will make your documentation faster to run and easier for people to use (they can modify your examples and re-run them quickly).
#
# Finalizing your contribution
# ----------------------------
#
# First, get Sphinx Gallery to build your documentation::
#
#     make --directory docs/src html
#
# This can take a while if your documentation uses a large dataset, or if you've changed many other tutorials or guides.
# Once this completes successfully, open ``docs/auto_examples/index.html`` in your browser.
# You should see your new tutorial or guide in the gallery.
#
# Once your documentation script is working correctly, it's time to add it to the git repository::
#
#     git add docs/src/gallery/tutorials/run_example.py
#     git add docs/src/auto_examples/tutorials/run_example.{py,py.md5,rst,ipynb}
#     git add docs/src/auto_examples/howtos/sg_execution_times.rst
#     git commit -m "enter a helpful commit message here"
#     git push origin branchname
#
# .. Note::
#   You may be wondering what all those other files are.
#   Sphinx Gallery puts a copy of your Python script in ``auto_examples/tutorials``.
#   The .md5 contains MD5 hash of the script to enable easy detection of modifications.
#   Gallery also generates .rst (RST for Sphinx) and .ipynb (Jupyter notebook) files from the script.
#   Finally, ``sg_execution_times.rst`` contains the time taken to run each example.
#
# Finally, open a PR at `github <https://github.com/RaRe-Technologies/gensim>`__.
# One of our friendly maintainers will review it, make suggestions, and eventually merge it.
# Your documentation will then appear in the `gallery <https://radimrehurek.com/gensim/auto_examples/index.html>`__,
# alongside the rest of the examples. Thanks a lot!
