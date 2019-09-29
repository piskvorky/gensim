r"""

.. _binder_howto:

How to Play with the Examples via Binder
========================================

If you don't have Jupyter installed locally, you can still play with the examples provided in the ``gensim`` documentation using Binder.
"""

import os.path

###############################################################################
# `Binder <https://mybinder.org>`__ is a free platform that enables you to run
# publicly available Jupyter notebooks, including all the ``gensim`` examples.
#
# To start, point your browser `here <https://mybinder.org/v2/gh/mpenkov/gensim/numfocus?filepath=docs/src/auto_examples/>`__.
# This will instruct Binder to set up a virtual environment to use with the examples.
# Binder will automatically pick up the required dependencies via the :file:`requirements.txt` at the top level of our git repository.
#
# .. Important::
#   It takes a few minutes to set up the virtual environment.
#
# Once the environment is ready, you will see a collection of Jupyter notebooks.
#
# - 000_core: core tutorials
# - 010_tutorials: tutorials
# - 020_howtos: how-to guides

def extract_title(path):
    with open(path) as fin:
        lines = [l for l in fin]
    for i, l in enumerate(lines):
        if l.startswith('=') and l.endswith('=\n'):
            return lines[i - 1].strip()

for subdir in ('000_core', '010_tutorials', '020_howtos'):
    for script in sorted(os.listdir(os.path.join('..', subdir))):
        if not script.endswith('.py'):
            continue
        path = os.path.join('..', subdir, script)
        title = extract_title(path)
        print(path)
        print('\t' + title)
    print()

###############################################################################
# Find the notebook you're interested in and click on it.
# A new browser tab will open.
# You can now play with the example the same way you would with a local Jupyter notebook, e.g. edit code, run everything via Kernel/Restart & Run All, etc.
#
# .. Important::
#   Avoid setting up more than virtual environment.
#   You can reuse the same environment for multiple examples.
#
# .. Important::
#   The virtual hardware provided by Binder may not be sufficient to run **all** the examples.
#   If a particular example requires gigabytes of memory or many CPU cycles, consider running it locally.
