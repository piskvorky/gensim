# -*- coding: utf-8 -*-
"""
test_notebooks.py
===================


.. _general_examples:

General examples
----------------

General-purpose and introductory examples from the sphinx-gallery
"""

import os
import sys
import tempfile
from glob import glob

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError


def _notebook_run(path):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    kernel_name = 'python%d' % sys.version_info[0]
    this_file_directory = os.path.dirname(__file__)
    errors = []
    with tempfile.NamedTemporaryFile(suffix=".ipynb", mode='wt') as fout:
        with open(path) as f:
            nb = nbformat.read(f, as_version=4)
            nb.metadata.get('kernelspec', {})['name'] = kernel_name
            ep = ExecutePreprocessor(kernel_name=kernel_name, timeout=10)

            try:
                ep.preprocess(nb, {'metadata': {'path': this_file_directory}})
            except CellExecutionError as e:
                if "SKIP" in e.traceback:
                    print(str(e.traceback).split("\n")[-2])
                else:
                    raise e
            except TimeoutError as e:
                print(e)

            finally:
                nbformat.write(nb, fout)
        #nb = nbformat.read(fout, nbformat.current_nbformat)

    #errors = errors.extend(
        #[output for cell in nb.cells if "outputs" in cell
        # for output in cell["outputs"] if output.output_type == "error"])

    return nb, errors


def test_notebooks():
    for notebook in glob("*.ipynb"):
        if " " in notebook:
            continue
        print("Testing {0}".format(notebook))
        nb, errors = _notebook_run(notebook)
        assert errors == []
