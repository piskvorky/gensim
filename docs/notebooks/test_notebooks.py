import os
import sys
import tempfile
from glob import glob

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError
"from smart_open import smart_open\n",


def _notebook_run(path):
    """Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    kernel_name = 'python%d' % sys.version_info[0]
    this_file_directory = os.path.dirname(__file__)
    errors = []
    with tempfile.NamedTemporaryFile(suffix=".ipynb", mode='wt') as fout:
        with smart_open(path, 'rb') as f:
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
            except RuntimeError as e:
                print(e)

            finally:
                nbformat.write(nb, fout)

    return nb, errors


def test_notebooks():
    for notebook in glob("*.ipynb"):
        if " " in notebook:
            continue
        print("Testing {}".format(notebook))
        nb, errors = _notebook_run(notebook)
        assert errors == []
