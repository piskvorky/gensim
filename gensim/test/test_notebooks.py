import sys
import tempfile
from glob import glob

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

import logging
import unittest
import os
import os.path
import tempfile

import numbers 

def _notebook_run(path):
    """
    Execute a notebook via nbconvert and collect output.
       :returns (parsed nb object, execution errors)
    """
    kernel_name = 'python%d' % sys.version_info[0]
    this_file_directory = os.path.dirname(__file__)
    errors = []

    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
        nb.metadata.get('kernelspec', {})['name'] = kernel_name
        ep = ExecutePreprocessor(kernel_name=kernel_name, timeout=300) # 

        try:
            ep.preprocess(nb, {'metadata': {'path': this_file_directory}})

        except CellExecutionError as e: 
            if "SKIP" in e.traceback:
                print(str(e.traceback).split("\n")[-2])
            else:
                raise e

    return nb, errors


class TestNotebooks(unittest.TestCase):
    def test_notebooks(self):
        num_notebooks_tested = 0
        for notebook in glob("./docs/notebooks/*.ipynb"):
            # temporary: ignore these files as they need to be corrected
            temporary_not_tested = ["topic_methods.ipynb", "distance_metrics.ipynb", "doc2vec-IMDB.ipynb", "lda_training_tips.ipynb", "Corpora_and_Vector_Spaces.ipynb", "online_w2v_tutorial.ipynb", "gensim_news_classification.ipynb", "Word2Vec_FastText_Comparison.ipynb"]
            
            if " " in notebook: 
                continue
            
            flag = 0
            for nb in temporary_not_tested:
                if nb in notebook:
                    flag = 1
                    break
            if flag == 1:
                continue
            print("Testing {}".format(notebook))
            nb, errors = _notebook_run(notebook)
            num_notebooks_tested = num_notebooks_tested + 1
            assert errors == []

        print('Tested {} notebooks in gensim/docs/notebooks').format(num_notebooks_tested)

#endclass TestNotebooks

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()


