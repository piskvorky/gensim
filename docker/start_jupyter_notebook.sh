#!/bin/bash

PORT=$1
NOTEBOOK_DIR=/gensim/docs/notebooks
DEFAULT_URL=/notebooks/gensim%20Quick%20Start.ipynb

jupyter notebook --no-browser --ip=* --port=$PORT --notebook-dir=$NOTEBOOK_DIR --NotebookApp.token=\"\" --NotebookApp.default_url=$DEFAULT_URL