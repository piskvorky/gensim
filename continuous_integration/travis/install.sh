#!/bin/bash

set -e

deactivate
wget 'http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh' -O miniconda.sh
chmod +x miniconda.sh && ./miniconda.sh -b
export PATH=/home/travis/miniconda2/bin:$PATH
conda update --yes conda


conda create --yes -n gensim-test python=${PYTHON_VERSION} pip atlas flake8 jupyter numpy==${NUMPY_VERSION} scipy==${SCIPY_VERSION} && source activate gensim-test
pip install . && pip install .[test]
