#!/bin/bash

set -e

pip freeze
python -c "import platform, sys, numpy, scipy, gensim; from gensim.models import word2vec; print('Platform: {}\nPython: {}\nNumpy: {}\nSciPy: {}\ngensim: {}\nFAST_VERSION: {}'.format(platform.platform(), sys.version, numpy.__version__, scipy.__version__, gensim.__version__, word2vec.FAST_VERSION))"

if [[ "$ONLY_CODESTYLE" == "yes" ]]; then
	continuous_integration/travis/flake8_diff.sh
else
	pytest -v gensim/test/
fi
