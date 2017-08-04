#!/bin/bash

set -e

pip freeze

if [[ "$ONLY_CODESTYLE" == "yes" ]]; then
	continuous_integration/travis/flake8_diff.sh
else
	pytest -v gensim/test/
fi
