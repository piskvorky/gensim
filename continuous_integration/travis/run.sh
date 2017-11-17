#!/bin/bash

set -e

pip freeze

if [[ "$ONLY_CODESTYLE" == "yes" ]]; then
	tox -e flake8
else
	tox -e ${PYTHON_VERSION}
fi
