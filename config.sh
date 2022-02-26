# Define custom utilities

function pre_build {
    # Any stuff that you need to do before you start building the wheels
    # Runs in the root directory of this repository.
    :
}

function pip_opts {
    [ -n "$MANYLINUX_URL" ] && echo "--find-links $MANYLINUX_URL"
    echo "-v"
}

function build_wheel_cmd {
    set -x
    local cmd=${1:-pip_wheel_cmd}
    local wheelhouse=$(abspath ${WHEEL_SDIR:-wheelhouse})
    start_spinner
    if [ -n "$(is_function "pre_build")" ]; then pre_build; fi
    stop_spinner
    if [ -n "$BUILD_DEPENDS" ]; then
	pip install $(pip_opts) $BUILD_DEPENDS
    fi
    echo "config.sh: build_wheel_cmd"
    pip install --upgrade pip setuptools
    pip --version
    pip freeze
    $cmd $wheelhouse
    repair_wheelhouse $wheelhouse
    set +x
}

function run_tests {
    # Runs tests on installed distribution from an empty directory
    set -x
    python --version
    echo "config.sh: run_tests"
    pip install --upgrade pip setuptools
    pip freeze
    pytest -rfxEXs --durations=20 --disable-warnings --showlocals --pyargs gensim
    set +x
}

function install_run {
    #
    # Overrides a function by the same name in multibuild/common_util.sh.
    # We do it here because we want to upgrade pip before installing the wheel.
    #
    pip install --upgrade pip setuptools
    install_wheel
    mkdir tmp_for_test
    (cd tmp_for_test && run_tests)
    rmdir tmp_for_test  2>/dev/null || echo "Cannot remove tmp_for_test"
}
