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
    pip --version
    pip freeze
    $cmd $wheelhouse
    repair_wheelhouse $wheelhouse
    set +x
}

function run_tests {
    # Runs tests on installed distribution from an empty directory
    set -x
    pip freeze
    pytest -rfxEXs --durations=20 --disable-warnings --showlocals --pyargs gensim
    set +x
}

#
# We do this here because we want to upgrade pip before the wheel gets installed.
# docker_test_wrap.sh sources this file before the wheel install.  The sourcing
# happens from multiple places, and some of the Python versions can be really
# ancient (e.g. when working outside a virtual environment, using the default
# Python install).
#
# We don't use pip to do the actual upgrade because something appears broken
# with the default pip on the Python 3.10 multibuild image.  This is really
# dodgy, but I couldn't work out a better way to get this done.
#
python continuous_integration/upgrade_pip_py310.py
