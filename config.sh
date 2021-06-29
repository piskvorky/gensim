# Define custom utilities

function pre_build {
    # Any stuff that you need to do before you start building the wheels
    # Runs in the root directory of this repository.
    :
}

function build_wheel_cmd {
    local cmd=${1:-pip_wheel_cmd}
    local wheelhouse=$(abspath ${WHEEL_SDIR:-wheelhouse})
    start_spinner
    if [ -n "$(is_function "pre_build")" ]; then pre_build; fi
    stop_spinner
    if [ -n "$BUILD_DEPENDS" ]; then
	pip install $(pip_opts) $BUILD_DEPENDS
    fi
    $cmd $wheelhouse
    repair_wheelhouse $wheelhouse
}

function run_tests {
    # Runs tests on installed distribution from an empty directory
    python --version
    pip freeze
    pytest -rfxEXs --durations=20 --disable-warnings --showlocals --pyargs gensim
}
