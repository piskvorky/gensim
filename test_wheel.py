#!/usr/bin/env python
"""Test a Gensim wheel stored on S3.

Downloads the wheel, installs it into a fresh working environment, and then runs gensim tests.

usage:

    python test_wheel.py <url> $(which python3.10)

where the URL comes from http://gensim-wheels.s3-website-us-east-1.amazonaws.com/
"""

import argparse
import io
import os
import subprocess
import tempfile
import urllib.parse
import urllib.request
import shutil
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))


def run(*command, **kwargs):
    print("-" * 70, file=sys.stderr)
    print(" ".join(command), file=sys.stderr)
    print("-" * 70, file=sys.stderr)
    subprocess.check_call(command, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel_path", help="The location of the wheel.  May be a URL or local path")
    parser.add_argument("python", help="Which python binary to use to test the wheel")
    parser.add_argument("--gensim-path", default=os.path.expanduser("~/git/gensim"), help="Where the gensim repo lives")
    parser.add_argument("--keep", action="store_true", help="Do not delete the sandbox after testing")
    parser.add_argument("--test", default="test", help="Specify which tests to run")
    args = parser.parse_args()

    _, python_version = subprocess.check_output([args.python, "--version"]).decode().strip().split(" ", 1)

    try:
        tmpdir = tempfile.mkdtemp(prefix=f"test_wheel-py{python_version}-")

        tmp_test_path = os.path.join(tmpdir, "test")
        shutil.copytree(os.path.join(args.gensim_path, "gensim/test"), tmp_test_path)

        if args.wheel_path.startswith("http://") or args.wheel_path.startswith("https://"):
            parsed = urllib.parse.urlparse(args.wheel_path)
            filename = parsed.path.split('/')[-1]
            wheel_path = os.path.join(tmpdir, filename)
            urllib.request.urlretrieve(args.wheel_path, wheel_path)
        else:
            wheel_path = args.wheel_path

        env_path = os.path.join(tmpdir, "env")
        run("virtualenv", "-p", args.python, env_path)

        python_exe = os.path.join(tmpdir, "env/bin/python")
        run(python_exe, "-m", "pip", "install", wheel_path)
        run(python_exe, "-m", "pip", "install", "mock", "pytest", "testfixtures")

        pytest_exe = os.path.join(tmpdir, "env/bin/pytest")
        run(pytest_exe, "-vvv", args.test, "--durations", "0", cwd=tmpdir)
    finally:
        if args.keep:
            print(f"keeping {tmpdir}, remove it yourself when done")
        else:
            shutil.rmtree(tmpdir)



if __name__ == "__main__":
    main()
