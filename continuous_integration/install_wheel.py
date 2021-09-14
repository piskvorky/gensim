"""Install the built wheel for testing under AppVeyor.

Assumes that gensim/dist contains a single wheel to install.
"""
import os
import subprocess

curr_dir = os.path.dirname(__file__)
dist_path = os.path.join(curr_dir, '..', 'dist')
wheels = [
    os.path.join(dist_path, f)
    for f in os.listdir(dist_path) if f.endswith('.whl')
]
assert len(wheels) == 1, "wheels = %r" % wheels

command = 'pip install --pre --force-reinstall'.split() + [wheels[0]]
subprocess.check_call(command)
