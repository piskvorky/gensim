"""Like pip-freeze, but writes to stderr instead of stdout.

Implemented as a Python script because we need it to work on Windows, etc.
"""
import subprocess
import sys
subprocess.call([sys.executable, '-m', 'pip', 'freeze'], stdout=sys.stderr)
