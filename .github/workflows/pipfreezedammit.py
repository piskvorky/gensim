"""Like pip-freeze, but writes to stderr instead of stdout.

Implemented as a Python script because we need it to work on Windows, etc.
"""
import os
import subprocess
import sys

pythonversion = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'

os.makedirs('pip-freeze', exist_ok=True)
with open(f'pip-freeze/{pythonversion}.txt', 'wt') as fout:
    subprocess.call([sys.executable, '-m', 'pip', 'freeze'], stdout=fout)
