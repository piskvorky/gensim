# This script needs to be able run under both Python 2 and 3 without crashing
# It only achieves the desired effect under Py3.10 on Linux and MacOS.
import subprocess
import sys
import tempfile
if sys.platform in ('linux', 'darwin') and sys.version_info[:2] == (3, 10):
    import urllib.request
    with tempfile.NamedTemporaryFile(suffix='.py') as fout:
        urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", fout.name)
        subprocess.call([sys.executable, fout.name])
