# This script needs to run under both Python 2 and 3
import sys
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve
urlretrieve(sys.argv[1], sys.argv[2])
