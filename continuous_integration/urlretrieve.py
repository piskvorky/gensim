# This script needs to be able run under both Python 2 and 3 without crashing
# It only achieves the desired effect under Py3.7 and above.
import sys
if sys.version_info[:2] < (3, 7):
    print("this script is a no-op for %s" % sys.version)
    sys.exit(0)
from urllib.request import urlretrieve
urlretrieve(sys.argv[1], sys.argv[2])
