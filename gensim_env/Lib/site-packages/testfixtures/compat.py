# compatibility module for different python versions
import sys

PY_VERSION = sys.version_info[:2]

PY_36_PLUS = PY_VERSION >= (3, 6)

if PY_VERSION > (3, 0):

    PY2 = False
    PY3 = True

    Bytes = bytes
    Unicode = str
    basestring = str
    BytesLiteral = lambda x: x.encode('latin1')
    UnicodeLiteral = lambda x: x

    class_type_name = 'class'
    ClassType = type
    exception_module = 'builtins'
    new_class = type
    self_name = '__self__'
    from io import StringIO
    xrange = range
    from itertools import zip_longest

else:

    PY2 = True
    PY3 = False

    Bytes = str
    Unicode = unicode
    basestring = basestring
    BytesLiteral = lambda x: x
    UnicodeLiteral = lambda x: x.decode('latin1')

    class_type_name = 'type'
    from types import ClassType
    exception_module = 'exceptions'
    from new import classobj as new_class
    self_name = 'im_self'
    from StringIO import StringIO
    xrange = xrange
    from itertools import izip_longest as zip_longest

try:
    from mock import call as mock_call
except ImportError:  # pragma: no cover
    class MockCall:
        pass
    mock_call = MockCall()

try:
    from unittest.mock import call as unittest_mock_call
except ImportError:
    class UnittestMockCall:
        pass
    unittest_mock_call = UnittestMockCall()
