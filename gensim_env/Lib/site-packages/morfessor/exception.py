from __future__ import unicode_literals


class MorfessorException(Exception):
    """Base class for exceptions in this module."""
    pass


class ArgumentException(Exception):
    """Exception in command line argument parsing."""
    pass


class InvalidCategoryError(MorfessorException):
    """Attempt to load data using a different categorization scheme."""
    def __init__(self, category):
        super(InvalidCategoryError, self).__init__(
            self, 'This model does not recognize the category {}'.format(
                category))


class InvalidOperationError(MorfessorException):
    def __init__(self, operation, function_name):
        super(InvalidOperationError, self).__init__(
            self, ('This model does not have a method {}, and therefore cannot'
                   ' perform operation "{}"').format(function_name, operation))


class UnsupportedConfigurationError(MorfessorException):
    def __init__(self, reason):
        super(UnsupportedConfigurationError, self).__init__(
            self, ('This operation is not supported in this program ' +
                   'configuration. Reason: {}.').format(reason))


class SegmentOnlyModelException(MorfessorException):
    def __init__(self):
        super(SegmentOnlyModelException, self).__init__(
            self, 'This model has been reduced to a segment-only model')
